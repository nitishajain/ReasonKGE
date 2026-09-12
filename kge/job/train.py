import itertools
import os
import math
import time
import traceback
from collections import defaultdict
import json
from pathlib import Path
import os.path as path
import csv
import glob

from dataclasses import dataclass

import torch
import torch.utils.data
import numpy as np
import pandas as pd

from kge import Config, Dataset
from kge.job import Job, TrainingOrEvaluationJob
from kge.model import KgeModel

from kge.util import KgeLoss, KgeOptimizer, KgeSampler, KgeLRScheduler
from kge.util.io import load_checkpoint
from kge.job.trace import format_trace_entry
from typing import Any, Callable, Dict, List, Optional
import kge.job.util
from kge.util.metric import Metric

from kge.util.sampler import DefaultBatchReasoningSample
from kge.util import KgeSamplerReasoning

from kge.job.pyowlapi import *


SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]


def _generate_worker_init_fn(config):
    "Initialize workers of a DataLoader"
    use_fixed_seed = config.get("random_seed.numpy") >= 0

    def worker_init_fn(worker_num):
        # ensure that NumPy uses different seeds at each worker
        if use_fixed_seed:
            # reseed based on current seed (same for all workers) and worker number
            # (different)
            base_seed = np.random.randint(2 ** 32 - 1)
            np.random.seed(base_seed + worker_num)
        else:
            # reseed fresh
            np.random.seed()

    return worker_init_fn


class TrainingJob(TrainingOrEvaluationJob):
    """Abstract base job to train a single model with a fixed set of hyperparameters.

    Also used by jobs such as :class:`SearchJob`.

    Subclasses for specific training methods need to implement `_prepare` and
    `_process_batch`.

    """


    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        parent_job: Job = None,
        model=None,
        forward_only=False,
    ) -> None:
        from kge.job import EvaluationJob

        super().__init__(config, dataset, parent_job)
        if model is None:
            self.model: KgeModel = KgeModel.create(config, dataset)
        else:
            self.model: KgeModel = model
        self.loss = KgeLoss.create(config)
        self.abort_on_nan: bool = config.get("train.abort_on_nan")
        self.batch_size: int = config.get("train.batch_size")
        self._subbatch_auto_tune: bool = config.get("train.subbatch_auto_tune")
        self._max_subbatch_size: int = config.get("train.subbatch_size")
        self.device: str = self.config.get("job.device")
        self.train_split = config.get("train.split")
        self.valid_split = config.get("valid.split")
        self.eval_split = config.get("eval.split")


        self.config.check("train.trace_level", ["batch", "epoch"])
        self.trace_batch: bool = self.config.get("train.trace_level") == "batch"
        self.epoch: int = 0
        self.is_forward_only = forward_only
        self.predict = None

        if not self.is_forward_only:
            self.model.train()
            self.optimizer = KgeOptimizer.create(config, self.model)
            self.kge_lr_scheduler = KgeLRScheduler(config, self.optimizer)

            self.valid_trace: List[Dict[str, Any]] = []
            valid_conf = config.clone()
            valid_conf.set("job.type", "eval")
            if self.config.get("valid.split") != "":
                valid_conf.set("eval.split", self.config.get("valid.split"))
            valid_conf.set("eval.trace_level", self.config.get("valid.trace_level"))
            self.valid_job = EvaluationJob.create(
                valid_conf, dataset, parent_job=self, model=self.model
            )

        # attributes filled in by implementing classes
        self.loader = None
        self.num_examples = None
        self.type_str: Optional[str] = None

        # Hooks run after validation. The corresponding valid trace entry can be found
        # in self.valid_trace[-1] Signature: job
        self.post_valid_hooks: List[Callable[[Job], Any]] = []

        if self.__class__ == TrainingJob:
            for f in Job.job_created_hooks:
                f(self)

    @staticmethod
    def create(
        config: Config,
        dataset: Dataset,
        parent_job: Job = None,
        model=None,
        forward_only=False,
    ) -> "TrainingJob":
        """Factory method to create a training job."""
        if config.get("train.type") == "KvsAll":
            return TrainingJobKvsAll(
                config, dataset, parent_job, model=model, forward_only=forward_only
            )
        elif config.get("train.type") == "negative_sampling":
            return TrainingJobNegativeSampling(
                config, dataset, parent_job, model=model, forward_only=forward_only
            )
        elif config.get("train.type") == "1vsAll":
            return TrainingJob1vsAll(
                config, dataset, parent_job, model=model, forward_only=forward_only
            )
        #elif config.get("train.type") == "reasoning_sampling.True" or "reasoning_sampling.False":
        elif config.get("train.type") == "reasoning_sampling.True.Predict" or "reasoning_sampling.True" or "reasoning_sampling.False":
            print("Reasoning_sampling")
            abstraction = config.get("train.type").split(".")[1]
            predict = None
            if len(config.get("train.type").split("."))>2:
                predict = config.get("train.type").split(".")[2]
                print("Predict:", predict)
            if (abstraction):
                print (abstraction)
            return TrainingJobReasoningSampling(
                config, dataset, abstraction, predict, parent_job, model=model, forward_only=forward_only)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("train.type")

    def _run(self) -> None:
        """Start/resume the training job and run to completion."""

        if self.is_forward_only:
            raise Exception(
                f"{self.__class__.__name__} was initialized for forward only. You can only call run_epoch()"
            )

        self.config.log("Starting training...")
        checkpoint_every = self.config.get("train.checkpoint.every")
        checkpoint_keep = self.config.get("train.checkpoint.keep")
        metric_name = self.config.get("valid.metric")
        patience = self.config.get("valid.early_stopping.patience")
        while True:
            # checking for model improvement according to metric_name
            # and do early stopping and keep the best checkpoint
            if (
                len(self.valid_trace) > 0
                and self.valid_trace[-1]["epoch"] == self.epoch
            ):
                best_index = Metric(self).best_index(
                    list(map(lambda trace: trace[metric_name], self.valid_trace))
                )
                if best_index == len(self.valid_trace) - 1:
                    self.save(self.config.checkpoint_file("best"))
                if (
                    patience > 0
                    and len(self.valid_trace) > patience
                    and best_index < len(self.valid_trace) - patience
                ):
                    self.config.log(
                        "Stopping early ({} did not improve over best result ".format(
                            metric_name
                        )
                        + "in the last {} validation runs).".format(patience)
                    )
                    break
                if self.epoch > self.config.get(
                    "valid.early_stopping.threshold.epochs"
                ):
                    achieved = self.valid_trace[best_index][metric_name]
                    target = self.config.get(
                        "valid.early_stopping.threshold.metric_value"
                    )
                    if Metric(self).better(target, achieved):
                        self.config.log(
                            "Stopping early ({} did not achieve threshold after {} epochs".format(
                                metric_name, self.epoch
                            )
                        )
                        break

            # should we stop?
            if self.epoch >= self.config.get("train.max_epochs"):
                self.config.log("Maximum number of epochs reached.")
                break


            if len(self.config.get("train.type").split("."))>2:
                self.predict = self.config.get("train.type").split(".")[2]
                print("Predict:", self.predict)

            # start a new epoch
            self.epoch += 1
            if self.epoch > 1 :
                if self.predict:
                    print("First epoch finished ..exiting..")
                    print("Predict...", self.predict)
                    break

            #print("Predict...", self.predict)
            self.config.log("Starting epoch {}...".format(self.epoch))
            trace_entry = self.run_epoch()
            self.config.log("Finished epoch {}.".format(self.epoch))

            # update model metadata
            self.model.meta["train_job_trace_entry"] = self.trace_entry
            self.model.meta["train_epoch"] = self.epoch
            self.model.meta["train_config"] = self.config
            self.model.meta["train_trace_entry"] = trace_entry

            # validate and update learning rate
            if (
                self.config.get("valid.every") > 0
                and self.epoch % self.config.get("valid.every") == 0
            ):
                self.valid_job.epoch = self.epoch
                trace_entry = self.valid_job.run()
                self.valid_trace.append(trace_entry)
                for f in self.post_valid_hooks:
                    f(self)
                self.model.meta["valid_trace_entry"] = trace_entry

                # metric-based scheduler step
                self.kge_lr_scheduler.step(trace_entry[metric_name])
            else:
                self.kge_lr_scheduler.step()

            # create checkpoint and delete old one, if necessary
            self.save(self.config.checkpoint_file(self.epoch))
            if self.epoch > 1:
                delete_checkpoint_epoch = -1
                if checkpoint_every == 0:
                    # do not keep any old checkpoints
                    delete_checkpoint_epoch = self.epoch - 1
                elif (self.epoch - 1) % checkpoint_every != 0:
                    # delete checkpoints that are not in the checkpoint.every schedule
                    delete_checkpoint_epoch = self.epoch - 1
                elif checkpoint_keep > 0:
                    # keep a maximum number of checkpoint_keep checkpoints
                    delete_checkpoint_epoch = (
                        self.epoch - 1 - checkpoint_every * checkpoint_keep
                    )
                if delete_checkpoint_epoch > 0:
                    if os.path.exists(
                        self.config.checkpoint_file(delete_checkpoint_epoch)
                    ):
                        self.config.log(
                            "Removing old checkpoint {}...".format(
                                self.config.checkpoint_file(delete_checkpoint_epoch)
                            )
                        )
                        os.remove(self.config.checkpoint_file(delete_checkpoint_epoch))
                    else:
                        self.config.log(
                            "Could not delete old checkpoint {}, does not exits.".format(
                                self.config.checkpoint_file(delete_checkpoint_epoch)
                            )
                        )

        self.trace(event="train_completed")

    def save(self, filename) -> None:
        """Save current state to specified file"""
        self.config.log("Saving checkpoint to {}...".format(filename))
        checkpoint = self.save_to({})
        torch.save(
            checkpoint, filename,
        )

    def save_to(self, checkpoint: Dict) -> Dict:
        """Adds trainjob specific information to the checkpoint"""
        train_checkpoint = {
            "type": "train",
            "epoch": self.epoch,
            "valid_trace": self.valid_trace,
            "model": self.model.save(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.kge_lr_scheduler.state_dict(),
            "job_id": self.job_id,
        }
        train_checkpoint = self.config.save_to(train_checkpoint)
        checkpoint.update(train_checkpoint)
        return checkpoint

    def _load(self, checkpoint: Dict) -> str:
        if checkpoint["type"] != "train":
            raise ValueError("Training can only be continued on trained checkpoints")
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in checkpoint:
            # new format
            self.kge_lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.valid_trace = checkpoint["valid_trace"]
        self.model.train()
        self.resumed_from_job_id = checkpoint.get("job_id")
        self.trace(
            event="job_resumed", epoch=self.epoch, checkpoint_file=checkpoint["file"],
        )
        self.config.log(
            "Resuming training from {} of job {}".format(
                checkpoint["file"], self.resumed_from_job_id
            )
        )

    def run_epoch(self) -> Dict[str, Any]:
        """ Runs an epoch and returns its trace entry. """

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type=self.type_str,
            scope="epoch",
            epoch=self.epoch,
            split=self.train_split,
            batches=len(self.loader),
            size=self.num_examples,
        )
        if not self.is_forward_only:
            self.current_trace["epoch"].update(
                lr=[group["lr"] for group in self.optimizer.param_groups],
            )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

        # variables that record various statitics
        sum_loss = 0.0
        sum_penalty = 0.0
        sum_penalties = defaultdict(lambda: 0.0)
        epoch_time = -time.time()
        prepare_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        optimizer_time = 0.0

        # process each batch
        for batch_index, batch in enumerate(self.loader):
            # create initial batch trace (yet incomplete)
            self.current_trace["batch"] = {
                "type": self.type_str,
                "scope": "batch",
                "epoch": self.epoch,
                "split": self.train_split,
                "batch": batch_index,
                "batches": len(self.loader),
            }
            if not self.is_forward_only:
                self.current_trace["batch"].update(
                    lr=[group["lr"] for group in self.optimizer.param_groups],
                )

            # run the pre-batch hooks (may update the trace)
            for f in self.pre_batch_hooks:
                f(self)

            # process batch (preprocessing + forward pass + backward pass on loss)
            done = False
            while not done:
                try:
                    # try running the batch
                    if not self.is_forward_only:
                        self.optimizer.zero_grad()
                    batch_result: TrainingJob._ProcessBatchResult = self._process_batch(
                        batch_index, batch
                    )
                    done = True
                except RuntimeError as e:
                    # is it a CUDA OOM exception and are we allowed to reduce the
                    # subbatch size on such an error? if not, raise the exception again
                    if (
                        "CUDA out of memory" not in str(e)
                        or not self._subbatch_auto_tune
                    ):
                        raise e

                    # try rerunning with smaller subbatch size
                    tb = traceback.format_exc()
                    self.config.log(tb)
                    self.config.log(
                        "Caught OOM exception when running a batch; "
                        "trying to reduce the subbatch size..."
                    )

                    if self._max_subbatch_size <= 0:
                        self._max_subbatch_size = self.batch_size
                    if self._max_subbatch_size <= 1:
                        self.config.log(
                            "Cannot reduce subbatch size "
                            f"(current value: {self._max_subbatch_size})"
                        )
                        raise e  # cannot reduce further

                    self._max_subbatch_size //= 2
                    self.config.set(
                        "train.subbatch_size", self._max_subbatch_size, log=True
                    )
            sum_loss += batch_result.avg_loss * batch_result.size

            # determine penalty terms (forward pass)
            batch_forward_time = batch_result.forward_time - time.time()
            penalties_torch = self.model.penalty(
                epoch=self.epoch,
                batch_index=batch_index,
                num_batches=len(self.loader),
                batch=batch,
            )
            batch_forward_time += time.time()

            # backward pass on penalties
            batch_backward_time = batch_result.backward_time - time.time()
            penalty = 0.0
            for index, (penalty_key, penalty_value_torch) in enumerate(penalties_torch):
                if not self.is_forward_only:
                    penalty_value_torch.backward()
                penalty += penalty_value_torch.item()
                sum_penalties[penalty_key] += penalty_value_torch.item()
            sum_penalty += penalty
            batch_backward_time += time.time()

            # determine full cost
            cost_value = batch_result.avg_loss + penalty

            # abort on nan
            if self.abort_on_nan and math.isnan(cost_value):
                raise FloatingPointError("Cost became nan, aborting training job")

            # TODO # visualize graph
            # if (
            #     self.epoch == 1
            #     and batch_index == 0
            #     and self.config.get("train.visualize_graph")
            # ):
            #     from torchviz import make_dot

            #     f = os.path.join(self.config.folder, "cost_value")
            #     graph = make_dot(cost_value, params=dict(self.model.named_parameters()))
            #     graph.save(f"{f}.gv")
            #     graph.render(f)  # needs graphviz installed
            #     self.config.log("Exported compute graph to " + f + ".{gv,pdf}")

            # print memory stats
            if self.epoch == 1 and batch_index == 0:
                if self.device.startswith("cuda"):
                    self.config.log(
                        "CUDA memory after first batch: allocated={:14,} "
                        "reserved={:14,} max_allocated={:14,}".format(
                            torch.cuda.memory_allocated(self.device),
                            torch.cuda.memory_reserved(self.device),
                            torch.cuda.max_memory_allocated(self.device),
                        )
                    )

            # update parameters
            batch_optimizer_time = -time.time()
            if not self.is_forward_only:
                self.optimizer.step()
            batch_optimizer_time += time.time()

            # update batch trace with the results
            self.current_trace["batch"].update(
                {
                    "size": batch_result.size,
                    "avg_loss": batch_result.avg_loss,
                    "penalties": [p.item() for k, p in penalties_torch],
                    "penalty": penalty,
                    "cost": cost_value,
                    "prepare_time": batch_result.prepare_time,
                    "forward_time": batch_forward_time,
                    "backward_time": batch_backward_time,
                    "optimizer_time": batch_optimizer_time,
                    "event": "batch_completed",
                }
            )

            # run the post-batch hooks (may modify the trace)
            for f in self.post_batch_hooks:
                f(self)

            # output, then clear trace
            if self.trace_batch:
                self.trace(**self.current_trace["batch"])
            self.current_trace["batch"] = None

            # print console feedback
            self.config.print(
                (
                    "\r"  # go back
                    + "{}  batch{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{}"
                    + ", avg_loss {:.4E}, penalty {:.4E}, cost {:.4E}, time {:6.2f}s"
                    + "\033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    batch_index,
                    len(self.loader) - 1,
                    batch_result.avg_loss,
                    penalty,
                    cost_value,
                    batch_result.prepare_time
                    + batch_forward_time
                    + batch_backward_time
                    + batch_optimizer_time,
                ),
                end="",
                flush=True,
            )

            # update epoch times
            prepare_time += batch_result.prepare_time
            forward_time += batch_forward_time
            backward_time += batch_backward_time
            optimizer_time += batch_optimizer_time

        # all done; now trace and log
        epoch_time += time.time()
        self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back

        other_time = (
            epoch_time - prepare_time - forward_time - backward_time - optimizer_time
        )

        # add results to trace entry
        self.current_trace["epoch"].update(
            dict(
                avg_loss=sum_loss / self.num_examples,
                avg_penalty=sum_penalty / len(self.loader),
                avg_penalties={
                    k: p / len(self.loader) for k, p in sum_penalties.items()
                },
                avg_cost=sum_loss / self.num_examples + sum_penalty / len(self.loader),
                epoch_time=epoch_time,
                prepare_time=prepare_time,
                forward_time=forward_time,
                backward_time=backward_time,
                optimizer_time=optimizer_time,
                other_time=other_time,
                event="epoch_completed",
            )
        )

        # run hooks (may modify trace)
        for f in self.post_epoch_hooks:
            f(self)

        # output the trace, then clear it
        trace_entry = self.trace(**self.current_trace["epoch"], echo=False, log=True)
        self.config.log(
            format_trace_entry("train_epoch", trace_entry, self.config), prefix="  "
        )
        self.current_trace["epoch"] = None

        return trace_entry

    def _prepare(self):
        """Prepare this job for running.

        Sets (at least) the `loader`, `num_examples`, and `type_str` attributes of this
        job to a data loader, number of examples per epoch, and a name for the trainer,
        repectively.

        Guaranteed to be called exactly once before running the first epoch.

        """
        super()._prepare()
        self.model.prepare_job(self)  # let the model add some hooks

    @dataclass
    class _ProcessBatchResult:
        """Result of running forward+backward pass on a batch."""

        avg_loss: float = 0.0
        size: int = 0
        prepare_time: float = 0.0
        forward_time: float = 0.0
        backward_time: float = 0.0

    def _process_batch(self, batch_index, batch) -> _ProcessBatchResult:
        "Breaks a batch into subbatches and processes them in turn."
        result = TrainingJob._ProcessBatchResult()
        self._prepare_batch(batch_index, batch, result)
        batch_size = result.size

        max_subbatch_size = (
            self._max_subbatch_size if self._max_subbatch_size > 0 else batch_size
        )
        for subbatch_start in range(0, batch_size, max_subbatch_size):
            # determine data used for this subbatch
            subbatch_end = min(subbatch_start + max_subbatch_size, batch_size)
            subbatch_slice = slice(subbatch_start, subbatch_end)
            self._process_subbatch(batch_index, batch, subbatch_slice, result)

        return result

    def _prepare_batch(self, batch_index, batch, result: _ProcessBatchResult):
        """Prepare the given batch for processing and determine the batch size.

        batch size must be written into result.size.
        """
        raise NotImplementedError

    def _process_subbatch(
        self, batch_index, batch, subbatch_slice, result: _ProcessBatchResult,
    ):
        """Run forward and backward pass on the given subbatch.

        Also update result.

        """
        raise NotImplementedError


class TrainingJobKvsAll(TrainingJob):
    """Train with examples consisting of a query and its answers.

    Terminology:
    - Query type: which queries to ask (sp_, s_o, and/or _po), can be configured via
      configuration key `KvsAll.query_type` (which see)
    - Query: a particular query, e.g., (John,marriedTo) of type sp_
    - Labels: list of true answers of a query (e.g., [Jane])
    - Example: a query + its labels, e.g., (John,marriedTo), [Jane]
    """

    from kge.indexing import KvsAllIndex

    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        self.label_smoothing = config.check_range(
            "KvsAll.label_smoothing", float("-inf"), 1.0, max_inclusive=False
        )
        if self.label_smoothing < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting label_smoothing to 0, "
                    "was set to {}.".format(self.label_smoothing)
                )
                self.label_smoothing = 0
            else:
                raise Exception(
                    "Label_smoothing was set to {}, "
                    "should be at least 0.".format(self.label_smoothing)
                )
        elif self.label_smoothing > 0 and self.label_smoothing <= (
            1.0 / dataset.num_entities()
        ):
            if config.get("train.auto_correct"):
                # just to be sure it's used correctly
                config.log(
                    "Setting label_smoothing to 1/num_entities = {}, "
                    "was set to {}.".format(
                        1.0 / dataset.num_entities(), self.label_smoothing
                    )
                )
                self.label_smoothing = 1.0 / dataset.num_entities()
            else:
                raise Exception(
                    "Label_smoothing was set to {}, "
                    "should be at least {}.".format(
                        self.label_smoothing, 1.0 / dataset.num_entities()
                    )
                )

        config.log("Initializing 1-to-N training job...")
        self.type_str = "KvsAll"

        if self.__class__ == TrainingJobKvsAll:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()
        # determine enabled query types
        self.query_types = [
            key
            for key, enabled in self.config.get("KvsAll.query_types").items()
            if enabled
        ]

        # corresponding indexes
        self.query_indexes: List[KvsAllIndex] = []

        #' for each query type (ordered as in self.query_types), index right after last
        #' example of that type in the list of all examples (over all query types)
        self.query_last_example = []

        # construct relevant data structures
        self.num_examples = 0
        for query_type in self.query_types:
            index_type = (
                "sp_to_o"
                if query_type == "sp_"
                else ("so_to_p" if query_type == "s_o" else "po_to_s")
            )
            index = self.dataset.index(f"{self.train_split}_{index_type}")
            self.query_indexes.append(index)
            self.num_examples += len(index)
            self.query_last_example.append(self.num_examples)

        # create dataloader
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _get_collate_fun(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a dictionary of:

            - queries: nx2 tensor, row = query (sp, po, or so indexes)
            - label_coords: for each query, position of true answers (an Nx2 tensor,
              first columns holds query index, second colum holds index of label)
            - query_type_indexes (vector of size n holding the query type of each query)
            - triples (all true triples in the batch; e.g., needed for weighted
              penalties)

            """

            # count how many labels we have across the entire batch
            num_ones = 0
            for example_index in batch:
                start = 0
                for query_type_index in range(len(self.query_types)):
                    end = self.query_last_example[query_type_index]
                    if example_index < end:
                        example_index -= start
                        num_ones += self.query_indexes[query_type_index]._values_offset[
                            example_index + 1
                        ]
                        num_ones -= self.query_indexes[query_type_index]._values_offset[
                            example_index
                        ]
                        break
                    start = end

            # now create the batch elements
            queries_batch = torch.zeros([len(batch), 2], dtype=torch.long)
            query_type_indexes_batch = torch.zeros([len(batch)], dtype=torch.long)
            label_coords_batch = torch.zeros([num_ones, 2], dtype=torch.int)
            triples_batch = torch.zeros([num_ones, 3], dtype=torch.long)
            current_index = 0
            for batch_index, example_index in enumerate(batch):
                start = 0
                for query_type_index, query_type in enumerate(self.query_types):
                    end = self.query_last_example[query_type_index]
                    if example_index < end:
                        example_index -= start
                        query_type_indexes_batch[batch_index] = query_type_index
                        queries = self.query_indexes[query_type_index]._keys
                        label_offsets = self.query_indexes[
                            query_type_index
                        ]._values_offset
                        labels = self.query_indexes[query_type_index]._values
                        if query_type == "sp_":
                            query_col_1, query_col_2, target_col = S, P, O
                        elif query_type == "s_o":
                            query_col_1, target_col, query_col_2 = S, P, O
                        else:
                            target_col, query_col_1, query_col_2 = S, P, O
                        break
                    start = end

                queries_batch[batch_index,] = queries[example_index]
                start = label_offsets[example_index]
                end = label_offsets[example_index + 1]
                size = end - start
                label_coords_batch[
                    current_index : (current_index + size), 0
                ] = batch_index
                label_coords_batch[current_index : (current_index + size), 1] = labels[
                    start:end
                ]
                triples_batch[
                    current_index : (current_index + size), query_col_1
                ] = queries[example_index][0]
                triples_batch[
                    current_index : (current_index + size), query_col_2
                ] = queries[example_index][1]
                triples_batch[
                    current_index : (current_index + size), target_col
                ] = labels[start:end]
                current_index += size

            # all done
            return {
                "queries": queries_batch,
                "label_coords": label_coords_batch,
                "query_type_indexes": query_type_indexes_batch,
                "triples": triples_batch,
            }

        return collate

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        # move labels to GPU for entire batch (else somewhat costly, but this should be
        # reasonably small)
        result.prepare_time -= time.time()
        batch["label_coords"] = batch["label_coords"].to(self.device)
        result.size = len(batch["queries"])
        result.prepare_time += time.time()

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        # prepare
        result.prepare_time -= time.time()
        queries_subbatch = batch["queries"][subbatch_slice].to(self.device)
        batch_size = len(batch["queries"])
        subbatch_size = len(queries_subbatch)
        label_coords_batch = batch["label_coords"]
        query_type_indexes_subbatch = batch["query_type_indexes"][subbatch_slice]

        # in this method, example refers to the index of an example in the batch, i.e.,
        # it takes values in 0,1,...,batch_size-1
        examples_for_query_type = {}
        for query_type_index, query_type in enumerate(self.query_types):
            examples_for_query_type[query_type] = (
                (query_type_indexes_subbatch == query_type_index)
                .nonzero(as_tuple=False)
                .to(self.device)
                .view(-1)
            )

        labels_subbatch = kge.job.util.coord_to_sparse_tensor(
            subbatch_size,
            max(self.dataset.num_entities(), self.dataset.num_relations()),
            label_coords_batch,
            self.device,
            row_slice=subbatch_slice,
        ).to_dense()
        labels_for_query_type = {}
        for query_type, examples in examples_for_query_type.items():
            if query_type == "s_o":
                labels_for_query_type[query_type] = labels_subbatch[
                    examples, : self.dataset.num_relations()
                ]
            else:
                labels_for_query_type[query_type] = labels_subbatch[
                    examples, : self.dataset.num_entities()
                ]

        if self.label_smoothing > 0.0:
            # as in ConvE: https://github.com/TimDettmers/ConvE
            for query_type, labels in labels_for_query_type.items():
                if query_type != "s_o":  # entity targets only for now
                    labels_for_query_type[query_type] = (
                        1.0 - self.label_smoothing
                    ) * labels + 1.0 / labels.size(1)

        result.prepare_time += time.time()

        # forward/backward pass (sp)
        for query_type, examples in examples_for_query_type.items():
            if len(examples) > 0:
                result.forward_time -= time.time()
                if query_type == "sp_":
                    scores = self.model.score_sp(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )
                elif query_type == "s_o":
                    scores = self.model.score_so(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )
                else:
                    scores = self.model.score_po(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )
                # note: average on batch_size, not on subbatch_size
                loss_value = (
                    self.loss(scores, labels_for_query_type[query_type]) / batch_size
                )
                result.avg_loss += loss_value.item()
                result.forward_time += time.time()
                result.backward_time -= time.time()
                if not self.is_forward_only:
                    loss_value.backward()
                result.backward_time += time.time()



class TrainingJob1vsAll(TrainingJob):
    """Samples SPO pairs and queries sp_ and _po, treating all other entities as negative."""

    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        config.log("Initializing spo training job...")
        self.type_str = "1vsAll"

        if self.__class__ == TrainingJob1vsAll:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=lambda batch: {
                "triples": self.dataset.split(self.train_split)[batch, :].long()
            },
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        result.size = len(batch["triples"])

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice].to(self.device)
        batch_size = len(triples)
        result.prepare_time += time.time()

        # forward/backward pass (sp)
        result.forward_time -= time.time()
        scores_sp = self.model.score_sp(triples[:, 0], triples[:, 1])
        loss_value_sp = self.loss(scores_sp, triples[:, 2]) / batch_size
        result.avg_loss += loss_value_sp.item()
        result.forward_time += time.time()
        result.backward_time = -time.time()
        if not self.is_forward_only:
            loss_value_sp.backward()
        result.backward_time += time.time()

        # forward/backward pass (po)
        result.forward_time -= time.time()
        scores_po = self.model.score_po(triples[:, 1], triples[:, 2])
        loss_value_po = self.loss(scores_po, triples[:, 0]) / batch_size
        result.avg_loss += loss_value_po.item()
        result.forward_time += time.time()
        result.backward_time -= time.time()
        if not self.is_forward_only:
            loss_value_po.backward()
        result.backward_time += time.time()



class TrainingJobNegativeSampling(TrainingJob):
    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        self._sampler = KgeSampler.create(config, "negative_sampling", dataset)
        self._implementation = self.config.check(
            "negative_sampling.implementation", ["triple", "all", "batch", "auto"],
        )
        if self._implementation == "auto":
            max_nr_of_negs = max(self._sampler.num_samples)
            if self._sampler.shared:
                self._implementation = "batch"
            elif max_nr_of_negs <= 30:
                self._implementation = "triple"
            elif max_nr_of_negs > 30:
                self._implementation = "batch"

        config.log(
            "Initializing negative sampling training job with "
            "'{}' scoring function ...".format(self._implementation)
        )
        self.type_str = "negative_sampling"

        if self.__class__ == TrainingJobNegativeSampling:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _get_collate_fun(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a tuple of:

            - triples (tensor of shape [n,3], ),
            - negative_samples (list of tensors of shape [n,num_samples]; 3 elements
              in order S,P,O)
            """

            triples = self.dataset.split(self.train_split)[batch, :].long()
            # labels = torch.zeros((len(batch), self._sampler.num_negatives_total + 1))
            # labels[:, 0] = 1
            # labels = labels.view(-1)

            negative_samples = list()
            for slot in [S, P, O]:
                negative_samples.append(self._sampler.sample(triples, slot))
            return {"triples": triples, "negative_samples": negative_samples}

        return collate

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        # move triples and negatives to GPU. With some implementaiton effort, this may
        # be avoided.
        result.prepare_time -= time.time()
        batch["triples"] = batch["triples"].to(self.device)
        for ns in batch["negative_samples"]:
            ns.positive_triples = batch["triples"]
        batch["negative_samples"] = [
            ns.to(self.device) for ns in batch["negative_samples"]
        ]

        batch["labels"] = [None] * 3  # reuse label tensors b/w subbatches
        result.size = len(batch["triples"])
        result.prepare_time += time.time()

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice]
        batch_negative_samples = batch["negative_samples"]
        batch_size = len(batch["triples"])
        subbatch_size = len(triples)
        result.prepare_time += time.time()
        labels = batch["labels"]  # reuse b/w subbatches

        # process the subbatch for each slot separately
        for slot in [S, P, O]:
            num_samples = self._sampler.num_samples[slot]
            if num_samples <= 0:
                continue

            # construct gold labels: first column corresponds to positives,
            # remaining columns to negatives
            if labels[slot] is None or labels[slot].shape != (
                subbatch_size,
                1 + num_samples,
            ):
                result.prepare_time -= time.time()
                labels[slot] = torch.zeros(
                    (subbatch_size, 1 + num_samples), device=self.device
                )
                labels[slot][:, 0] = 1
                result.prepare_time += time.time()

            # compute the scores
            result.forward_time -= time.time()
            scores = torch.empty((subbatch_size, num_samples + 1), device=self.device)
            scores[:, 0] = self.model.score_spo(
                triples[:, S], triples[:, P], triples[:, O], direction=SLOT_STR[slot],
            )
            result.forward_time += time.time()
            scores[:, 1:] = batch_negative_samples[slot].score(
                self.model, indexes=subbatch_slice
            )
            result.forward_time += batch_negative_samples[slot].forward_time
            result.prepare_time += batch_negative_samples[slot].prepare_time

            # compute loss for slot in subbatch (concluding the forward pass)
            result.forward_time -= time.time()
            loss_value_torch = (
                self.loss(scores, labels[slot], num_negatives=num_samples) / batch_size
            )
            result.avg_loss += loss_value_torch.item()
            result.forward_time += time.time()

            # backward pass for this slot in the subbatch
            result.backward_time -= time.time()
            if not self.is_forward_only:
                loss_value_torch.backward()
            result.backward_time += time.time()


class TrainingJobReasoningSampling(TrainingJob):
    def __init__(
        self, config, dataset, abstraction=None, predict=None, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        self._sampler = KgeSamplerReasoning.create(config, "negative_sampling", dataset)
        self._implementation = self.config.check(
            "negative_sampling.implementation", ["triple", "all", "batch", "auto"],
        )

        if self._implementation == "auto":
            max_nr_of_negs = max(self._sampler.num_samples)
            if self._sampler.shared:
                self._implementation = "batch"
            elif max_nr_of_negs <= 30:
                self._implementation = "triple"
            elif max_nr_of_negs > 30:
                self._implementation = "batch"

        config.log(
            "Initializing reasoning sampling training job with "
            "'{}' scoring function ...".format(self._implementation)
        )
        self.type_str = "reasoning_sampling"

        #print("Value of epoch received", epoch)

        if self.__class__ == TrainingJobReasoningSampling:
            for f in Job.job_created_hooks:
                f(self)

        self.abstraction = abstraction
        print("predict:", predict)
        if (predict):
            self.first = True
        else:
            self.first = False
        print("First epoch:", self.first)

        #initialize all parametrs from params file
        name = config.get("dataset.name")
        print(name)

        parent_folder = path.abspath(path.join(__file__, "../../.."))
        #print(parent_folder)
        #print(config.folder)

        self.folder_name = parent_folder+'/'+ config.folder
        params_file = os.path.join(self.folder_name, "params.json")
        print(params_file)

        self.params = json.load(open(params_file, 'r'))
        #print(self.params)

        # no of times prediction was found in saved dict
        self.found_predict = None
        self.predict_file = os.path.join(self.folder_name, self.params["predict_file"])
        #if predict_file.is_file():
        if os.path.exists(self.predict_file):
            print("Loading predict_dict:", self.predict_file)
            self.predict_dict = json.load(open(self.predict_file, 'r'))
        else:
            self.predict_dict = dict()

        self.entity_mapping, self.relation_mapping, self.rev_entity_mapping, self.rev_relation_mapping = self.create_mappings()
        self.subject_dict = dict()
        self.object_dict = dict()
        self.subject_dict, self.object_dict = self.create_load_dict()
        self.IRIstring = self.params['IRIstring']

        if config.get("dataset.name") == "uobm_3":
            self.sepstr = "-"

        elif (config.get("dataset.name") == "yago3-10"):
            self.sepstr = "|"

        elif (config.get("dataset.name") == "dbpedia15k"):
            print("Setting sepstr")
            self.sepstr = "|"

        if self.abstraction == "False":
            #load the simple samples
            self.triple_samples_map_file = os.path.join(self.folder_name, self.params["triple_samples_map_file"])
            if os.path.exists(self.triple_samples_map_file):
                print("Loading triple_samples_map dict:", self.triple_samples_map_file)
                self.triple_samples_map = json.load(open(self.triple_samples_map_file, 'r'))
            else:
                self.triple_samples_map = dict()

        else: #when abstraction is true
            # load the abstract samples
            self.triple_samples_map_file = os.path.join(self.folder_name, self.params["abstract_triple_samples_map_file"])
            if os.path.exists(self.triple_samples_map_file):
                print("Loading abstract_triple_samples_map dict:", self.triple_samples_map_file)
                self.triple_samples_map = json.load(open(self.triple_samples_map_file, 'r'))
            else:
                self.triple_samples_map = dict()

            #create or load the local types for whole dataset
            self.all_local_types_file = os.path.join(self.folder_name, self.params["all_local_types_file"])
            if os.path.exists(self.all_local_types_file):
                print("Loading local_type for dataset:", self.all_local_types_file)
                self.all_local_types = pd.read_csv(self.all_local_types_file, index_col=False)

            else:
                print("Creating local_type for dataset")
                train_triples = self.dataset.split(self.train_split).long()
                valid_triples = self.dataset.split(self.valid_split).long()
                all_triples_list = train_triples.numpy().tolist()+valid_triples.numpy().tolist()
                print("all triples", len(all_triples_list))
                self.all_local_types = self.find_local_type(all_triples_list)
                self.all_local_types.to_csv(self.all_local_types_file, index = False)

            # print("All local types: ", len(self.all_local_types))
            # print(self.all_local_types[:2])

        checkpoint = load_checkpoint(os.path.join(self.folder_name, self.params['checkpoint']))
        print("Loading from ", self.params['checkpoint'])
        self.prev_model = KgeModel.create_from(checkpoint)
        self.path_onto = os.path.join(self.folder_name, self.params['path_onto'])

        self.measure_inconsistency = self.params['measure_inconsistent']
        print("measure_inconsistent", self.measure_inconsistency)
        self.sub_inconsistent = None
        self.obj_inconsistent = None

        self.static_sampling = self.params['static_sampling']
        if self.static_sampling == "True":
            print("static_sampling")
            if (config.get("dataset.name") == "dbpedia15k"):
                self.entity_corruptentity_file = os.path.join(self.folder_name, self.params["entity_corruptentity_file"])
                self.entity_corruptentity_map = json.load(open(self.entity_corruptentity_file, 'r'))
                #for dbpedia15k dataset
                self.entity_static_samples_map = self.find_static_samples_transrowl(self.entity_corruptentity_map)

            else:
                #first create the extended local type, if not exists already
                # create or load the extended local types for whole dataset
                extended_local_types_file = os.path.join(self.folder_name, self.params["extended_local_types_file"])
                if os.path.exists(extended_local_types_file):
                    print("Loading extended_local_type for dataset:", extended_local_types_file)
                    self.extended_local_types = pd.read_csv(extended_local_types_file, index_col=False)

                else:
                    #this will create and write the files, then I can read
                    self.extended_local_types = self.find_extended_local_type(self.all_local_types)
                    self.extended_local_types.to_csv(extended_local_types_file, index=False)
                    #the column names are added while writing to csv, reading from csv is better in both cases
                    self.extended_local_types = pd.read_csv(extended_local_types_file, index_col=False)

                #then get the entity to corrupt entity mapping
                self.entity_static_samples_map = self.find_static_samples(self.extended_local_types)




    def create_load_dict(self):
        subject = 0
        object = 0

        subject_dict_file = os.path.join(self.folder_name, self.params["subject_dict_file"])
        # print("Subject_dict:", subject_dict_file)

        if os.path.exists(subject_dict_file):
            subject = 1
            print("Loading subject_dict")
            subject_dict = json.load(open(subject_dict_file, 'r'))

        object_dict_file = os.path.join(self.folder_name, self.params['object_dict_file'])

        if os.path.exists(object_dict_file):
            object = 1
            print("Loading object_dict")
            object_dict = json.load(open(object_dict_file, 'r'))

        if (subject == 0) or (object == 0):
            print("Creating subject,object dicts")

            train_triples = self.dataset.split(self.train_split).long()
            train_triples_list = train_triples.numpy().tolist()

            val_triples = self.dataset.split(self.valid_split).long()
            val_triples_list = val_triples.numpy().tolist()

            triples_list = train_triples_list + val_triples_list
            # print("Triples list ", triples_list[0])
            print("Total", len(triples_list))

            subject_dict = dict()
            object_dict = dict()

            for triple in triples_list:
                subject = triple[0]
                object = triple[2]
                relation = triple[1]

                # subject_str = self.dataset.entity_strings(subject)
                # rel_str = self.dataset.relation_strings(relation)
                # object_str = self.dataset.entity_strings(object)

                subject_str = self.entity_mapping[subject]
                rel_str = self.relation_mapping[relation]
                object_str = self.entity_mapping[object]

                if rel_str == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                    continue

                triple_str = [subject_str, rel_str, object_str]

                if subject_str not in subject_dict.keys():
                    subject_dict[subject_str] = [
                        triple_str]  # triple is already a list, now Im making list of list here
                else:
                    if triple_str not in subject_dict[subject_str]:  # to avoid duplicates
                        subject_dict[subject_str].append(triple_str)  # addng another list element to list of lists
                        # print(subject_dict[subject_str])

                if object_str not in object_dict.keys():
                    object_dict[object_str] = [triple_str]
                else:
                    if triple_str not in object_dict[object_str]:  # to avoid duplicates
                        object_dict[object_str].append(triple_str)


            # print (subject_dict.keys())
            # print (object_dict)
            print("Subject, object dictionaries created and saved")
            # print(list(subject_dict.items())[:1])
            # print(list(object_dict.items())[:1])

            print("Writing dict files")
            json.dump(subject_dict, open(subject_dict_file, 'w'))
            json.dump(object_dict, open(object_dict_file, 'w'))

        return subject_dict, object_dict

    def create_mappings(self):

        # get list of all entity ids
        entities = torch.Tensor(range(0, self.dataset.num_entities())).long()
        entity_indices = entities.tolist()
        # print(entity_indices[:10])

        # to get list of all entities from the model
        entity_list = self.dataset.entity_ids(entities)
        # print(entity_list[:10])

        entity_mapping = dict(zip(entity_indices, entity_list))
        # print(list(entity_mapping.items())[:10])
        # print(len(entity_mapping))

        rev_entity_mapping = dict(zip(entity_list,entity_indices))

        relations = torch.Tensor(range(0, self.dataset.num_relations())).long()
        relation_indices = relations.tolist()
        # print(relation_indices[:10])

        # to get list of all relations from the model
        relation_list = self.dataset.relation_ids(relations)
        # print(relation_list[:10])

        relation_mapping = dict(zip(relation_indices, relation_list))
        # print(list(relation_mapping.items())[:10])
        # print(len(relation_mapping))

        rev_relation_mapping = dict(zip(relation_list,relation_indices))

        return entity_mapping, relation_mapping, rev_entity_mapping, rev_relation_mapping


    def find_static_samples(self, extended_local_types_csv):

        print("Making entity mapping")
        entity_corruptIDs = defaultdict(list)

        #print(extended_local_types_csv.head())
        extended_local_types = extended_local_types_csv.replace(np.nan, '', regex=True)
        print(extended_local_types.head())

        for row in extended_local_types.itertuples():
            #print(row)
            entities = row.individual.split(self.sepstr)
            if row.disjoint_types =="nan" or pd.isnull(row.disjoint_types):
                continue

            #else:
            disjoint_types = row.disjoint_types.split(self.sepstr)

            corrupt_entities = []
            for disjoint in disjoint_types:
                matched_types = self.extended_local_types[self.extended_local_types['extended_types'].str.contains(disjoint, na=False)]
                #print("Found matching rows:", len(matched_types))
                #print(matched_types.head())

                for each_row in matched_types.itertuples():
                    matched_entities = each_row.individual.split(self.sepstr)
                    #print("Found corrupt entities", len(matched_entities))
                    corrupt_entities.extend(matched_entities)

            for entity in entities:
                entity_corruptIDs[entity] = corrupt_entities

            #print(entity_corruptIDs)

        # for key, values in entity_corruptIDs.items():
        #     print (key, len(values))

        for key, values in entity_corruptIDs.items():
            print(key, values[:5])
            break



        print(len(list(entity_corruptIDs.keys())), "have corrupt entities")

        return entity_corruptIDs


    def find_extended_local_type(self, all_local_types):

        #read from extended files folder and combine all
        print("Loading extended types file:")
        folder = self.folder_name+"/extended_files"
        #all_files = os.path.join(folder + "/*.csv")
        all_files = glob.glob(folder + "/*.csv")
        #print(all_files)

        li = []
        for filename in all_files:
            print(filename)
            df = pd.read_csv(filename, index_col=None, header=0)
            print(len(df))
            li.append(df)

        extended_local_type_set_df = pd.concat(li, axis=0, ignore_index=True)
        extended_local_type_set_df = extended_local_type_set_df.drop_duplicates()
        print("No of extended sets combined", len(extended_local_type_set_df))

        unique_extended_local_sets = extended_local_type_set_df.groupby(['extended_types', 'disjoint_types']).agg(
            lambda x: self.sepstr.join(set(x))).reset_index()

        print("No of unique extended sets ", len(unique_extended_local_sets))
        unique_extended_local_sets.to_csv(self.folder_name + "/extended_local_types_file_combined.csv", index=False)

        return unique_extended_local_sets



        print("Finding extended types")
        #https://stackoverflow.com/questions/17611477/getting-superclasses-in-imported-owl-ontology

        reasoner = PyExplanationReasoner()
        ontology_from_file = reasoner.load_ontology_from_file(self.path_onto)
        extended_local_type_set = []

        extended_type = set()
        print('extended_type', extended_type)
        print(type(extended_type))

        print("No of all_local_types", len(all_local_types))

        count = 0
        for row in all_local_types.itertuples():
            #print(count)

            if count<9501:
                count+=1
                continue
            #print(count)

            class_set = set()
            if not pd.isnull(row.classes):
                #print(row.classes)
                for cls in row.classes:
                    #print(cls)
                    class_set.add(cls) #ad the current class already, then its superclasses
                    superclasses = reasoner.get_super_classes(self.IRIstring+cls)
                    #print(superclasses)
                    if superclasses:
                        class_set = class_set | superclasses

                    #extended_type = extended_type | superclasses
                    #print("extended_type", extended_type)
                    #print("Class_set", class_set)

            # cls = "wordnet_actor_109765278"
            # extended_type.add(cls)
            # superclasses = reasoner.get_super_classes(self.IRIstring+cls)
            # print(superclasses)
            # extended_type.update(superclasses)
            # print("extended_type", extended_type)

            range_class_set = set()
            if not pd.isnull(row.incoming):
                #print(row.incoming)
                for pred in row.incoming:
                    #print(pred)
                    in_superproperties = reasoner.get_super_properties(self.IRIstring+pred)
                    in_superproperties.add(pred) #add the original propoerty as well
                    #print("in_superproperties",in_superproperties)

                    #now find the ranges of all incoming superproperties
                    #print("Finding ranges")
                    for pred in in_superproperties:
                        #print(pred)
                        ranges = reasoner.get_ranges(self.IRIstring+pred)
                        #print("Ranges", ranges)
                        if ranges:
                            range_class_set = range_class_set | ranges
                        #print("range_class_set", range_class_set)

            domain_class_set = set()
            if not pd.isnull(row.outgoing):
                #print(row.outgoing)
                for pred in row.outgoing.split(self.sepstr):
                    #print(pred)
                    #pred = 'happenedIn'
                    out_superproperties = reasoner.get_super_properties(self.IRIstring+pred)
                    out_superproperties.add(pred)  # add the original propoerty as well
                    #print("out_superproperties", out_superproperties)

                    # now find the domains of all outgoing superproperties
                    for pred in out_superproperties:
                        #print(pred)
                        domains = reasoner.get_domains(self.IRIstring+pred)
                        #print("Domain", domains)
                        if domains:
                            #extended_type.update(domains)
                            #print(type(extended_type))
                            for domain in domains:
                                domain = domain.replace(self.IRIstring, "")
                                domain_class_set.add(domain)

                        #print("domain_class_set", domain_class_set)
                        #break

            extended_type = domain_class_set | range_class_set | class_set
            #print("All extended types for this local type:", extended_type)

            #now find the disjoint classes

            all_disjoint_types = set()
            for ex_type in extended_type:
                if self.IRIstring not in ex_type:
                    ex_type = self.IRIstring+ex_type
                #print("Finding disjoint for", ex_type)
                disjoint_types = reasoner.get_disjoint(ex_type)
                # print(len(disjoint_types))
                # print(list(disjoint_types)[0])
                all_disjoint_types = all_disjoint_types | disjoint_types

            #print("total disjoint types:", len(all_disjoint_types))

            extended_type = sorted(extended_type)
            extended_type_str = self.sepstr.join(extended_type)
            all_disjoint_types = sorted(all_disjoint_types)
            all_disjoint_types_str = self.sepstr.join(all_disjoint_types)

            extended_local_type_set.append([row.individual, extended_type_str, all_disjoint_types_str])
            #print([row.individual, extended_type_str, all_disjoint_types_str])

            if count % 100 == 0:
                print(count)
                print(row)
                # break

            if count % 500 == 0:
                print(count)
                extended_local_type_set_df = pd.DataFrame.from_records(extended_local_type_set,
                                                                       columns=['individual', 'extended_types',
                                                                                'disjoint_types'])
                print("No of extended sets ", len(extended_local_type_set_df))

                unique_extended_local_sets = extended_local_type_set_df.groupby(
                    ['extended_types', 'disjoint_types']).agg(
                    lambda x: self.sepstr.join(set(x))).reset_index()

                print("No of unique extended sets ", len(unique_extended_local_sets))
                filename = self.folder_name + "/extended_local_types_file_" +str(count) +".csv"

                unique_extended_local_sets.to_csv(filename, index=False)

            count += 1
            #break

        #when all rows have been processed
        extended_local_type_set_df = pd.DataFrame.from_records(extended_local_type_set,
                                                      columns=['individual', 'extended_types', 'disjoint_types'])
        print("No of extended sets ", len(extended_local_type_set_df))


        unique_extended_local_sets = extended_local_type_set_df.groupby(['extended_types', 'disjoint_types']).agg(
            lambda x: self.sepstr.join(set(x))).reset_index()

        print("No of unique extended sets ", len(unique_extended_local_sets))
        unique_extended_local_sets.to_csv(self.folder_name + "/extended_local_types_file_all.csv", index=False)


        return unique_extended_local_sets


    def find_static_samples_transrowl(self, entity_corruptentity_map):

        entity_corruptIDs = defaultdict(list)

        #read entity_corruptEntity fiel here, convert strings to ids

        print("No of keys in dict:" , len(entity_corruptentity_map.keys()))

        for key, entityList in entity_corruptentity_map.items():
            entityKey = key
            entityKeyId = self.rev_entity_mapping.get(entityKey, "")
            if not entityKeyId:
                continue

            for sublist in entityList:
                for idx in range(len(sublist)):
                    entity = sublist[idx]
                    entityId = self.rev_entity_mapping.get(entity, "")
                    sublist[idx] = entityId #it might be empty string

                sublist[:] = [x for x in sublist if x]
                #print("Sublist", sublist)

            entity_corruptIDs[entityKeyId].extend(entityList)

        print("No of keys in Id dict:", len(entity_corruptIDs.keys()))

        # for key, value in entity_corruptIDs.items():
        #     print(key, value)
        #     break

        return entity_corruptIDs


        #then slotting code will work just fine
        return


    def find_local_type(self,triple_set):
        #print("Finding local types")
        #print("Triples", len(triple_set))
        # abstract type triples here
        # for each entity, first find its type, outgoing relations, incoming relations

        individuals = []
        for triple in triple_set:
            subject = triple[0]
            # if type(subject) == int:
            #     subject = self.entity_mapping(subject)
            object = triple[2]
            # if type(object) == int:
            #     object = self.entity_mapping(object)

            if subject not in individuals:
                individuals.append(subject)
                # print (subject)
            if object not in individuals:
                individuals.append(object)
                # print (object)

        #print("individuals", len(individuals), individuals[:3])

        local_type_set = []
        for individual in individuals:
            # print (individual)
            class_set = []
            incoming_set = []
            outgoing_set = []
            for triple in triple_set:
                if individual in triple:
                    #print(type(individual))
                    if isinstance(individual, str):
                        individual_str = individual
                    else:
                        individual_str = self.entity_mapping[individual]

                    subject = triple[0]
                    if not isinstance(subject, str):
                        subject = self.entity_mapping[triple[0]]

                    object = triple[2]
                    if not isinstance(object, str):
                        object = self.entity_mapping[triple[2]]

                    relation = triple[1]
                    if not isinstance(relation, str):
                        relation = self.relation_mapping[triple[1]]

                    #print(subject, relation, object)

                    if relation == "type":
                        if object not in class_set:
                            class_set.append(object)

                    elif subject == individual_str:
                        if relation not in outgoing_set:
                            outgoing_set.append(relation)

                    elif object == individual_str:
                        if relation not in incoming_set:
                            incoming_set.append(relation)


            class_set.sort()
            class_set = list(set(class_set))
            incoming_set.sort()
            outgoing_set.sort()

            #self.sepstr = "|"
            #print("self.sepstr:", self.sepstr)

            #class_set_str = '-'.join(class_set)
            class_set_str = self.sepstr.join(class_set)

            # print (class_set_str)
            # incoming_set_str = '-'.join(incoming_set)
            # outgoing_set_str = '-'.join(outgoing_set)
            incoming_set_str = self.sepstr.join(incoming_set)
            outgoing_set_str = self.sepstr.join(outgoing_set)

            local_type_set.append([individual_str, class_set_str, incoming_set_str, outgoing_set_str])
            #print([individual_str, class_set_str, incoming_set_str, outgoing_set_str])


        local_type_set_df = pd.DataFrame.from_records(local_type_set,
                                                      columns=['individual', 'classes', 'incoming', 'outgoing'])
        #print(local_type_set_df)
        #print("No of local sets ", len(local_type_set_df))
       # unique_local_sets = local_type_set_df.groupby(['classes', 'incoming', 'outgoing']).agg(lambda x: '-'.join(set(x))).reset_index()
        unique_local_sets = local_type_set_df.groupby(['classes', 'incoming', 'outgoing']).agg(
            lambda x: self.sepstr.join(set(x))).reset_index()
        #unique_local_sets = local_type_set_df.drop_duplicates(subset = ['classes', 'incoming', 'outgoing'], keep='first')
        # print(unique_local_sets[['individual', 'classes', 'incoming', 'outgoing']])

        #print("No of grouped local sets ", len(unique_local_sets))
        #print(unique_local_sets)


        return unique_local_sets

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _get_collate_fun(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a tuple of:

            - triples (tensor of shape [n,3], ),
            - negative_samples (list of tensors of shape [n,num_samples]; 3 elements
              in order S,P,O)
            """
            #generating the usual way first, will be replaced where needed
            triples = self.dataset.split(self.train_split)[batch, :].long()
            triples_list = triples.numpy().tolist()

            negative_samples = list()
            self.found_predict = 0 #set to 0 here since want to print value for each batch


            #for slot in [S, P, O]:
                #negative_samples.append(self._sampler.sample(triples, slot))
            #return {"triples": triples, "negative_samples": negative_samples}

            #set triples to test_set if measuring inconsitency
            if self.measure_inconsistency == "True":
                triples = self.dataset.split(self.eval_split).long()
                triples_list = triples.numpy().tolist()[:5000]
                total = len(triples_list)
                print("Choosing eval_split set to test ", total)

                self.sub_inconsistent =  []
                self.obj_inconsistent = []

                #print(self.sub_inconsistent)


            def rel_set_extract_fast(model, subject, object, subject_dict, object_dict):
                #print("Creating rel set")
                rel_set = []
               # print ("Looking in dict:", subject)
                if subject in subject_dict.keys():
                    subject_triples = subject_dict[subject]
                    #print("Subject matches from subject_dict", len(subject_triples))
                    # for item in subject_triples:
                    #     print(item)
                    rel_set.extend(subject_triples)
                    #print(len(rel_set))

                if subject in object_dict.keys():
                    subject_triples = object_dict[subject]
                    # print("Subject matches from object_dict", len(subject_triples))
                    # for item in subject_triples:
                    #     print(item)
                    rel_set.extend(subject_triples)
                    #print(len(rel_set))

               # print("Looking in dict:", object)
                if object in subject_dict.keys():
                    object_triples = subject_dict[object]
                    # print("Object matches from subject_dict", len(object_triples))
                    # for item in object_triples:
                    #     print(item)
                    rel_set.extend(object_triples)
                    #print(len(rel_set))

                if object in object_dict.keys():
                    object_triples = object_dict[object]
                    # print("Object matches from object_dict", len(object_triples))
                    # for item in object_triples:
                    #     print(item)
                    rel_set.extend(object_triples)
                    #print(len(rel_set))

                #print("Rel_set", rel_set[:5])

                #dicts are with strings now, no need for changes here
                rel_set_all = rel_set

                # https://stackoverflow.com/questions/2213923/removing-duplicates-from-a-list-of-lists
                # when creating owl ontology for reasoner, duplicates will be handled
                # rel_set.sort()
                # rel_set = list(rel_set for rel_set, _ in itertools.groupby(rel_set))

                return rel_set_all

            def get_abstract_samples(expl_local_sets, predicted, all_local_types):

                samples = []

                #code to find samples from matching, finding supersets of expl_local_types
                # check hee if dataframe is empty, if so, then dont fo any further
                if expl_local_sets.empty:
                    #print("No expl local types found")
                    return samples

                #find the local type of precited form expl_local_types

                #to solve re.error: multiple repeat at position x
                #https://blog.finxter.com/python-regex-multiple-repeat-error/
                if "++" in predicted:
                    predicted = predicted.replace("++","")

                predicted_local_type = expl_local_sets[expl_local_sets['individual'].str.contains(predicted)]
                #print(predicted_local_type)

                if predicted_local_type.empty:
                    # print("No predicted local type found")
                    return samples

                # predicted_classes = predicted_local_type['classes'].to_string().split("-")
                # predicted_incoming = predicted_local_type['incoming'].to_string()
                # predicted_outgoing = predicted_local_type['outgoing'].to_string()

                for row in predicted_local_type.itertuples(index=False):
                    #print(row)
                    # predicted_classes = row.classes.split('-')
                    # predicted_incoming = row.incoming.split('-')
                    # predicted_outgoing = row.outgoing.split('-')

                    predicted_classes = row.classes.split(self.sepstr)
                    predicted_incoming = row.incoming.split(self.sepstr)
                    predicted_outgoing = row.outgoing.split(self.sepstr)


                # print("classes", type(predicted_classes), predicted_classes)
                # print("outgoing",type(predicted_outgoing), predicted_outgoing)
                # print("incoming",type(predicted_incoming),predicted_incoming)

                #https://stackoverflow.com/questions/17071871/how-to-select-rows-from-a-dataframe-based-on-column-values

                # predicted_local_type = all_local_types[
                #     all_local_types['individual'].str.contains("-" + predicted + "-")]
              
               # target_local_type= all_local_types[all_local_types['classes'].str.split("-").contains(predicted_classes)]


                #                                        (all_local_types['outgoing'].tolist() ==predicted_outgoing)]
                #


                #print("Comparing with all_local_types")
                found = 0
                for row in all_local_types.itertuples(index=False):
                    #print(row)

                    if not type(row.classes) == float:
                        classes = row.classes.split(self.sepstr)
                    else :
                        classes = ['']
                    if not type(row.incoming) == float:
                        incoming = row.incoming.split(self.sepstr)
                    else :
                        incoming = ['']
                    if not type(row.outgoing) == float:
                        outgoing = row.outgoing.split(self.sepstr)
                    else:
                        outgoing = ['']


                    # print("classes", type(classes), classes)
                    # print("incoming", type(incoming), incoming)
                    # print("outgoing", type(outgoing), outgoing)
                    #

                    if (set(classes)>= set(predicted_classes) and set(incoming)>= set(predicted_incoming) and set(outgoing)>= set(predicted_outgoing)):
                        #print("Found match or superset ")
                        found = 1
                        # print("classes", type(classes), classes)
                        # print("outgoing", type(outgoing), outgoing)
                        # print("incoming", type(incoming), incoming)
                        matches = row.individual
                        matches = matches.split(self.sepstr)
                        samples.extend(matches)
                        #print("Matches", len(matches), matches[0])


                #if found == 0:
                    #print("Matches not found")

                # print('Found: ', target_local_type)
                # samples = target_local_type['individual']
                # print (samples)
                return samples

            def check_consistency(triples, predicted, relation, input_entity):
                #print("***Example 1: for checking inconsistency")
                reasoner = PyExplanationReasoner()

                #Load ontology from file
                ontology_from_file = reasoner.load_ontology_from_file(self.path_onto)

                #triples = {('john', 'rdf:type', 'Person'), ('john', 'livesIn', 'germany')}
                #triples = relevant_set_ids

                #print(triples)
                reasoner.load_ontology_from_tbox_and_assertions(ontology_from_file, triples, self.IRIstring)

                # print("=== Axioms in the ontology loaded from file:")
                # for axiom in reasoner.get_ontology().getAxioms():
                #     print(axiom.toString())

                reasoner.check_consistency()
                #print("The onto ", path_onto, "is consistent?", str(reasoner.is_consistent()))

                #print("=== Axioms added to the ontology loaded from file:")
                #for axiom in reasoner.get_ontology().getAxioms():
                    #print(axiom.toString())

                # output from reasoner
                consistent = reasoner.is_consistent()
                #print ("consistent:", str(consistent))

                abstract_samples = []

                if self.measure_inconsistency == "True":
                    #no need to calculate anything, just eturn here
                    return consistent, abstract_samples

                #consistent = False
                if consistent == False and self.abstraction == 'True' :
                    #print("consistent:", str(consistent))
                    abstract_samples = get_explanations(reasoner, predicted, relation, input_entity)

                    if len(abstract_samples) == 0:
                        abstract_samples.append(predicted)

                    #here, after getting the abstrac samples form all case,s now convrt strings to ids
                    abstract_samples_ids = []
                    for str_sample in abstract_samples:
                        # print(str_sample)
                        # id_sample = self.rev_entity_mapping[str_sample]
                        # https://stackoverflow.com/questions/42297801/keyerror-when-using-non-ascii-characters-as-keys-in-a-python-dictionary
                        id_sample = self.rev_entity_mapping.get(str_sample, "")
                        if not id_sample == "":
                            abstract_samples_ids.append(id_sample)

                        # id_sample = self.rev_entity_mapping[str_sample]
                        # abstract_samples_ids.append(id_sample)

                    #print("After converting to ids:", abstract_samples_ids)
                    abstract_samples = abstract_samples_ids
                    # print("Abstract samples obtained-", len(abstract_samples), abstract_samples[:1])

                reasoner.cleanup()
                OwlAPI.owl_manager.removeOntology(ontology_from_file)

                return consistent, abstract_samples

            def get_explanations(expl_reasoner, predicted, relation, input_entity):
                #expl_reasoner = PyExplanationReasoner(reasoner)
                #expl_reasoner.load_ontology_from_file(path_onto)
                # print()
                # print("Subject", input_entity)
                # print("Relation", relation)
                # print(predicted, "caused inconsistency")

                abstract_samples = []

                #find all_local_types for whole dataset -
                # TODO - calculate once and load eveyrtime like subject_dict
                #do it at epoch level nd use every time
                # all_triples = self.dataset.split(self.train_split).long()
                # all_triples_list = all_triples.numpy().tolist()
                #
                # all_local_types = find_local_type(all_triples_list)
                #print("All local types: ", len(self.all_local_types))
                #print(self.all_local_types[:2])

                explanations_list = expl_reasoner.get_explanations_for_inconsistency(5, 100)
                count = 0
                #print("number of explanations: ", len(explanations_list))

                #explanations_list = []
                if not explanations_list:
                    #print("No explanations found:")

                    samples = []
                    # in case we did not get any explanations, time out etc
                    # then find samples from directly the local type of predicted entity and add to samples

                    # just to avoid any wrong matches, adding "-" to predicted to avoid any partial macthes here
                    #predicted_local_type = self.all_local_types[self.all_local_types['individual'].str.contains("-" + predicted + "-")]

                    # to solve re.error: multiple repeat at position x
                    # https://blog.finxter.com/python-regex-multiple-repeat-error/
                    if "++" in predicted:
                        predicted = predicted.replace("++", "")

                    predicted_local_type = self.all_local_types[self.all_local_types['individual']
                        .str.contains(predicted)]
                    # print('Local type of predicted:')
                    # print(predicted_local_type)
                    #  print(type(predicted_local_type))

                    # there should be only local_type row tat shoudl contain the predcited entity
                    # so only row should be obtained
                    if len(predicted_local_type.index) > 1:
                        #print("More than one row")
                        return []

                    for idx, row in predicted_local_type.iterrows():
                        matches = row['individual']
                        # print("Matches", type(matches))
                        matches = matches.split(self.sepstr)
                        #print("Matches", len(matches), matches[0])
                        samples.extend(matches)

                    #print("No of samples obtained", len(samples), samples[:1])
                    abstract_samples = list(set(samples))

                    return abstract_samples


                for each_explanation in explanations_list:
                    count += 1
                    # print()
                    #print("Explanation: ", count)
                    #print("Get TBox/ontology axioms:", each_explanation[0])

                    #print("Get concept assertions/type triples:")
                    type_triples = each_explanation[1]
                    #print("No of type triples:", len(type_triples))

                    samples = []
                    for each_type_triple in type_triples:
                        # print("Concept/type:", each_type_triple[0])
                        # print("entity:", each_type_triple[1])

                        if (each_type_triple[1] == predicted):
                            #if this is the predicted entity, then take all entities who have same type
                            target_type = each_type_triple[0]

                            #now, search in dataset for entities with tis type
                            if target_type in object_dict.keys():
                                object_triples = object_dict[object]

                            #from these triples, extract the subject entities, these are samples
                            for triple in object_triples:
                                samples.append(triple[0])

                    #print(samples)
                    #abstract_samples.append(samples)
                    #here, the samples should be added as a list directly, need to make list of lists

                    #print("Get relation/property triples:")
                    property_triples = each_explanation[2]
                    #print("No of property triples:", len(property_triples))

                    clean_property_triples=[]
                    for each_pr_triple in property_triples:
                        #print(list(each_pr_triple))
                        subject = each_pr_triple[0].replace('<', '').replace('>', '').replace(self.IRIstring, '')
                        predicate = each_pr_triple[1].replace('<', '').replace('>', '').replace(self.IRIstring, '')
                        object = each_pr_triple[2].replace('<', '').replace('>', '').replace(self.IRIstring, '')

                        #print([subject, predicate, object])
                        clean_triple = [subject, predicate, object]
                        clean_property_triples.append(clean_triple)

                    predicted_triple = None
                    #print("Predicted", [predicted, relation, input_entity])
                    for each_pr_triple in clean_property_triples:
                        #print(each_pr_triple)

                        if (each_pr_triple == [predicted, relation, input_entity]):
                            #print("Incorrect prediction is :", [subject, predicate, object])
                            predicted_triple = each_pr_triple
                            #print("Found predicted triple:",predicted_triple)


                    if predicted_triple:
                        #print("Property triples" )
                        #print(property_triples)
                        clean_property_triples.remove(predicted_triple)
                        #print("Final local type context:", clean_property_triples)


                    #find local_type for predicted here from local_type_context
                    # and get entities from local_type_dict in samples


                    expl_local_sets = self.find_local_type(clean_property_triples)
                    # print('Local type of property triples:')
                    # print(expl_local_sets)


                    samples = get_abstract_samples(expl_local_sets, predicted, self.all_local_types)
                    #print(samples)

                    abstract_samples.extend(samples)
                    abstract_samples = list(set(abstract_samples))

                    #print("Abstract samples obtained-", len(abstract_samples), abstract_samples[:1])


                    #print("End of this explanation:===========")

                #abstract_samples = samples

                return abstract_samples

            def get_sub_negatives(triples_list, model):
                # create map here for triple index to negative sample, if generated

                reasoner_samples_map = dict()

               # print(len(triples_list[:5]))

                for t_index, triple in enumerate(triples_list):
                    #start = time.process_time()

                    s_id = triple[2]
                    o_id = triple[0]
                    p_id = triple[1]

                    # print("")
                    # print("")
                    # print(triple)
                    #
                    # print("Relation", p_id)
                    # print("Input entity", s_id)


                    map_key = str(s_id) + "_" + str(p_id)
                    #for static map, the entity being predicted should eb the key now,
                    # here triple[0] is being predicted
                    static_map_key = o_id

                    if not self.first:
                        # when its not the first epoch, then look for reasoner_samples values in the map
                        # if the key is found, then the values need to be updated, else just leave it
                        # in every case, no need to make predictions, so continue always for all triples

                        #adding the static_sampling logic here as well
                        #when if self.static_sampling == "True":, then load the static samples instead
                        #no need to to do anything aftr loading, so conitnue still needed
                        #print("first epoch", self.first)

                        if self.static_sampling == "True":
                            #print("Static sampling")
                            if static_map_key in self.entity_static_samples_map.keys():
                                #print("Found key", static_map_key)
                                reasoner_samples_map[t_index] = self.entity_static_samples_map[static_map_key]
                                #print(reasoner_samples_map[t_index])

                        else : #if not static sampling, conitnue as before to update reaoner_samples
                            if map_key in self.triple_samples_map.keys():
                                #print("Previous found")
                                reasoner_samples_map[t_index] = self.triple_samples_map[map_key]
                                #print(map_key, self.triple_samples_map[map_key])
                                #print(reasoner_samples_map[t_index])

                        continue  # for all triples in both cases

                    input_entity = self.entity_mapping[s_id]
                    relation = self.relation_mapping[p_id]
                    target_entity = self.entity_mapping[o_id]

                    # print("Relation", relation)
                    # print("Input entity", input_entity)
                    # print("Target entity", target_entity)


                    # skip useless relation triples - those that dont connect 2 entities, also skip type triples
                    if relation in ["emailAddress", "firstName", "lastName"]:
                        #print("skipping")
                        continue

                    #print(time.process_time() - start)

                    p = torch.Tensor([p_id, ]).long()  # relation indexes
                    # s is now the object entity
                    s = torch.Tensor([s_id, ]).long()  # subject indexes
                    scores = self.prev_model.score_sp(s, p)  # scores of all objects for (s,p,?)
                    o = torch.argmax(scores, dim=-1)
                    # print("Top predicted", o)

                    # perhaps the value of k should be decided based on num_sample for s, o
                    k = int(self._sampler.num_samples[slot])
                    #print("No of predictions,num_samples:", k)

                    score, idx = torch.topk(scores, k, -1)
                    # print (idx)
                    # print("Predictions")
                    topk = idx.numpy().flatten().tolist()  # convert tensor to numpy , flatten and create 1d list

                   # print("Before predictions", time.process_time() - start)

                    for idx, predict_id in enumerate(topk):
                        #print("")
                        #print(predict_id)

                        predicted = self.entity_mapping[predict_id]
                        # print(idx)
                        #print("Predicted", predicted)

                        key = str(input_entity) + "_" + str(relation) + "_" + str(predicted)

                        if key in self.predict_dict and self.measure_inconsistency == "False":
                            #print("Prediction found in dict")
                            consistent = self.predict_dict[key]
                            self.found_predict += 1

                            if consistent == True:
                                abstract_samples = []

                            #else:
                                #pass
                            #hanlde here th ecase where consistent is false and we need abstract samples

                        else : #calculate if either key not found or if consitent is False

                            if (predicted == target_entity):
                                #print("Correct prediction")
                                # no need to check anything if this is already correct prediciton
                                # it has to be consistent as well, no neg samples here
                                self.predict_dict[key] = True
                                continue

                            relevant_set_ids = []
                            # find rest of them from ()
                            # relevant_set_ids = rel_set_extract_ids(s_id, predict_id, all_triples)

                            relevant_set_ids = rel_set_extract_fast(self.prev_model, input_entity, predicted, self.subject_dict, self.object_dict)

                            # adding the prediction also to rel_set
                            # prediction = [input_entity, relation, predicted]
                            # need to invert the s, o here since all rel set triples are not inverted

                            relevant_set_ids.insert(0, [predicted, relation, input_entity])

                            #print("Total no of triples in rel_set", len(relevant_set_ids))
                            #print(relevant_set_ids[:1])

                            # here call the reasoner with owl file and the relevant_set - ids or strings

                            consistent, abstract_samples = check_consistency(relevant_set_ids, predicted, relation, input_entity)

                            #print(consistent, type(consistent))

                            #here, add this triple, predcition and its result to dict
                            self.predict_dict[key] = consistent

                            #consistent = False
                            if consistent == False:

                                if self.measure_inconsistency == "True":
                                    #print("Found subject inconsistency at", t_index)
                                    self.sub_inconsistent.append(str(idx) + "_" + str(s_id) + "_" + str(p_id))
                                    #print("Sub inconsistency", self.sub_inconsistent)
                                    continue #nothing further needed

                                #print("Inconsistency found, t_index", t_index)
                                # triple_negativesample_map[t_index] = [s_id, p_id, predict_id]
                                elif self.abstraction == 'False':

                                    if t_index in reasoner_samples_map.keys():
                                        reasoner_samples_map[t_index].append(predict_id)
                                    else:
                                        reasoner_samples_map[t_index] = [predict_id]

                                    # also write the predict_id to other map with proper key
                                    if map_key in self.triple_samples_map.keys():
                                        self.triple_samples_map[map_key].append(predict_id)
                                        # for k, v in self.triple_samples_map.items():
                                        #     print((k, v))
                                    else:
                                        self.triple_samples_map[map_key] = [predict_id]
                                        # for k, v in self.triple_samples_map.items():
                                        #     print((k, v))

                                else : #abstarction is true
                                    #abstract_samples is a list of subjects/objects as negative samples
                                    #add the abstract_samples list as a list itself, the values of dict will be list of lists
                                    #print("Found subject inconsistency at", t_index, input_entity, relation, predicted)
                                    if t_index in reasoner_samples_map.keys():
                                        reasoner_samples_map[t_index].append(abstract_samples)
                                    else:
                                        reasoner_samples_map[t_index] = [abstract_samples]
                                    #this list may become longer than the num_samples
                                    #print("Len of reasoner_samples_map at", t_index, len(reasoner_samples_map[t_index]))
                                    #print(reasoner_samples_map[t_index])

                                    #break

                                    # also write the predict_id to te triples map with proper key
                                    if map_key in self.triple_samples_map.keys():
                                        self.triple_samples_map[map_key].append(abstract_samples)
                                        # for k, v in self.triple_samples_map.items():
                                        #     print((k, v))
                                    else:
                                        self.triple_samples_map[map_key] = [abstract_samples]
                                        # for k, v in self.triple_samples_map.items():
                                        #     print((k, v))

                return reasoner_samples_map


            def get_obj_negatives(triples_list, model):
                # create map here for triple index to negative sample, if generated
                reasoner_samples_map = dict()

                count = 0
                #print(len(triples_list))

                for t_index, triple in enumerate(triples_list):

                    s_id = triple[0]
                    o_id = triple[2]
                    p_id = triple[1]

                    # print("")
                    # print("")
                    # print(triple)
                    #
                    # print("Relation", p_id)
                    # print("Input entity", s_id)

                    map_key = str(s_id) + "_" + str(p_id)
                    static_map_key = o_id

                    if not self.first:
                        #when its not the first epoch, then look for reasoner_samples values in the map
                        # if the key is found, then the values need to be updated, else just leave it
                        # in every case, no need to make predictions, so continue always for all triples

                        # adding the static_sampling logic here as well
                        # when if self.static_sampling == "True":, then load the static samples instead
                        # no need to to do anything aftr loading, so conitnue still needed

                        #print("first epoch", self.first)
                        if self.static_sampling == "True":
                            #print("Static sampling")
                            if static_map_key in self.entity_static_samples_map.keys():
                                #print("Found key", static_map_key)
                                reasoner_samples_map[t_index] = self.entity_static_samples_map[static_map_key]

                        else:  # if not static sampling, conitnue as before to update reaoner_samples

                            if map_key in self.triple_samples_map.keys():
                                # print("Previous found")
                                reasoner_samples_map[t_index] = self.triple_samples_map[map_key]
                                # print(map_key, self.triple_samples_map[map_key])
                                # print(reasoner_samples_map[t_index])

                        continue  # for all triples in both cases
                        # the rest of the loop doesnt run at all if not first epoch


                    # input_entity = self.dataset.entity_strings(s_id)
                    # relation = self.dataset.relation_strings(p_id)
                    # target_entity = self.dataset.entity_strings(o_id)

                    input_entity = self.entity_mapping[s_id]
                    relation = self.relation_mapping[p_id]
                    target_entity = self.entity_mapping[o_id]

                    # input_entity = 'http://dbpedia.org/resource/Alex_Borstein'
                    # relation = 'http://dbpedia.org/ontology/almaMater'
                    # target_entity = 'http://dbpedia.org/resource/San_Francisco_State_University'


                    # skip useless relation triples - those that dont connect 2 entities, also skip type triples
                    if relation in ["emailAddress", "firstName", "lastName", ]:
                        continue

                    if relation == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                        continue

                    # print("Input entity", input_entity)
                    # print("Relation", relation)
                    # print("Target entity", target_entity)

                    p = torch.Tensor([p_id, ]).long()  # relation indexes
                    # for object prediction
                    s = torch.Tensor([s_id, ]).long()  # subject indexes
                    scores = self.prev_model.score_sp(s, p)  # scores of all objects for (s,p,?)
                    o = torch.argmax(scores, dim=-1)
                    # print("Top predicted", o)

                    #perhaps the value of k should be decided based on num_sample for s, o
                    k = int(self._sampler.num_samples[slot])
                    #print("No of predictions,num_samples:", k)

                    score, idx = torch.topk(scores, k, -1)
                    # print (idx)
                    # print("Predictions")
                    topk = idx.numpy().flatten().tolist()  # convert tensor to numpy , flatten and create 1d list

                    for idx, predict_id in enumerate(topk):
                        #print("")
                        # print(idx)
                        #print(predict_id)
                        #predicted = model.dataset.entity_strings(predict_id)
                        predicted = self.entity_mapping[predict_id]

                        # predicted = 'http://dbpedia.org/resource/New_York_University'
                        # print("Predicted", predicted)

                        #key = str(s_id) + "_" + str(p_id) + "_" + str(predict_id)
                        key = str(input_entity) + "_" + str(relation) + "_" + str(predicted)

                        if key in self.predict_dict and self.measure_inconsistency =="False":
                            # print("Prediction found in dict")
                            consistent = self.predict_dict[key]
                            self.found_predict +=1

                            if consistent == True:
                                abstract_samples = []

                            else:
                                pass
                            # hanlde here th ecase where consistent is false but we do need abstract samples

                        else : #calculate

                            if (predicted == target_entity):
                                #print("Correct prediction")
                                # no need to check anything if this is already correct prediciton
                                # it has to be consistent as well, no neg samples here
                                self.predict_dict[key] = True
                                continue

                            relevant_set_ids = []
                            # find rest of them from ()
                            # relevant_set_ids = rel_set_extract_ids(s_id, predict_id, all_triples)
                            #relevant_set_ids = rel_set_extract_fast(model, s_id, predict_id, subject_dict, object_dict)

                            relevant_set_ids = rel_set_extract_fast(self.prev_model, input_entity, predicted, self.subject_dict,self.object_dict)
                            # adding the prediction also to rel_set
                            # prediction = [input_entity, relation, predicted]
                            # need to invert the s, o here since all rel set triples are not inverted
                            # relevant_set_ids.insert(0, [s_id, relation, predict_id])
                            relevant_set_ids.insert(0, [input_entity, relation, predicted])

                            # relevant_set_ids.insert(0,["Luke_Christopher", "type", "Person"])
                            # relevant_set_ids.insert(0, ["Parker_Self", "type", "Person"])
                            #
                            # print("Total no of triples in rel_set", len(relevant_set_ids))
                            # print(relevant_set_ids)

                            #rel_set_local_types = find_local_type(relevant_set)
                            #relevant_set_ids = get_rel_set_abstract(rel_set_local_types)

                            consistent, abstract_samples = check_consistency(relevant_set_ids, predicted, relation,
                                                                             input_entity)

                            #print(consistent, type(consistent))
                            # print("abstract samples")
                            # print(abstract_samples)

                            # here, add this triple, predcition and its result to dict
                            self.predict_dict[key] = consistent


                            if consistent == False:
                                # print("Inconsistency found, t_index", t_index)
                                # triple_negativesample_map[t_index] = [s_id, p_id, predict_id]

                                if self.measure_inconsistency == "True":
                                    #print("Found object inconsistency at", t_index)
                                    self.obj_inconsistent.append(str(idx) + "_" + str(s_id) + "_" + str(p_id))
                                    #print("Obj inconsistency", self.obj_inconsistent)
                                    continue  # nothing further needed


                                elif self.abstraction == 'False':
                                    if t_index in reasoner_samples_map.keys():
                                        reasoner_samples_map[t_index].append(predict_id)
                                    else:
                                        reasoner_samples_map[t_index] = [predict_id]

                                    # also write the predict_id to other map with proper key
                                    if map_key in self.triple_samples_map.keys():
                                        self.triple_samples_map[map_key].append(predict_id)
                                        # for k, v in self.triple_samples_map.items():
                                        #     print((k, v))
                                    else:
                                        self.triple_samples_map[map_key] = [predict_id]
                                        # for k, v in self.triple_samples_map.items():
                                        #     print((k, v))

                                else:  # abstarction is true
                                    # abstract_samples is a list of subjects/objects as negative samples
                                    # add the abstract_samples list as a list itself, the values of dict will be list of lists
                                    #print("Found object inconsistency at", t_index, input_entity, relation, predicted)
                                    if t_index in reasoner_samples_map.keys():
                                        reasoner_samples_map[t_index].append(abstract_samples)
                                    else:
                                        reasoner_samples_map[t_index] = [abstract_samples]
                                    # this list may become longer than the num_samples
                                    # print("Len of reasoner_samples_map at", t_index, len(reasoner_samples_map[t_index]))
                                    # print(reasoner_samples_map[t_index])

                                    # also write the predict_id to other map with proper key
                                    if map_key in self.triple_samples_map.keys():
                                        self.triple_samples_map[map_key].append(abstract_samples)
                                        # for k, v in self.triple_samples_map.items():
                                        #     print((k, v))
                                    else:
                                        self.triple_samples_map[map_key] = [abstract_samples]
                                        # for k, v in self.triple_samples_map.items():
                                        #     print((k, v))

                return reasoner_samples_map

            ###### code end ######

            for slot in [S, P, O]:
                #print('Calling sampler')
                #print('Sampler is ', self._sampler)
                #print("slot:", slot)
                reasoner_samples_map= dict() #initialize here, but updated only for slot 0 and 2

                if (slot == 0):
                    start = time.process_time()
                    #print("No of predictions,num_samples:", int(self._sampler.num_samples[slot]))
                    reasoner_samples_map = get_sub_negatives(triples_list, self.prev_model)
                    #if self.first:
                    #print(time.process_time() - start)
                    #print(self.predict_dict)

                if (slot == 2):
                    start = time.process_time()
                    #print("No of predictions,num_samples:", int(self._sampler.num_samples[slot]))
                    reasoner_samples_map = get_obj_negatives(triples_list, self.prev_model)
                    #if self.first:
                    #print(time.process_time() - start)
                    #print(self.predict_dict)

                #print(self._sampler.sample(triple_negativesample_map, triple_negativesample_map, triples, slot))
                negative_samples.append(self._sampler.sample(reasoner_samples_map, self.abstraction, self.first, triples, slot))


            #it makes sense to write these files only when its is the first epoch with predictions
            if self.first and self.measure_inconsistency == "False":
                print("Found in predict_file:", self.found_predict)
                print("Writing predict_dict file")

                json.dump(self.predict_dict,
                          open(self.predict_file, 'w'))

                print("Writing triple_samples_map file")
                json.dump(self.triple_samples_map,
                          open(self.triple_samples_map_file, 'w'))


            if self.measure_inconsistency == "True":
                file = open(self.folder_name + '/sub_consistent_results.csv', 'w')
                with file:
                    writer = csv.writer(file, delimiter='\n')
                    writer.writerow(self.sub_inconsistent)
                print("Total subject_inconsistent", len(self.sub_inconsistent))

                file2 = open(self.folder_name + '/obj_consistent_results.csv', 'w')
                with file2:
                    writer = csv.writer(file2, delimiter='\n')
                    writer.writerow(self.obj_inconsistent)
                print("Total object_inconsistent", len(self.obj_inconsistent))

                #
                # import csv
                # wtr = csv.writer(open(self.folder_name + 'sub_consistent_results.csv', 'w'), delimiter=',',
                #                  lineterminator='\n')
                # subject_inconsistent = 0
                # for idx, value in self.sub_inconsistent:
                #     if value:
                #         print (value)
                #         subject_inconsistent +=1
                #         wtr.writerow(idx, value)
                # print("Total subject_inconsistent", subject_inconsistent)

                exit(0)  # stop at one batch , it has all test triples needed already, no training needed

            return {"triples": triples, "negative_samples": negative_samples}

        return collate

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        # move triples and negatives to GPU. With some implementaiton effort, this may
        # be avoided.
        result.prepare_time -= time.time()
        batch["triples"] = batch["triples"].to(self.device)
        for ns in batch["negative_samples"]:
            ns.positive_triples = batch["triples"]
        batch["negative_samples"] = [
            ns.to(self.device) for ns in batch["negative_samples"]
        ]

        batch["labels"] = [None] * 3  # reuse label tensors b/w subbatches
        result.size = len(batch["triples"])
        result.prepare_time += time.time()

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice]
        batch_negative_samples = batch["negative_samples"]
        batch_size = len(batch["triples"])
        subbatch_size = len(triples)
        result.prepare_time += time.time()
        labels = batch["labels"]  # reuse b/w subbatches

        # process the subbatch for each slot separately
        for slot in [S, P, O]:
            num_samples = self._sampler.num_samples[slot]
            if num_samples <= 0:
                continue

            # construct gold labels: first column corresponds to positives,
            # remaining columns to negatives
            if labels[slot] is None or labels[slot].shape != (
                subbatch_size,
                1 + num_samples,
            ):
                result.prepare_time -= time.time()
                labels[slot] = torch.zeros(
                    (subbatch_size, 1 + num_samples), device=self.device
                )
                labels[slot][:, 0] = 1
                result.prepare_time += time.time()

            # compute the scores
            result.forward_time -= time.time()
            scores = torch.empty((subbatch_size, num_samples + 1), device=self.device)
            scores[:, 0] = self.model.score_spo(
                triples[:, S], triples[:, P], triples[:, O], direction=SLOT_STR[slot],
            )
            result.forward_time += time.time()
            scores[:, 1:] = batch_negative_samples[slot].score(
                self.model, indexes=subbatch_slice
            )
            result.forward_time += batch_negative_samples[slot].forward_time
            result.prepare_time += batch_negative_samples[slot].prepare_time

            # compute loss for slot in subbatch (concluding the forward pass)
            result.forward_time -= time.time()
            loss_value_torch = (
                self.loss(scores, labels[slot], num_negatives=num_samples) / batch_size
            )
            result.avg_loss += loss_value_torch.item()
            result.forward_time += time.time()

            # backward pass for this slot in the subbatch
            result.backward_time -= time.time()
            if not self.is_forward_only:
                loss_value_torch.backward()
            result.backward_time += time.time()

