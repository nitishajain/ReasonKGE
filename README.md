# ReasonKGE

ReasonKGE uses ontological reasoning to identify inconsistent predictions from knowledge graph embedding models and reuse them as negative samples during training.

This repository contains recovered source code and research assets for **Improving Knowledge Graph Embeddings with Ontological Reasoning** (ISWC 2021).

## Contents

- `kge/` — ReasonKGE implementation based on LibKGE
- `configs/` — archived Yago3-10/TransE configuration examples
- `data/` — prepared LUBM3U, Yago3-10, and DBpedia15K datasets
- `ontologies/` — ontologies used with the datasets

## Installation

The recovered Linux environment uses Python 3.7, PyTorch 1.7, Java, PyJNIus, OWLAPI, HermiT, and the OWL explanation library.

```sh
conda env create -f environment.yml
conda activate libkge
pip install -e .
```

Place the Java dependencies listed in `java/README.md` in one directory and set:

```sh
export REASONKGE_JAVA_CLASSPATH="/path/to/java-libs/*"
```

## Usage

LibKGE commands are available through:

```sh
kge --help
```

For example, the archived baseline configuration can be started with:

```sh
kge start configs/yago3-10/transe-baseline.yaml
```

The remaining configuration files are archived reference snapshots from subsequent ReasonKGE iterations.

## Citation

```bibtex
@inproceedings{jain2021reasonkge,
  title     = {Improving Knowledge Graph Embeddings with Ontological Reasoning},
  author    = {Jain, Nitisha and Tran, Trung-Kien and Gad-Elrab, Mohamed H. and Stepanova, Daria},
  booktitle = {The Semantic Web -- ISWC 2021},
  year      = {2021},
  pages     = {410--426},
  doi       = {10.1007/978-3-030-88361-4_24}
}
```

ReasonKGE includes a modified version of [LibKGE](https://github.com/uma-pi1/kge). See `THIRD_PARTY_NOTICE`.
