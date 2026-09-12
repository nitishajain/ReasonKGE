import jnius_config
import os
import copy
from typing import Set, List, Dict, Tuple
import logging

''' ***SUPER IMPORTANT***
Issue with loading:
https://github.com/kivy/pyjnius/issues/216
Issue with nested class:
https://stackoverflow.com/questions/41660690/how-to-implement-nested-class-and-abstract-class-with-pyjniuskivy
'''
#
# os.environ['JDK_HOME'] = "/usr/lib/jvm/java-1.8.0-openjdk-amd64/"
# os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-1.8.0-openjdk-amd64/"
# os.environ[
#     'PATH'] += ';/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/bin/;/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/bin/;'
#
# print ("setenv JAVA_HOME", os.environ["JAVA_HOME"])

if os.environ.get("REASONKGE_JAVA_CLASSPATH"):
    os.environ["CLASSPATH"] = os.environ["REASONKGE_JAVA_CLASSPATH"]

from jnius import autoclass
MyJavaFile = autoclass('java.io.File')
OWLManager = autoclass('org.semanticweb.owlapi.apibinding.OWLManager')
OWLOntology = autoclass('org.semanticweb.owlapi.model.OWLOntology')
OWLOntologyManager = autoclass('org.semanticweb.owlapi.model.OWLOntologyManager')
IRI = autoclass('org.semanticweb.owlapi.model.IRI')
OWLAxiom = autoclass('org.semanticweb.owlapi.model.OWLAxiom')
OWLClass = autoclass('org.semanticweb.owlapi.model.OWLClass')
OWLDataFactory = autoclass('org.semanticweb.owlapi.model.OWLDataFactory')
OWLOntologyCreationException = autoclass('org.semanticweb.owlapi.model.OWLOntologyCreationException')
OWLReasonerFactory = autoclass('org.semanticweb.owlapi.reasoner.OWLReasonerFactory')
OWLReasoner = autoclass('org.semanticweb.owlapi.reasoner.OWLReasoner')
ReasonerFactory = autoclass('org.semanticweb.HermiT.Reasoner$ReasonerFactory')
Reasoner = autoclass('org.semanticweb.HermiT.Reasoner')
InconsistentOntologyExplanationGeneratorFactory = autoclass(
    'org.semanticweb.owl.explanation.impl.blackbox.checker.InconsistentOntologyExplanationGeneratorFactory')
# Assetions:
OWLClassAssertionAxiom = autoclass('org.semanticweb.owlapi.model.OWLClassAssertionAxiom')
OWLObjectPropertyAssertionAxiom = autoclass('org.semanticweb.owlapi.model.OWLObjectPropertyAssertionAxiom')
InferenceType = autoclass('org.semanticweb.owlapi.reasoner.InferenceType')

'''====================finished importing java classes======================'''


#
def setup_logging():
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.INFO
                        )

setup_logging()

class PyReasoner:
    """OWLReasoner"""
    owl_file_path = None
    ontology = None
    initialized = False
    manager = None
    consistent = None
    consistency_is_checked = False
    reasoner = None
    def load_ontology_from_file(self, owl_file_path_):
        self.owl_file_path = owl_file_path_
        self.manager = OWLManager.createOWLOntologyManager()
        #print(self.owl_file_path)
        onto_file = MyJavaFile(self.owl_file_path)
        self.ontology = self.manager.loadOntologyFromOntologyDocument(onto_file)
        self.initialized = True
        self.consistent = None
        self.consistency_is_checked = False
        return self.ontology

    def load_ontology_from_tbox_and_assertions_old(self, tbox_ontology, assertions: Set):
        """ tbox_ontology: ontology loaded from .owl file
            assertions: a set of tuples; each tuple is of the form (subject_string, predicate_string, object_string); predicate string could also be rdf:type
            Note: tbox_ontology will be remain unchanged. This means you can load the ontology from file just one and reuse it with different set of assertions
        """
        self.manager = OWLManager.createOWLOntologyManager()
        self.ontology = self.manager.createOntology(IRI.create("reasoning_ontology.owl"))
        self.manager.addAxioms(self.ontology, tbox_ontology.getAxioms())

        '''add triples to the ontology'''
        for each_tuple in assertions:

            print(each_tuple)
            relation = each_tuple[1]
            print(relation)
            if 'rdf:type' in relation:
            #if relation == 'rdf:type' or relation == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                print("Adding class assertion")
                owl_assertion = OwlAPI.create_class_assertion(each_tuple[0], each_tuple[2])
                self.manager.addAxiom(self.ontology, owl_assertion)
            else:
                print("Adding property assertion")
                owl_property_assertion = OwlAPI.create_property_assertion(each_tuple[0], each_tuple[1], each_tuple[2])
                self.manager.addAxiom(self.ontology, owl_property_assertion)

        self.initialized = True
        self.consistent = None
        self.consistency_is_checked = False

    def load_ontology_from_tbox_and_assertions(self, tbox_ontology, assertions: Set, IRIstring):
        """ tbox_ontology: ontology loaded from .owl file
            assertions: a set of tuples; each tuple is of the form (subject_string, predicate_string, object_string); predicate string could also be rdf:type
            Note: tbox_ontology will be remain unchanged. This means you can load the ontology from file just one and reuse it with different set of assertions
        """
        self.manager = OWLManager.createOWLOntologyManager()
        #self.ontology = self.manager.createOntology(IRI.create("reasoning_ontology.owl"))
        self.manager.addAxioms(self.ontology, self.ontology.getAxioms())

        '''add triples to the ontology'''

       # print("Assertions are: ", assertions)
        #print("IRIstring", IRIstring, len(IRIstring))

        for each_tuple in assertions:

            #print (each_tuple)
            property = each_tuple[1]
            # subject_entity = factory.getOWLNamedIndividual(IRI.create(IRIstring + str(each_tuple[0])))
            # object_entity = factory.getOWLClass(IRI.create(IRIstring + str(each_tuple[2])))
            # relation = factory.getOWLObjectProperty(IRI.create(IRIstring + property))

            subject_entity = IRIstring + str(each_tuple[0])
            object_entity = IRIstring + str(each_tuple[2])
            relation = IRIstring + property

            # print(subject_entity)
            # print(object_entity)
            #print(property)

            if 'type' in property or 'Type' in property:
                # if relation == 'rdf:type' or relation == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                #print("Adding class assertion")
                owl_assertion = OwlAPI.create_class_assertion(subject_entity, object_entity)
                #print(owl_assertion)
                self.manager.addAxiom(self.ontology, owl_assertion)
                #print(subject_entity, object_entity)
                #print(owl_assertion.toString())

            else:
                #print("Adding property assertion")
                owl_property_assertion = OwlAPI.create_property_assertion(subject_entity, relation, object_entity)
                #print(owl_property_assertion)
                self.manager.addAxiom(self.ontology, owl_property_assertion)
                #print(subject_entity, relation, object_entity)

        self.initialized = True
        self.consistent = None
        self.consistency_is_checked = False

    def check_consistency(self):
        """Must be called after loading ontology. """
        if not self.initialized:
            logging.error("There is no ontology loaded. Please do that before calling this method")
            return
        if not self.consistency_is_checked:
            self.reasoner = Reasoner(self.ontology)
            # logging.info("Ontology ", str(self.ontology.getAxioms().toString()))
            for axiom in self.ontology.getAxioms():
                logging.debug(axiom.toString())
            if self.reasoner.isConsistent():
                self.consistent = True
            else:
                self.consistent = False

    def is_consistent(self):
        if self.consistent is not None:
            return self.consistent
        else:
            logging.error("You must call check_consistency first")

    def cleanup(self):
        if self.reasoner is not None:
            self.reasoner.dispose()
        if self.initialized:
            self.manager.removeOntology(self.ontology)

    def get_ontology(self):
        return self.ontology

    def get_super_classes(self,class_string):
        # self.precomputeInferences(InferenceType.CLASS_HIERARCHY)
        # self.precomputeInferences(InferenceType.OBJECT_PROPERTY_HIERARCHY)
        #class_string = IRIstring + classname
        #print(class_string)
        class_name = OwlAPI.data_factory.getOWLClass(IRI.create(str(class_string)))

        superClasses = set()
        owl_reasoner_factory = ReasonerFactory()
        reasoner = owl_reasoner_factory.createReasoner(self.ontology)

        superClasses = reasoner.getSuperClasses(class_name,0).getFlattened()
        classes = set()
        for val in superClasses:
            #https://stackoverflow.com/questions/24821322/using-owl-api-how-to-get-class-or-individual-name
            supercls = val.getIRI().getFragment()
            if 'Thing' not in supercls:
                #print(supercls)
                classes.add(supercls)


        if reasoner is not None:
            reasoner.dispose()

        return classes

    def get_ranges(self, predicate):

        #print("Range for ", predicate)
        property = OwlAPI.data_factory.getOWLObjectProperty(IRI.create(str(predicate)))
        range_axioms = self.ontology.getObjectPropertyRangeAxioms(property)
        # print("Range axioms", range_axioms)
        # OWLClassExpression
        range = set()
        for axiom in range_axioms:
            # print('axiom',axiom)
            #c = axiom.getRange()
            # print(c)
            range_str = axiom.getSignature()
            #print(range_str.toString())
            for cls in range_str:
                #print(cls.toString())
                if predicate not in cls.toString() and 'Thing' not in cls.toString():
                    cleaned_cls = cls.toString().strip('<>')
                    #cleaned_cls = cleaned_cls.lstrip('http://yago-knowledge.org/resource/')
                    #print(cleaned_cls)
                    range.add(cleaned_cls)

        #print(range)

        return range

    def get_domains(self, predicate):
        #print("Domain for ", predicate)
        property = OwlAPI.data_factory.getOWLObjectProperty(IRI.create(str(predicate)))
        domain_axioms = self.ontology.getObjectPropertyDomainAxioms(property)
        # print(domain_axioms)
        # OWLClassExpression
        domain = set()
        for axiom in domain_axioms:
            # print(axiom)
            # c = axiom.getDomain()
            # print(c.asOWLClass().getIRI().getFragment())
            domain_str = axiom.getSignature()
            #print(domain_str)

            for cls in domain_str:
                #print(cls.toString())
                #print(cls.getFragment())
                if predicate not in cls.toString() and 'Thing' not in cls.toString():
                    cleaned_cls = cls.toString().strip('<>')
                    #cleaned_cls = cleaned_cls.lstrip(IRIstring)
                    #print(cleaned_cls)
                    domain.add(cleaned_cls)


        #print(domain)
        return domain

    def get_disjoint(self, class_string):
        owl_reasoner_factory = ReasonerFactory()
        reasoner = owl_reasoner_factory.createReasoner(self.ontology)

        class_name = OwlAPI.data_factory.getOWLClass(IRI.create(str(class_string)))
        disjoint_set = reasoner.getDisjointClasses(class_name)

        disjoints = set()
        for disjoint_node in disjoint_set:
            #print(disjoint_node.toString())
            for owlclassimpl in disjoint_node:
                disjoint_cls = owlclassimpl.getIRI().getFragment()
                #print(disjoint_cls)
                disjoints.add(disjoint_cls)

            #break

        if reasoner is not None:
            reasoner.dispose()
        return disjoints

    def get_super_properties(self, predicate):
        # self.precomputeInferences(InferenceType.CLASS_HIERARCHY)
        # self.precomputeInferences(InferenceType.OBJECT_PROPERTY_HIERARCHY)
        #superclasses = self.getSuperProperty(pred)

        #predicate = 'http://yago-knowledge.org/resource/happenedIn'
        #predicate = IRIstring + pred
        #print(predicate) #http://yago-knowledge.org/resource/isLocatedIn
        property = OwlAPI.data_factory.getOWLObjectProperty(IRI.create(str(predicate)))
        #superproperties = self.ontology.getObjectSubPropertyAxiomsForSubProperty(property)

        owl_reasoner_factory = ReasonerFactory()
        reasoner = owl_reasoner_factory.createReasoner(self.ontology)
        superRelations = set()
        #Set<OWLObjectPropertyExpression> superRelations
        superRelations = reasoner.getSuperObjectProperties(property, 0).getFlattened()
        #print("Super properties")
        relations = set()
        for val in superRelations:
            #print(val)
            # print(val.getIRI())
            # print(val.getNamedProperty())
            # print(val.getSimplifival.getIRI().getFragment()ed())
            try:
                relation = val.getIRI().getFragment()
                #print(relation)
                if 'topObjectProperty' not in relation:
                    #print(relation)
                    relations.add(relation)

            except Exception as e:
                pass
                #print("error", e, "for", predicate)

        if reasoner is not None:
            reasoner.dispose()
        return relations


class OwlAPI:
    """ Class with static functions to work with OWL"""
    owl_manager = OWLManager.createOWLOntologyManager()
    #owl_manager = OWLOntologyManager.createOWLOntologyManager()
    data_factory = owl_manager.getOWLDataFactory()

    @classmethod
    def load_owlontology_from_file(cls, ontology_file_path):
        """ load java OWLAPI ontology from a file """
        #print(ontology_file_path)
        onto_file = MyJavaFile(ontology_file_path)
        #print(onto_file)
        owlapi_ontology = cls.owl_manager.loadOntologyFromOntologyDocument(onto_file)
        return owlapi_ontology

    @classmethod
    def create_class(cls, class_string):
        return cls.data_factory.getOWLClass(IRI.create(str(class_string)))

    @classmethod
    def create_individual(cls, individual_string):
        return cls.data_factory.getOWLNamedIndividual(IRI.create(str(individual_string)))

    @classmethod
    def create_object_property(cls, predicate):
        return cls.data_factory.getOWLObjectProperty(IRI.create(str(predicate)))

    @classmethod
    def create_class_assertion(cls, class_string, instance_string):
        class_owl = cls.create_class(class_string)
        instance = cls.create_individual(instance_string)
        return cls.data_factory.getOWLClassAssertionAxiom(class_owl, instance)

    @classmethod
    def create_property_assertion(cls, subject, predicate, object):
        s = cls.create_individual(subject)
        o = cls.create_individual(object)
        p = cls.create_object_property(predicate)
        return cls.data_factory.getOWLObjectPropertyAssertionAxiom(p, s, o)


    @classmethod
    def cleanup(cls):
        """Remove all ontologies currently open"""
        for opening_onto in cls.owl_manager.getOntologies():
            print(opening_onto.toString())
            cls.owl_manager.removeOntology(OWLOntology(opening_onto))


class PyExplanationReasoner(PyReasoner):
    """Inconsistency explanation Reasoner"""

    def get_explanations_for_inconsistency(self, max_number_of_explanations, timeout_in_seconds):
        """return a set of explanations"""

        if not self.initialized:
            logging.error("There is no ontology loaded. Please do that before calling this method")
            return
        manager = OWLManager.createOWLOntologyManager()
        data_factory = manager.getOWLDataFactory()

        owl_thing = data_factory.getOWLThing()
        owl_nothing = data_factory.getOWLNothing()
        thing_implies_nothing = data_factory.getOWLSubClassOfAxiom(owl_thing, owl_nothing)
        #
        owl_reasoner_factory = ReasonerFactory()
        expl_generator_fac = InconsistentOntologyExplanationGeneratorFactory(owl_reasoner_factory, timeout_in_seconds)
        expl_generator = expl_generator_fac.createExplanationGenerator(self.ontology)

        try:
            expl = expl_generator.getExplanations(thing_implies_nothing, max_number_of_explanations)

        except Exception as e:
            #print(e)
            return []


        explanation_list = []
        for ex in expl:
            ''' each explanatino if a tuple tbox_axioms, class_assertions, property_assertions'''
            each_explanation = ()
            tbox_axioms = []
            concept_assertions = []
            role_assertions = []
            for axiom in ex.getAxioms():
                if isinstance(axiom, OWLClassAssertionAxiom):
                    concept = axiom.getClassExpression().asOWLClass().toString()
                    entity = axiom.getIndividual().toString()
                    concept_assertions.append((concept, entity))
                elif isinstance(axiom, OWLObjectPropertyAssertionAxiom):
                    predicate = axiom.getProperty().toString()
                    subject = axiom.getSubject().toString()
                    obj = axiom.getObject().toString()
                    role_assertions.append((subject, predicate, obj))
                else:
                    tbox_axioms.append(axiom.toString())
            each_explanation = (tbox_axioms, concept_assertions, role_assertions)
            explanation_list.append(each_explanation)

        return explanation_list

