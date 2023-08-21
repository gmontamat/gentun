#!/usr/bin/env python
"""
Classes which define the individuals of a population with
its characteristic genes, generation, crossover and
mutation processes.
"""

from __future__ import annotations
import math
import pprint
import random
import multiprocessing as mp
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile
from typing import List, Tuple
from .populations import split_list

import numpy as np

try:
    from .models.xgboost_models import XgboostModel
except ImportError:
    pass

try:
    from .models.keras_models import GeneticCnnModel
except ImportError:
    pass


def random_log_uniform(minimum, maximum, base, eps=1e-12):
    """Generate a random number which is uniform in a
    logarithmic scale. If base > 0 scale goes from minimum
    to maximum, if base < 0 vice versa, and if base is 0,
    use a uniform scale.
    """
    if base == 0:
        return random.uniform(minimum, maximum)
    minimum += eps  # Avoid math domain error when minimum is zero
    if base > 0:
        return base ** random.uniform(math.log(minimum, base), math.log(maximum, base))
    base = abs(base)
    return maximum - base ** random.uniform(math.log(eps, base), math.log(maximum - minimum, base))


class Individual(ABC):
    """Basic definition of an individual containing
    reproduction and mutation methods. Do not instantiate,
    use a subclass which extends this object by defining a
    genome and a random individual generator.
    """

    def __init__(self, x_train: np.memmap, y_train, genome, genes, crossover_rate, mutation_rate, additional_parameters=None):
        self.x_train = x_train
        self.y_train = y_train
        self.genome = genome
        self.validate_genome()
        self.genes = genes
        self.validate_genes()
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.fitness = None  # Until evaluated an individual fitness is unknown
        assert additional_parameters is None

    def validate_genome(self):
        """Check genome structure."""
        if type(self.genome) != dict:
            raise TypeError("Genome must be a dictionary.")
        for gene, properties in self.genome.items():
            if type(gene) != str:
                raise TypeError("Gene names must be strings.")

    def validate_genes(self):
        """Check that genes are compatible with genome."""
        if set(self.genome.keys()) != set(self.genes.keys()):
            raise ValueError("Genes passed don't correspond to individual's genome.")

    def get_genes(self):
        """Return individual's genes."""
        return self.genes

    def get_genome(self):
        """Return individual's genome."""
        return self.genome

    def set_genes(self, new_genes):
        self.genes = new_genes

    @staticmethod
    @abstractmethod
    def generate_random_genes(genome):
        pass


    @abstractmethod
    def evaluate_fitness(self):
        pass

    @abstractmethod
    def get_additional_parameters(self):
        pass

    def get_fitness(self):
        """Compute individual's fitness if necessary and return it."""
        if self.fitness is None:
            self.evaluate_fitness()
        return self.fitness

    def reproduce(self, partner):
        """Mix genes from self and partner randomly and
        return a new instance of an individual. Do not
        mutate parents.
        """
        assert self.__class__ == partner.__class__  # Can only reproduce if they're the same species
        child_genes = {}
        for name, value in self.get_genes().items():
            if random.random() < self.crossover_rate:
                child_genes[name] = partner.get_genes()[name]
            else:
                child_genes[name] = value
        return self.__class__(
            self.x_train, self.y_train, self.genome, child_genes, self.crossover_rate, self.mutation_rate,
            **self.get_additional_parameters()
        )

    def crossover(self, partner):
        """Mix genes from self and partner randomly.
        Mutates each parent instead of producing a
        new instance (child).
        """
        assert self.__class__ == partner.__class__  # Can only cross if they're the same species
        for name in self.get_genes().keys():
            if random.random() < self.crossover_rate:
                self.get_genes()[name], partner.get_genes()[name] = partner.get_genes()[name], self.get_genes()[name]
                self.set_fitness(None)
                partner.set_fitness(None)

    def mutate(self):
        """Mutate instance's genes with a certain probability."""
        for name, value in self.get_genes().items():
            if random.random() < self.mutation_rate:
                default, minimum, maximum, log_scale = self.get_genome()[name]
                if type(default) == int:
                    self.get_genes()[name] = random.randint(minimum, maximum)
                else:
                    self.get_genes()[name] = round(random_log_uniform(minimum, maximum, log_scale), 4)
                self.set_fitness(None)  # The mutation produces a new individual

    def get_fitness_status(self):
        """Return True if individual's fitness in known."""
        return self.fitness is not None

    def set_fitness(self, value):
        """Assign fitness."""
        self.fitness = value

    def copy(self):
        """Copy instance."""
        individual_copy = self.__class__(
            self.x_train, self.y_train, self.genome, self.genes.copy(), self.crossover_rate,
            self.mutation_rate, **self.get_additional_parameters()
        )
        individual_copy.set_fitness(self.fitness)
        return individual_copy

    def __str__(self):
        """Return genes which identify the individual."""
        return pprint.pformat(self.genes)

    def clear_large_data(self):
        """To be overwritten in children"""

    @staticmethod
    def evaluate_fitness_and_return_results(individuals: List[Individual], queue: mp.Queue, indices: List[int], *args):
        for individual in enumerate(individuals):
            queue.put(individual.get_fitness())

    def prepare_data_sharing(self, n_workers: int, *args) -> Tuple:
        with NamedTemporaryFile(prefix='gentun-results-') as memfile:
            outputs = np.memmap(filename=memfile.name, shape=(n_workers,), mode='w+', dtype=np.float64)

        return outputs,

    @staticmethod
    def update_individuals_from_remote_data(individuals: List[Individual], outputs: np.memmap, *args):
        for i, individual in enumerate(individuals):
            individual.set_fitness(outputs[i])

class XgboostIndividual(Individual):

    def __init__(self, x_train, y_train, genome=None, genes=None,
                 crossover_rate=0.5, mutation_rate=0.015,
                 fixed_genes=None, model: XgboostModel = None):
        if genome is None:
            genome = {
                # name: (default, min, max, logarithmic-scale-base)
                'eta': (0.3, 0.001, 1.0, 10),
                'min_child_weight': (1, 0, 10, None),
                'max_depth': (6, 3, 10, None),
                'gamma': (0.0, 0.0, 10.0, 10),
                'max_delta_step': (0, 0, 10, None),
                'subsample': (1.0, 0.0, 1.0, -10),
                'colsample_bytree': (1.0, 0.0, 1.0, -10),
                'colsample_bylevel': (1.0, 0.0, 1.0, -10),
                'lambda': (1.0, 0.1, 10.0, 10),
                'alpha': (0.0, 0.0, 10.0, 10),
                'scale_pos_weight': (1.0, 0.0, 10.0, 0)
            }
        if genes is None:
            genes = self.generate_random_genes(genome)

        # overwrite random genes with fixed ones
        if fixed_genes is not None:
            genes = {**genes, **fixed_genes}
        # Set individual's attributes
        super(XgboostIndividual, self).__init__(x_train, y_train, genome, genes, crossover_rate, mutation_rate)

        assert model is not None, 'Model has to be provided'
        self.model = model
        self.best_ntree_limit = None
        self.fixed_genes = fixed_genes
        self.oof_dict = {}

    def mutate(self):
        """Mutate instance's genes with a certain probability."""
        super(XgboostIndividual, self).mutate()
        # overwrite random genes with fixed ones
        if self.fixed_genes is not None:
            genes = self.get_genes()
            new_genes = {**genes, **self.fixed_genes}
            self.set_genes(new_genes=new_genes)

    @staticmethod
    def generate_random_genes(genome):
        """Create and return random genes."""
        genes = {}
        for name, (default, minimum, maximum, log_scale) in genome.items():
            if type(default) == int:
                genes[name] = random.randint(minimum, maximum)
            else:
                genes[name] = round(random_log_uniform(minimum, maximum, log_scale), 4)
        return genes

    def evaluate_fitness(self):
        """perform cross-validation."""
        self.model.update(self.genes)
        self.fitness = self.model.cross_validate()
        self.best_ntree_limit = self.model.best_ntree_limit
        self.oof_dict = self.model.oof_dict.copy()

    def get_additional_parameters(self):
        return {
            'model': self.model,
        }

    def clear_large_data(self):
        self.model.d_train = None

    @staticmethod
    def evaluate_fitness_and_return_results(individuals: List[Individual], queue: mp.Queue, indices: List[int],
                                            outputs: np.memmap, ntree_limits: np.memmap, cv_preds: np.memmap,
                                            cv_trues: np.memmap, model: XgboostModel, *args):
        for i, result_index in enumerate(indices):
            individuals[i].model = model
            fitness = individuals[i].get_fitness()
            outputs[result_index] = fitness
            ntree_limits[result_index] = individuals[i].best_ntree_limit

            start = 0
            for x in individuals[i].oof_dict['cv_preds']:
                cv_preds[result_index, start:start + len(x)] = x
                start += len(x)

            start = 0
            for x in individuals[i].oof_dict['cv_trues']:
                cv_trues[result_index, start:start + len(x)] = x
                start += len(x)

            queue.put(fitness)

    def prepare_data_sharing(self, n_workers: int, n_colums: int) -> Tuple:
        with NamedTemporaryFile(prefix='gentun-results-') as memfile:
            outputs = np.memmap(filename=memfile.name, shape=(n_workers,), mode='w+', dtype=np.float64)

        with NamedTemporaryFile(prefix='gentun-tree-limits-') as memfile2:
            ntree_limits = np.memmap(filename=memfile2.name, shape=(n_workers,), mode='w+', dtype=np.int64)

        with NamedTemporaryFile(prefix='gentun-cv_preds-') as memfile3:
            cv_preds = np.memmap(filename=memfile3.name, shape=(n_workers, n_colums), mode='w+', dtype=np.float64)

        with NamedTemporaryFile(prefix='gentun-cv_trues-') as memfile4:
            cv_trues = np.memmap(filename=memfile4.name, shape=(n_workers, n_colums), mode='w+', dtype=np.int64)

        return  outputs, ntree_limits, cv_preds, cv_trues, self.model

    @staticmethod
    def update_individuals_from_remote_data(individuals: List[XgboostIndividual], outputs: np.memmap,
                                           ntree_limits: np.memmap, cv_preds: np.memmap, cv_trues: np.memmap, *args):

        super(XgboostIndividual, XgboostIndividual).update_individuals_from_remote_data(individuals, outputs)

        for i, individual in enumerate(individuals):
            individual.best_ntree_limit = ntree_limits[i]
            individual.oof_dict['cv_preds']= split_list(cv_preds[i,:], individual.model.kfold)
            individual.oof_dict['cv_trues'] = split_list(cv_trues[i, :], individual.model.kfold)



class GeneticCnnIndividual(Individual):

    def __init__(self, x_train, y_train, genome=None, genes=None, crossover_rate=0.3, mutation_rate=0.1, nodes=(3, 5),
                 input_shape=(28, 28, 1), kernels_per_layer=(20, 50), kernel_sizes=((5, 5), (5, 5)), dense_units=500,
                 dropout_probability=0.5, classes=10, kfold=5, epochs=(3,), learning_rate=(1e-3,), batch_size=32):
        if genome is None:
            genome = {'S_{}'.format(i + 1): int(K_s * (K_s - 1) / 2) for i, K_s in enumerate(nodes)}
        if genes is None:
            genes = self.generate_random_genes(genome)
        # Set individual's attributes
        super(GeneticCnnIndividual, self).__init__(x_train, y_train, genome, genes, crossover_rate, mutation_rate)
        # Set additional parameters which are not tuned
        assert len(nodes) == len(kernels_per_layer) and len(kernels_per_layer) == len(kernel_sizes)
        self.nodes = nodes
        self.input_shape = input_shape
        self.kernels_per_layer = kernels_per_layer
        self.kernel_sizes = kernel_sizes
        self.dense_units = dense_units
        self.dropout_probability = dropout_probability
        self.classes = classes
        self.kfold = kfold
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    @staticmethod
    def generate_random_genes(genome):
        """Create and return random genes."""
        genes = {}
        for name, connections in genome.items():
            genes[name] = ''.join([random.choice(['0', '1']) for _ in range(connections)])
        return genes

    def evaluate_fitness(self):
        """Create model and perform cross-validation."""
        model = GeneticCnnModel(
            self.x_train, self.y_train, self.genes, self.nodes, self.input_shape, self.kernels_per_layer,
            self.kernel_sizes, self.dense_units, self.dropout_probability, self.classes,
            self.kfold, self.epochs, self.learning_rate, self.batch_size
        )
        self.fitness = model.cross_validate()

    def get_additional_parameters(self):
        return {
            'nodes': self.nodes,
            'input_shape': self.input_shape,
            'kernels_per_layer': self.kernels_per_layer,
            'kernel_sizes': self.kernel_sizes,
            'dense_units': self.dense_units,
            'dropout_probability': self.dropout_probability,
            'classes': self.classes,
            'kfold': self.kfold,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }

    def mutate(self):
        """Mutate instance's genes with a certain probability."""
        for name, connections in self.get_genes().items():
            new_connections = ''.join([
                str(int(int(byte) != (random.random() < self.mutation_rate))) for byte in connections
            ])
            if new_connections != connections:
                self.set_fitness(None)  # A mutation means the individual has to be re-evaluated
                self.get_genes()[name] = new_connections
