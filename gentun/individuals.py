#!/usr/bin/env python
"""
Classes which define the individuals of a population with
its characteristic genes, generation, crossover and
mutation processes.
"""

import pprint
import random

from models import XgboostRegressor


class Individual(object):
    """Basic definition of an individual containing
    reproduction and mutation methods. Do not instantiate,
    use a subclass which extends this object by defining a
    genome and a random individual generator.
    """

    def __init__(self, x_train, y_train, genome, genes, uniform_rate, mutation_rate, additional_parameters=None):
        self.x_train = x_train
        self.y_train = y_train
        self.genome = genome
        self.validate_genome()
        self.genes = genes
        self.validate_genes()
        self.uniform_rate = uniform_rate
        self.mutation_rate = mutation_rate
        self.fitness = None  # Until evaluated an individual is unfit
        assert additional_parameters is None

    def validate_genome(self):
        """Check genome structure."""
        if type(self.genome) != dict:
            raise TypeError("Genome must be a dictionary.")
        for gene, properties in self.genome.iteritems():
            if type(gene) != str:
                raise TypeError("Gene names must be strings.")
            if type(properties) != tuple:
                raise TypeError("Gene attributes must be a tuple.")
            if len(properties) != 4:
                raise TypeError(
                    "A gene must have 4 attributes: a default value, "
                    "minimum value, maximum value, and a precision or None."
                )

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

    @staticmethod
    def generate_random_genes(genome):
        raise NotImplementedError("Use a subclass with genes definition.")

    def evaluate_fitness(self):
        raise NotImplementedError("Use a subclass with genes definition.")

    def get_additional_parameters(self):
        raise NotImplementedError("Use a subclass with genes definition.")

    def get_fitness(self):
        """Compute individual's fitness if necessary and return it."""
        if self.fitness is None:
            self.evaluate_fitness()
        return self.fitness

    def reproduce(self, partner):
        """Mix genes from self and partner randomly and
        return a new instance of an individual.
        """
        assert self.__class__ == partner.__class__  # Can only reproduce if they're the same species
        child_genes = {}
        for name, value in self.get_genes().iteritems():
            if random.random() < self.uniform_rate:
                child_genes[name] = value
            else:
                child_genes[name] = partner.get_genes()[name]
        return self.__class__(
            self.x_train, self.y_train, self.genome, child_genes, self.uniform_rate, self.mutation_rate,
            **self.get_additional_parameters()
        )

    def mutate(self):
        """Mutate instance's genes with a certain probability."""
        for name, value in self.get_genes().iteritems():
            if random.random() < self.mutation_rate:
                default, minimum, maximum, precision = self.get_genome()[name]
                if type(default) == int:
                    self.get_genes()[name] = random.randint(minimum, maximum)
                else:
                    self.get_genes()[name] = round(random.uniform(minimum, maximum), precision)
                self.fitness = None  # The mutation produces a new individual

    def __str__(self):
        """Return genes which identify the individual."""
        return pprint.pformat(self.genes)


class XgboostIndividual(Individual):

    def __init__(self, x_train, y_train, genome=None, genes=None, uniform_rate=0.5, mutation_rate=0.015,
                 eval_metric='rmse', nfold=5, num_boost_round=5000, early_stopping_rounds=100):
        if genome is None:
            genome = {
                # name: (default, min, max, precision)
                'eta': (0.3, 0.0, 1.0, 6),
                'min_child_weight': (1, 0, 10, None),
                'max_depth': (6, 3, 10, None),
                'gamma': (0.0, 0.0, 10.0, 2),
                'max_delta_step': (0, 0, 10, None),
                'subsample': (1.0, 0.5, 1.0, 2),
                'colsample_bytree': (1.0, 0.5, 1.0, 2),
                'colsample_bylevel': (1.0, 0.5, 1.0, 2),
                'lambda': (1.0, 0.0, 10.0, 2),
                'alpha': (0.0, 0.0, 10.0, 2),
                'scale_pos_weight': (1.0, 0.0, 10.0, 2)
            }
        if genes is None:
            genes = self.generate_random_genes(genome)
        # Set individual's attributes
        super(XgboostIndividual, self).__init__(x_train, y_train, genome, genes, uniform_rate, mutation_rate)
        # Set additional parameters which are not tuned
        self.eval_metric = eval_metric
        self.nfold = nfold
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    @staticmethod
    def generate_random_genes(genome):
        """Create and return random genes."""
        genes = {}
        for name, (default, minimum, maximum, precision) in genome.iteritems():
            if type(default) == int:
                genes[name] = random.randint(minimum, maximum)
            else:
                genes[name] = round(random.uniform(minimum, maximum), precision)
        return genes

    def evaluate_fitness(self):
        """Create model and perform cross-validation."""
        model = XgboostRegressor(
            self.x_train, self.y_train, self.genes, eval_metric=self.eval_metric,
            nfold=self.nfold, num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds
        )
        self.fitness = model.cross_validate()

    def get_additional_parameters(self):
        return {
            'eval_metric': self.eval_metric,
            'nfold': self.nfold,
            'num_boost_round': self.num_boost_round,
            'early_stopping_rounds': self.early_stopping_rounds
        }
