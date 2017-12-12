#!/usr/bin/env python
"""
Classes which define the individuals of a population
with its characteristic genes, generation, crossover
and mutation processes.
"""

import random

from models import XgboostRegressor


class Individual(object):
    """Basic definition of an individual containing
    reproduction and mutation methods. Not to be
    instantiated, use a subclass which extends this object
    by defining a genome, item getters and a random
    individual generator.
    """

    def __init__(self, x_train, y_train, genes, uniform_rate, mutation_rate):
        self.x_train = x_train
        self.y_train = y_train
        self.fitness = None  # Until evaluated an individual is unfit
        self.genes = genes
        self.uniform_rate = uniform_rate
        self.mutation_rate = mutation_rate

    def generate_random_genes(self):
        raise NotImplementedError("Use a subclass with genes definition.")

    def get_genes(self):
        raise NotImplementedError("Use a subclass with genes definition.")

    def get_genome(self):
        raise NotImplementedError("Use a subclass with genes definition.")

    def evaluate_fitness(self):
        raise NotImplementedError("Use a subclass with genes definition.")

    def get_fitness(self):
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
                child_genes[name] = partner.get_genes[name]
        return self.__class__(child_genes, self.uniform_rate, self.mutation_rate)

    def mutate(self):
        """Mutate instance's genes with a certain
        probability.
        """
        for name, value in self.get_genes().iteritems():
            if random.random() < self.mutation_rate:
                default, minimum, maximum = self.get_genome()[name]
                if type(default) == int:
                    self.get_genes()[name] = random.randint(minimum, maximum)
                else:
                    self.get_genes()[name] = random.uniform(minimum, maximum)
                self.fitness = float('inf')  # The mutation produces a new individual


class XgboostIndividual(Individual):

    def __init__(self, x_train, y_train, genes=None, uniform_rate=0.5, mutation_rate=0.015):
        super(XgboostIndividual, self).__init__(x_train, y_train, genes, uniform_rate, mutation_rate)
        self.genome = {
            # name: (default, min, max)
            'eta': (0.3, 0.0, 1.0),
            'min_child_weight': (1, 0, 10),
            'max_depth': (6, 3, 10),
            'gamma': (0.0, 0.0, 10.0),
            'max_delta_step': (0, 0, 10),
            'subsample': (1.0, 0.5, 1.0),
            'colsample_bytree': (1.0, 0.5, 1.0),
            'colsample_bylevel': (1.0, 0.5, 1.0),
            'lambda': (1.0, 0.0, 10.0),
            'alpha': (0.0, 0.0, 10.0),
            'scale_pos_weight': (1.0, 0.0, 10.0)
        }
        if genes is None:
            self.genes = self.generate_random_genes()
        if set(self.genome.keys()) != set(self.genes.keys()):
            raise ValueError("Genes passed don't correspond to individual's genome")

    def generate_random_genes(self):
        genes = {}
        for name, (default, minimum, maximum) in self.genome.iteritems():
            if type(default) == int:
                genes[name] = random.randint(minimum, maximum)
            else:
                genes[name] = random.uniform(minimum, maximum)
        return genes

    def get_genes(self):
        return self.genes

    def get_genome(self):
        return self.genome

    def evaluate_fitness(self):
        model = XgboostRegressor(self.x_train, self.y_train, self.genes)
        self.fitness = model.cross_validate()
