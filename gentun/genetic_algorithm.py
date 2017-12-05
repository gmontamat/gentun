#!/usr/bin/env python
"""
Basic genetic algorithm in Python
"""

import operator

# Custom definitions of individuals and its genes
import individuals


class Population(object):

    def __init__(self, species, population_size):
        self.population_size = population_size
        self.species = getattr(individuals, species)
        self.individuals = [self.species() for _ in xrange(population_size)]

    def get_individual(self, item):
        return self.individuals[item]

    def get_fittest(self):
        return min(self.individuals, key=operator.attrgetter('fitness'))

    def __getitem__(self, item):
        return self.get_individual(item)


if __name__ == '__main__':
    population = Population('XgboostIndividual', 100)
