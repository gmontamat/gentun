#!/usr/bin/env python
"""
Basic genetic algorithm in Python
"""

import operator
import random

# Custom definitions of individuals and its genes
import individuals


class Population(object):

    def __init__(self, species, x_train, y_train, individual_list=None, size=None,
                 uniform_rate=0.5, mutation_rate=0.015, additional_parameters=None):
        self.x_train = x_train
        self.y_train = y_train
        self.species = getattr(individuals, species)
        if individual_list is None and size is None:
            raise ValueError("Either pass a list of individuals or set a population size for a random creation")
        elif individual_list is None:
            if additional_parameters is None:
                additional_parameters = {}
            self.population_size = size
            self.individuals = [
                self.species(
                    self.x_train, self.y_train, uniform_rate=uniform_rate,
                    mutation_rate=mutation_rate, **additional_parameters
                )
                for _ in xrange(size)
            ]
        else:
            assert all([self.species == individual.__class__ for individual in individual_list])
            self.population_size = len(individual_list)
            self.individuals = individual_list

    def add_individual(self, individual):
        assert self.species == individual.__class__
        self.individuals.append(individual)
        self.population_size += 1

    def get_species(self):
        return self.species.__name__

    def get_size(self):
        return self.population_size

    def get_fittest(self):
        return min(self.individuals, key=operator.methodcaller('get_fitness'))

    def get_data(self):
        return self.x_train, self.y_train

    def __getitem__(self, item):
        return self.individuals[item]


class GeneticAlgorithm(object):

    def __init__(self, population, tournament_size=5, elitism=True):
        self.population = population
        self.x_train, self.y_train = self.population.get_data()
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.generation = 0

    def run(self, max_generations):
        print 'Starting genetic algorithm...'
        print
        while self.generation < max_generations:
            self.evolve_population()
            self.generation += 1

    def evolve_population(self):
        print 'Generation #{}, fittest individual is:'.format(self.generation)
        print self.population.get_fittest()
        print
        new_population = Population(self.population.get_species(), self.x_train, self.y_train, individual_list=[])
        if self.elitism:
            new_population.add_individual(self.population.get_fittest())
        while new_population.get_size() < self.population.get_size():
            child = self.tournament_select().reproduce(self.tournament_select())
            child.mutate()
            new_population.add_individual(child)
        self.population = new_population

    def tournament_select(self):
        tournament = Population(
            self.population.get_species(), self.x_train, self.y_train, individual_list=[
                self.population[i] for i in random.sample(range(self.population.get_size()), self.tournament_size)
            ]
        )
        return tournament.get_fittest()


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv('../tests/wine-quality/winequality-white.csv', delimiter=';')
    y_train = data['quality']
    x_train = data.drop(['quality'], axis=1)
    pop = Population('XgboostIndividual', x_train, y_train, size=100, additional_parameters={'nfold': 3})
    ga = GeneticAlgorithm(pop)
    ga.run(10)
