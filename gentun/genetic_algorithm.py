#!/usr/bin/env python
"""
Genetic algorithm and population classes
"""

import operator
import random


class Population(object):
    """Group of individuals of the same species, that is,
    with the same genome. Can be initialized either with a
    list of individuals or a population size so that
    random individuals are created. The get_fittest method
    returns the strongest individual.
    """

    def __init__(self, species, x_train, y_train, individual_list=None, size=None,
                 uniform_rate=0.5, mutation_rate=0.015, additional_parameters=None):
        self.x_train = x_train
        self.y_train = y_train
        self.species = species
        if individual_list is None and size is None:
            raise ValueError("Either pass a list of individuals or set a population size for a random one.")
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
            assert all([type(individual) is self.species for individual in individual_list])
            self.population_size = len(individual_list)
            self.individuals = individual_list

    def add_individual(self, individual):
        assert type(individual) is self.species
        self.individuals.append(individual)
        self.population_size += 1

    def get_species(self):
        return self.species

    def get_size(self):
        return self.population_size

    def get_fittest(self):
        return min(self.individuals, key=operator.methodcaller('get_fitness'))

    def get_data(self):
        return self.x_train, self.y_train

    def __getitem__(self, item):
        return self.individuals[item]


class GeneticAlgorithm(object):
    """Evolve a population iteratively to find better
    individuals on each generation. If elitism is set, the
    fittest individual of a generation will be part of the
    next one.
    """

    def __init__(self, population, tournament_size=5, elitism=True):
        self.population = population
        self.x_train, self.y_train = self.population.get_data()
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.generation = 1

    def get_population_type(self):
        return self.population.__class__

    def run(self, max_generations):
        print("Starting genetic algorithm...\n")
        while self.generation <= max_generations:
            self.evolve_population()
            self.generation += 1

    def evolve_population(self):
        print("Evaluating generation #{}...".format(self.generation))
        fittest = self.population.get_fittest()
        print("Fittest individual is:")
        print(fittest)
        print("Fitness value is: {}\n".format(round(fittest.get_fitness(), 4)))
        new_population = self.get_population_type()(
            self.population.get_species(), self.x_train, self.y_train, individual_list=[]
        )
        if self.elitism:
            new_population.add_individual(self.population.get_fittest())
        while new_population.get_size() < self.population.get_size():
            child = self.tournament_select().reproduce(self.tournament_select())
            child.mutate()
            new_population.add_individual(child)
        self.population = new_population

    def tournament_select(self):
        tournament = self.get_population_type()(
            self.population.get_species(), self.x_train, self.y_train, individual_list=[
                self.population[i] for i in random.sample(range(self.population.get_size()), self.tournament_size)
            ]
        )
        return tournament.get_fittest()


if __name__ == '__main__':
    import pandas as pd
    from individuals import XgboostIndividual

    data = pd.read_csv('../tests/wine-quality/winequality-white.csv', delimiter=';')
    y_train = data['quality']
    x_train = data.drop(['quality'], axis=1)
    pop = Population(XgboostIndividual, x_train, y_train, size=100, additional_parameters={'nfold': 3})
    ga = GeneticAlgorithm(pop)
    ga.run(10)
