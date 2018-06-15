#!/usr/bin/env python
"""
Population class
"""

import itertools
import operator


class Population(object):
    """Group of individuals of the same species, that is,
    with the same genome. Can be initialized either with a
    list of individuals or a population size so that
    random individuals are created. The get_fittest method
    returns the strongest individual.
    """

    def __init__(self, species, x_train, y_train, individual_list=None, size=None,
                 uniform_rate=0.5, mutation_rate=0.015, additional_parameters=None,
                 minimize=True):
        self.x_train = x_train
        self.y_train = y_train
        self.species = species
        self.minimize = minimize
        if individual_list is None and size is None:
            raise ValueError("Either pass a list of individuals or a population size for a random population.")
        elif individual_list is None:
            if additional_parameters is None:
                additional_parameters = {}
            self.population_size = size
            self.individuals = [
                self.species(
                    self.x_train, self.y_train, uniform_rate=uniform_rate,
                    mutation_rate=mutation_rate, **additional_parameters
                )
                for _ in range(size)
            ]
            print("Using a random population. Size: {}".format(size))
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
        if self.minimize:
            return min(self.individuals, key=operator.methodcaller('get_fitness'))
        return max(self.individuals, key=operator.methodcaller('get_fitness'))

    def get_data(self):
        return self.x_train, self.y_train

    def __getitem__(self, item):
        return self.individuals[item]


class GridPopulation(Population):
    """Population whose individuals are created based on a
     grid search approach instead of randomly. Can be
     initialized either with a list of individuals (in
     which case it behaves like a Population) or with a
     dictionary of genes and grid values pairs.
     """

    def __init__(self, species, x_train, y_train, individual_list=None, genes_grid=None,
                 uniform_rate=0.5, mutation_rate=0.015, additional_parameters=None,
                 minimize=True):
        if individual_list is None and genes_grid is None:
            raise ValueError("Either pass a list of individuals or a grid definition.")
        elif genes_grid is not None:
            genome = species(None, None).get_genome()  # Get species' genome
            if not set(genes_grid.keys()).issubset(set(genome.keys())):
                raise ValueError("Some grid parameters do not belong to the species' genome")
            # Fill genes_grid with default parameters
            for gene, properties in genome.iteritems():
                if gene not in genes_grid:
                    genes_grid[gene] = [properties[0]]  # Use default value
            individual_list = [
                species(
                    x_train, y_train, genes=genes, uniform_rate=uniform_rate,
                    mutation_rate=mutation_rate, **additional_parameters
                )
                for genes in (
                    dict(zip(genes_grid, x))
                    for x in itertools.product(*genes_grid.itervalues())
                )
            ]
            print("Using a grid population. Size: {}".format(len(individual_list)))
        super(GridPopulation, self).__init__(
            species, x_train, y_train, individual_list, None, uniform_rate, mutation_rate,
            additional_parameters, minimize
        )
