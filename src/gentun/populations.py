#!/usr/bin/env python
"""
Population class
"""

import itertools
import operator

from typing import Any, Dict, List, Optional, Type

from .individuals import Individual
from .models import Model
from .genes import Gene


class Population:
    """
    Group of individuals that share the same parameters.
    Can be initialized either with a list of individuals
    or a population size so that random individuals are
    created. The get_fittest method returns the strongest
    individual.
    """

    def __init__(self,
                 model: Type[Model],
                 genes: List[Gene],
                 x_train: Any,
                 y_train: Any,
                 individuals: Optional[List[Dict[str, Any]]] = None,
                 size: Optional[int] = None,
                 crossover_rate: float = 0.5,
                 mutation_rate: float = 0.015,
                 maximize: bool = True,
                 **kwargs):
        self.model = model
        self.genes = genes
        self.x_train = x_train
        self.y_train = y_train
        self.maximize = maximize  # if True, we maximize fitness
        self.parameters = kwargs
        if individuals is None and size is None:
            raise ValueError("Pass a list of individuals or define the population size to create a random population.")
        elif individuals is None:
            # Create a random population
            self.population_size = size
            self.individuals = [
                Individual(
                    self.model,
                    self.x_train,
                    self.y_train,
                    {},
                    **kwargs
                )
                for _ in range(size)
            ]
            print(f"Initializing a random population of size: {size}")
        else:
            self.population_size = len(individuals)
            self.individuals = [
                Individual(
                    self.model,
                    self.x_train,
                    self.y_train,
                    hyperparameters=hyperparameters,
                    **kwargs
                )
                for hyperparameters in individuals
            ]

    def add_individual(self, individual):
        assert type(individual) is self.species
        self.individuals.append(individual)
        self.population_size += 1

    def get_size(self):
        return self.population_size

    def get_fittest(self):
        if self.maximize:
            return max(self.individuals, key=operator.methodcaller('evaluate_fitness'))
        return min(self.individuals, key=operator.methodcaller('evaluate_fitness'))

    def __getitem__(self, item):
        return self.individuals[item]

    def reproduce(self, individual1, individual2):
        """Mix genes from self and partner randomly and
        return a new instance of an individual. Do not
        mutate parents.
        """
        child_genes = {}
        for name, value in self.get_genes().items():
            if random.random() < self.crossover_rate:
                child_genes[name] = partner.get_genes()[name]
            else:
                child_genes[name] = value
        return Individual(

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


class GridPopulation(Population):
    """Population whose individuals are created based on a
     grid search approach instead of randomly. Can be
     initialized either with a list of individuals (in
     which case it behaves like a Population) or with a
     dictionary of genes and grid values pairs.
     """

    def __init__(self, species: Type[Individual], x_train: Any, y_train: Any,
                 individual_list=None,
                 genes_grid=None,
                 crossover_rate: float = 0.5,
                 mutation_rate: float = 0.015,
                 maximize: bool = True,
                 additional_parameters: Optional[Dict[str, Any]] = None):
        if individual_list is None and genes_grid is None:
            raise ValueError("Pass a list of individuals or a grid definition.")
        elif genes_grid is not None:
            genome = species(None, None).get_genome()  # Get species' genome
            if not set(genes_grid.keys()).issubset(set(genome.keys())):
                raise ValueError("Some grid parameters do not belong to the species' genome")
            # Fill genes_grid with default parameters
            for gene, properties in genome.items():
                if gene not in genes_grid:
                    genes_grid[gene] = [properties[0]]  # Use default value
            individual_list = [
                species(
                    x_train, y_train, genes=genes, crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate, **additional_parameters
                )
                for genes in (
                    dict(zip(genes_grid, x))
                    for x in itertools.product(*genes_grid.values())
                )
            ]
            print("Initializing a grid population. Size: {}".format(len(individual_list)))
        super(GridPopulation, self).__init__(
            species, x_train, y_train, individual_list, None, crossover_rate, mutation_rate,
            maximize, additional_parameters
        )
