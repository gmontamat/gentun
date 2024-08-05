"""
Define a group of individuals
"""

import itertools
import operator

from typing import Any, Dict, List, Optional, Type

from .individuals import Individual
from .models import Model
from .genes import Gene


class Population:
    """
    Group of individuals that share the same genes.
    Can be initialized either with a list of individuals
    or a population size so that random individuals are
    created. The get_fittest method returns the strongest
    individual.
    """

    def __init__(
        self,
        model: Type[Model],
        genes: List[Gene],
        x_train: Any,
        y_train: Any,
        individuals: Optional[List[Dict[str, Any]]] = None,
        size: Optional[int] = None,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.015,
        maximize: bool = True,
        **kwargs
    ):
        self.model = model
        self.genes = genes
        self.x_train = x_train
        self.y_train = y_train
        self.maximize = maximize  # if True, we maximize fitness
        self.parameters = kwargs
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
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
                    {str(gene): gene() for gene in self.genes},
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

    def add_individual(self, individual: Individual):
        self.individuals.append(individual)
        self.population_size += 1

    def get_size(self) -> int:
        return self.population_size

    def get_fittest(self) -> Individual:
        if self.maximize:
            return max(self.individuals, key=operator.methodcaller('evaluate_fitness'))
        return min(self.individuals, key=operator.methodcaller('evaluate_fitness'))

    def __getitem__(self, item):
        return self.individuals[item]


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
