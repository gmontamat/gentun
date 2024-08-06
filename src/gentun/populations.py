"""
Population
"""

import itertools
import operator

from typing import Any, Dict, List, Optional, Type, Union

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
        genes: List[Gene],
        model: Type[Model],
        x_train: Any,
        y_train: Any,
        individuals: Optional[Union[List[Dict[str, Any]], int]] = None,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.015,
        maximize: bool = True,
        **kwargs
    ):
        self.genes = genes
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        # Evolution parameters of the population
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.maximize = maximize  # if True, maximize fitness
        # Static parameters used to create model
        self.parameters = kwargs
        # Create individuals
        if isinstance(individuals, int):
            # Random population
            self.individuals = [
                self.create_individual()
                for _ in range(individuals)
            ]
        elif isinstance(individuals, list):
            self.individuals = [
                self.create_individual(hyperparams)
                for hyperparams in individuals
            ]
        else:
            raise ValueError("'individuals' must be a `int` or a `list`.")

    def create_individual(self, hyperparameters: Optional[Dict[str, Any]] = None) -> Individual:
        if hyperparameters is None:
            # Random individual
            hyperparameters = {str(gene): gene() for gene in self.genes}
        else:
            # Hyperparameters passed, check them
            for gene in self.genes:
                if str(gene) not in hyperparameters:
                    raise KeyError(f"Missing hyperparameter '{str(gene)}'.")
                if not gene.validate(hyperparameters[str(gene)]):
                    raise ValueError(
                        f"Invalid value `{hyperparameters[str(gene)]}` for gene '{str(gene)}'."
                    )
        return Individual(
            self.genes,
            self.model,
            self.x_train,
            self.y_train,
            hyperparameters=hyperparameters,
            **kwargs
        )

    def add_individual(self, hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        self.individuals.append(self.create_individual(hyperparameters))

    def get_fittest(self) -> Individual:
        if self.maximize:
            return max(self.individuals, key=operator.methodcaller('evaluate_fitness'))
        return min(self.individuals, key=operator.methodcaller('evaluate_fitness'))

    def __len__(self) -> int:
        return len(self.individuals)

    def __getitem__(self, item) -> Individual:
        return self.individuals[item]


# TODO: re-implement
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
