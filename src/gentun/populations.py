"""
Population
"""
from __future__ import annotations

import itertools
import operator
import random
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from .genes import Gene
from .individuals import Individual
from .models.base import Model


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
        individuals: Optional[Union[List[Dict[str, Any]], List[Individual], int]] = None,
        **kwargs,
    ):
        self.genes = genes
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        # Static parameters used to create model
        self.kwargs = kwargs
        # Create individuals
        if isinstance(individuals, int):
            # Random population
            self.individuals = [self.spawn() for _ in range(individuals)]
        elif isinstance(individuals, list):
            self.individuals = []
            for individual in individuals:
                # Here an individual can be an instance or the hyperparameters
                self.add_individual(individual)
        else:
            raise ValueError("'individuals' must be a `int` or a `list`.")

    def spawn(self, hyperparameters: Optional[Dict[str, Any]] = None) -> Individual:
        """Return an individual from this population."""
        if hyperparameters is None:
            # Create a random individual
            hyperparameters = {str(gene): gene() for gene in self.genes}
        else:
            # Hyperparameters passed, check for missing ones
            for gene in self.genes:
                if str(gene) not in hyperparameters:
                    raise KeyError(f"Missing hyperparameter '{str(gene)}'.")
        return Individual(
            self.genes, self.model, self.x_train, self.y_train, hyperparameters=hyperparameters, **self.kwargs
        )

    def add_individual(self, individual: Optional[Union[Dict[str, Any], Individual]] = None) -> None:
        """Add an individual to this population."""
        if isinstance(individual, dict) or individual is None:
            self.individuals.append(self.spawn(individual))
        elif isinstance(individual, Individual):
            self.individuals.append(individual)
        else:
            raise ValueError

    def get_fittest(self, maximize: bool = True) -> Individual:
        if maximize:
            return max(self.individuals, key=operator.methodcaller("evaluate_fitness"))
        return min(self.individuals, key=operator.methodcaller("evaluate_fitness"))

    def get_genes(self) -> List[Gene]:
        return self.genes

    def duplicate(self, sample_size: int = 0) -> Population:
        """
        Creates an identical population. If sample_size > 0,
        sample random individuals from population without
        replacement.
        """
        individuals = random.sample(self.individuals, sample_size)
        return Population(self.genes, self.model, self.x_train, self.y_train, individuals, **self.kwargs)

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self) -> Iterator[Individual]:
        return iter(self.individuals)

    def __getitem__(self, item: Union[int, slice]) -> Union[Individual, List[Individual]]:
        return self.individuals[item]


# TODO: re-implement
class GridPopulation(Population):
    """Population whose individuals are created based on a
    grid search approach instead of randomly. Can be
    initialized either with a list of individuals (in
    which case it behaves like a Population) or with a
    dictionary of genes and grid values pairs.
    """

    def __init__(
        self,
        species: Type[Individual],
        x_train: Any,
        y_train: Any,
        individual_list=None,
        genes_grid=None,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.015,
        maximize: bool = True,
        additional_parameters: Optional[Dict[str, Any]] = None,
    ):
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
                    x_train,
                    y_train,
                    genes=genes,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    **additional_parameters,
                )
                for genes in (dict(zip(genes_grid, x)) for x in itertools.product(*genes_grid.values()))
            ]
            print("Initializing a grid population. Size: {}".format(len(individual_list)))
        super(GridPopulation, self).__init__(
            species,
            x_train,
            y_train,
            individual_list,
            None,
            crossover_rate,
            mutation_rate,
            maximize,
            additional_parameters,
        )
