"""
A collection of individuals with
different gene values (hyperparameters)
that will be evaluated with train data.
"""
from __future__ import annotations

import itertools
import operator
import random
from typing import Any, Dict, Iterator, Optional, Sequence, Type, Union

import numpy as np

from .genes import Gene
from .individuals import Individual
from .models.base import Handler
from .services import RedisController


class Population:
    """
    Group of individuals that share the same genes. Can be initialized
    with a sequence of individuals or a population size, in which case,
    random individuals are created. The get_fittest() method returns
    the strongest individual.
    """

    def __init__(
        self,
        genes: Sequence[Gene],
        handler: Type[Handler],
        individuals: Union[Sequence[Dict[str, Any]], Sequence[Individual], int],
        x_train: Any = None,
        y_train: Any = None,
        controller: Optional[RedisController] = None,
        **kwargs,
    ):
        self.genes = genes
        self.handler = handler
        # Static parameters used to create model
        self.kwargs = kwargs
        # Handle either train data or queueing server
        if controller is not None:
            if x_train is not None or y_train is not None:
                print("Warning: `x_train` and `y_train` ignored, using server instead.")
        else:
            if x_train is None:
                raise ValueError("Missing `x_train`.")
            if y_train is None:
                raise ValueError("Missing `y_train`.")
        self.x_train = x_train
        self.y_train = y_train
        self.controller = controller
        # Create individuals
        if isinstance(individuals, int):
            # Random population
            self.individuals = [self.spawn() for _ in range(individuals)]
        elif isinstance(individuals, Sequence):
            self.individuals = []
            for individual in individuals:
                # Here an individual can be an instance or the hyperparameters
                self.add_individual(individual)
        else:
            raise ValueError("'individuals' must be a `int` or a sequence.")

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
            self.genes, self.handler, self.x_train, self.y_train, hyperparameters=hyperparameters, **self.kwargs
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
        """Return the fittest individual of the population."""
        if self.controller is not None:
            # Evaluate a population in parallel
            for individual in self.individuals:
                if individual.get_fitness() is None:
                    individual.send_to_queue(self.controller)
            for individual in self.individuals:
                if individual.get_fitness() is None:
                    individual.read_from_queue(self.controller)
        if maximize:
            return max(self.individuals, key=operator.methodcaller("evaluate_fitness"))
        return min(self.individuals, key=operator.methodcaller("evaluate_fitness"))

    def duplicate(self, sample_size: int = 0) -> Population:
        """
        Create an identical population. If sample_size > 0, sample
        random individuals from population without replacement.
        """
        individuals = random.sample(self.individuals, sample_size)
        return Population(
            self.genes, self.handler, individuals, self.x_train, self.y_train, self.controller, **self.kwargs
        )

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self) -> Iterator[Individual]:
        """Enable `for individual in population` usage."""
        return iter(self.individuals)

    def __getitem__(self, item: Union[int, slice]) -> Union[Individual, Sequence[Individual]]:
        return self.individuals[item]


class Grid(Population):
    """
    Population whose individuals are created based on a grid search
    approach instead of at random.
    """

    def __init__(
        self,
        genes: Sequence[Gene],
        handler: Type[Handler],
        gene_samples: Union[int, Sequence[int]],
        x_train: Any = None,
        y_train: Any = None,
        controller: Optional[Handler] = None,
        **kwargs,
    ):
        super().__init__(genes, handler, [], x_train, y_train, controller, **kwargs)
        # Define the grid and add individuals
        if isinstance(gene_samples, Sequence):
            assert len(gene_samples) == len(genes), "`genes` and `gene_samples` must have the same length."
        else:
            gene_samples = [gene_samples] * len(genes)
        genes_grid = {}
        for gene, samples in zip(genes, gene_samples):
            genes_grid[str(gene)] = [gene.sample(percentile) for percentile in np.linspace(0.0, 1.0, samples)]
        for hyperparams in (dict(zip(genes_grid, value)) for value in itertools.product(*genes_grid.values())):
            self.add_individual(hyperparams)
