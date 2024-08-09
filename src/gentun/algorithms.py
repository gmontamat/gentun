"""
Genetic algorithms
"""

import random

from typing import Optional

from .populations import Population
from .individuals import Individual


class GeneticAlgorithm:

    def __init__(self, population: Population):
        self.population = population
        self.current_generation = 1

    def run(
        self,
        generations: int,
        maximize: bool = True,
        patience: Optional[int] = None,
        verbose: bool = True
    ) -> None:
        """Run genetic algorithm for generations."""
        if patience:
            best_fitness = -float("inf") if maximize else float("inf")
            current_strike = 1
        while self.current_generation <= generations:
            if verbose:
                print(f"Running generation #{self.current_generation}...")
            self.evolve(maximize)
            fittest = self.population.get_fittest()
            fitness = fittest.evaluate_fitness()
            if verbose:
                print("Fittest individual:")
                print(fittest)
                print(f"Fitness value: {round(fitness, 4)}")
            if patience:
                if fitness <= best_fitness * (1. if maximize else -1.):
                    current_strike += 1
                else:
                    best_fitness = fitness
                    current_strike = 1
                if current_strike == patience:
                    if verbose:
                        print("Ran out of patience...")
                    break
            self.current_generation += 1
        print("Complete! Fittest individual:")
        print(fittest)

    def evolve(self, maximize: bool) -> None:
        """Run a single generation."""
        raise NotImplementedError


class Tournament(GeneticAlgorithm):
    """
    Evolve a population by running small-group
    tournaments to find fittest individuals on
    each generation. If elitism is set, the
    fittest individual of a generation will be
    part of the next one.
    TODO: add reference
    """

    def __init__(
        self,
        population: Population,
        tournament_size: int = 5,
        reproduction_rate: float = 0.5,
        mutation_rate: float = 0.015,
        elitism: bool = True
    ):
        super().__init__(population)
        self.tournament_size = tournament_size
        self.reproduction_rate = reproduction_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism  # if True, fittest individual survives
        assert len(self.population) > self.tournament_size, \
            "Population size must be larger than tournament size."

    def evolve(self, maximize: bool) -> None:
        # Define the new population
        new_population = self.population.duplicate()
        if self.elitism:
            new_population.add_individual(self.population.get_fittest(maximize))
        while len(new_population) < len(self.population):
            # Select offspring from tournament and mutate
            parent1 = self.run_tournament(maximize)
            parent2 = self.run_tournament(maximize)
            child = parent1.reproduce(parent2, self.reproduction_rate)
            child.mutate(self.mutation_rate)
            new_population.add_individual(child)
        self.population = new_population  # Garbage collection here?

    def run_tournament(self, maximize: bool) -> Individual:
        """Define a small random population and return the fittest individual."""
        tournament = self.population.duplicate(self.tournament_size)
        return tournament.get_fittest(maximize)


class RussianRoulette(GeneticAlgorithm):
    """
    Algorithm used by the Genetic CNN paper.
    http://arxiv.org/pdf/1703.01513
    """

    def __init__(
        self,
        population: Population,
        crossover_probability: float = 0.2,
        crossover_rate: float = 0.3,
        mutation_probability: float = 0.8,
        mutation_rate: float = 0.1
    ):
        super().__init__(population)
        self.crossover_probability = crossover_probability
        self.crossover_rate = crossover_rate
        self.mutation_probability = mutation_probability
        self.mutation_rate = mutation_rate

    def evolve(self, maximize: bool, eps: float = 1e-15):
        # Evaluate all individuals in this population
        _ = self.population.get_fittest(maximize)
        # Get weights for russian roulette
        if maximize:
            weights = [
                individual.evaluate_fitness()
                for individual in self.population
            ]
        else:
            weights = [
                1. / (individual.evaluate_fitness() + eps)
                for individual in self.population
            ]
        min_weight = min(weights)
        weights = [weight - min_weight for weight in weights]
        if sum(weights) == .0:
            weights = [1. for _ in self.population]
        # Sample with replacement using weights
        new_population = self.population.duplicate()
        for i in random.choices(range(len(self.population)), weights=weights, k=len(self.population)):
            # We have to make copies of the individuals we
            # select, since they may be selected again.
            new_population.add_individual(self.population[i].duplicate())
        # Crossover and mutation
        for i in range(len(new_population) // 2):
            if random.random() < self.crossover_probability:
                new_population[2 * i - 1].crossover(
                    new_population[2 * i], self.crossover_rate
                )
            else:
                if random.random() < self.mutation_probability:
                    new_population[2 * i - 1].mutate(self.mutation_rate)
                if random.random() < self.mutation_probability:
                    new_population[2 * i].mutate(self.mutation_rate)
        self.population = new_population  # Garbage collection here?
