"""
Genetic algorithms
"""

import random

from .populations import Population
from .individuals import Individual


class GeneticAlgorithm:

    def __init__(self, population: Population):
        self.population = population
        self.genes = self.population.get_genes()
        self.current_generation = 1

    def run(self, generations: int, verbose: bool = True) -> None:
        """Run genetic algorithm for generations."""
        while self.current_generation <= generations:
            if verbose:
                print(f"Running generation #{self.current_generation}...")
            self.evolve()
            if verbose:
                fittest = self.population.get_fittest()
                print("Fittest individual:")
                print(fittest)
                print(f"Fitness value: {round(fittest.evaluate_fitness(), 4)}")
            self.current_generation += 1

    def evolve(self) -> None:
        """Run a single generation."""
        raise NotImplementedError


class Tournament(GeneticAlgorithm):
    """
    Evolve a population by running small-group
    tournaments to find fittest individuals on
    each generation. If elitism is set, the
    fittest individual of a generation will be
    part of the next one.
    TODO: missing reference
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

    def evolve(self) -> None:
        # Define the new population
        new_population = self.population.duplicate()
        if self.elitism:
            new_population.add_individual(self.population.get_fittest())
        while len(new_population) < len(self.population):
            # Select offspring from tournament and mutate
            parent1 = self.run_tournament()
            parent2 = self.run_tournament()
            child = parent1.reproduce(parent2, self.reproduction_rate)
            child.mutate(self.mutation_rate)
            new_population.add_individual(child)
        self.population = new_population  # Garbage collection here?

    def run_tournament(self) -> Individual:
        """Define a small random population and return the fittest individual."""
        tournament = self.population.duplicate(self.tournament_size)
        return tournament.get_fittest()


# TODO: re-implement
class RussianRoulette(GeneticAlgorithm):
    """
    Algorithm used by the Genetic CNN paper.
    TODO: arxiv
    """

    def __init__(self, population: Population,
                 crossover_probability: int = 0.2,
                 mutation_probability: int = 0.8):
        super().__init__(population)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability

    def evolve_population(self, eps: float = 1e-15):
        print(f"Evaluating generation #{self.generation}...")
        fittest = self.population.get_fittest()
        print(f"Fittest individual is: {fittest}")
        print(f"Fitness value is: {round(fittest.get_fitness(), 4)}")
        print()
        # Russian roulette selection
        if self.population.get_fitness_criteria():
            weights = [self.population[i].get_fitness() for i in range(self.population.get_size())]
        else:
            weights = [1 / (self.population[i].get_fitness() + eps) for i in range(self.population.get_size())]
        min_weight = min(weights)
        weights = [weight - min_weight for weight in weights]
        if sum(weights) == .0:
            weights = [1. for _ in range(self.population.get_size())]
        new_population = self.get_population_type()(
            self.population.get_species(), self.x_train, self.y_train, individual_list=[
                self.population[i].copy() for i in random.choices(
                    range(self.population.get_size()), weights=weights, k=self.population.get_size()
                )
            ], maximize=self.population.get_fitness_criteria()
        )
        # Crossover and mutation
        for i in range(new_population.get_size() // 2):
            if random.random() < self.crossover_probability:
                new_population[i].crossover(new_population[i + 1])
            else:
                if random.random() < self.mutation_probability:
                    new_population[i].mutate()
                if random.random() < self.mutation_probability:
                    new_population[i + 1].mutate()
        self.population = new_population
