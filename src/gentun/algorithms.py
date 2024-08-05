"""
Genetic algorithm class
"""

import random

from .populations import Population


class GeneticAlgorithm:
    """
    Evolve a population iteratively to find better
    individuals on each generation. If elitism is set, the
    fittest individual of a generation will be part of the
    next one.
    """

    def __init__(self,
                 population: Population,
                 tournament_size: int = 5,
                 elitism: bool = True):
        self.population = population
        self.x_train, self.y_train = self.population.get_data()
        self.tournament_size = tournament_size
        self.elitism = elitism  # if True, fittest individual survives
        self.generation = 1

    def get_population_type(self):
        return self.population.__class__

    def run(self, max_generations: int):
        print("Starting genetic algorithm...")
        print()
        while self.generation <= max_generations:
            self.evolve_population()
            self.generation += 1

    def evolve_population(self):
        if self.population.get_size() < self.tournament_size:
            raise ValueError("Population size is smaller than tournament size.")
        print(f"Evaluating generation #{self.generation}...")
        fittest = self.population.get_fittest()
        print(f"Fittest individual is: {fittest}")
        print(f"Fitness value is: {round(fittest.get_fitness(), 4)}")
        print()
        new_population = self.get_population_type()(
            self.population.get_species(), self.x_train, self.y_train, individual_list=[],
            maximize=self.population.get_fitness_criteria()
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
            ], maximize=self.population.get_fitness_criteria()
        )
        return tournament.get_fittest()


class RussianRouletteGA(GeneticAlgorithm):
    """
    Simpler genetic algorithm used in the Genetic CNN paper.
    """

    def __init__(self, population: Population,
                 crossover_probability: int = 0.2,
                 mutation_probability: int = 0.8):
        super(RussianRouletteGA, self).__init__(population)
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
