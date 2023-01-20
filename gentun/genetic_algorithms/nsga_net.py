#!/usr/bin/env python
"""
NSGANet class.
"""

import random

try:
    from .genetic_algorithm import GeneticAlgorithm
except ImportError:
    pass


class NSGANet(GeneticAlgorithm):
    """
    Genetic algorithm used in the Neural Architecture Search using Multi-Objective Genetic Algorithm paper.
    """

    def __init__(self, population, crossover_probability=0.2, mutation_probability=0.8):
        super(NSGANet, self).__init__(population)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
    
    def run(self, max_generations):
        print("Starting genetic algorithm...\n")
        print("Start exploration phase...\n")
        while self.generation <= max_generations:
            self.evolve_population()
            self.generation += 1

        print("Start exploitation phase...\n")
        self.bayesian_optimization_algorithm()

    def evolve_population(self, eps=1e-15):  # TODO: change it to NSGA-II
        print("Evaluating generation #{}...".format(self.generation))
        fittest = self.population.get_fittest()
        print("Fittest individual is:")
        print(fittest)
        print("Fitness value is: {}\n".format(round(fittest.get_fitness(), 4)))
        
        # NSGA selection
        if self.population.get_fitness_criteria():
            weights = [self.population[i].get_fitness() for i in range(self.population.get_size())]
        else:
            weights = [1 / (self.population[i].get_fitness() + eps) for i in range(self.population.get_size())]
        
        min_weight = min(weights)
        weights = [weight - min_weight for weight in weights]
        # TODO: why we are doing this?
        if sum(weights) == .0:
            weights = [1. for _ in range(self.population.get_size())]
        
        new_population = self.get_population_type()(
            self.population.get_species(), 
            self.x_train, 
            self.y_train, 
            individual_list=[
                self.population[i].copy() for i in random.choices(
                    range(self.population.get_size()), 
                    weights=weights, 
                    k=self.population.get_size()
                )
            ], 
            maximize=self.population.get_fitness_criteria()
        )

        # Crossover and mutation
        for i in range(new_population.get_size() // 2):
            if random.random() < self.crossover_probability:
                new_population[i].crossover(new_population[i + 1])
            else:  # chance to mutate both crossover chromosoms
                if random.random() < self.mutation_probability:
                    new_population[i].mutate()
                if random.random() < self.mutation_probability:
                    new_population[i + 1].mutate()
        self.population = new_population
    
    def bayesian_optimization_algorithm(self):
        """Sampling from the Bayesian Network (BN) constructed by NSGA-Net."""
        # TODO: add exploitation phase - BOA -> https://arxiv.org/pdf/1810.03522.pdf -> 3.2
        pass
