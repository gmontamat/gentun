#!/usr/bin/env python
"""
Implementation of a distributed version of the Genetic CNN
algorithm on MNIST data. The rabbitmq server should be
running locally.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    from gentun import RussianRouletteGA, DistributedPopulation, GeneticCnnIndividual

    pop = DistributedPopulation(
        GeneticCnnIndividual, size=20, crossover_rate=0.3, mutation_rate=0.1,
        additional_parameters={
            'nfold': 5, 'epochs': (20, 4, 1), 'learning_rate': (1e-3, 1e-4, 1e-5), 'batch_size': 32
        }, maximize=True
    )
    ga = RussianRouletteGA(pop, crossover_probability=0.2, mutation_probability=0.8)
    ga.run(50)
