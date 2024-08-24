#!/usr/bin/env python
"""
Test the genetic algorithm on a single
node using the dummy model which sums
hyperparameter values.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

if __name__ == "__main__":
    from gentun.algorithms import RussianRoulette, Tournament
    from gentun.genes import RandomChoice
    from gentun.models.base import Dummy
    from gentun.populations import Population

    x_train = []
    y_train = []

    genes = [RandomChoice(f"hyperparam_{i}", [0, 1, 2]) for i in range(10)]

    # Run russian roulette with a population of 20 for 50 generations
    population = Population(genes, Dummy, 50, x_train, y_train)
    algorithm = RussianRoulette(population)
    algorithm.run(50)

    # Run tournament select with a population of 50 for 20 generations
    population = Population(genes, Dummy, 50, x_train, y_train)
    algorithm = Tournament(population)
    algorithm.run(20, patience=3)
