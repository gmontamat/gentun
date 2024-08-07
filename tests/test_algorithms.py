#!/usr/bin/env python
"""
Test the genetic algorithm on a single node
using the dummy model which sums
hyperparameter values.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

if __name__ == '__main__':
    from gentun.algorithms import Tournament
    from gentun.genes import RandomChoice
    from gentun.populations import Population
    from gentun.models import DummyModel

    y_train = None
    x_train = None

    genes = [
        RandomChoice(f"hyperparam_{i}", [0, 1, 2])
        for i in range(10)
    ]

    population = Population(
        genes,
        DummyModel,
        x_train,
        y_train,
        50
    )

    algorithm = Tournament(population)
    algorithm.run(10)
