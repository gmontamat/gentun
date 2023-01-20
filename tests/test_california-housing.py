#!/usr/bin/env python
"""
Test the genetic algorithm on a single machine over the
California Housing data using a random population.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    from sklearn.datasets import fetch_california_housing
    from gentun import Population
    from gentun.genetic_algorithms.genetic_algorithm import GeneticAlgorithm
    from gentun.individuals.xgboost_individual import XgboostIndividual

    data = fetch_california_housing()
    y_train = data.target
    x_train = data.data

    pop = Population(
        XgboostIndividual, x_train, y_train, size=100,
        additional_parameters={'kfold': 3}, maximize=False
    )
    ga = GeneticAlgorithm(pop)
    ga.run(10)
