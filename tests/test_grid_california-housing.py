#!/usr/bin/env python
"""
Test the genetic algorithm on a single machine over the
California Housing data using a grid as an initial
population.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    from sklearn.datasets import fetch_california_housing
    from gentun import GridPopulation
    from gentun.genetic_algorithms.genetic_algorithm import GeneticAlgorithm
    from gentun.individuals.xgboost_individual import XgboostIndividual

    data = fetch_california_housing()
    y_train = data.target
    x_train = data.data

    grid = {
        'eta': [0.001, 0.005, 0.01, 0.015, 0.2],
        'max_depth': range(3, 11),
        'colsample_bytree': [0.80, 0.85, 0.90, 0.95, 1.0],
    }
    pop = GridPopulation(
        XgboostIndividual, x_train, y_train, genes_grid=grid,
        additional_parameters={'kfold': 3}, maximize=False
    )
    ga = GeneticAlgorithm(pop)
    ga.run(10)
