#!/usr/bin/env python
"""
Test the genetic algorithm on a single machine over the
'wine-quality' data using a grid as an initial population.
"""

import os
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    from gentun import GeneticAlgorithm, GridPopulation, XgboostIndividual

    data = pd.read_csv('./data/winequality-white.csv', delimiter=';')
    y_train = data['quality']
    x_train = data.drop(['quality'], axis=1)
    grid = {
        'eta': [0.001, 0.005, 0.01, 0.015, 0.2],
        'max_depth': range(3, 11),
        'colsample_bytree': [0.80, 0.85, 0.90, 0.95, 1.0],
    }
    pop = GridPopulation(
        XgboostIndividual, x_train, y_train, genes_grid=grid,
        additional_parameters={'nfold': 3}, maximize=False
    )
    ga = GeneticAlgorithm(pop)
    ga.run(10)
