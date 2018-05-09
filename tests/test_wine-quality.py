#!/usr/bin/env python
"""
Test the genetic algorithm on a single machine over the
'wine-quality' data using a random population as starter.
"""

import os
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    from gentun import GeneticAlgorithm, Population, XgboostIndividual

    data = pd.read_csv('./data/winequality-white.csv', delimiter=';')
    y_train = data['quality']
    x_train = data.drop(['quality'], axis=1)
    pop = Population(XgboostIndividual, x_train, y_train, size=100, additional_parameters={'nfold': 3})
    ga = GeneticAlgorithm(pop)
    ga.run(10)
