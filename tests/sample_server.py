#!/usr/bin/env python
"""
Create a DistributedPopulation which generates a queue of
jobs to evaluate individuals in parallel. The rabbitmq
service should be running in 'localhost'.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    from gentun import GeneticAlgorithm, DistributedPopulation, XgboostIndividual

    pop = DistributedPopulation(
        XgboostIndividual, size=100, additional_parameters={'nfold': 3}, maximize=False,
        host='localhost', user='guest', password='guest'
    )
    ga = GeneticAlgorithm(pop)
    ga.run(10)
