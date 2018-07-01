#!/usr/bin/env python
"""
Create a 'DistributedPopulation' which generates a local
queue of jobs to run individuals in parallel. The rabbitmq
daemon should be running locally and at least one sample
worker too.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    from gentun import GeneticAlgorithm, DistributedPopulation, XgboostIndividual

    pop = DistributedPopulation(XgboostIndividual, size=100, additional_parameters={'nfold': 3}, maximize=False)
    ga = GeneticAlgorithm(pop)
    ga.run(10)
