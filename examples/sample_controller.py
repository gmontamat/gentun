#!/usr/bin/env python
"""
Test the genetic algorithm on multiple nodes using the Dummy model
which sums hyperparameter values.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

if __name__ == "__main__":
    from gentun.algorithms import RussianRoulette, Tournament
    from gentun.genes import RandomChoice
    from gentun.models.base import Dummy
    from gentun.populations import Population
    from gentun.services import RedisController

    genes = [RandomChoice(f"hyperparam_{i}", [0, 1, 2]) for i in range(10)]
    # This assumes you're running a Redis server on localhost in port 6379
    # The simplest way to set it up is via docker:
    # docker run -d --rm --name gentun-redis -p 6379:6379 redis
    controller = RedisController("test", host="localhost", port=6379)

    # Run russian roulette with a population of 20 for 50 generations
    population = Population(genes, Dummy, 20, controller=controller)
    algorithm = RussianRoulette(population)
    algorithm.run(50)

    # Run tournament select with a population of 50 for 20 generations
    population = Population(genes, Dummy, 50, controller=controller)
    algorithm = Tournament(population)
    algorithm.run(20, patience=3)
