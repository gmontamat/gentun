#!/usr/bin/env python
"""
Test the genetic algorithm on multiple nodes using the Dummy model
which sums hyperparameter values.
"""

from gentun.models.base import Dummy
from gentun.services import RedisWorker

if __name__ == "__main__":
    # This assumes you're running a Redis server on localhost in port 6379
    # The simplest way to set it up is via docker:
    # docker run -d --rm --name gentun-redis -p 6379:6379 redis
    worker = RedisWorker("test", Dummy, host="localhost", port=6379)

    x_train, y_train = [], []
    x_test, y_test = [], []

    # Start worker process
    worker.run(x_train, y_train, x_test, y_test)
