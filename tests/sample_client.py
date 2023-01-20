#!/usr/bin/env python
"""
Create a client which loads California Housing data and
waits for jobs to evaluate models. The rabbitmq service
should be running in 'localhost'.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    from sklearn.datasets import fetch_california_housing
    from gentun import GentunClient
    from gentun.individuals.xgboost_individual import XgboostIndividual

    data = fetch_california_housing()
    y_train = data.target
    x_train = data.data

    gc = GentunClient(XgboostIndividual, x_train, y_train, host='localhost', user='guest', password='guest')
    gc.work()
