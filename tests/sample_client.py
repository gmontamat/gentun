#!/usr/bin/env python
"""
Create a client which loads California Housing data and
waits for a job. The rabbitmq server should be running
locally.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    from sklearn.datasets import fetch_california_housing
    from gentun import GentunClient, XgboostIndividual

    data = fetch_california_housing()
    y = data.target
    x = data.data
    gw = GentunClient(XgboostIndividual, x, y)
    gw.work()
