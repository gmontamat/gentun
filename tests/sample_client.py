#!/usr/bin/env python
"""
Create a client which loads 'wine-quality' data and waits
for a job. The rabbitmq server should be running locally.
"""

import os
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    from gentun import GentunClient, XgboostIndividual

    data = pd.read_csv('../tests/data/winequality-white.csv', delimiter=';')
    y = data['quality']
    x = data.drop(['quality'], axis=1)
    gw = GentunClient(XgboostIndividual, x, y)
    gw.work()
