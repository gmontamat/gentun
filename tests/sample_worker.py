#!/usr/bin/env python
"""
Create a worker which loads 'wine-quality' data and waits
for a job. The rabbitmq daemon should be running locally.
"""

import os
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    from gentun import GentunWorker, XgboostModel

    data = pd.read_csv('../tests/data/winequality-white.csv', delimiter=';')
    y = data['quality']
    x = data.drop(['quality'], axis=1)
    gw = GentunWorker(XgboostModel, x, y)
    gw.work()
