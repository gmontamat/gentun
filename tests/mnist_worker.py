#!/usr/bin/env python
"""
Create a worker which loads MNIST data and waits
for a job. The rabbitmq daemon should be running
locally.
"""

import os
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    import random
    from sklearn.datasets import fetch_mldata
    from sklearn.preprocessing import LabelBinarizer
    from gentun import GentunWorker, GeneticCnnModel

    mnist = fetch_mldata('MNIST original', data_home='./data')
    lb = LabelBinarizer()
    lb.fit(range(max(mnist.target.astype('int')) + 1))
    selection = random.sample(range(mnist.data.shape[0]), 10000)
    y_train = lb.transform(mnist.target.astype('int'))[selection]
    x_train = mnist.data.reshape(mnist.data.shape[0], 28, 28, 1)[selection]
    x_train = x_train / 255  # Normalize train data

    gw = GentunWorker(GeneticCnnModel, x_train, y_train)
    gw.work()
