#!/usr/bin/env python
"""
Test the GeneticCnnModel using the MNIST dataset
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    from sklearn.datasets import fetch_mldata
    from sklearn.preprocessing import LabelBinarizer
    from gentun import GeneticCnnModel

    mnist = fetch_mldata('MNIST original', data_home='./data')
    lb = LabelBinarizer()
    lb.fit(range(max(mnist.target.astype('int')) + 1))
    y_train = lb.transform(mnist.target.astype('int'))
    x_train = mnist.data.reshape(mnist.data.shape[0], 28, 28, 1)
    model = GeneticCnnModel(x_train, y_train, {'S_1': '', 'S_2': ''}, (20, 50), ((5, 5), (5, 5)), (28, 28, 1), 10)
    model.cross_validate()
