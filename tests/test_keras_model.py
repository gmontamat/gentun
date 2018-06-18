#!/usr/bin/env python
"""
Test the GeneticCnnModel using the MNIST dataset
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    import random
    from sklearn.datasets import fetch_mldata
    from sklearn.preprocessing import LabelBinarizer
    from gentun import GeneticCnnModel

    mnist = fetch_mldata('MNIST original', data_home='./data')
    lb = LabelBinarizer()
    lb.fit(range(max(mnist.target.astype('int')) + 1))
    selection = random.sample(range(mnist.data.shape[0]), 10000)
    y_train = lb.transform(mnist.target.astype('int'))[selection]
    x_train = mnist.data.reshape(mnist.data.shape[0], 28, 28, 1)[selection]
    x_train = x_train / 255  # Normalize train data

    model = GeneticCnnModel(
        x_train, y_train,
        {'S_1': '000000', 'S_2': '000000'},  # Genes to test
        (28, 28, 1),  # Shape of input data
        (20, 50),  # Number of kernels per layer
        ((5, 5), (5, 5)),  # Sizes of kernels per layer
        500,  # Number of units in Dense layer
        0.5,  # Dropout probability
        10,  # Number of classes to predict
        nfold=5,
        epochs=8,
        batch_size=32
    )
    print(model.cross_validate())
