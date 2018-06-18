#!/usr/bin/env python
"""
Implementation of GeneticCNN on MNIST data
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    from sklearn.datasets import fetch_mldata
    from sklearn.preprocessing import LabelBinarizer
    from gentun import Population, GeneticCnnIndividual, GeneticAlgorithm

    mnist = fetch_mldata('MNIST original', data_home='./data')
    lb = LabelBinarizer()
    lb.fit(range(max(mnist.target.astype('int')) + 1))
    y_train = lb.transform(mnist.target.astype('int'))
    x_train = mnist.data.reshape(mnist.data.shape[0], 28, 28, 1)
    x_train = x_train / 255  # Normalize train data

    pop = Population(
        GeneticCnnIndividual, x_train, y_train, size=10,
        additional_parameters={'epochs': 3, 'nfold': 5}
    )
    ga = GeneticAlgorithm(pop)
    ga.run(10)
