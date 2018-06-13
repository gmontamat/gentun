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
    from gentun import Population, GeneticCnnIndividual, GeneticAlgorithm

    mnist = fetch_mldata('MNIST original', data_home='./data')
    lb = LabelBinarizer()
    lb.fit(range(max(mnist.target.astype('int')) + 1))
    y_train = lb.transform(mnist.target.astype('int')[:1000])
    x_train = mnist.data.reshape(mnist.data.shape[0], 28, 28, 1)[:1000]
    # model = GeneticCnnModel(
    #     x_train, y_train,
    #     {'S_1': '001100', 'S_2': '101100'},
    #     (20, 50),
    #     ((5, 5), (5, 5)),
    #     (28, 28, 1),
    #     10
    # )
    # print(model.cross_validate())
    pop = Population(GeneticCnnIndividual, x_train, y_train, size=10)
    ga = GeneticAlgorithm(pop)
    ga.run(10)
