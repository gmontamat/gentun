#!/usr/bin/env python
"""
Implementation of Genetic CNN on MNIST data.
This is a replica of the algorithm described
on section 4.1.1 of the Genetic CNN paper.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    import random
    from sklearn.datasets import fetch_mldata
    from sklearn.preprocessing import LabelBinarizer
    from gentun import Population, GeneticCnnIndividual, RussianRouletteGA, GeneticAlgorithm

    mnist = fetch_mldata('MNIST original', data_home='./data')
    lb = LabelBinarizer()
    lb.fit(range(max(mnist.target.astype('int')) + 1))
    selection = random.sample(range(mnist.data.shape[0]), 10000)
    y_train = lb.transform(mnist.target.astype('int'))[selection]
    x_train = mnist.data.reshape(mnist.data.shape[0], 28, 28, 1)[selection]
    x_train = x_train / 255  # Normalize train data

    pop = Population(
        GeneticCnnIndividual, x_train, y_train, size=20,
        uniform_rate=0.3, mutation_rate=0.1,
        additional_parameters={
            'nfold': 5, 'epochs': (20, 4, 1), 'learning_rate': (1e-3, 1e-4, 1e-5), 'batch_size': 128
        },
        minimize=False
    )
    ga = RussianRouletteGA(pop, crossover_probability=0.2, mutation_probability=0.8)
    ga.run(50)
