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
    import mnist
    import random

    from sklearn.preprocessing import LabelBinarizer
    from gentun import Population
    from gentun.individuals.genetic_cnn_with_skip_individual import GeneticCnnWithSkipIndividual
    from gentun.genetic_algorithms.nsga_net import NSGANet

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    n = train_images.shape[0]
    lb = LabelBinarizer()
    lb.fit(range(10))
    selection = random.sample(range(n), 10000)  # Use only a subsample
    y_train = lb.transform(train_labels[selection])  # One-hot encodings
    x_train = train_images.reshape(n, 28, 28, 1)[selection]
    x_train = x_train / 255  # Normalize train data

    pop = Population(
        GeneticCnnWithSkipIndividual, x_train, y_train, size=20, crossover_rate=0.3, mutation_rate=0.1,
        additional_parameters={
            'kfold': 5, 'epochs': (20, 4, 1), 'learning_rate': (1e-3, 1e-4, 1e-5), 'batch_size': 32
        }, maximize=True
    )
    ga = NSGAnet(pop, crossover_probability=0.2, mutation_probability=0.8)

    ga.run(50)
