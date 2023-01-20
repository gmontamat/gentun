#!/usr/bin/env python
"""
Implementation of Genetic CNN on MNIST data.
This is a replica of the algorithm described
on section 4.1.1 of the Genetic CNN paper.
"""

import os
import sys
import operator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    import mnist
    import random

    from sklearn.preprocessing import LabelBinarizer
    from gentun.populations import Population
    from gentun.individuals.genetic_cnn_with_skip_individual import GeneticCnnWithSkipIndividual

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    n = train_images.shape[0]
    lb = LabelBinarizer()
    lb.fit(range(10))
    selection = random.sample(range(n), 10000)  # Use only a subsample
    y_train = lb.transform(train_labels[selection])  # One-hot encodings
    x_train = train_images.reshape(n, 28, 28, 1)[selection]
    x_train = x_train / 255  # Normalize train data

    population_size = 20

    population = Population(
        GeneticCnnWithSkipIndividual, x_train, y_train, size=population_size, crossover_rate=0.3, mutation_rate=0.1,
        additional_parameters={
            'kfold': 5, 'epochs': (20, 4, 1), 'learning_rate': (1e-3, 1e-4, 1e-5), 'batch_size': 32
        }, maximize=True
    )

    assert(population.get_species() == GeneticCnnWithSkipIndividual)
    assert(population.get_size() == population_size)
    assert(population.get_data() == (x_train, y_train))
    assert(population.get_fitness_criteria() == True)
    assert(population.get_fittest() == max(population.individuals, key=operator.methodcaller('get_fitness')))
