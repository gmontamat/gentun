#!/usr/bin/env python
"""
Implementation of Genetic CNN on MNIST data.
This is a replica of the algorithm described
on section 4.1.1 of the Genetic CNN paper.
http://arxiv.org/pdf/1703.01513
"""

import os
import numpy as np
import random
import sys

from typing import Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def load_mnist(file_name: str, sample_size: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load, sample, one-hot encode,
    and normalize MNIST dataset.
    """
    mnist = np.load(file_name)
    x_train = mnist["x_train"].reshape(mnist["x_train"].shape[:-2] + (-1,))
    y_train = mnist["y_train"]
    n = x_train.shape[0]
    # Normalize and reshape input
    x_train = x_train / 255
    x_train = x_train.reshape(n, 28, 28, 1)
    # One-hot encode the output
    y_onehot = np.zeros((n, 10))
    y_onehot[np.arange(n), y_train] = 1
    selection = random.sample(range(n), sample_size)
    return x_train[selection], y_onehot[selection]


if __name__ == '__main__':
    from gentun.algorithms import RussianRoulette
    from gentun.genes import RandomChoice, RandomUniform, RandomLogUniform
    # from gentun.models.tensorflow import GeneticCnn
    from gentun.populations import Population

    x, y = load_mnist("mnist.npz")

    # pop = Population(
    #     GeneticCnnIndividual, x_train, y_train, size=20, crossover_rate=0.3, mutation_rate=0.1,
    #     additional_parameters={
    #         'kfold': 5, 'epochs': (20, 4, 1), 'learning_rate': (1e-3, 1e-4, 1e-5), 'batch_size': 32
    #     }, maximize=True
    # )
    # ga = RussianRouletteGA(pop, crossover_probability=0.2, mutation_probability=0.8)
    # ga.run(50)
