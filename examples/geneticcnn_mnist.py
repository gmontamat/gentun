#!/usr/bin/env python
"""
Implementation of Genetic CNN on MNIST data.
This is a replica of the algorithm described
on section 4.1.1 of the Genetic CNN paper.
http://arxiv.org/pdf/1703.01513
"""

import random
from typing import Tuple

import numpy as np

from gentun.algorithms import RussianRoulette
from gentun.genes import Binary
from gentun.models.tensorflow import GeneticCNN
from gentun.populations import Population


def load_mnist(file_name: str, sample_size: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load, sample, one-hot encode,
    and normalize MNIST dataset.
    """
    mnist = np.load(file_name)
    x = mnist["x_train"].reshape(mnist["x_train"].shape[:-2] + (-1,))
    y_raw = mnist["y_train"]
    size = x.shape[0]
    # Normalize and reshape input
    x = x / 255
    x = x.reshape(size, 28, 28, 1)
    # One-hot encode the output
    y = np.zeros((size, 10))
    y[np.arange(size), y_raw] = 1
    # TODO: stratified selection or random (check paper)?
    selection = random.sample(range(size), sample_size)
    return x[selection], y[selection]


if __name__ == "__main__":
    # Genetic CNN static parameters
    kwargs = {
        "nodes": (3, 5),
        "input_shape": (28, 28, 1),
        "kernels_per_layer": (20, 50),
        "kernel_sizes": ((5, 5), (5, 5)),
        "dense_units": 500,
        "dropout_probability": 0.5,
        "classes": 10,
        "kfold": 5,
        "epochs": (20, 4, 1),
        "learning_rate": (1e-3, 1e-4, 1e-5),
        "batch_size": 32,
        "plot": True,
    }
    # Genetic CNN hyperparameters
    genes = [Binary(f"S_{i + 1}", int(K_s * (K_s - 1) / 2)) for i, K_s in enumerate(kwargs["nodes"])]

    x_train, y_train = load_mnist("mnist.npz")
    population = Population(genes, GeneticCNN, 20, x_train, y_train, **kwargs)
    algorithm = RussianRoulette(
        population,
        crossover_probability=0.2,  # p_C
        crossover_rate=0.3,  # q_C
        mutation_probability=0.8,  # p_M
        mutation_rate=0.1,  # q_M
    )
    algorithm.run(50)
