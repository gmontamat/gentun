#!/usr/bin/env python
"""
Implementation of Genetic CNN on CIFAR-10 data. This is a replica of
the algorithm described on section 4.2 of the Genetic CNN paper.
http://arxiv.org/pdf/1703.01513
"""

import os
import pickle
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from gentun.algorithms import RussianRoulette
from gentun.genes import Binary
from gentun.models.tensorflow import GeneticCNN
from gentun.populations import Population


def load_cifar10(data_dir: str, test_size: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, sample, one-hot encode, and normalize CIFAR-10."""

    def unpickle(file_name: str) -> dict:
        with open(file_name, "rb") as fo:
            data = pickle.load(fo, encoding="bytes")
        return data

    x_train = []
    y_raw = []
    for batch in range(1, 6):
        batch_file = os.path.join(data_dir, f"data_batch_{batch}")
        batch_data = unpickle(batch_file)
        x_train.append(batch_data[b"data"])
        y_raw.extend(batch_data[b"labels"])
    x_train = np.concatenate(x_train).astype(np.float32)
    x_train = np.moveaxis(x_train.reshape(-1, 3, 32, 32), 1, -1) / 255
    y_raw = np.array(y_raw, dtype=np.int32)
    # One-hot encode the output
    y_train = np.zeros((y_raw.size, 10))
    y_train[np.arange(y_raw.size), y_raw] = 1
    return train_test_split(x_train, y_train, test_size=test_size, shuffle=True, stratify=y_raw)


if __name__ == "__main__":
    # Genetic CNN static parameters
    kwargs = {
        "nodes": (3, 4, 5),
        "input_shape": (32, 32, 3),
        "kernels_per_layer": (8, 16, 32),
        "kernel_sizes": ((5, 5), (5, 5), (5, 5)),
        "pool_sizes": ((3, 3), (3, 3), (3, 3)),
        "dense_units": 128,
        "dropout_probability": 0.5,
        "classes": 10,
        "epochs": (120, 60, 40, 20),
        "learning_rate": (1e-2, 1e-3, 1e-4, 1e-5),
        "batch_size": 32,  # Not mentioned in the paper, but 32 is a good default for most cases
        "plot": False,  # if True, graphviz needs to be installed on your system
    }
    # Genetic CNN hyperparameters
    genes = [Binary(f"S_{i + 1}", int(K_s * (K_s - 1) / 2)) for i, K_s in enumerate(kwargs["nodes"])]

    x_train, x_test, y_train, y_test = load_cifar10("cifar-10-batches-py")
    population = Population(genes, GeneticCNN, 20, x_train, y_train, x_test, y_test, **kwargs)
    algorithm = RussianRoulette(
        population,
        crossover_probability=0.2,  # p_C
        crossover_rate=0.2,  # q_C
        mutation_probability=0.8,  # p_M
        mutation_rate=0.05,  # q_M
    )
    algorithm.run(50)
