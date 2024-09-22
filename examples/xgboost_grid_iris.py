#!/usr/bin/env python
"""
Test a grid search on a single node
with the iris dataset using xgboost.
"""

import csv
from typing import Tuple

import numpy as np

from gentun.algorithms import Tournament
from gentun.genes import RandomChoice, RandomLogUniform
from gentun.models.xgboost import XGBoost
from gentun.populations import Grid


def parse_iris(file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load iris.data into X, y arrays."""
    x = []
    y = []
    with open(file_name, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            features = [float(val) for val in row[:-1]]
            targets = 0 if "setosa" in row[-1] else 1 if "versicolor" in row[-1] else 2
            x.append(features)
            y.append(targets)
    x = np.array(x).astype(np.float64)
    y = np.array(y)
    return x, y


if __name__ == "__main__":
    # xgboost hyperparameters
    genes = [
        RandomLogUniform("learning_rate", minimum=0.001, maximum=0.1, base=10),
        RandomChoice("max_depth", range(3, 11)),
        RandomChoice("min_child_weight", range(11)),
    ]
    gene_samples = [10, 8, 11]
    # xgboost static parameters
    kwargs = {
        "booster": "gbtree",
        "device": "cpu",
        "objective": "multi:softmax",
        "metrics": "mlogloss",  # The metric we want to minimize with the algorithm
        "num_class": 3,
        "nfold": 5,
        "num_boost_round": 5000,
        "early_stopping_rounds": 100,
    }

    # Fetch training data
    x_train, y_train = parse_iris("iris.data")
    # Run genetic algorithm on a grid population for 1 generation
    population = Grid(genes, XGBoost, gene_samples, x_train, y_train, **kwargs)
    algorithm = Tournament(population)
    algorithm.run(1, maximize=False)
