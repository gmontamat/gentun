#!/usr/bin/env python
"""
Test the genetic algorithm on a single node
with the iris dataset using xgboost.
"""

import csv
from typing import Tuple

import numpy as np

from gentun.algorithms import Tournament
from gentun.genes import RandomChoice, RandomLogUniform, RandomUniform
from gentun.models.xgboost import XGBoost
from gentun.populations import Population


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
        RandomLogUniform("eta", minimum=0.001, maximum=0.1, base=10),  # aka. learning_rate
        RandomLogUniform("gamma", minimum=0.0, maximum=10.0, base=10),  # aka. min_split_loss
        RandomChoice("max_depth", range(3, 11)),
        RandomChoice("min_child_weight", range(11)),
        RandomChoice("max_delta_step", range(11)),
        RandomLogUniform("subsample", minimum=0.0, maximum=1.0, base=10, reverse=True),
        RandomLogUniform("colsample_bytree", minimum=0.0, maximum=1.0, base=10, reverse=True),
        RandomLogUniform("colsample_bylevel", minimum=0.0, maximum=1.0, base=10, reverse=True),
        RandomLogUniform("lambda", minimum=0.1, maximum=10.0, base=10),  # L-2 regularization
        RandomLogUniform("alpha", minimum=0.0, maximum=10.0, base=10),  # L-1 regularization
        RandomUniform("scale_pos_weight", minimum=0.0, maximum=10.0),
    ]
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
    # Run genetic algorithm on a population of 50 for 100 generations
    population = Population(genes, XGBoost, 50, x_train, y_train, **kwargs)
    algorithm = Tournament(population)
    algorithm.run(100, maximize=False)
