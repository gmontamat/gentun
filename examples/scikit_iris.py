#!/usr/bin/env python
"""
Test the genetic algorithm on a single node
with the iris dataset using scikit-learn.
"""

import csv
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from gentun.algorithms import Tournament
from gentun.genes import RandomChoice
from gentun.models.sklearn import Sklearn
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
    # RandomForestClassifier hyperparameters
    genes = [
        RandomChoice("n_estimators", [10, 20, 30, 40, 50]),
        RandomChoice("max_depth", [None, 4, 5, 6, 7, 8, 9, 10]),
        RandomChoice("criterion", ["gini", "entropy"]),
    ]
    # RandomForestClassifier static parameters
    kwargs = {
        "sklearn_model": RandomForestClassifier,
        "sklearn_metric": f1_score,
        "metric_kwargs": {"average": "macro"},
        "folds": 5,
    }

    # Fetch training data
    x_train, y_train = parse_iris("iris.data")
    # Run genetic algorithm on a population of 10 for 10 generations
    population = Population(genes, Sklearn, 10, x_train, y_train, **kwargs)
    algorithm = Tournament(population)
    algorithm.run(10)
