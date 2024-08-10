#!/usr/bin/env python
"""
Test the genetic algorithm on a single node
with the iris dataset using xgboost.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_iris(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    # TODO
    return X, y

if __name__ == '__main__':
    from gentun.algorithms import Tournament
    from gentun.genes import RandomChoice, RandomUniform, RandomLogUniform
    from gentun.models.xgboost import XGBoostModel

    # Hyperparameters
    genes = [
        RandomLogUniform("eta", minimum=0.001, maximum=0.1, base=10),
        RandomChoice("min_child_weight", range(11)),
        RandomChoice("max_depth", range(3, 11)),
        RandomLogUniform("gamma", minimum=0.0, maximum=10., base=10),
        RandomChoice("max_delta_step", range(11)),
        RandomLogUniform("subsample", minimum=0.0, maximum=1.0, base=10, reverse=True),
        RandomLogUniform("colsample_bytree", minimum=0.0, maximum=1.0, base=10, reverse=True),
        RandomLogUniform("colsample_bylevel", minimum=0.0, maximum=1.0, base=10, reverse=True),
        RandomLogUniform("lambda", minimum=0.1, maximum=10.0, base=10),
        RandomLogUniform("alpha", minimum=0.0, maximum=10.0, base=10),
        RandomUniform("scale_pos_weight", minimum=0.0, maximum=10.0)
    ]
    # Static parameters
    kwargs = {
        "booster": "gbtree",
        "objective": "reg:linear",
        "eval_metric": "rmse",
        "kfold": 5,
        "num_boost_round": 5000,
        "early_stopping_rounds": 100,
    }

    # Get training data
    X, y = parse_iris("iris.data")

    population = Population(genes, XGBoostModel, X, y, 50)
    algorithm = Tournament(population)
    algorithm.run(100, maximize=False)
