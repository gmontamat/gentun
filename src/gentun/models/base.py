"""
Base class for the models compatible with the genetic algorithms.
"""

import logging
from typing import Any

from sklearn.model_selection import KFold, StratifiedKFold


class Handler:
    """
    Wrapper for a machine learning model builder which receives
    a train set and fits a model.
    """

    def __init__(self, **kwargs: Any):
        self.model_params = kwargs

    def create_train_evaluate(self, x_train: Any, y_train: Any, x_test: Any, y_test: Any) -> float:
        """
        Create an instance of your model. Train model with
        x_train, y_train. Evaluate with x_test, y_test and
        return the evaluated metric.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, x_train: Any, y_train: Any, x_test: Any, y_test: Any) -> float:
        """Compute and return fitness."""
        return self.create_train_evaluate(x_train, y_train, x_test, y_test)


class KFoldCrossValidation(Handler):
    """
    Wrapper that performs k-fold cross-validation to return a robust
    estimation of the evaluated metric.
    """

    def __init__(self, folds: int = 5, stratified: bool = True, shuffle: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        self.folds = folds
        self.fold_method = StratifiedKFold if stratified else KFold
        self.shuffle = shuffle

    def __call__(self, x_train: Any, y_train: Any, x_test: Any = None, y_test: Any = None) -> float:
        """Use k-Fold Cross-Validation to evaluate."""
        if x_test is not None or y_test is not None:
            logging.warning("`x_test` and `y_test` are ignored.")
        # Convert one-hot encoded labels to class labels if necessary
        if len(y_train.shape) == 2 and (y_train.sum(axis=1) == 1).all():
            y = y_train.argmax(axis=1)
        else:
            y = y_train
        metric = 0.0
        cross_validation = self.fold_method(n_splits=self.folds, shuffle=self.shuffle)
        for fold, (train, validation) in enumerate(cross_validation.split(x_train, y)):
            logging.debug("KFold %d of %d", fold + 1, self.folds)
            metric += (
                self.create_train_evaluate(x_train[train], y_train[train], x_train[validation], y_train[validation])
                / self.folds
            )
        return metric


class Dummy(Handler):
    """
    Use this model to test algorithms only. Ignores x_train, y_train
    to evaluate, just returns the sum of its hyperparameter values.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.fitness = sum(kwargs.values())

    def create_train_evaluate(self, x_train: Any, y_train: Any, x_test: Any, y_test: Any) -> float:
        """Ignore x_train, y_train; return sum of parameter values."""
        return self.fitness
