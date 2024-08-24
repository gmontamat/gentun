"""
Base class for the models compatible
with the genetic algorithms.
"""

from typing import Any


class Handler:
    """
    Template definition of a machine learning model
    which receives a train set and fits a model using
    k-fold cross-validation to avoid overfitting.
    """

    def __init__(self, **kwargs: Any):
        self.model_params = kwargs

    def evaluate(self, x_train: Any, y_train: Any):
        """
        Create an instance of your model.
        Train model with x_train, y_train.
        Use cross-validation to evaluate.
        """
        raise NotImplementedError("Use a subclass with a defined model.")


class Dummy(Handler):
    """
    Use this model to test algorithms only.
    Ignores x_train, y_train to evaluate, just
    returns the sum of its hyperparameter values.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.fitness = sum(kwargs.values())

    def evaluate(self, x_train: Any, y_train: Any):
        """Ignore x_train, y_train; return sum of parameter values."""
        return self.fitness
