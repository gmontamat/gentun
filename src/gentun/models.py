"""
Machine Learning models compatible with the Genetic Algorithm
"""

# from .xgboost_models import XgboostModel
# from .keras_models import GeneticCnnModel

from typing import Any, Dict


class Model:
    """
    Template definition of a machine learning model
    which receives a train set and fits a model using
    n-fold cross-validation to avoid over-fitting.
    """

    def __init__(self, **kwargs: Dict[str, Any]):
        """Create an instance of your model."""
        self.model_params = kwargs

    def evaluate(self, x_train: Any, y_train: Any):
        """
        Train model with x_train, y_train.
        Use cross-validation to evaluate.
        """
        raise NotImplementedError("Use a subclass with a defined model.")


class DummyModel(Model):
    """
    Use this model to test algorithms only.
    Ignores x_train, y_train to evaluate, just
    returns the sum of its hyperparameters.
    """

    def __init__(self, **kwargs: Dict[str, float]):
        super().__init__(**kwargs)
        self.fitness = sum([value for key, value in kwargs.items()])

    def evaluate(self, x_train: Any, y_train: Any):
        """Ignore x_train, y_train; return sum of parameter values."""
        return self.fitness
