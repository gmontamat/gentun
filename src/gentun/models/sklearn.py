"""
Models implemented with scikit-learn
"""

from typing import Callable, Type

import numpy as np
from sklearn.base import BaseEstimator

from .base import KFoldCrossValidation


class Sklearn(KFoldCrossValidation):
    """
    Perform cross-validation with scikit-learn.
    This model can be used as classifier or
    regressor depending on the model passed.
    """

    def __init__(
        self,
        sklearn_model: Type[BaseEstimator],
        sklearn_metric: Callable[[np.ndarray, np.ndarray], float],
        folds: int = 3,
        stratified: bool = True,
        shuffle: bool = True,
        **kwargs,
    ):
        assert hasattr(sklearn_model, "fit"), f"`{sklearn_model}` has not fit method."
        assert hasattr(sklearn_model, "predict"), f"`{sklearn_model}` has not predict method."
        self.sklearn_model = sklearn_model
        self.sklearn_metric = sklearn_metric  # scikit-learn metric to score the model
        if "metric_kwargs" in kwargs:
            self.metric_params = kwargs["metric_kwargs"]
            super().__init__(
                folds=folds,
                stratified=stratified,
                shuffle=shuffle,
                **{k: v for k, v in kwargs.items() if k != "metric_kwargs"},
            )
        else:
            self.metric_params = {}
            super().__init__(folds=folds, stratified=stratified, shuffle=shuffle, **kwargs)

    def create_train_evaluate(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
    ) -> float:
        """
        Split the passed data into k-folds, fit the model,
        score with the passed metric, and average results.
        """
        model = self.sklearn_model(**self.model_params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return self.sklearn_metric(y_test, y_pred, **self.metric_params)
