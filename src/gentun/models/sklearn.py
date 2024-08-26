"""
Models implemented with scikit-learn
"""

from typing import Callable, Type

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold

from .base import Handler


class SklearnCV(Handler):
    """
    Perform cross-validation with scikit-learn.
    This model can be used as classifier or
    regressor depending on the model passed.
    """

    def __init__(
        self,
        sklearn_model: Type[BaseEstimator],
        sklearn_metric: Callable[[np.ndarray, np.ndarray], float],
        kfold: int = 3,
        stratified: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert hasattr(sklearn_model, "fit"), f"`{sklearn_model}` has not fit method."
        assert hasattr(sklearn_model, "predict"), f"`{sklearn_model}` has not predict method."
        self.sklearn_model = sklearn_model
        self.sklearn_metric = sklearn_metric  # scikit-learn metric to score the model
        self.kfold = kfold
        self.stratified = stratified
        if "metric_kwargs" in kwargs:
            self.metric_params = kwargs["metric_kwargs"]
            self.model_params = {k: v for k, v in kwargs.items() if k != "metric_kwargs"}
        else:
            self.metric_params = {}
            self.model_params = kwargs

    def evaluate(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Split the passed data into k-folds, fit the model,
        score with the passed metric, and average results.
        """
        metric = 0.0
        if self.stratified:
            cross_validation = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        else:
            cross_validation = KFold(n_splits=self.kfold, shuffle=True)
        if len(y_train.shape) == 2:
            # Convert one-hot to int
            y = np.where(y_train == 1)[1]
        else:
            y = y_train
        for train, validation in cross_validation.split(x_train, y):
            model = self.sklearn_model(**self.model_params)
            model.fit(x_train[train], y_train[train])
            y_pred = model.predict(x_train[validation])
            metric += self.sklearn_metric(y_train[validation], y_pred, **self.metric_params) / self.kfold
        return metric
