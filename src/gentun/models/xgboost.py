"""
Models implemented with xgboost
"""

import logging
from typing import Any, Optional, Sequence, Union

import numpy as np
import xgboost as xgb

from .base import KFoldCrossValidation


class XGBoost(KFoldCrossValidation):
    """
    Perform cross-validation with xgboost.
    This model can be used as classifier or
    regressor depending on the kwargs passed.
    """

    def __init__(
        self,
        metrics: Union[str, Sequence[str]],
        num_boost_round: int = 10,
        nfold: int = 3,
        stratified: bool = False,
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ):
        """
        Booster params reference:
            - https://xgboost.readthedocs.io/en/stable/parameter.html#general-parameters
            - https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster
            - https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
        """
        super().__init__(stratified=stratified, folds=nfold, shuffle=True, **kwargs)
        self.metrics = metrics  # XGBoost will evaluate this metric
        self.num_boost_round = num_boost_round
        self.is_stratified = stratified
        self.early_stopping_rounds = early_stopping_rounds
        if "verbosity" not in self.model_params:
            self.model_params["verbosity"] = 0  # By default, be silent

    def create_train_evaluate(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: Any = None, y_test: Any = None
    ) -> float:
        """
        Use xgboost cross-validation with given parameters.
        xgboost.cv API reference:
            - https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.cv
        """
        d_train = xgb.DMatrix(x_train, label=y_train)
        cv_result = xgb.cv(
            self.model_params,  # Booster params
            d_train,
            num_boost_round=self.num_boost_round,
            nfold=self.folds,
            stratified=self.is_stratified,
            metrics=self.metrics,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        if isinstance(self.metrics, str):
            metric = self.metrics
        else:
            metric = self.metrics[-1]
        return cv_result[f"test-{metric}-mean"][-1]

    def __call__(self, x_train: np.ndarray, y_train: np.ndarray, x_test: Any = None, y_test: Any = None) -> float:
        """Override default k-Fold Cross-Validation with xgboost's."""
        if x_test is not None or y_test is not None:
            logging.warning("`x_test` and `y_test` are ignored.")
        return self.create_train_evaluate(x_train, y_train, x_test, y_test)
