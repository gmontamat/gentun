"""
Models implemented with xgboost
"""

from typing import Optional, Sequence, Union

import numpy as np
import xgboost as xgb

from .base import Handler


class XGBoostCV(Handler):
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
        super().__init__()
        # Cross-validation params
        self.metrics = metrics  # XGBoost will evaluate this metric
        self.num_boost_round = num_boost_round
        self.nfold = nfold
        self.stratified = stratified
        self.early_stopping_rounds = early_stopping_rounds
        self.booster_params = kwargs  # Booster params
        if "verbosity" not in self.booster_params:
            self.booster_params["verbosity"] = 0  # By default, be silent

    def evaluate(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Use xgboost cross-validation with given parameters.
        xgboost.cv API reference:
            - https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.cv
        """
        d_train = xgb.DMatrix(x_train, label=y_train)
        cv_result = xgb.cv(
            self.booster_params,
            d_train,
            num_boost_round=self.num_boost_round,
            nfold=self.nfold,  # the "k" in k-fold cross-validation
            stratified=self.stratified,
            metrics=self.metrics,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        if isinstance(self.metrics, str):
            metric = self.metrics
        else:
            metric = self.metrics[-1]
        return cv_result[f"test-{metric}-mean"][-1]
