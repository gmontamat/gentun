"""
Models implemented with xgboost
"""

import numpy as np

from typing import Union, Tuple, Optional

from .base import Model


class XGBoostCV(Model):

    def __init__(
            self,
            num_boost_round: int = 10,
            nfold: int = 3,
            stratified: bool = False,
            metrics: Union[str, Tuple[str, ...]] = "",
            early_stopping_rounds: Optional[int] = None,
            **kwargs):
        """
        Booster params reference:
        - https://xgboost.readthedocs.io/en/stable/parameter.html#general-parameters
        - https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster
        - https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
        """
        super().__init__()
        # Cross-validation params
        self.num_boost_round = num_boost_round
        self.nfold = nfold
        self.stratified = stratified
        self.metrics = metrics
        self.early_stopping_rounds = early_stopping_rounds
        self.kwargs = kwargs  # Booster params
        if "verbosity" not in self.kwargs:
            self.kwargs["verbosity"] = 0

    def evaluate(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Use xgboost cross-validation with given parameters.
        API reference:
            - https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.cv
        """
        import xgboost as xgb  # We import here so that xgboost is optional
        d_train = xgb.DMatrix(x_train, label=y_train)
        cv_result = xgb.cv(
            self.kwargs,
            d_train,
            num_boost_round=self.num_boost_round,
            nfold=self.nfold,  # the "k" in k-fold cross-validation
            stratified=self.stratified,
            metrics=self.metrics,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        return cv_result[f'test-{self.metrics[-1]}-mean'][-1]
