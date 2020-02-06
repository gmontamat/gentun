#!/usr/bin/env python
"""
Machine Learning models compatible with the Genetic Algorithm implemented using xgboost
"""
import os

import numpy as np
import xgboost as xgb

from .generic_models import GentunModel


class XgboostModel(GentunModel):

    def __init__(self, x_train, y_train, hyperparameters,
                 booster='gbtree', objective='reg:linear',
                 eval_metric='rmse', kfold=5, num_class=None,
                 num_boost_round=5000, early_stopping_rounds=100,
                 missing=np.nan, nthread=8):
        super(XgboostModel, self).__init__(x_train, y_train)
        self.nthread = min(os.cpu_count(), nthread)
        self.params = {
            'booster': booster,
            'objective': objective,
            'eval_metric': eval_metric,
            'nthread': self.nthread,
            'silent': 1
        }
        if num_class is not None:
            self.params['num_class'] = num_class
        self.params.update(hyperparameters)
        self.eval_metric = eval_metric
        self.kfold = kfold
        self.num_class = num_class
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.missing = missing
        self.best_ntree_limit = None

    def cross_validate(self):
        """Train model using k-fold cross validation and
        return mean value of validation metric.
        """
        d_train = xgb.DMatrix(self.x_train, label=self.y_train, missing=self.missing,
                              nthread=self.nthread)
        # xgb calls its k-fold cross-validation parameter 'nfold'
        cv_result = xgb.cv(
            self.params, d_train, num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds, nfold=self.kfold
        )
        self.best_ntree_limit = len(cv_result)
        return cv_result['test-{}-mean'.format(self.eval_metric)].values[-1]
