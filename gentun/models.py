#!/usr/bin/env python
"""
Machine Learning models compatible with the Genetic Algorithm
"""

import pandas as pd
import xgboost as xgb

class GentunModel(object):

    def __init__(self, x_train, y_train, hyperparameters):
        self.x_train = x_train
        self.y_train = y_train
        self.hyperparameters = hyperparameters
    
    def cross_validate(self):
        raise NotImplementedError("Use a subclass with a defined model.")


class XgboostRegressor(GentunModel):

    def __init__(self, x_train, y_train, hyperparameters, eval_metric='rmse', nfold=5):
        hyperparameters.update({
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'eval_metric': eval_metric,
            'silent': 1
        })
        super(XgboostRegressor, self).__init__(x_train, y_train, hyperparameters)
        self.eval_metric = eval_metric
        self.nfold = nfold

    def cross_validate(self):
        """Train model using n-fold cross validation and
        return mean value of validation metric.
        """
        d_train = xgb.DMatrix(self.x_train, label=self.y_train)
        cv_result = xgb.cv(
            self.params, d_train, num_boost_round=1000, early_stopping_rounds=50, nfold=self.nfold
        )
        return cv_result['test-{}-mean'.format(self.eval_metric)][cv_result.index[-1]]


if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    y = train['y']
    x = train.drop(['y'], axis=1)
    hyper = {
        'eta': 0.3, 'min_child_weight': 1, 'max_depth': 6, 'gamma': 0.0, 'max_delta_step': 0,
        'subsample': 1.0, 'colsample_bytree': 1.0, 'colsample_bylevel': 1.0, 'lambda': 1.0,
        'alpha': 0.0, 'scale_pos_weight': 1.0
    }
    model = XgboostRegressor(x, y, hyper)
    model.cross_validate()
