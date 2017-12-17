#!/usr/bin/env python
"""
Machine Learning models compatible with the Genetic Algorithm
"""

import xgboost as xgb


class GentunModel(object):
    """Template definition of a machine learning model which
    is passed a train set and fits the model using n-fold
    cross-validation to avoid overfitting.
    """

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def cross_validate(self):
        raise NotImplementedError("Use a subclass with a defined model.")


class XgboostRegressor(GentunModel):

    def __init__(self, x_train, y_train, hyperparameters, eval_metric='rmse', nfold=5,
                 num_boost_round=5000, early_stopping_rounds=100):
        super(XgboostRegressor, self).__init__(x_train, y_train)
        self.params = {
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'eval_metric': eval_metric,
            'silent': 1
        }
        self.params.update(hyperparameters)
        self.eval_metric = eval_metric
        self.nfold = nfold
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    def cross_validate(self):
        """Train model using n-fold cross validation and
        return mean value of validation metric.
        """
        d_train = xgb.DMatrix(self.x_train, label=self.y_train)
        cv_result = xgb.cv(
            self.params, d_train, num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds, nfold=self.nfold
        )
        return cv_result['test-{}-mean'.format(self.eval_metric)][cv_result.index[-1]]


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv('../tests/wine-quality/winequality-white.csv', delimiter=';')
    y = data['quality']
    x = data.drop(['quality'], axis=1)
    genes = {
        'eta': 0.3, 'min_child_weight': 1, 'max_depth': 6, 'gamma': 0.0, 'max_delta_step': 0,
        'subsample': 1.0, 'colsample_bytree': 1.0, 'colsample_bylevel': 1.0, 'lambda': 1.0,
        'alpha': 0.0, 'scale_pos_weight': 1.0
    }
    model = XgboostRegressor(x, y, genes, nfold=3)
    print model.cross_validate()
