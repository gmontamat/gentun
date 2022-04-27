#!/usr/bin/env python
"""
Machine Learning models compatible with the Genetic Algorithm implemented using xgboost
"""
import os
from typing import Tuple, Optional, List

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold

from .generic_models import GentunModel


class OofGetterCallback(xgb.callback.TrainingCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_history = {}

    def after_iteration(self, model, epoch, evals_log):
        cv_preds = []
        cv_trues = []
        for i in range(len(model.cvfolds)):
            cv_preds.append(np.array(model.cvfolds[i].bst.predict(model.cvfolds[i].dtest)))
            cv_trues.append(np.array(model.cvfolds[i].dtest.get_label()))
        where_to_add = self.eval_history.setdefault('cv', [])
        where_to_add.append({'cv_preds': cv_preds, 'cv_trues': cv_trues})


class XgboostModel(GentunModel):
    def __init__(self, x_train, y_train,
                 y_weights=None, booster='gbtree', objective='reg:linear',
                 eval_metric='rmse', kfold=5, num_class=None,
                 num_boost_round=5000, early_stopping_rounds=100,
                 missing=np.nan, nthread=8, feval = None, maximize = False, disable_default_eval_metric: bool = False,
                 splits:Optional[List[np.ndarray]] = None):
        super(XgboostModel, self).__init__(x_train, y_train)
        self.y_weights = y_weights
        self.nthread = min(os.cpu_count(), nthread)
        self.params = {
            'booster': booster,
            'objective': objective,
            'eval_metric': eval_metric,
            'nthread': self.nthread,
            'silent': 1
        }

        if disable_default_eval_metric:
            self.params['disable_default_eval_metric']=1

        if num_class is not None:
            self.params['num_class'] = num_class
        self.eval_metric = eval_metric
        self.feval = feval
        self.maximize = maximize
        if self.feval is not None:
            del self.params['eval_metric']
        self.kfold = kfold
        self.num_class = num_class
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.missing = missing
        self.best_ntree_limit = None
        self.oof_dict = None

        self.d_train = None
        self.splits = splits

    def _create_dmatrix(self):
        #print(f'NNN {self.nthread}')
        return xgb.DMatrix(
            self.x_train,
            label=self.y_train,
            weight=self.y_weights,
            missing=self.missing,
            nthread=1
            # for whatever reason, creation of DMatrix got deadlocked when
            # attempted on multiple processes on xgboost v. 1.0.0
            # with nthread>=2, checking xgboost source code in core.py
            #         if nthread is None:
            #             _check_call(_LIB.XGDMatrixCreateFromMat(....
            #          else:
            #             _check_call(_LIB.XGDMatrixCreateFromMat_omp(
            # looks like a different method is being called when nthread>1... which might be thread unsafe
         )

    def update(self, hyperparameters):
        self.params.update(hyperparameters)

    def cross_validate(self):
        """Train model using k-fold cross validation and
        return mean value of validation metric.
        """
        ogc = OofGetterCallback()

        if self.splits is None:
            if self.y_train[0].dtype.kind == 'f':  # continuous labels
                skf = KFold(n_splits=self.kfold, shuffle=True)
            else: # categorical labels
                skf = StratifiedKFold(n_splits=self.kfold, shuffle=True)

            splits = list(skf.split(X=np.zeros_like(self.y_train), y=self.y_train))
        else:
            splits = self.splits

        if self.d_train is None:
            self.d_train = self._create_dmatrix()
        cv_result = xgb.cv(
            params=self.params, dtrain=self.d_train, nfold=self.kfold, folds=splits,
            early_stopping_rounds=self.early_stopping_rounds, num_boost_round=self.num_boost_round, feval= self.feval,
            maximize=self.maximize, callbacks=[ogc]
        )
        if self.maximize:
            self.best_ntree_limit = cv_result['test-{}-mean'.format(self.eval_metric)].argmax() + 1
        else:
            self.best_ntree_limit = cv_result['test-{}-mean'.format(self.eval_metric)].argmin() + 1
        self.oof_dict = ogc.eval_history['cv'][self.best_ntree_limit - 1]
        return cv_result['test-{}-mean'.format(self.eval_metric)][self.best_ntree_limit - 1]
