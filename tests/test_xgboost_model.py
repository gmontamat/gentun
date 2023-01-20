#!/usr/bin/env python
"""
Test the XgboostModel using the California Housing data
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    from sklearn.datasets import fetch_california_housing
    from gentun.models.xgboost_models import XgboostModel

    data = fetch_california_housing()
    y_train = data.target
    x_train = data.data

    genes = {
        'eta': 0.3, 'min_child_weight': 1, 'max_depth': 6, 'gamma': 0.0, 'max_delta_step': 0,
        'subsample': 1.0, 'colsample_bytree': 1.0, 'colsample_bylevel': 1.0, 'lambda': 1.0,
        'alpha': 0.0, 'scale_pos_weight': 1.0
    }
    model = XgboostModel(x_train, y_train, genes, kfold=3)
    print(model.cross_validate())
