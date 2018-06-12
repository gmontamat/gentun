#!/usr/bin/env python
"""
Machine Learning models compatible with the Genetic Algorithm
"""


class GentunModel(object):
    """Template definition of a machine learning model
    which receives a train set and fits a model using
    n-fold cross-validation to avoid over-fitting.
    """

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def cross_validate(self):
        raise NotImplementedError("Use a subclass with a defined model.")
