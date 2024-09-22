import numpy as np
import pytest

from src.gentun.models.base import Handler, KFoldCrossValidation


class MockModel(Handler):
    def create_train_evaluate(self, x_train, y_train, x_test, y_test):
        return np.mean(y_test)  # Dummy evaluation metric


@pytest.fixture
def data():
    x_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, size=(100,))
    return x_train, y_train


def test_initialization():
    params = {"param1": 1, "param2": 2}
    handler = Handler(**params)
    assert handler.model_params == params


def test_evaluate_not_implemented():
    handler = Handler()
    with pytest.raises(NotImplementedError):
        handler(None, None, None, None)


def test_kfold_cross_validation(data):
    x_train, y_train = data
    model = KFoldCrossValidation(folds=5, stratified=True, shuffle=True)
    model.create_train_evaluate = MockModel().create_train_evaluate
    metric = model(x_train, y_train)
    assert isinstance(metric, float)
    assert metric >= 0


def test_kfold_cross_validation_with_one_hot_labels(data):
    x_train, y_train = data
    y_train_one_hot = np.eye(2)[y_train]  # Convert to one-hot encoding
    model = KFoldCrossValidation(folds=5, stratified=True, shuffle=True)
    model.create_train_evaluate = MockModel().create_train_evaluate
    metric = model(x_train, y_train_one_hot)
    assert isinstance(metric, float)
    assert metric >= 0


def test_kfold_cross_validation_ignores_test_data(data):
    x_train, y_train = data
    x_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 2, size=(20,))
    model = KFoldCrossValidation(folds=5, stratified=True, shuffle=True)
    model.create_train_evaluate = MockModel().create_train_evaluate
    metric = model(x_train, y_train, x_test, y_test)
    assert isinstance(metric, float)
    assert metric >= 0
