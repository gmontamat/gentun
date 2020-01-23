from sklearn.datasets import fetch_california_housing
from gentun import XgboostModel, GeneticAlgorithm, Population, GridPopulation, XgboostIndividual


def test_xgboost_model():
    data = fetch_california_housing()
    y_train = data.target
    x_train = data.data

    genes = {
        'eta': 0.3, 'min_child_weight': 1, 'max_depth': 6, 'gamma': 0.0, 'max_delta_step': 0,
        'subsample': 1.0, 'colsample_bytree': 1.0, 'colsample_bylevel': 1.0, 'lambda': 1.0,
        'alpha': 0.0, 'scale_pos_weight': 1.0
    }
    model = XgboostModel(x_train, y_train, genes, kfold=3, num_boost_round=10)
    print(model.cross_validate())


def test_california_housing_xgb():
    data = fetch_california_housing()
    y_train = data.target
    x_train = data.data

    pop = Population(
        XgboostIndividual, x_train, y_train, size=10,
        additional_parameters={'kfold': 3,
                               'num_boost_round': 10}, maximize=False
    )
    ga = GeneticAlgorithm(pop)
    ga.run(3)


def test_california_housing_xgb_grid():
    data = fetch_california_housing()
    y_train = data.target
    x_train = data.data

    grid = {
        'eta': [0.001, 0.005, 0.01, 0.015, 0.2],
        'max_depth': range(3, 11),
        'colsample_bytree': [0.80, 0.85, 0.90, 0.95, 1.0],
    }
    pop = GridPopulation(
        XgboostIndividual, x_train, y_train, genes_grid=grid,
        additional_parameters={'kfold': 3,
                               'num_boost_round': 10}, maximize=False
    )
    ga = GeneticAlgorithm(pop)
    ga.run(3)
