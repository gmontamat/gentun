import pprint
from typing import Union
from unittest.mock import MagicMock

import pytest

from src.gentun.genes import Gene, RandomChoice
from src.gentun.individuals import Individual
from src.gentun.models.base import Handler
from src.gentun.services import RedisController


class MockHandler(Handler):
    def __init__(self, param1: int, param2: str = "default", param3: Union[int, str] = 1, param4: int = 4):
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.param4 = param4

    def create_train_evaluate(self, x_train, y_train, x_test, y_test):
        return 0.9


def test_individual_init():
    genes = [Gene("gene1"), Gene("gene2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]
    hyperparameters = {"param1": 10, "param2": "value"}
    individual = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters)
    assert individual.genes == genes
    assert individual.handler == handler
    assert individual.x_train == x_train
    assert individual.y_train == y_train
    assert individual.x_test == x_test
    assert individual.y_test == y_test
    assert individual.hyperparameters == hyperparameters
    assert individual.fitness is None
    assert individual.job_id is None


def test_validate_params():
    genes = [Gene("gene1"), Gene("gene2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]

    # Should not raise any exceptions
    hyperparameters = {"param1": 10, "param2": "value", "param3": 3}
    kwargs = {"param4": 4}
    individual = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters, **kwargs)
    individual.validate_params()

    # Should not raise any exceptions
    hyperparameters = {"param1": 10, "param2": "value"}
    kwargs = {"param3": "3"}
    individual = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters, **kwargs)
    individual.validate_params()

    # Test missing required parameter
    hyperparameters = {"param2": "value"}
    kwargs = {"param3": "3"}
    with pytest.raises(ValueError):
        individual = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters, **kwargs)
        individual.validate_params()

    # Test hyperparameter type mismatch
    hyperparameters = {"param1": "wrong_type", "param2": "value"}
    with pytest.raises(TypeError):
        individual = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters)
        individual.validate_params()

    # Test kwarg type mismatch
    hyperparameters = {"param1": 10, "param2": "value"}
    kwargs = {"param4": "4"}
    with pytest.raises(TypeError):
        individual = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters, **kwargs)
        individual.validate_params()


def test_evaluate_fitness():
    genes = [Gene("gene1"), Gene("gene2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]
    hyperparameters = {"param1": 10, "param2": "value"}
    individual = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters)
    fitness = individual.evaluate_fitness()
    assert fitness == 0.9
    assert individual.fitness == 0.9


def test_get_fitness():
    genes = [Gene("gene1"), Gene("gene2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]
    hyperparameters = {"param1": 10, "param2": "value"}
    individual = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters)
    assert individual.get_fitness() is None
    individual.evaluate_fitness()
    assert individual.get_fitness() == 0.9


def test_send_to_queue():
    genes = [Gene("gene1"), Gene("gene2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]
    hyperparameters = {"param1": 10, "param2": "value"}
    individual = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters)
    server = MagicMock(spec=RedisController)
    server.send_job.return_value = "job_id"
    individual.send_to_queue(server)
    assert individual.job_id == "job_id"
    server.send_job.assert_called_once_with(handler, param1=10, param2="value")


def test_read_from_queue():
    genes = [Gene("gene1"), Gene("gene2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]
    hyperparameters = {"param1": 10, "param2": "value"}
    individual = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters)
    server = MagicMock(spec=RedisController)
    server.wait_for_result.return_value = 0.9
    individual.job_id = "job_id"
    individual.read_from_queue(server)
    assert individual.fitness == 0.9
    server.wait_for_result.assert_called_once_with("job_id")


def test_getitem_setitem():
    genes = [Gene("gene1"), Gene("gene2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]
    hyperparameters = {"param1": 10, "param2": "value"}
    individual = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters)
    assert individual["param1"] == 10
    individual["param1"] = 20
    assert individual["param1"] == 20
    assert individual.fitness is None  # Fitness should be reset


def test_reproduce():
    genes = [Gene("gene1"), Gene("gene2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]
    hyperparameters1 = {"param1": 10, "param2": "value1"}
    hyperparameters2 = {"param1": 20, "param2": "value2"}
    individual1 = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters1)
    individual2 = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters2)
    child = individual1.reproduce(individual2, rate=0.5)
    assert isinstance(child, Individual)
    assert child.handler == handler
    assert child.x_train == x_train
    assert child.y_train == y_train
    assert child.genes == genes


def test_crossover():
    genes = [Gene("gene1"), Gene("gene2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]
    hyperparameters1 = {"param1": 10, "param2": "value1"}
    hyperparameters2 = {"param1": 20, "param2": "value2"}
    individual1 = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters1)
    individual2 = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters2)
    individual1.crossover(individual2, rate=1.0)
    assert individual1["param1"] == 20
    assert individual1["param2"] == "value2"
    assert individual2["param1"] == 10
    assert individual2["param2"] == "value1"


def test_mutate():
    gene1 = RandomChoice("param1", [1, 2, 3])
    gene2 = RandomChoice("param2", ["4", "5", "6"])
    genes = [gene1, gene2]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]
    hyperparameters = {"param1": 1, "param2": "4"}
    individual = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters)
    individual.mutate(rate=1.0)
    assert individual["param1"] in [1, 2, 3]
    assert individual["param2"] in ["4", "5", "6"]


def test_duplicate():
    genes = [Gene("gene1"), Gene("gene2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]
    hyperparameters = {"param1": 10, "param2": "value"}
    individual = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters)
    duplicate = individual.duplicate()
    assert isinstance(duplicate, Individual)
    assert duplicate.genes == genes
    assert duplicate.handler == handler
    assert duplicate.x_train == x_train
    assert duplicate.y_train == y_train
    assert duplicate.hyperparameters == hyperparameters
    assert duplicate.kwargs == individual.kwargs


def test_str():
    genes = [Gene("gene1"), Gene("gene2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]
    hyperparameters = {"param1": 10, "param2": "value"}
    individual = Individual(genes, handler, x_train, y_train, x_test, y_test, hyperparameters)
    assert str(individual) == str(hyperparameters)
