from unittest.mock import MagicMock, patch

import pytest

from src.gentun.genes import Gene
from src.gentun.individuals import Individual
from src.gentun.models.base import Handler
from src.gentun.populations import Grid, Population
from src.gentun.services import RedisController


class MockHandler(Handler):
    def __init__(self, param1: int, param2: str = "default"):
        self.param1 = param1
        self.param2 = param2

    def create_train_evaluate(self, x_train, y_train, x_test, y_test):
        return 0.9


class MockIntGene(Gene):
    def __init__(self, name: str):
        super().__init__(name)

    def __call__(self):
        return 1

    def sample(self, percentile: float):
        return int(percentile * 100)


class MockStrGene(Gene):
    def __init__(self, name: str):
        super().__init__(name)

    def __call__(self):
        return "1"

    def sample(self, percentile: float):
        return str(int(percentile * 100))


def test_population_init():
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    population = Population(genes, handler, 10, x_train, y_train)
    assert len(population) == 10
    assert all(isinstance(individual, Individual) for individual in population)


@patch("src.gentun.services.redis.StrictRedis")
def test_population_init_with_redis(mock_redis):
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    controller = RedisController("test")
    # x_train and y_train ignored
    population = Population(genes, handler, 10, x_train, y_train, controller=controller)
    assert len(population) == 10
    assert all(isinstance(individual, Individual) for individual in population)
    # No data passed
    population = Population(genes, handler, 5, controller=controller)
    assert len(population) == 5
    assert all(isinstance(individual, Individual) for individual in population)


def test_population_init_error_cases():
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    # Missing data
    with pytest.raises(ValueError):
        Population(genes, handler, 10, x_train=x_train)
    with pytest.raises(ValueError):
        Population(genes, handler, 10, y_train=y_train)
    # Invalid population type
    with pytest.raises(ValueError):
        Population(genes, handler, {"param1": 1}, x_train, y_train)


def test_population_init_with_individuals():
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]
    individuals = [
        Individual(genes, handler, x_train, y_train, x_test, y_test, {"param1": 1, "param2": "2"}) for _ in range(5)
    ]
    population = Population(genes, handler, individuals, x_train, y_train)
    assert len(population) == 5
    assert all(isinstance(individual, Individual) for individual in population)


def test_population_spawn():
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    population = Population(genes, handler, 0, x_train, y_train)
    individual = population.spawn()
    assert isinstance(individual, Individual)
    assert individual.hyperparameters == {"param1": 1, "param2": "1"}


def test_population_spawn_error():
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    population = Population(genes, handler, 0, x_train, y_train)
    with pytest.raises(KeyError):
        individual = population.spawn({"param1": 1})


def test_population_add_individual():
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    population = Population(genes, handler, 0, x_train, y_train)
    population.add_individual({"param1": 1, "param2": "2"})
    assert len(population) == 1
    assert isinstance(population[0], Individual)


def test_population_add_individual_error():
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    population = Population(genes, handler, 0, x_train, y_train)
    with pytest.raises(ValueError):
        population.add_individual("param1")


def test_population_get_fittest():
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]
    individuals = [
        Individual(genes, handler, x_train, y_train, x_test, y_test, {"param1": 1, "param2": "2"}) for _ in range(5)
    ]
    population = Population(genes, handler, individuals, x_train, y_train)
    fittest = population.get_fittest()
    assert isinstance(fittest, Individual)


@patch("src.gentun.services.redis.StrictRedis")
@patch.object(Individual, "send_to_queue", return_value="job_id")
@patch.object(Individual, "read_from_queue", return_value=0.9)
def test_population_get_fittest_with_redis(mock_send_to_queue, mock_read_from_queue, mock_redis):
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    controller = RedisController("test")
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    x_test, y_test = [6, 5, 4], [3, 2, 1]
    individuals = [
        Individual(genes, handler, x_train, y_train, x_test, y_test, {"param1": 1, "param2": "2"}) for _ in range(5)
    ]
    population = Population(genes, handler, individuals, x_train, y_train, x_test, y_test, controller=controller)
    fittest = population.get_fittest()
    assert isinstance(fittest, Individual)
    assert mock_send_to_queue.call_count == 5
    assert mock_read_from_queue.call_count == 5
    # No new calls
    fittest = population.get_fittest()
    assert mock_send_to_queue.call_count == 5
    assert mock_read_from_queue.call_count == 5


def test_population_duplicate():
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    population = Population(genes, handler, 10, x_train, y_train)
    duplicate = population.duplicate(5)
    assert len(duplicate) == 5
    assert all(isinstance(individual, Individual) for individual in duplicate)


def test_population_len():
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    population = Population(genes, handler, 10, x_train, y_train)
    assert len(population) == 10


def test_population_iter():
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    population = Population(genes, handler, 10, x_train, y_train)
    for individual in population:
        assert isinstance(individual, Individual)


def test_population_getitem():
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    population = Population(genes, handler, 10, x_train, y_train)
    assert isinstance(population[0], Individual)
    assert isinstance(population[:5], list)


def test_grid_init():
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    grid = Grid(genes, handler, 3, x_train, y_train)
    assert len(grid) == 9  # 3 samples per gene, 3x3 = 9 individuals
    assert all(isinstance(ind, Individual) for ind in grid)


def test_grid_init_with_sequence():
    genes = [MockIntGene("param1"), MockStrGene("param2")]
    handler = MockHandler
    x_train, y_train = [1, 2, 3], [4, 5, 6]
    grid = Grid(genes, handler, [2, 3], x_train, y_train)
    assert len(grid) == 6  # 2 samples for gene1, 3 samples for gene2, 2x3 = 6 individuals
    assert all(isinstance(ind, Individual) for ind in grid)
