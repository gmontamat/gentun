import pytest

from gentun.algorithms import GeneticAlgorithm, RussianRoulette, Tournament
from gentun.genes import RandomChoice
from gentun.models.base import Dummy
from gentun.populations import Population


@pytest.fixture
def setup_genes():
    return [RandomChoice(f"hyperparam_{i + 1}", [0, 1, 2]) for i in range(10)]


@pytest.fixture
def setup_individuals():
    return [{f"hyperparam_{i + 1}": 1 for i in range(10)}] * 50


@pytest.fixture
def setup_data():
    return [], []


def test_russian_roulette(setup_genes, setup_data):
    genes = setup_genes
    x_train, y_train = setup_data
    population = Population(genes, Dummy, 50, x_train, y_train)
    algorithm = RussianRoulette(population)
    fitness = algorithm.run(100, verbose=False)
    assert algorithm.current_generation == 101
    assert fitness == 20


def test_russian_roulette_minimize(setup_genes, setup_data):
    genes = setup_genes
    x_train, y_train = setup_data
    population = Population(genes, Dummy, 50, x_train, y_train)
    algorithm = RussianRoulette(population)
    fitness = algorithm.run(100, maximize=False, patience=20)
    assert algorithm.current_generation <= 101
    assert fitness == 0


def test_russian_roulette_weights(setup_genes, setup_individuals, setup_data):
    genes = setup_genes
    individuals = setup_individuals
    x_train, y_train = setup_data
    population = Population(genes, Dummy, [], x_train, y_train)
    for individual in individuals:
        population.add_individual(individual)
    algorithm = RussianRoulette(population)
    fitness = algorithm.run(100, maximize=False, patience=20)
    assert algorithm.current_generation <= 101
    assert fitness == 0


def test_tournament(setup_genes, setup_data):
    genes = setup_genes
    x_train, y_train = setup_data
    population = Population(genes, Dummy, 50, x_train, y_train)
    algorithm = Tournament(population, elitism=False)
    fitness = algorithm.run(50, patience=5, verbose=False)
    assert algorithm.current_generation <= 51
    assert fitness == 20


def test_tournament_minimize(setup_genes, setup_data):
    genes = setup_genes
    x_train, y_train = setup_data
    population = Population(genes, Dummy, 50, x_train, y_train)
    algorithm = Tournament(population)
    fitness = algorithm.run(20, maximize=False)
    assert algorithm.current_generation == 21
    assert fitness == 0


def test_genetic_algorithm_evolve_error(setup_genes, setup_data):
    genes = setup_genes
    x_train, y_train = setup_data
    population = Population(genes, Dummy, 50, x_train, y_train)
    algorithm = GeneticAlgorithm(population)
    # Calling the evolve method raises NotImplementedError
    with pytest.raises(NotImplementedError):
        algorithm.evolve(maximize=True)
