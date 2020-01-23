import pytest

run_distributed = True
try:
    import mnist
    import random

    from sklearn.preprocessing import LabelBinarizer
    from sklearn.datasets import fetch_california_housing
    from gentun import GentunClient, GeneticAlgorithm, XgboostIndividual, GeneticCnnIndividual
    from gentun import RussianRouletteGA, DistributedPopulation
except ImportError:
    run_distributed = False


@pytest.mark.skipif(not run_distributed, reason='Extras not installed.')
def test_sample_client():
    data = fetch_california_housing()
    y_train = data.target
    x_train = data.data

    gc = GentunClient(XgboostIndividual, x_train, y_train, host='localhost', user='guest', password='guest')
    gc.work()


@pytest.mark.skipif(not run_distributed, reason='Extras not installed.')
def test_sample_server():
    pop = DistributedPopulation(
        XgboostIndividual, size=100, additional_parameters={'kfold': 3}, maximize=False,
        host='localhost', user='guest', password='guest'
    )
    ga = GeneticAlgorithm(pop)
    ga.run(10)


@pytest.mark.skipif(not run_distributed, reason='Extras not installed.')
def test_mnist_client():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    n = train_images.shape[0]
    lb = LabelBinarizer()
    lb.fit(range(10))
    selection = random.sample(range(n), 10000)  # Use only a subsample
    y_train = lb.transform(train_labels[selection])  # One-hot encodings
    x_train = train_images.reshape(n, 28, 28, 1)[selection]
    x_train = x_train / 255  # Normalize train data

    gc = GentunClient(GeneticCnnIndividual, x_train, y_train, host='localhost', user='guest',
                      password='guest')
    gc.work()


@pytest.mark.skipif(not run_distributed, reason='Extras not installed.')
def test_mnist_server():
    pop = DistributedPopulation(
        GeneticCnnIndividual, size=20, crossover_rate=0.3, mutation_rate=0.1,
        additional_parameters={
            'kfold': 5, 'epochs': (20, 4, 1), 'learning_rate': (1e-3, 1e-4, 1e-5), 'batch_size': 32
        }, maximize=True, host='localhost', user='guest', password='guest'
    )
    ga = RussianRouletteGA(pop, crossover_probability=0.2, mutation_probability=0.8)
    ga.run(50)
