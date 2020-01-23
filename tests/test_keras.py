import pytest

run_keras = True
try:
    import mnist
    import random

    from sklearn.preprocessing import LabelBinarizer
    from gentun import GeneticCnnModel, Population, GeneticCnnIndividual, RussianRouletteGA
except ImportError:
    run_keras = False


@pytest.mark.skipif(not run_keras, reason='Extras not installed.')
def test_keras_model():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    n = train_images.shape[0]
    lb = LabelBinarizer()
    lb.fit(range(10))

    selection = random.sample(range(n), 10000)
    y_train = lb.transform(train_labels[selection])
    x_train = train_images.reshape(n, 28, 28, 1)[selection]
    x_train = x_train / 255  # Normalize train data

    model = GeneticCnnModel(
        x_train, y_train,
        {'S_1': '000', 'S_2': '0000000000'},  # Genes to test
        (3, 5),  # Number of nodes per DAG (corresponds to gene bytes)
        (28, 28, 1),  # Shape of input data
        (20, 50),  # Number of kernels per layer
        ((5, 5), (5, 5)),  # Sizes of kernels per layer
        500,  # Number of units in Dense layer
        0.5,  # Dropout probability
        10,  # Number of classes to predict
        kfold=5,
        epochs=(20, 4, 1),
        learning_rate=(1e-3, 1e-4, 1e-5),
        batch_size=128
    )
    print(model.cross_validate())


@pytest.mark.skipif(not run_keras, reason='Extras not installed.')
def test_keras_mnist():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    n = train_images.shape[0]
    lb = LabelBinarizer()
    lb.fit(range(10))
    selection = random.sample(range(n), 10000)  # Use only a subsample
    y_train = lb.transform(train_labels[selection])  # One-hot encodings
    x_train = train_images.reshape(n, 28, 28, 1)[selection]
    x_train = x_train / 255  # Normalize train data

    pop = Population(
        GeneticCnnIndividual, x_train, y_train, size=20, crossover_rate=0.3, mutation_rate=0.1,
        additional_parameters={
            'kfold': 5, 'epochs': (20, 4, 1), 'learning_rate': (1e-3, 1e-4, 1e-5), 'batch_size': 32
        }, maximize=True
    )
    ga = RussianRouletteGA(pop, crossover_probability=0.2, mutation_probability=0.8)
    ga.run(50)
