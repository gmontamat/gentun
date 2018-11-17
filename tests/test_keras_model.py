#!/usr/bin/env python
"""
Test the GeneticCnnModel using the MNIST dataset
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    import mnist
    import random

    from sklearn.preprocessing import LabelBinarizer
    from gentun import GeneticCnnModel

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
        nfold=5,
        epochs=(20, 4, 1),
        learning_rate=(1e-3, 1e-4, 1e-5),
        batch_size=128
    )
    print(model.cross_validate())
