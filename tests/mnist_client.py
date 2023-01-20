#!/usr/bin/env python
"""
Create a client which loads MNIST data and waits for jobs
to evaluate models. The rabbitmq service should be running
in 'localhost'.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    import mnist
    import random

    from sklearn.preprocessing import LabelBinarizer
    from gentun import GentunClient
    from gentun.individuals.genetic_cnn_individual import GeneticCnnIndividual

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    n = train_images.shape[0]
    lb = LabelBinarizer()
    lb.fit(range(10))
    selection = random.sample(range(n), 10000)  # Use only a subsample
    y_train = lb.transform(train_labels[selection])  # One-hot encodings
    x_train = train_images.reshape(n, 28, 28, 1)[selection]
    x_train = x_train / 255  # Normalize train data

    gc = GentunClient(GeneticCnnIndividual, x_train, y_train, host='localhost', user='guest', password='guest')
    gc.work()
