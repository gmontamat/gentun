#!/usr/bin/env python
"""
Test the GeneticCnnWithSkipModel using the MNIST dataset.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    import mnist
    import random

    from sklearn.preprocessing import LabelBinarizer
    from gentun import GeneticCnnWithSkipModel

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    n = train_images.shape[0]
    lb = LabelBinarizer()
    lb.fit(range(10))

    selection = random.sample(range(n), 10000)
    y_train = lb.transform(train_labels[selection])
    x_train = train_images.reshape(n, 28, 28, 1)[selection]
    x_train = x_train / 255  # Normalize train data

    model_without_skip = GeneticCnnWithSkipModel(
        x_train=x_train, 
        y_train=y_train,
        genes={'Stage_1': '0000', 'Stage_2': '00000000000'},  # Genes to test
        nodes_per_stage=(3, 5),  # Number of nodes per Directed Acyclic Graph (corresponds to gene bytes)
        input_shape=(28, 28, 1),  # Shape of input data
        kernels_per_layer=(20, 50),  # Number of kernels per layer
        kernel_sizes=((5, 5), (5, 5)),  # Sizes of kernels per layer
        dense_units=500,  # Number of units in Dense layer
        dropout_probability=0.5,  # Dropout probability
        classes=10,  # Number of classes to predict
        kfold=5,
        epochs=(5, 4),
        learning_rate=(1e-3, 1e-4),
        batch_size=128
    )
    model_with_skip = GeneticCnnWithSkipModel(
        x_train=x_train, 
        y_train=y_train,
        genes={'Stage_1': '0001', 'Stage_2': '00000000001'},  # Genes to test
        nodes_per_stage=(3, 5),  # Number of nodes per Directed Acyclic Graph (corresponds to gene bytes)
        input_shape=(28, 28, 1),  # Shape of input data
        kernels_per_layer=(20, 50),  # Number of kernels per layer
        kernel_sizes=((5, 5), (5, 5)),  # Sizes of kernels per layer
        dense_units=500,  # Number of units in Dense layer
        dropout_probability=0.5,  # Dropout probability
        classes=10,  # Number of classes to predict
        kfold=5,
        epochs=(5, 4),
        learning_rate=(1e-3, 1e-4),
        batch_size=128
    )

    print("\n---------------------------------------------------------")
    print("Validation accuracy for model without skip: ", model_without_skip.cross_validate())
    print("Validation accuracy for model with skip: ", model_with_skip.cross_validate())
