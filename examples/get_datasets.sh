#!/usr/bin/env bash

if [ ! -e iris.data ]; then
    echo "Downloading iris dataset..."
    wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
fi

if [ ! -e mnist.npz ]; then
    echo "Downloading MNIST dataset..."
    wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
fi

echo "Done!"
