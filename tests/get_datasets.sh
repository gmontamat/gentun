#!/usr/bin/env bash

echo "Downloading iris dataset..."
wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

echo "Downloading MNIST dataset..."
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

echo "Unzipping MNIST dataset..."
gzip -f -d train-images-idx3-ubyte.gz
gzip -f -d train-labels-idx1-ubyte.gz

echo "Done!"
