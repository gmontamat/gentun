#!/usr/bin/env python
"""
Machine Learning models compatible with the Genetic Algorithm implemented using Keras
"""

import keras.backend as K

from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Model

from .generic_models import GentunModel

K.set_image_data_format('channels_last')


class GeneticCnnModel(GentunModel):

    def __init__(self, x_train, y_train, genes, kernels_per_layer, kernel_sizes):
        super(GeneticCnnModel, self).__init__(x_train, y_train)
        self.model = self.build_model(genes, kernels_per_layer, kernel_sizes)

    @staticmethod
    def build_model(genes, kernels_per_layer, kernel_sizes):
        X_input = Input((28, 28, 1))
        X = X_input
        for layer, kernels in enumerate(kernels_per_layer):
            connections = genes['S_{}'.format(layer + 1)]
            X = Conv2D(kernels, kernel_size=kernel_sizes[layer], strides=(1, 1), padding='same')(X)
            X = Activation('relu')(X)
            # TODO: complete internal connections
            X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)
        X = Flatten()(X)
        X = Dense(500, activation='relu')(X)
        X = Dropout(0.5)(X)
        X = Dense(10, activation='softmax')(X)
        return Model(inputs=X_input, outputs=X, name='GeneticCNN')

    def cross_validate(self):
        """Train model using n-fold cross validation and
        return mean value of validation metric.
        """
        self.model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=2, batch_size=128)
