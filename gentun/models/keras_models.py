#!/usr/bin/env python
"""
Machine Learning models compatible with the Genetic Algorithm implemented using Keras
"""

import keras.backend as K

from keras.layers import Input, Conv2D, Activation, Add, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model

from .generic_models import GentunModel

K.set_image_data_format('channels_last')


class GeneticCnnModel(GentunModel):

    def __init__(self, x_train, y_train, genes, kernels_per_layer, kernel_sizes, input_shape, classes):
        super(GeneticCnnModel, self).__init__(x_train, y_train)
        self.model = self.build_model(genes, kernels_per_layer, kernel_sizes, input_shape, classes)

    @staticmethod
    def build_model(genes, kernels_per_layer, kernel_sizes, input_shape, classes):
        x_input = Input(input_shape)
        x = x_input
        for layer, kernels in enumerate(kernels_per_layer):
            # Default input node
            x = Conv2D(kernels, kernel_size=kernel_sizes[layer], strides=(1, 1), padding='same')(x)
            x = Activation('relu')(x)
            # Decode internal connections
            connections = genes['S_{}'.format(layer + 1)]
            if all([not bool(int(connection)) for connection in connections]):
                # TODO: DAG
                # TODO: Default output node
                pass
            x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(500, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(classes, activation='softmax')(x)
        return Model(inputs=x_input, outputs=x, name='GeneticCNN')

    def cross_validate(self):
        """Train model using n-fold cross validation and
        return mean value of validation metric.
        """
        self.model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=2, batch_size=128)
