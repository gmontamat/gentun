#!/usr/bin/env python
"""
Machine Learning models compatible with the Genetic Algorithm implemented using Keras
"""

import keras.backend as K

from keras.layers import Input, Conv2D, Activation, Add, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model

from .generic_models import GentunModel

K.set_image_data_format('channels_last')


def build_dag(x, connections, kernels):
    # Get number of nodes (K_s) using the fact that K_s*(K_s-1)/2 == #bits
    nodes = int((1 + (1 + 8 * len(connections)) ** 0.5) / 2)
    # Separate bits by whose input they represent (GeneticCNN paper uses a dash)
    ctr = 0
    idx = 0
    separated_connections = []
    while idx + ctr < len(connections):
        ctr += 1
        separated_connections.append(connections[idx:idx + ctr])
        idx += ctr
    # Get outputs by node (dummy output ignored)
    outputs = []
    for node in range(nodes - 1):
        node_outputs = []
        for i, node_connections in enumerate(separated_connections[node:]):
            if node_connections[node] == '1':
                node_outputs.append(node + i + 1)
        outputs.append(node_outputs)
    outputs.append([])
    # Get inputs by node (dummy input, x, ignored)
    inputs = [[]]
    for node in range(1, nodes):
        node_inputs = []
        for i, connection in enumerate(separated_connections[node - 1]):
            if connection == '1':
                node_inputs.append(i)
        inputs.append(node_inputs)
    # Build DAG
    output_vars = []
    all_vars = [None] * nodes
    for i, (ins, outs) in enumerate(zip(inputs, outputs)):
        if ins or outs:
            if not ins:
                tmp = x
            else:
                add_vars = [all_vars[i] for i in ins]
                if len(add_vars) > 1:
                    tmp = Add()(add_vars)
                else:
                    tmp = add_vars[0]
            tmp = Conv2D(kernels, kernel_size=(3, 3), strides=(1, 1), padding='same')(tmp)
            tmp = Activation('relu')(tmp)
            all_vars[i] = tmp
            if not outs:
                output_vars.append(tmp)
    if len(output_vars) > 1:
        return Add()(output_vars)
    return output_vars[0]


class GeneticCnnModel(GentunModel):

    def __init__(self, x_train, y_train, genes, kernels_per_layer, kernel_sizes, input_shape, classes):
        super(GeneticCnnModel, self).__init__(x_train, y_train)
        self.model = self.build_model(genes, kernels_per_layer, kernel_sizes, input_shape, classes)
        # Draw model to validate gene-to-DAG
        # from keras.utils import plot_model
        # plot_model(self.model, to_file='model.png')

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
            # If at least one bit is 1, then we need to construct the Directed Acyclic Graph
            if not all([not bool(int(connection)) for connection in connections]):
                x = build_dag(x, connections, kernels)
                # Output node
                x = Conv2D(kernels, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
                x = Activation('relu')(x)
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
        # TODO: cross-validations or at least train/test split
        return self.model.fit(self.x_train, self.y_train, epochs=2, batch_size=128, verbose=0)
