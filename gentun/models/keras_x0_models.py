#!/usr/bin/env python
"""
Machine Learning models compatible with the Genetic Algorithm implemented using Keras
"""

import keras.backend as K
import numpy as np

from keras.layers import Input, Conv2D, Activation, Add, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import StratifiedKFold

from .generic_models import GentunModel

K.set_image_data_format('channels_last')


class GeneticCnnX0Model(GentunModel):  # TODO: add typing and docstring
    """
    Model of Neural Network leyars proposed in article:
    NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm
    by Zhichao Lu, Ian Whalen, Vishnu Boddeti, Yashesh Dhebar, Kalyanmoy Deb, Erik Goodman and Wolfgang Banzhaf: 
    https://arxiv.org/pdf/1703.01513.pdf 

    It is variation of model of Neural Network leyars proposed in article: 
    Genetic CNN 
    by Lingxi Xie, Alan Yuille: 
    https://arxiv.org/pdf/1703.01513.pdf 
    """

    def __init__(
        self, 
        x_train,  # TODO: add typing
        y_train,  # TODO: add typing
        genes: dict, 
        nodes_per_stage: tuple, 
        input_shape: tuple, 
        kernels_per_layer: tuple, 
        kernel_sizes: tuple, 
        dense_units: int,
        dropout_probability: float, 
        classes: int, 
        kfold: int = 5, 
        epochs: tuple = (3,), 
        learning_rate: tuple = (1e-3,), 
        batch_size: int = 32
    ):
        # Validate if we can proceed and set model's attributes
        if type(epochs) is int and type(learning_rate) is int:
            self.epochs = (epochs,)
            self.learning_rate = (learning_rate,)
        elif type(epochs) is tuple and type(learning_rate) is tuple:
            self.epochs = epochs
            self.learning_rate = learning_rate
        else:
            print(epochs, learning_rate)
            raise ValueError("epochs and learning_rate must be both either integers or tuples of integers.")

        # Set model's attributes
        super(GeneticCnnX0Model, self).__init__(x_train, y_train)
        self.model = self.build_model(
            genes, nodes_per_stage, input_shape, kernels_per_layer, kernel_sizes,
            dense_units, dropout_probability, classes
        )
        self.name = '-'.join(gene for gene in genes.values())
        self.kfold = kfold

        self.batch_size = batch_size

    def plot(self) -> None:
        """Draw model to validate gene-to-DAG."""
        from keras.utils import plot_model
        plot_model(self.model, to_file='{}.png'.format(self.name))

    @staticmethod
    def build_dag(x, nodes_per_stage, connections, kernels):  # TODO: add typing
        # Get number of nodes (K_s) using the fact that K_s*(K_s-1)/2 == #bits
        # nodes = int((1 + (1 + 8 * len(connections)) ** 0.5) / 2)
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
        for node in range(nodes_per_stage - 1):
            node_outputs = []
            for i, node_connections in enumerate(separated_connections[node:]):
                if node_connections[node] == '1':
                    node_outputs.append(node + i + 1)
            outputs.append(node_outputs)
        outputs.append([])
        # Get inputs by node (dummy input, x, ignored)
        inputs = [[]]
        for node in range(1, nodes_per_stage):
            node_inputs = []
            for i, connection in enumerate(separated_connections[node - 1]):
                if connection == '1':
                    node_inputs.append(i)
            inputs.append(node_inputs)
        # Build DAG
        output_vars = []
        all_vars = [None] * nodes_per_stage
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

    def build_model(self, genes, nodes_per_stage, input_shape, kernels_per_layer, kernel_sizes,
                    dense_units, dropout_probability, classes):  # TODO: add typing
        x_input = Input(input_shape)
        x = x_input
        for layer, kernels in enumerate(kernels_per_layer):
            # Default input node
            x = Conv2D(kernels, kernel_size=kernel_sizes[layer], strides=(1, 1), padding='same')(x)
            x = Activation('relu')(x)
            # Decode internal connections
            connections = genes['Stage_{}'.format(layer + 1)]
            # If at least one bit is 1, then we need to construct the Directed Acyclic Graph
            if not all([not bool(int(connection)) for connection in connections]):
                x = self.build_dag(x, nodes_per_stage[layer], connections, kernels)
                # Output node
                x = Conv2D(kernels, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
                x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(dropout_probability)(x)
        x = Dense(classes, activation='softmax')(x)
        return Model(inputs=x_input, outputs=x, name='GeneticCNN')

    def reset_weights(self) -> None:
        """Initialize model weights."""
        session = K.get_session()
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

    def cross_validate(self):  # TODO: add typing
        """Train model using k-fold cross validation and
        return mean value of the validation accuracy.
        """
        acc = .0
        kfold = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        for fold, (train, validation) in enumerate(kfold.split(self.x_train, np.where(self.y_train == 1)[1])):
            print("KFold {}/{}".format(fold + 1, self.kfold))
            self.reset_weights()
            for epochs, learning_rate in zip(self.epochs, self.learning_rate):
                print("Training {} epochs with learning rate {}".format(epochs, learning_rate))
                self.model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
                self.model.fit(
                    self.x_train[train], self.y_train[train], epochs=epochs, batch_size=self.batch_size, verbose=1
                )
            acc += self.model.evaluate(self.x_train[validation], self.y_train[validation], verbose=0)[1] / self.kfold
        return acc
