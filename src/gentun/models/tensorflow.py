"""
Models implemented in tensorflow
"""

import logging
import os
from typing import Any, Sequence, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPool2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from .base import Handler

K.set_image_data_format("channels_last")


class GeneticCNN(Handler):
    """
    Implement Genetic CNN model.
    http://arxiv.org/pdf/1703.01513
    """

    def __init__(
        self,
        nodes: Sequence[int],
        kernels_per_layer: Sequence[int],
        kernel_sizes: Sequence[Union[Sequence[int], int]],
        pool_sizes: Sequence[Union[Sequence[int], int]],
        dense_units: int = 500,
        dropout_probability: float = 0.5,
        input_shape: Sequence[int] = (28, 28, 1),
        num_classes: int = 10,
        epochs: Union[int, Sequence[int]] = (3,),
        learning_rate: Union[int, Sequence[int]] = (1e-3,),
        batch_size: int = 32,
        plot: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert (
            len(nodes) == len(kernels_per_layer) == len(kernel_sizes) == len(pool_sizes)
        ), "`nodes`, `kernels_per_layer`, `kernel_sizes`, and `pool_sizes` should have the same length (#layers)."
        # Define node connections
        connections = []
        for i in range(len(nodes)):
            connections.append(kwargs[f"S_{i + 1}"])
        self.name = f"GeNet__{'-'.join(connection for connection in connections)}"
        self.model = self.build_model(
            connections,
            nodes,
            input_shape,
            kernels_per_layer,
            kernel_sizes,
            pool_sizes,
            dense_units,
            dropout_probability,
            num_classes,
        )
        if plot:
            self.plot()
        self.batch_size = batch_size
        self.epochs = (epochs,) if isinstance(epochs, int) else epochs
        self.learning_rate = (learning_rate,) if isinstance(learning_rate, float) else learning_rate
        assert len(epochs) == len(learning_rate), "`epochs` and `learning_rate` should have the same dimensions."

    def plot(self):
        """
        Draw model to validate gene-to-DAG.
        Install graphviz (apt install graphviz) to use.
        """
        if not os.path.isdir("models"):
            os.mkdir("models")
        plot_model(
            self.model, to_file=f"models/{self.name}.png", show_shapes=True, show_layer_names=True, expand_nested=True
        )

    @staticmethod
    def build_dag(x: Any, nodes: int, connections: str, kernels: int):
        """
        Decode the binary representation into the
        network stages (Genetic CNN - Section 3.1).
        """
        # Separate bits by whose input they represent (Genetic CNN paper uses a dash)
        ctr = 0
        idx = 0
        separated_connections = []
        while idx + ctr < len(connections):
            ctr += 1
            separated_connections.append(connections[idx : idx + ctr])
            idx += ctr
        # Get outputs by node (dummy output ignored)
        outputs = []
        for node in range(nodes - 1):
            node_outputs = []
            for i, node_connections in enumerate(separated_connections[node:]):
                if node_connections[node] == "1":
                    node_outputs.append(node + i + 1)
            outputs.append(node_outputs)
        outputs.append([])
        # Get inputs by node (dummy input, x, ignored)
        inputs = [[]]
        for node in range(1, nodes):
            node_inputs = []
            for i, connection in enumerate(separated_connections[node - 1]):
                if connection == "1":
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
                    add_vars = [all_vars[j] for j in ins]
                    if len(add_vars) > 1:
                        tmp = Add()(add_vars)
                    else:
                        tmp = add_vars[0]
                tmp = Conv2D(kernels, kernel_size=(3, 3), strides=(1, 1), padding="same")(tmp)
                tmp = BatchNormalization()(tmp)
                tmp = Activation("relu")(tmp)
                all_vars[i] = tmp
                if not outs:
                    output_vars.append(tmp)
        if len(output_vars) > 1:
            return Add()(output_vars)
        return output_vars[0]

    def build_model(
        self,
        connections: Sequence[str],
        nodes: Sequence[int],
        input_shape: Sequence[int],
        kernels_per_layer: Sequence[int],
        kernel_sizes: Sequence[Union[Sequence[int], int]],
        pool_sizes: Sequence[Union[Sequence[int], int]],
        dense_units: int,
        dropout_probability: float,
        num_classes: int,
    ) -> Model:
        """Create the Convolutional Neural Network."""
        x_input = Input(input_shape)
        x = x_input
        for layer, kernels in enumerate(kernels_per_layer):
            # Default input node
            x = Conv2D(kernels, kernel_size=kernel_sizes[layer], strides=(1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            # Decode internal connections
            # If at least one bit is 1, then we need to construct the Directed Acyclic Graph
            if not all(not bool(int(bit)) for bit in connections[layer]):
                x = self.build_dag(x, nodes[layer], connections[layer], kernels)
                # Output node
                x = Conv2D(kernels, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
            x = MaxPool2D(pool_size=pool_sizes[layer], strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(dense_units, activation="relu")(x)
        x = Dropout(dropout_probability)(x)
        x = Dense(num_classes, activation="softmax")(x)
        return Model(inputs=x_input, outputs=x, name=f"{self.name}")

    def reset_weights(self):
        """Initialize model weights."""
        for layer in self.model.layers:
            if hasattr(layer, "kernel_initializer") and hasattr(layer, "bias_initializer"):
                layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))
                layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))
            elif hasattr(layer, "kernel_initializer"):
                layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))

    def create_train_evaluate(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
    ) -> float:
        """
        Train model using k-fold cross validation and
        return mean value of the validation accuracy.
        """
        self.reset_weights()
        for epochs, learning_rate in zip(self.epochs, self.learning_rate):
            logging.debug("Training %d epochs with learning rate %4.1g", epochs, learning_rate)
            self.model.compile(
                optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"]
            )
            self.model.fit(x_train, y_train, epochs=epochs, batch_size=self.batch_size, verbose=1)
        return self.model.evaluate(x_test, y_test, verbose=0)[1]
