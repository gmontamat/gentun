"""
Models implemented in tensorflow
"""

import numpy as np
import os
import tensorflow.compat.v1 as tf

from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Activation, Add, Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D
from tensorflow.python.keras.models import Model as KerasModel
from tensorflow.python.keras.optimizer_v1 import Adam
from tensorflow.keras.utils import plot_model
from typing import List, Tuple, Union

from .base import Model

# Compatibility with TF1
tf.disable_eager_execution()
tf.experimental.output_all_intermediates(True)
K.set_image_data_format("channels_last")


class GeneticCNN(Model):

    def __init__(
            self,
            nodes: Tuple[int, ...],
            kernels_per_layer: Tuple[int, ...],
            kernel_sizes: Tuple[Tuple[int, ...], ...],
            dense_units: int = 500,
            dropout_probability: float = 0.5,
            input_shape: Tuple[int, ...] = (28, 28, 1),
            num_classes: int = 10,
            kfold: int = 5,
            epochs: Union[int, Tuple[int, ...]] = (3,),
            learning_rate: Union[int, Tuple[int, ...]] = (1e-3,),
            batch_size: int = 32,
            plot: bool = False,
            **kwargs
    ):
        super().__init__()
        assert len(nodes) == len(kernels_per_layer) == len(kernel_sizes), \
            "`nodes`, `kernels_per_layer`, and `kernel_sizes` should have the same length (#layers)."
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
            dense_units,
            dropout_probability,
            num_classes
        )
        if plot:
            self.plot()
        self.kfold = kfold
        self.batch_size = batch_size
        assert (
                (isinstance(epochs, int) and isinstance(learning_rate, int)) or
                (len(epochs) == len(learning_rate))
        ), "`epochs` and `learning_rate` should have the same dimensions."
        self.epochs = epochs
        self.learning_rate = learning_rate

    def plot(self):
        """
        Draw model to validate gene-to-DAG.
        Install graphviz (apt install graphviz) to use.
        """
        if not os.path.isdir("models"):
            os.mkdir("models")
        plot_model(self.model, to_file=f"models/{self.name}.png", show_shapes=True, show_layer_names=True, expand_nested=True)

    @staticmethod
    def build_dag(x, nodes, connections, kernels):
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
                    add_vars = [all_vars[i] for i in ins]
                    if len(add_vars) > 1:
                        tmp = Add()(add_vars)
                    else:
                        tmp = add_vars[0]
                tmp = Conv2D(kernels, kernel_size=(3, 3), strides=(1, 1), padding="same")(tmp)
                tmp = Activation("relu")(tmp)
                all_vars[i] = tmp
                if not outs:
                    output_vars.append(tmp)
        if len(output_vars) > 1:
            return Add()(output_vars)
        return output_vars[0]

    def build_model(
            self,
            connections: List[str],
            nodes: Tuple[int, ...],
            input_shape: Tuple[int, ...],
            kernels_per_layer: Tuple[int, ...],
            kernel_sizes: Tuple[Tuple[int, ...], ...],
            dense_units: int,
            dropout_probability: float,
            num_classes: int
    ) -> KerasModel:
        x_input = Input(input_shape)
        x = x_input
        for layer, kernels in enumerate(kernels_per_layer):
            # Default input node
            x = Conv2D(kernels, kernel_size=kernel_sizes[layer], strides=(1, 1), padding="same")(x)
            x = Activation("relu")(x)
            # Decode internal connections
            # If at least one bit is 1, then we need to construct the Directed Acyclic Graph
            if not all([not bool(int(bit)) for bit in connections[layer]]):
                x = self.build_dag(x, nodes[layer], connections[layer], kernels)
                # Output node
                x = Conv2D(kernels, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
                x = Activation("relu")(x)
            x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(dense_units, activation="relu")(x)
        x = Dropout(dropout_probability)(x)
        x = Dense(num_classes, activation="softmax")(x)
        return KerasModel(inputs=x_input, outputs=x, name=f"{self.name}")

    def reset_weights(self):
        """Initialize model weights."""
        session = K.get_session()
        for layer in self.model.layers:
            if hasattr(layer, "kernel_initializer"):
                layer.kernel.initializer.run(session=session)

    def evaluate(self, x_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Train model using k-fold cross validation and
        return mean value of the validation accuracy.
        """
        acc = .0
        cross_validation = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        for fold, (train, validation) in enumerate(cross_validation.split(x_train, np.where(y_train == 1)[1])):
            print(f"KFold {fold + 1}/{self.kfold}")
            self.reset_weights()
            for epochs, learning_rate in zip(self.epochs, self.learning_rate):
                print(f"Training {epochs} epochs with learning rate {learning_rate}")
                self.model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
                self.model.fit(
                    x_train[train], y_train[train], epochs=epochs, batch_size=self.batch_size, verbose=1
                )
            acc += self.model.evaluate(x_train[validation], y_train[validation], verbose=0)[1] / self.kfold
        return acc
