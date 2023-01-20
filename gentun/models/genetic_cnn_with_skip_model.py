#!/usr/bin/env python
"""
Machine Learning models compatible with the Genetic Algorithm implemented using Keras.
"""

import keras.backend as K
import numpy as np

from keras.layers import Input, Conv2D, Activation, Add, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import plot_model
from sklearn.model_selection import StratifiedKFold
import keras

from .generic_models import GentunModel

K.set_image_data_format('channels_last')


class GeneticCnnWithSkipModel(GentunModel):
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
        x_train: np.ndarray,
        y_train: np.ndarray,
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
        """
        Note:
        (NN) - means term is related to neural networks.

        :param x_train (numpy.ndarray): data input set (NN).
        :param y_train (numpy.ndarray): data output set (NN).
        :param genes (dict): dict containing stages and strings containing 0 and 1 to represent connections between nodes. Genes is the same thing as chromosome.
        :param nodes_per_stage (tuple): number of nodes for each stage. Default value is only example for easier class usage.
        :param input_shape (tuple): shape of input image (NN). Default value for digit recognition example using.
        :param kernels_per_layer (tuple): number of kernels for each layer (NN). Kernel is a matrix, which is slid across the image and multiplied with the input such that the output is enhanced in a certain desirable manner. Default value is only example for easier class usage.
        :param kernel_sizes (tuple): size of kernels (NN). Kernel is a matrix, which is slid across the image and multiplied with the input such that the output is enhanced in a certain desirable manner. Default value is only example for easier class usage.
        :param dense_units (int): represents the output size of the dense layer (NN). Default value is only example for easier class usage.
        :param dropout_probability (float): probability of dropout (NN). Dropout operation remove values of some weights. Default value is only example for easier class usage.
        :param classes (int): number of classes possible images (NN). Default value for digit recognition example using CAFIR10.
        :param kfold (int): Number of folds for K-Fold Cross-Validation (NN). It is way of validating neural network results. Must be at least 2. Default value is the same as one in used later in the code function sklearn.model_selection.StratifiedKFold.
        :param epochs (tuple): number of epochs neural network will be trained (NN). Epochs means how many times we will train neural network.  Default value is only example for easier class usage (NN).
        :param learning_rate (float): rate of learning means how much we will modyfie weights (NN). Default value is only example for easier class usage.
        :param batch_size (int):  defines the number of samples that will be propagated through the network (NN). Default value is only example for easier class usage.
        """
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
        super(GeneticCnnWithSkipModel, self).__init__(x_train, y_train)
        self.model = self.build_model(
            genes, nodes_per_stage, input_shape, kernels_per_layer, kernel_sizes,
            dense_units, dropout_probability, classes
        )
        self.name = '-'.join(gene for gene in genes.values())
        self.kfold = kfold

        self.batch_size = batch_size

    def plot(self) -> None:
        """Draw model to validate gene-to-directed_acyclic_graph."""
        plot_model(self.model, to_file='{}.png'.format(self.name))

    @staticmethod
    def build_directed_acyclic_graph(
        x: keras.engine.keras_tensor.KerasTensor, 
        nodes_per_stage: tuple, 
        connections: str, 
        kernels: int
    ) -> keras.engine.keras_tensor.KerasTensor:  # TODO: add docstring
        """
        Build directed acyclic graph from Conv2D and Activation layers as realisation of computational block.

        :param x (keras.engine.keras_tensor.KerasTensor): already created neural network architecture.
        :param nodes_per_stage (tuple): number of nodes for each stage.
        :param connections (str): string containing 0 and 1 to represent connections between nodes.
        :param kernels (int): number of kernels for layer. Kernel is a matrix, which is slid across the image and multiplied with the input such that the output is enhanced in a certain desirable manner.

        :return keras.engine.keras_tensor.KerasTensor: builded directed acyclic graph from Conv2D and Activation layers.
        """
        # Get number of nodes (K_s) using the fact that K_s*(K_s-1)/2 + 1 == #bits
        # nodes = int((1 + (1 + 8 * len(connections)) ** 0.5) / 2)
        # Separate bits by whose input they represent (GeneticCNN paper uses a dash)
        ctr = 0
        idx = 0
        separated_connections = []
        while idx + ctr < len(connections) - 1:  # To not include last skip bit
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
        
        # Build Directed Acyclic Graph
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
        
        # Add builded Directed Acyclic Graph to current exited architecture
        if len(output_vars) > 1:
            return Add()(output_vars)
        
        return output_vars[0]

    def build_model(
        self, 
        genes: dict, 
        nodes_per_stage: tuple, 
        input_shape: tuple, 
        kernels_per_layer: tuple, 
        kernel_sizes: tuple, 
        dense_units: int,
        dropout_probability: float, 
        classes: int, 
    ) -> Model:
        """
        :param genes (dict): dict containing stages and strings containing 0 and 1 to represent connections between nodes. Genes is the same thing as chromosome.
        :param nodes_per_stage (tuple): number of nodes for each stage.
        :param input_shape (tuple): shape of input image. Default value for digit recognition example using.
        :param kernels_per_layer (tuple): number of kernels for each layer. Kernel is a matrix, which is slid across the image and multiplied with the input such that the output is enhanced in a certain desirable manner. Default value is only example for easier class usage.
        :param kernel_sizes (tuple): size of kernels. Kernel is a matrix, which is slid across the image and multiplied with the input such that the output is enhanced in a certain desirable manner. Default value is only example for easier class usage.
        :param dense_units (int): represents the output size of the dense layer.
        :param dropout_probability (float): probability of dropout. Dropout operation remove values of some weights.
        :param classes (int): number of classes possible images. Default value for digit recognition example using CAFIR10.

        :return tensorflow.keras.models.Model: builded tensorflow model object.
        """
        x_input = Input(input_shape)
        x = x_input

        # Building 
        for layer, kernels in enumerate(kernels_per_layer):
            # Decode internal connections
            connections = genes['Stage_{}'.format(layer + 1)]

            # Check last bit of internal connections to check if we need to skip layer 
            # 0 - no direct connection input-output (creat other connections)
            # 1 - direct connection input-output (skip layer)
            if int(connections) == 0:

                # Default input node
                x = Conv2D(kernels, kernel_size=kernel_sizes[layer], strides=(1, 1), padding='same')(x)
                x = Activation('relu')(x)
                
                # If at least one bit is 1, then we need to construct the Directed Acyclic Graph
                if not all([not bool(int(connection)) for connection in connections]):
                    
                    x = self.build_directed_acyclic_graph(x, nodes_per_stage[layer], connections, kernels)
                    
                    # Output node
                    x = Conv2D(kernels, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
                    x = Activation('relu')(x)
                
                x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        # Adding final output layers
        x = Flatten()(x)
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(dropout_probability)(x)
        x = Dense(classes, activation='softmax')(x)

        return Model(inputs=x_input, outputs=x, name='GeneticCNNWithSkip')

    def reset_weights(self) -> None:
        """Initialize model weights."""
        session = K.get_session()
        for layer in self.model.layers: 
            for layer_type in layer.__dict__:
                layer_type_arg = getattr(layer, layer_type)

                if hasattr(layer_type_arg,'kernel_initializer'):
                    initializer_method = getattr(layer_type_arg, 'kernel_initializer')
                    initializer_method.run(session=session)
                    print('reinitializing layer {}.{}'.format(layer.name, layer_type))

    def cross_validate(self) -> float:
        """
        Train model using k-fold cross validation and
        return mean value of the validation accuracy.

        :return float: mean value of the validation accuracy.
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
                    self.x_train[train], 
                    self.y_train[train], 
                    epochs=epochs, 
                    batch_size=self.batch_size, 
                    verbose=1
                )

            acc += self.model.evaluate(self.x_train[validation], self.y_train[validation], verbose=0)[1] / self.kfold
        return acc
