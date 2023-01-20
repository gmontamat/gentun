#!/usr/bin/env python
"""
Classes which define the individuals of a population with
its characteristic genes, generation, crossover and
mutation processes.
"""

import math
import pprint
import random
import numpy

try:
    from .models.xgboost_models import XgboostModel
except ImportError:
    pass

try:
    from .models.keras_models import GeneticCnnModel
except ImportError:
    pass

try:
    from .models.genetic_cnn_with_skip_model import GeneticCnnWithSkipModel
except ImportError:
    pass


def random_log_uniform(minimum, maximum, base, eps=1e-12):
    """Generate a random number which is uniform in a
    logarithmic scale. If base > 0 scale goes from minimum
    to maximum, if base < 0 vice versa, and if base is 0,
    use a uniform scale.
    """
    if base == 0:
        return random.uniform(minimum, maximum)
    minimum += eps  # Avoid math domain error when minimum is zero
    if base > 0:
        return base ** random.uniform(math.log(minimum, base), math.log(maximum, base))
    base = abs(base)
    return maximum - base ** random.uniform(math.log(eps, base), math.log(maximum - minimum, base))


class Individual(object):
    """Basic definition of an individual containing
    reproduction and mutation methods. Do not instantiate,
    use a subclass which extends this object by defining a
    genome and a random individual generator.
    """

    def __init__(self, x_train, y_train, genome, genes, crossover_rate, mutation_rate, additional_parameters=None):
        self.x_train = x_train
        self.y_train = y_train
        self.genome = genome
        self.validate_genome()
        self.genes = genes
        self.validate_genes()
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.fitness = None  # Until evaluated an individual fitness is unknown
        assert additional_parameters is None

    def validate_genome(self):
        """Check genome structure."""
        if type(self.genome) != dict:
            raise TypeError("Genome must be a dictionary.")
        for gene, properties in self.genome.items():
            if type(gene) != str:
                raise TypeError("Gene names must be strings.")

    def validate_genes(self):
        """Check that genes are compatible with genome."""
        if set(self.genome.keys()) != set(self.genes.keys()):
            raise ValueError("Genes passed don't correspond to individual's genome.")

    def get_genes(self):
        """Return individual's genes."""
        return self.genes

    def get_genome(self):
        """Return individual's genome."""
        return self.genome

    @staticmethod
    def generate_random_genes(genome):
        raise NotImplementedError("Use a subclass with genes definition.")

    def evaluate_fitness(self):
        raise NotImplementedError("Use a subclass with genes definition.")

    def get_additional_parameters(self):
        raise NotImplementedError("Use a subclass with genes definition.")

    def get_fitness(self):
        """Compute individual's fitness if necessary and return it."""
        if self.fitness is None:
            self.evaluate_fitness()
        return self.fitness

    def reproduce(self, partner):
        """Mix genes from self and partner randomly and
        return a new instance of an individual. Do not
        mutate parents.
        """
        assert self.__class__ == partner.__class__  # Can only reproduce if they're the same species
        child_genes = {}
        for name, value in self.get_genes().items():
            if random.random() < self.crossover_rate:
                child_genes[name] = partner.get_genes()[name]
            else:
                child_genes[name] = value
        return self.__class__(
            self.x_train, self.y_train, self.genome, child_genes, self.crossover_rate, self.mutation_rate,
            **self.get_additional_parameters()
        )

    def crossover(self, partner):
        """Mix genes from self and partner randomly.
        Mutates each parent instead of producing a
        new instance (child).
        """
        assert self.__class__ == partner.__class__  # Can only cross if they're the same species
        for name in self.get_genes().keys():
            if random.random() < self.crossover_rate:
                self.get_genes()[name], partner.get_genes()[name] = partner.get_genes()[name], self.get_genes()[name]
                self.set_fitness(None)
                partner.set_fitness(None)

    def mutate(self):
        """Mutate instance's genes with a certain probability."""
        for name, value in self.get_genes().items():
            if random.random() < self.mutation_rate:
                default, minimum, maximum, log_scale = self.get_genome()[name]
                if type(default) == int:
                    self.get_genes()[name] = random.randint(minimum, maximum)
                else:
                    self.get_genes()[name] = round(random_log_uniform(minimum, maximum, log_scale), 4)
                self.set_fitness(None)  # The mutation produces a new individual

    def get_fitness_status(self):
        """Return True if individual's fitness in known."""
        return self.fitness is not None

    def set_fitness(self, value):
        """Assign fitness."""
        self.fitness = value

    def copy(self):
        """Copy instance."""
        individual_copy = self.__class__(
            self.x_train, self.y_train, self.genome, self.genes.copy(), self.crossover_rate,
            self.mutation_rate, **self.get_additional_parameters()
        )
        individual_copy.set_fitness(self.fitness)
        return individual_copy

    def __str__(self):
        """Return genes which identify the individual."""
        return pprint.pformat(self.genes)


class XgboostIndividual(Individual):

    def __init__(self, x_train, y_train, genome=None, genes=None, crossover_rate=0.5, mutation_rate=0.015,
                 booster='gbtree', objective='reg:linear', eval_metric='rmse', kfold=5,
                 num_boost_round=5000, early_stopping_rounds=100):
        if genome is None:
            genome = {
                # name: (default, min, max, logarithmic-scale-base)
                'eta': (0.3, 0.001, 1.0, 10),
                'min_child_weight': (1, 0, 10, None),
                'max_depth': (6, 3, 10, None),
                'gamma': (0.0, 0.0, 10.0, 10),
                'max_delta_step': (0, 0, 10, None),
                'subsample': (1.0, 0.0, 1.0, -10),
                'colsample_bytree': (1.0, 0.0, 1.0, -10),
                'colsample_bylevel': (1.0, 0.0, 1.0, -10),
                'lambda': (1.0, 0.1, 10.0, 10),
                'alpha': (0.0, 0.0, 10.0, 10),
                'scale_pos_weight': (1.0, 0.0, 10.0, 0)
            }
        if genes is None:
            genes = self.generate_random_genes(genome)
        # Set individual's attributes
        super(XgboostIndividual, self).__init__(x_train, y_train, genome, genes, crossover_rate, mutation_rate)
        # Set additional parameters which are not tuned
        self.booster = booster
        self.objective = objective
        self.eval_metric = eval_metric
        self.kfold = kfold
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    @staticmethod
    def generate_random_genes(genome):
        """Create and return random genes."""
        genes = {}
        for name, (default, minimum, maximum, log_scale) in genome.items():
            if type(default) == int:
                genes[name] = random.randint(minimum, maximum)
            else:
                genes[name] = round(random_log_uniform(minimum, maximum, log_scale), 4)
        return genes

    def evaluate_fitness(self):
        """Create model and perform cross-validation."""
        model = XgboostModel(
            self.x_train, self.y_train, self.genes, booster=self.booster, objective=self.objective,
            eval_metric=self.eval_metric, kfold=self.kfold, num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds
        )
        self.fitness = model.cross_validate()

    def get_additional_parameters(self):
        return {
            'booster': self.booster,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'kfold': self.kfold,
            'num_boost_round': self.num_boost_round,
            'early_stopping_rounds': self.early_stopping_rounds
        }


class GeneticCnnIndividual(Individual):

    def __init__(self, x_train, y_train, genome=None, genes=None, crossover_rate=0.3, mutation_rate=0.1, nodes=(3, 5),
                 input_shape=(28, 28, 1), kernels_per_layer=(20, 50), kernel_sizes=((5, 5), (5, 5)), dense_units=500,
                 dropout_probability=0.5, classes=10, kfold=5, epochs=(3,), learning_rate=(1e-3,), batch_size=32):
        if genome is None:
            genome = {'S_{}'.format(i + 1): int(K_s * (K_s - 1) / 2) for i, K_s in enumerate(nodes)}
        if genes is None:
            genes = self.generate_random_genes(genome)
        # Set individual's attributes
        super(GeneticCnnIndividual, self).__init__(x_train, y_train, genome, genes, crossover_rate, mutation_rate)
        # Set additional parameters which are not tuned
        assert len(nodes) == len(kernels_per_layer) and len(kernels_per_layer) == len(kernel_sizes)
        self.nodes = nodes
        self.input_shape = input_shape
        self.kernels_per_layer = kernels_per_layer
        self.kernel_sizes = kernel_sizes
        self.dense_units = dense_units
        self.dropout_probability = dropout_probability
        self.classes = classes
        self.kfold = kfold
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    @staticmethod
    def generate_random_genes(genome):
        """Create and return random genes."""
        genes = {}
        for name, connections in genome.items():
            genes[name] = ''.join([random.choice(['0', '1']) for _ in range(connections)])
        return genes

    def evaluate_fitness(self):
        """Create model and perform cross-validation."""
        model = GeneticCnnModel(
            self.x_train, self.y_train, self.genes, self.nodes, self.input_shape, self.kernels_per_layer,
            self.kernel_sizes, self.dense_units, self.dropout_probability, self.classes,
            self.kfold, self.epochs, self.learning_rate, self.batch_size
        )
        self.fitness = model.cross_validate()

    def get_additional_parameters(self):
        return {
            'nodes': self.nodes,
            'input_shape': self.input_shape,
            'kernels_per_layer': self.kernels_per_layer,
            'kernel_sizes': self.kernel_sizes,
            'dense_units': self.dense_units,
            'dropout_probability': self.dropout_probability,
            'classes': self.classes,
            'kfold': self.kfold,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }

    def mutate(self):
        """Mutate instance's genes with a certain probability."""
        for name, connections in self.get_genes().items():
            new_connections = ''.join([
                str(int(int(byte) != (random.random() < self.mutation_rate))) for byte in connections
            ])
            if new_connections != connections:
                self.set_fitness(None)  # A mutation means the individual has to be re-evaluated
                self.get_genes()[name] = new_connections


class GeneticCnnWithSkipIndividual(Individual):
    """
    Individual of individauls for Neural Network leyars proposed in article:
    NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm
    by Zhichao Lu, Ian Whalen, Vishnu Boddeti, Yashesh Dhebar, Kalyanmoy Deb, Erik Goodman and Wolfgang Banzhaf: 
    https://arxiv.org/pdf/1703.01513.pdf 

    It is variation of individual of Neural Network leyars proposed in article: 
    Genetic CNN 
    by Lingxi Xie, Alan Yuille: 
    https://arxiv.org/pdf/1703.01513.pdf 


    Notes about nodes:
    Node is a basic computational unit, which can be a single operation like convolution, pooling,
    batch-normalization or a sequence of operations.

    The nodes within each stage are ordered, and we only allow connections from a lower-numbered node to a highernumbered node.
    Example: node_1 may be connected to node_2 and/or node_3. Node_2 can be connected to node_3 but not to node_1.

    To make created architecture valid there is the default input node (node_0), receives data from the previous stage, 
    performs convolution, and sends its output to every node without a predecessor, e.g., node_1. 
    User never interact node_0. 


    Reminder:
    Kernel is amatrix, which is slid across the image and multiplied with the input 
    such that the output is enhanced in a certain desirable manner.

    """
    def __init__(
        self, 
        x_train: numpy.ndarray,
        y_train: numpy.ndarray,
        genome: dict = None,
        genes: str = None,
        crossover_rate: float = 0.3, 
        mutation_rate: float = 0.1, 
        nodes_per_stage: tuple = (3, 5),
        input_shape: tuple = (28, 28, 1), 
        kernels_per_layer: tuple = (20, 50),
        kernel_sizes: tuple = ((5, 5), (5, 5)),
        dense_units: int = 500,
        dropout_probability: float = 0.5,
        classes: int = 10,
        kfold: int = 5,
        epochs: tuple =(3,),
        learning_rate: float = (1e-3,),
        batch_size: int = 32 
    ):
        """
        Note:
        (GA) - means term is related to genetic algorithms.
        (NN) - means term is related to neural networks.

        :param x_train (numpy.ndarray): data input set.
        :param y_train (numpy.ndarray): data output set.
        :param genome (dict): stage with a number of nodes for it. Basing on that we will create neural network architecture. Genome is other name for individual.
        :param genes (str): string containing 0 and 1 to represent connections between nodes. Genes is the same thing as chromosome.
        :param crossover_rate (float): probability of crossover (GA). Crossover operation crossover two individuals. Default value is only example for easier class usage.
        :param mutation_rate (float): probability of mutation (GA). Mutation change random bits in individuals. Default value is only example for easier class usage.
        :param nodes_per_stage (tuple): .number of nodes for each stage. Default value is only example for easier class usage.
        :param input_shape (tuple): shape of input image (NN). Default value for digit recognition example using.
        :param kernels_per_layer (tuple): number of kernels for each layer (NN). Kernel is a matrix, which is slid across the image and multiplied with the input such that the output is enhanced in a certain desirable manner. Default value is only example for easier class usage.
        :param kernel_sizes (tuple): size of kernels (NN). Kernel is a matrix, which is slid across the image and multiplied with the input such that the output is enhanced in a certain desirable manner. Default value is only example for easier class usage.
        :param dense_units (int): represents the output size of the dense layer (NN). Default value is only example for easier class usage.
        :param dropout_probability (float): probability of dropout (NN). Dropout operation remove values of some weights . Default value is only example for easier class usage.
        :param classes (int): number of classes possible images (NN). Default value for digit recognition example using CAFIR10.
        :param kfold (int): Number of folds for K-Fold Cross-Validation (NN). It is way of validating neural network results. Must be at least 2. Default value is the same as one in used later in the code function sklearn.model_selection.StratifiedKFold.
        :param epochs (tuple): number of epochs neural network will be trained (NN). Epochs means how many times we will train neural network.  Default value is only example for easier class usage (NN).
        :param learning_rate (float): rate of learning means how much we will modyfie weights (NN). Default value is only example for easier class usage.
        :param batch_size (int):  defines the number of samples that will be propagated through the network. (NN). Default value is only example for easier class usage.
        """
        # Validate if we can proceed
        assert len(nodes_per_stage) == len(kernels_per_layer) and len(kernels_per_layer) == len(kernel_sizes)

        # Set genomes and genes if none (we need to have something to work with)
        if genome is None:
            genome = self.generate_random_genome(nodes_per_stage)
        if genes is None:
            genes = self.generate_random_genes(genome)

        # Set individual's attributes
        super(GeneticCnnWithSkipIndividual, self).__init__(x_train, y_train, genome, genes, crossover_rate, mutation_rate)

        self.nodes_per_stage = nodes_per_stage
        self.input_shape = input_shape
        self.kernels_per_layer = kernels_per_layer
        self.kernel_sizes = kernel_sizes
        self.dense_units = dense_units
        self.dropout_probability = dropout_probability
        self.classes = classes
        self.kfold = kfold
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    @staticmethod
    def generate_random_genome(nodes_per_stage: tuple) -> dict:
        """
        Create and return random genome.
        Genome containing string with stage name and number (template: stage_i) and
        number of bits to encode connections between layers.
        Stage is one computational block. From them we are creating architecture. It may be more then one neural network layer.
        We are calculating number of bits based on node number for each state using formula:
        1/2K_s*(K_s - 1)+1 where K_s is number of nodes for stage.
        We area adding 1 to create a bit for skipping stage/computational block and go right from input to output.

        Example:
        For two stage network S=2 and nodes=(3,5) we will create pairs:
        Stage_1: 3
        Stage_2: 5
        and after calculation of number of bits, we will have:
        Stage_1: 3
        Stage_2: 10
        
        Friendly reminder:
        1. Genome is other name for individual.
        2. Remember we are operating on computational blocks. They are set of convolution layers ended with pool layer.
        3. More details in article Genetic CNN by Lingxi Xie, Alan Yuille, section 3.1 Binary Network Representation

        :param nodes_per_stage (tuple): number of nodes for each stage.

        :return dict: dictionary of stages with a number of nodes for it, organise as {stage_1: x_1, stage_2, x_2, ..., stahe_n, x_n}.
        """
        return {'Stage_{}'.format(i + 1): int(K_s * (K_s - 1) / 2)+1 for i, K_s in enumerate(nodes_per_stage)}
    
    @staticmethod
    def generate_random_genes(genome: dict) -> str:
        """
        Create and return random genes.
        We are calculating number of bits based on node number for each state using formula:
        1/2K_s*(K_s - 1)+1 where K_s is number of nodes for stage.

        Example:
        For two stage network S=2 and nodes_per_stage=(3,5) we will create pairs:
        Stage_1: 3
        Stage_2: 5
        and after calculation of number of bits, we will have:
        Stage_1: 4
        Stage_2: 11
        and after coding genes we will get (example):
        Stage_1: 1000
        Stage_2: 11011000000

        So for first stage with 3 nodes node_1, node_2 and node_3
        the first bit represents the connection between (node_1, node_2),
        then the following two bits represent the connection between (node_1, node_3) and (node_2, node_3), etc.
        if the code corresponding to (node_1, node_2) is 1.
        As effect if there is an edge connecting node_1 and node_2, i.e., 
        node_2 takes the output of node_1 as a part of the element-wise summation, and vice versa.
        Last bit is for skipping stage/computational block and go right from input to output.
        
        Friendly reminder:
        1. Chromosome is a set og genes. So refereing to genes we refering to chromosome.
        2. Convolution layers may be not connect linear.
        3. More details in article Genetic CNN by Lingxi Xie, Alan Yuille, section 3.1 Binary Network Representation

        :param genome (dict): dictionary of stages with a number of nodes for it, organise as {stage_1: x_1, stage_2, x_2, ..., stahe_n, x_n}.

        :return str: string containing 0 and 1 to represent connections between nodes.
        """
        genes = {}
        for name, connections in genome.items():
            genes[name] = ''.join([random.choice(['0', '1']) for _ in range(connections)])
        return genes

    def evaluate_fitness(self) -> None:
        """Create model and perform cross-validation."""
        model = GeneticCnnWithSkipModel(
            self.x_train, self.y_train, self.genes, self.nodes_per_stage, self.input_shape, self.kernels_per_layer,
            self.kernel_sizes, self.dense_units, self.dropout_probability, self.classes,
            self.kfold, self.epochs, self.learning_rate, self.batch_size
        )

        self.fitness = model.cross_validate()

    def get_additional_parameters(self) -> dict:
        """
        Return dictionary with all individual's parameters.
        
        :return dict: dictionary with all individual's parameters.
        """
        return {
            'nodes_per_stage': self.nodes_per_stage,
            'input_shape': self.input_shape,
            'kernels_per_layer': self.kernels_per_layer,
            'kernel_sizes': self.kernel_sizes,
            'dense_units': self.dense_units,
            'dropout_probability': self.dropout_probability,
            'classes': self.classes,
            'kfold': self.kfold,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }

    def mutate(self) -> None:
        """Mutate instance's genes with a certain probability."""
        for name, connections in self.get_genes().items():
            new_connections = ''.join([
                str(int(int(byte) != (random.random() < self.mutation_rate))) for byte in connections
            ])
            if new_connections != connections:
                self.set_fitness(None)  # A mutation means the individual has to be re-evaluated
                self.get_genes()[name] = new_connections
