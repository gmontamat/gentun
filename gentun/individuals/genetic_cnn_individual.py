#!/usr/bin/env python
"""
Classes which define the individuals of a population with
its characteristic genes, generation, crossover and
mutation processes.
"""

import random

try:
    from .individual import Individual
except ImportError:
    pass

try:
    from ..models.keras_models import GeneticCnnModel
except ImportError:
    pass


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
