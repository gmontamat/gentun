#!/usr/bin/env python
"""
Classes which define the individuals of a population with
its characteristic genes, generation, crossover and
mutation processes.
"""

import math
import pprint
import random

try:
    from .models.xgboost_models import XgboostModel
except ImportError:
    pass

try:
    from .models.keras_models import GeneticCnnModel
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

    def __init__(self, x_train, y_train, genome, genes, uniform_rate, mutation_rate, additional_parameters=None):
        self.x_train = x_train
        self.y_train = y_train
        self.genome = genome
        self.validate_genome()
        self.genes = genes
        self.validate_genes()
        self.uniform_rate = uniform_rate
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
            # if type(properties) != tuple:
            #     raise TypeError("Gene attributes must be a tuple.")
            # if len(properties) != 4:
            #     raise TypeError(
            #         "A gene must have 4 attributes: a default value, "
            #         "minimum value, maximum value, and a logarithm scale "
            #         "(or None for integers)."
            #     )

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
        return a new instance of an individual.
        """
        assert self.__class__ == partner.__class__  # Can only reproduce if they're the same species
        child_genes = {}
        for name, value in self.get_genes().items():
            if random.random() < self.uniform_rate:
                child_genes[name] = value
            else:
                child_genes[name] = partner.get_genes()[name]
        return self.__class__(
            self.x_train, self.y_train, self.genome, child_genes, self.uniform_rate, self.mutation_rate,
            **self.get_additional_parameters()
        )

    def mutate(self):
        """Mutate instance's genes with a certain probability."""
        for name, value in self.get_genes().items():
            if random.random() < self.mutation_rate:
                default, minimum, maximum, log_scale = self.get_genome()[name]
                if type(default) == int:
                    self.get_genes()[name] = random.randint(minimum, maximum)
                else:
                    self.get_genes()[name] = round(random_log_uniform(minimum, maximum, log_scale), 4)
                self.fitness = None  # The mutation produces a new individual

    def get_fitness_status(self):
        """Return True if individual's fitness in known."""
        return self.fitness is not None

    def set_fitness(self, value):
        """Assign fitness. Only to be used by DistributedPopulation."""
        self.fitness = value

    def __str__(self):
        """Return genes which identify the individual."""
        return pprint.pformat(self.genes)


class XgboostIndividual(Individual):

    def __init__(self, x_train, y_train, genome=None, genes=None, uniform_rate=0.5, mutation_rate=0.015,
                 booster='gbtree', objective='reg:linear', eval_metric='rmse', nfold=5,
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
        super(XgboostIndividual, self).__init__(x_train, y_train, genome, genes, uniform_rate, mutation_rate)
        # Set additional parameters which are not tuned
        self.booster = booster
        self.objective = objective
        self.eval_metric = eval_metric
        self.nfold = nfold
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
            eval_metric=self.eval_metric, nfold=self.nfold, num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds
        )
        self.fitness = model.cross_validate()

    def get_additional_parameters(self):
        return {
            'booster': self.booster,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'nfold': self.nfold,
            'num_boost_round': self.num_boost_round,
            'early_stopping_rounds': self.early_stopping_rounds
        }


class GeneticCnnIndividual(Individual):

    def __init__(self, x_train, y_train, genome=None, genes=None, uniform_rate=0.5, mutation_rate=0.015,
                 nodes=(3, 5), input_shape=(28, 28, 1), kernels_per_layer=(20, 50), kernel_sizes=((5, 5), (5, 5)),
                 dense_units=500, dropout_probability=0.5, classes=10, nfold=5, epochs=3, batch_size=128):
        if genome is None:
            genome = {'S_{}'.format(i + 1): int(K_s * (K_s - 1) / 2) for i, K_s in enumerate(nodes)}
        if genes is None:
            genes = self.generate_random_genes(genome)
        # Set individual's attributes
        super(GeneticCnnIndividual, self).__init__(x_train, y_train, genome, genes, uniform_rate, mutation_rate)
        # Set additional parameters which are not tuned
        assert len(nodes) == len(kernels_per_layer) and len(kernels_per_layer) == len(kernel_sizes)
        self.nodes = nodes
        self.input_shape = input_shape
        self.kernels_per_layer = kernels_per_layer
        self.kernel_sizes = kernel_sizes
        self.dense_units = dense_units
        self.dropout_probability = dropout_probability
        self.classes = classes
        self.nfold = nfold
        self.epochs = epochs
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
            self.x_train, self.y_train, self.genes, self.input_shape, self.kernels_per_layer,
            self.kernel_sizes, self.dense_units, self.dropout_probability, self.classes,
            self.nfold, self.epochs, self.batch_size
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
            'nfold': self.nfold,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }

    def mutate(self):
        """Mutate instance's genes with a certain probability."""
        for name, connections in self.get_genes().items():
            new_connections = ''.join([
                str(int((not int(byte)) != (random.random() < self.mutation_rate))) for byte in connections
            ])
            if new_connections != connections:
                self.fitness = None  # The mutation produces a new individual
            self.get_genes()[name] = new_connections
