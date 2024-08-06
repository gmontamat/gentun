"""
Classes which define the individuals of a population with
its characteristic genes, generation, crossover and
mutation processes.
"""

import inspect
import pprint
import random

from typing import Any, Dict, Type

from .models import Model
from .genes import Gene


class Individual:
    """
    Member of a population with specific gene
    values (hyperparameters of the model).
    """

    def __init__(
        self,
        model: Type[Model],
        genes: List[Gene],
        x_train: Any,
        y_train: Any,
        hyperparameters: Dict[str, Any],
        **kwargs
    ):
        self.model = model
        self.genes = genes
        self.x_train = x_train
        self.y_train = y_train
        self.hyperparameters = hyperparameters
        self.parameters = kwargs  # model parameters that remain unchanged
        self.validate_params()
        self.fitness = None  # Until evaluated an individual fitness is unknown

    @staticmethod
    def get_init_params(_class: Type[Model]) -> Dict[str, Dict[str, Any]]:
        init_signature = inspect.signature(_class.__init__)
        params_info = {}
        for param_name, param in init_signature.parameters.items():
            if param_name == 'self':
                continue
            param_info = {
                'type': param.annotation if param.annotation != inspect.Parameter.empty else None,
                'default': param.default if param.default != inspect.Parameter.empty else None
            }
            params_info[param_name] = param_info
        return params_info

    def validate_params(self):
        """Check all parameters against model."""
        for param_name, param_info in self.get_init_params(self.model):
            if ((param_name not in self.hyperparameters.keys() or param_name not in self.parameters.keys()) and
                    param_info["default"] is not None):
                raise ValueError(f"Missing model parameter: {param_name}")
            elif param_name in self.hyperparameters.keys():
                if not isinstance(self.hyperparameters[param_name], param_info["type"]):
                    raise TypeError(
                        f"Type missmatch with hyperparameter `{param_name}`. "
                        f"Expected `{param_info['type']}`, got `{type(self.hyperparameters[param_name])}`."
                    )
            else:
                if not isinstance(self.parameters[param_name], param_info["type"]):
                    raise TypeError(
                        f"Type missmatch with parameter `{param_name}`. "
                        f"Expected `{param_info['type']}`, got `{type(self.parameters[param_name])}`."
                    )

    def evaluate_fitness(self) -> float:
        """Create instance of model and evaluate."""
        if self.fitness is not None:
            return self.fitness
        self.fitness = self.model(
            **{**self.hyperparameters, **self.parameters}
        ).evaluate(self.x_train, self.y_train)
        return self.fitness

    def __getitem__(self, key: str) -> Any:
        """Select a hyperparameter."""
        return self.hyperparameters[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Change a hyperparameter. Reset fitness."""
        self.hyperparameters[key] = value
        self.fitness = None

    def reproduce(self, partner: Individual, rate: float) -> Individual:
        """
        Mix genes from self and partner at random
        and return a new instance of an individual.
        Does not mutate parents.
        """
        child = {}
        for param, value in self.hyperparameters.items():
            if random.random() < rate:
                child[param] = partner[param]
            else:
                child[param] = value
        return Individual(
            self.model,
            self.genes,
            self.x_train,
            self.y_train,
            hyperparameters=child,
            **self.parameters
        )

    def crossover(self, partner: Individual, rate: float) -> None:
        """
        Swap genes from self and partner at random.
        Mutates each parent.
        """
        for param, value in self.hyperparameters.items():
            if random.random() < rate:
                partner_value = partner[param]
                partner[param] = value
                self[param] = partner_value

    def mutate(self, rate: float) -> None:
        """Mutate individual."""
        for gene in self.genes:
            if random.random() < rate:
                self[str(gene)] = gene()

    def __copy__(self):
        """Copy instance."""
        return Individual(
            self.model,
            self.x_train,
            self.y_train,
            self.hyperparameters.copy(),
            **self.parameters
        )

    def __str__(self):
        """Return hyperparameters which identify the individual."""
        return pprint.pformat(self.hyperparameters)


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
