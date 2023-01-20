#!/usr/bin/env python
"""
Classes which define the individuals of a population with
its characteristic genes, generation, crossover and
mutation processes.
"""
import random

try:
    from .individual import Individual
    from .individual import random_log_uniform
except ImportError:
    pass

try:
    from ..models.xgboost_models import XgboostModel
except ImportError:
    pass

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
