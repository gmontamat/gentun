"""
Define an individual with its genes and genetic operations.
This module provides the Individual class, which represents a member
of a population in genetic algorithms. It includes methods for
duplication, reproduction, crossover, and mutation processes.
"""
from __future__ import annotations

import inspect
import logging
import random
from typing import Any, Dict, Optional, Sequence, Type, Union

from .genes import Gene
from .models.base import Handler
from .services import RedisController


class Individual:
    """
    Member of a population with specific gene values, which are the
    hyperparameters of the model.
    """

    _params_validated = False

    def __init__(
        self,
        genes: Sequence[Gene],
        handler: Type[Handler],
        x_train: Any,
        y_train: Any,
        x_test: Any,
        y_test: Any,
        hyperparameters: Dict[str, Any],
        **kwargs: Any,
    ):
        self.genes = genes
        self.handler = handler
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.hyperparameters = hyperparameters
        self.kwargs = kwargs  # model parameters that remain unchanged
        if not Individual._params_validated:
            self.validate_params()
            Individual._params_validated = True
        self.fitness = None  # Until evaluated an individual fitness is unknown
        self.job_id = None

    @staticmethod
    def get_init_params(_class: Type[Handler]) -> Dict[str, Dict[str, Any]]:
        """Get parameters defined in the Handler class used."""
        init_signature = inspect.signature(_class.__init__)
        params_info = {}
        for param_name, param in init_signature.parameters.items():
            if param_name == "self":
                continue
            param_info = {
                "type": param.annotation if param.annotation != inspect.Parameter.empty else None,
                "default": param.default if param.default != inspect.Parameter.empty else None,
                "empty_default": param.default == inspect.Parameter.empty,
            }
            params_info[param_name] = param_info
        return params_info

    def validate_params(self) -> None:
        """Check all parameters against model handler."""
        for param_name, param_info in self.get_init_params(self.handler).items():
            if param_name == "kwargs":
                continue
            # Convert typing hint types into their original type (e.g. typing.List -> list)
            try:
                param_type = param_info["type"].__origin__
            except AttributeError:
                param_type = param_info["type"]
            if param_name in self.hyperparameters:
                if param_type is Union:
                    logging.warning("Not checking type for '%s' with type `Union`.", param_name)
                elif not isinstance(self.hyperparameters[param_name], param_type):
                    raise TypeError(
                        f"Type missmatch with hyperparameter '{param_name}'. "
                        f"Expected `{param_type}`, got `{type(self.hyperparameters[param_name])}`."
                    )
            elif param_name in self.kwargs:
                if param_type is Union:
                    logging.warning("Not checking type for '%s' with type `Union`.", param_name)
                elif not isinstance(self.kwargs[param_name], param_type):
                    raise TypeError(
                        f"Type missmatch with parameter '{param_name}'. "
                        f"Expected `{param_type}`, got `{type(self.kwargs[param_name])}`."
                    )
            elif param_info["empty_default"]:
                raise ValueError(f"Missing `{self.handler.__name__}` parameter: '{param_name}'.")
            else:
                logging.debug("Using `%s`'s default value for '%s'", self.handler.__name__, param_name)

    def evaluate_fitness(self) -> float:
        """Instantiate model and evaluate."""
        if self.fitness is not None:
            return self.fitness
        self.fitness = self.handler(**{**self.hyperparameters, **self.kwargs})(
            self.x_train, self.y_train, self.x_test, self.y_test
        )
        logging.info("Individual evaluated: %s; fitness: %s", self, self.fitness)
        return self.fitness

    def get_fitness(self) -> Optional[float]:
        """Return the current value of individual's fitness."""
        return self.fitness

    def send_to_queue(self, server: RedisController) -> None:
        """Send individual to be evaluated."""
        self.job_id = server.send_job(self.handler, **{**self.hyperparameters, **self.kwargs})

    def read_from_queue(self, server: RedisController) -> None:
        """Read fitness results from queue."""
        self.fitness = server.wait_for_result(self.job_id)
        logging.info("Individual received: %s; fitness: %s", self, self.fitness)

    def __getitem__(self, key: str) -> Any:
        """Select a hyperparameter."""
        return self.hyperparameters[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Change a hyperparameter. Reset fitness if needed."""
        if value != self.hyperparameters[key]:
            self.fitness = None
        self.hyperparameters[key] = value

    def reproduce(self, partner: Individual, rate: float = 0.5) -> Individual:
        """
        Mix genes from self and partner at random and return a new
        instance of an individual. Does not mutate parents.
        """
        child = {}
        for param, value in self.hyperparameters.items():
            if random.random() < rate:
                child[param] = partner[param]
            else:
                child[param] = value
        return Individual(
            self.genes,
            self.handler,
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
            hyperparameters=child,
            **self.kwargs,
        )

    def crossover(self, partner: Individual, rate: float = 0.5) -> None:
        """
        Swap genes from self and partner at random.
        Mutates each parent.
        """
        for param, value in self.hyperparameters.items():
            if random.random() < rate:
                partner_value = partner[param]
                partner[param] = value
                self[param] = partner_value

    def mutate(self, rate: float = 1.0) -> None:
        """Mutate individual."""
        for gene in self.genes:
            self[str(gene)] = gene.mutate(self[str(gene)], rate)

    def duplicate(self) -> Individual:
        """
        Create a copy of the individual. Required when algorithms sample
        with replacement.
        """
        individual = Individual(
            self.genes,
            self.handler,
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
            hyperparameters=self.hyperparameters.copy(),
            **self.kwargs,
        )
        individual.fitness = self.fitness
        return individual

    def __str__(self) -> str:
        """Return hyperparameters which identify the individual."""
        combined_params = {**self.hyperparameters, **self.kwargs}
        return str(combined_params)
