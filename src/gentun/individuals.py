"""
Define an individual with its genes,
duplication, reproduction, crossover
and mutation processes.
"""
from __future__ import annotations

import inspect
import pprint
import random
from typing import Any, Dict, List, Type, Union

from .genes import Gene
from .wrappers.base import ModelWrapper


class Individual:
    """
    Member of a population with specific gene
    values (hyperparameters of the model).
    """

    def __init__(
        self,
        genes: List[Gene],
        model_wrapper: Type[ModelWrapper],
        x_train: Any,
        y_train: Any,
        hyperparameters: Dict[str, Any],
        **kwargs: Any,
    ):
        self.genes = genes
        self.model_wrapper = model_wrapper
        self.x_train = x_train
        self.y_train = y_train
        self.hyperparameters = hyperparameters
        self.kwargs = kwargs  # model parameters that remain unchanged
        self.validate_params()
        self.fitness = None  # Until evaluated an individual fitness is unknown

    @staticmethod
    def get_init_params(_class: Type[ModelWrapper]) -> Dict[str, Dict[str, Any]]:
        """Get parameters defined in the ModelWrapper class used."""
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
        """Check all parameters against wrapper."""
        for param_name, param_info in self.get_init_params(self.model_wrapper).items():
            if param_name == "kwargs":
                continue
            # Convert typing hint types into their original type (e.g. typing.List -> list)
            try:
                param_type = param_info["type"].__origin__
            except AttributeError:
                param_type = param_info["type"]
            if param_name in self.hyperparameters:
                if param_type is Union:
                    # print(f"Warning: cannot check type for `{param_name}` with type `Union`.")
                    pass
                elif not isinstance(self.hyperparameters[param_name], param_type):
                    raise TypeError(
                        f"Type missmatch with hyperparameter `{param_name}`. "
                        f"Expected `{param_type}`, got `{type(self.hyperparameters[param_name])}`."
                    )
            elif param_name in self.kwargs:
                if param_type is Union:
                    # print(f"Warning: cannot check type for `{param_name}` with type `Union`.")
                    pass
                elif not isinstance(self.kwargs[param_name], param_type):
                    raise TypeError(
                        f"Type missmatch with parameter `{param_name}`. "
                        f"Expected `{param_type}`, got `{type(self.kwargs[param_name])}`."
                    )
            elif param_info["empty_default"]:
                raise ValueError(f"Missing `{self.model_wrapper}` parameter: `{param_name}`.")
            else:
                # print(f"Warning: using `{self.model_wrapper}`'s default value for `{param_name}`.")
                pass

    def evaluate_fitness(self) -> float:
        """Create instance of model and evaluate."""
        if self.fitness is not None:
            return self.fitness
        self.fitness = self.model_wrapper(**{**self.hyperparameters, **self.kwargs}).evaluate(
            self.x_train, self.y_train
        )
        return self.fitness

    def __getitem__(self, key: str) -> Any:
        """Select a hyperparameter."""
        return self.hyperparameters[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Change a hyperparameter. Reset fitness if needed."""
        if value != self.hyperparameters[key]:
            self.fitness = None
        self.hyperparameters[key] = value

    def reproduce(self, partner: Individual, rate: float = 1.0) -> Individual:
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
            self.genes, self.model_wrapper, self.x_train, self.y_train, hyperparameters=child, **self.kwargs
        )

    def crossover(self, partner: Individual, rate: float = 1.0) -> None:
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
        Create a copy of the individual.
        Useful when algorithms sample
        with replacement.
        """
        return Individual(
            self.genes,
            self.model_wrapper,
            self.x_train,
            self.y_train,
            hyperparameters=self.hyperparameters.copy(),
            **self.kwargs,
        )

    def __str__(self):
        """Return hyperparameters which identify the individual."""
        return pprint.pformat(self.hyperparameters)
