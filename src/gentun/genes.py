"""
Define the genes of an individual which
represent the hyperparameters we want to
optimize through the genetic algorithm.
"""

import math
import random

from typing import Any, List


class Gene:
    """
    Hyperparameter we want to optimize.

    """

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __call__(self):
        raise NotImplementedError


class RandomChoiceGene(Gene):
    """Get random value from a list."""

    def __init__(self, name: str, values: List[float]):
        super().__init__(name)
        self.values = values

    def __call__(self) -> float:
        return random.choice(self.values)


class RandomUniformGene(Gene):

    def __init__(self, name: str, minimum: float, maximum: float):
        super().__init__(name)
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self) -> float:
        return random.uniform(self.minimum, self.maximum)


class RandomLogUniformGene(Gene):

    def __init__(self, name: str, minimum: float, maximum: float, base: float, eps: float = 1e-12):
        super().__init__(name)
        self.minimum = minimum + eps
        self.maximum = maximum
        self.base = base

    def __call__(self) -> float:
        return self.base ** random.uniform(
            math.log(self.minimum, self.base),
            math.log(self.maximum, self.base)
        )
