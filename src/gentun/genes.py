"""
The genes of a population represent
the hyperparameters we want to
optimize with the algorithm.
"""

import math
import random

from typing import Any, List


class Gene:
    """
    A hyperparameter we want to optimize.
    Define its name and distribution from
    which to sample values when called.
    """

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __call__(self):
        """Return a sample value following the gene specification."""
        raise NotImplementedError


class RandomChoice(Gene):
    """Get random value from a list."""

    def __init__(self, name: str, values: List[Any]):
        super().__init__(name)
        self.values = values

    def __call__(self) -> Any:
        return random.choice(self.values)


class RandomUniform(Gene):
    """
    Sample random uniform number
    between minimum and maximum.
    """

    def __init__(self, name: str, minimum: float, maximum: float):
        super().__init__(name)
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self) -> float:
        return random.uniform(self.minimum, self.maximum)


class RandomLogUniform(Gene):
    """
    Uniform random number in log scale.
    Useful for parameters such as lr.
    """

    def __init__(self, name: str, minimum: float, maximum: float, base: float = 10, eps: float = 1e-12):
        super().__init__(name)
        self.minimum = minimum + eps
        self.maximum = maximum
        self.base = base

    def __call__(self) -> float:
        return self.base ** random.uniform(
            math.log(self.minimum, self.base),
            math.log(self.maximum, self.base)
        )
