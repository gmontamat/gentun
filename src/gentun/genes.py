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
        """Return a random sample value following the gene specification."""
        raise NotImplementedError

    def sample(self, percentile: float):
        """
        Return a sample value given the percentile.
        This method enables grid search.
        """
        raise NotImplementedError

    def mutate(self, value: Any, rate: float):
        """
        Mutate a gene. The default behavior is
        to re-sample with probability 'rate'.
        Subclasses may want to refine this method.
        """
        if random.random() < rate:
            return self()
        return value


class RandomChoice(Gene):
    """
    Get random value from a list.
    """

    def __init__(self, name: str, values: List[Any]):
        super().__init__(name)
        self.values = values

    def __call__(self) -> Any:
        return random.choice(self.values)

    def sample(self, percentile: float) -> Any:
        try:
            return self.values[int(len(self.values) * percentile)]
        except IndexError:
            return self.values[-1]


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

    def sample(self, percentile: float) -> float:
        return self.minimum + percentile * (self.maximum - self.minimum)


class RandomLogUniform(Gene):
    """
    Uniform random number in log scale.
    Useful for parameters such as lr.
    """

    def __init__(
        self, name: str, minimum: float, maximum: float, base: float = 10, reverse: bool = False, eps: float = 1e-12
    ):
        super().__init__(name)
        self.minimum = minimum + eps
        self.maximum = maximum
        self.eps = eps
        self.base = base
        self.reverse = reverse

    def __call__(self) -> float:
        if self.reverse:
            return self.maximum - math.pow(
                self.base,
                random.uniform(math.log(self.eps, self.base), math.log(self.maximum - self.minimum, self.base)),
            )
        return math.pow(self.base, random.uniform(math.log(self.minimum, self.base), math.log(self.maximum, self.base)))

    def sample(self, percentile: float) -> float:
        if self.reverse:
            return self.maximum - math.pow(
                self.base,
                math.log(self.eps, self.base)
                + percentile * (math.log(self.maximum - self.minimum, self.base) - math.log(self.eps, self.base)),
            )
        return math.pow(
            self.base,
            math.log(self.minimum, self.base)
            + percentile * (math.log(self.maximum, self.base) - math.log(self.minimum, self.base)),
        )


class Binary(Gene):
    """
    Gene used in Genetic CNN paper
    http://arxiv.org/pdf/1703.01513
    """

    def __init__(self, name: str, length: int):
        super().__init__(name)
        self.length = length

    def __call__(self) -> str:
        return "".join(["0" if random.random() < 0.5 else "1" for _ in range(self.length)])

    def mutate(self, value: str, rate: float) -> str:
        """Toggle each bit with probability 'rate'."""
        return "".join([str(int(int(bit) != (random.random() < rate))) for bit in value])
