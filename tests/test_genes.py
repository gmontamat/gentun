import math

import pytest

from gentun.genes import Binary, Gene, RandomChoice, RandomLogUniform, RandomUniform


def test_gene_init():
    gene = Gene("test_gene")
    assert gene.name == "test_gene"


def test_gene_str():
    gene = Gene("test_gene")
    assert str(gene) == "test_gene"


def test_gene_call():
    gene = Gene("test_gene")
    with pytest.raises(NotImplementedError):
        gene()


def test_gene_sample():
    gene = Gene("test_gene")
    with pytest.raises(NotImplementedError):
        gene.sample(0.5)


def test_random_choice_init():
    gene = RandomChoice("test_choice", [1, 2, 3])
    assert gene.name == "test_choice"
    assert gene.values == [1, 2, 3]


def test_random_choice_call():
    gene = RandomChoice("test_choice", [1, 2, 3])
    assert gene() in [1, 2, 3]


def test_random_choice_sample():
    gene = RandomChoice("test_choice", [1, 2, 3])
    assert gene.sample(0.0) == 1
    assert gene.sample(0.5) in [2]
    assert gene.sample(1.0) in [3]


def test_random_uniform_init():
    gene = RandomUniform("test_uniform", 0.0, 1.0)
    assert gene.name == "test_uniform"
    assert gene.minimum == 0.0
    assert gene.maximum == 1.0


def test_random_uniform_call():
    gene = RandomUniform("test_uniform", 0.0, 1.0)
    value = gene()
    assert 0.0 <= value <= 1.0


def test_random_uniform_sample():
    gene = RandomUniform("test_uniform", 0.0, 1.0)
    assert gene.sample(0.0) == 0.0
    assert gene.sample(0.5) == 0.5
    assert gene.sample(1.0) == 1.0


def test_random_log_uniform_init():
    gene = RandomLogUniform("test_log_uniform", 1.0, 10.0)
    assert gene.name == "test_log_uniform"
    assert math.isclose(gene.minimum, 1.0 + gene.eps, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(gene.maximum, 10.0, rel_tol=1e-9, abs_tol=1e-12)


def test_random_log_uniform_call():
    gene = RandomLogUniform("test_log_uniform", 1.0, 10.0)
    value = gene()
    assert 1.0 <= value <= 10.0
    gene = RandomLogUniform("test_log_uniform", 1.0, 10.0, reverse=True)
    value = gene()
    assert 1.0 <= value <= 10.0


def test_random_log_uniform_sample():
    gene = RandomLogUniform("test_log_uniform", 1.0, 10.0)
    assert math.isclose(gene.sample(0.0), 1.0 + gene.eps, rel_tol=1e-9, abs_tol=1e-12)
    assert 1.0 <= gene.sample(0.5) <= 10.0
    assert math.isclose(gene.sample(1.0), 10.0, rel_tol=1e-9, abs_tol=1e-12)
    gene = RandomLogUniform("test_log_uniform", 1.0, 10.0, reverse=True)
    assert 1.0 <= gene.sample(0.5) <= 10.0


def test_binary_init():
    gene = Binary("test_binary", 4)
    assert gene.name == "test_binary"
    assert gene.length == 4


def test_binary_call():
    gene = Binary("test_binary", 4)
    value = gene()
    assert len(value) == 4
    assert all(bit in "01" for bit in value)


def test_binary_mutate():
    gene = Binary("test_binary", 4)
    value = "1010"
    mutated_value = gene.mutate(value, 0.5)
    assert len(mutated_value) == 4
    assert all(bit in "01" for bit in mutated_value)
