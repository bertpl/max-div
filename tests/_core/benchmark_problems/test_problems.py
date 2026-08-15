"""Structural guarantees of the built-in problem family: derived-dimension formulas and constraint feasibility.

The formulas asserted here are the published contract (the docs' problem-overview table); pinning
them keeps code and table from drifting apart.
"""

import math

import numpy as np
import pytest

from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.metrics import DiversityMetric

ODD_N_VALUES = [20, 21, 37, 55, 101, 137, 149, 314]

METRIC = DiversityMetric.MIN_SEPARATION


@pytest.mark.parametrize("n", [20, 137, 1000])
@pytest.mark.parametrize(
    ("name", "d_formula", "k_formula"),
    [
        ("U1", lambda n: 2, lambda n: math.ceil(n / 10)),
        ("U2", lambda n: math.ceil(n / 100), lambda n: math.ceil(n / 10)),
        ("U3", lambda n: math.ceil(n / 100), lambda n: math.ceil(n / 10)),
        ("U4", lambda n: math.ceil(n / 100), lambda n: math.ceil(n / 10)),
        ("C1", lambda n: 2, lambda n: math.ceil(n / 10)),
        ("C2", lambda n: 2, lambda n: math.ceil(n / 10)),
        ("C3", lambda n: math.ceil(n / 150), lambda n: math.ceil(n / 15)),
        ("C4", lambda n: math.ceil(n / 150), lambda n: math.ceil(n / 15)),
    ],
)
def test_dimension_formulas(name, d_formula, k_formula, n):
    # --- act ---------------------------------------------
    d, _, k, m, _ = BenchmarkProblemFactory.get_problem_dimensions(name, n=n)

    # --- assert ------------------------------------------
    assert d == d_formula(n)
    assert k == k_formula(n)
    if name in ("C1", "C2"):
        assert m == math.ceil(k / 5)
    elif name == "C3":
        assert m == 2 * d
    elif name == "C4":
        assert m == 3 * d


@pytest.mark.parametrize("n", ODD_N_VALUES)
def test_u1_component_structure(n):
    """U1's mixture components always sum to exactly n, and the geometry is deterministic."""
    # --- act ---------------------------------------------
    problem = BenchmarkProblemFactory.construct_problem("U1", n=n, diversity_metric=METRIC)
    problem_again = BenchmarkProblemFactory.construct_problem("U1", n=n, diversity_metric=METRIC)

    # --- assert ------------------------------------------
    assert problem.vectors.shape == (n, 2)
    assert np.array_equal(problem.vectors, problem_again.vectors)


@pytest.mark.parametrize("n", ODD_N_VALUES)
def test_c1_exact_quotas_feasible(n):
    """C1's bands partition the population and their exact quotas sum to k with every band large enough."""
    # --- act ---------------------------------------------
    problem = BenchmarkProblemFactory.construct_problem("C1", n=n, diversity_metric=METRIC)

    # --- assert ------------------------------------------
    all_indices: set[int] = set()
    quota_sum = 0
    for constraint in problem.constraints:
        assert constraint.min_count == constraint.max_count  # exact quotas
        assert constraint.min_count >= 1
        assert len(constraint.int_set) >= constraint.min_count  # band can fill its quota
        assert all_indices.isdisjoint(constraint.int_set)  # non-overlapping
        all_indices |= constraint.int_set
        quota_sum += constraint.min_count
    assert all_indices == set(range(n))  # partition covers the population
    assert quota_sum == problem.k


@pytest.mark.parametrize("n", ODD_N_VALUES)
def test_c2_lower_bounds_feasible(n):
    """C2's per-band lower bounds always sum to at most k, each satisfiable within its band."""
    # --- act ---------------------------------------------
    problem = BenchmarkProblemFactory.construct_problem("C2", n=n, diversity_metric=METRIC)

    # --- assert ------------------------------------------
    min_sum = 0
    for constraint in problem.constraints:
        assert constraint.min_count >= 1
        assert constraint.max_count == problem.k
        assert len(constraint.int_set) >= constraint.min_count
        min_sum += constraint.min_count
    assert min_sum <= problem.k


@pytest.mark.parametrize("name", ["C3", "C4"])
@pytest.mark.parametrize("n", ODD_N_VALUES)
def test_c3_c4_bounds_satisfiable_per_constraint(name, n):
    """C3/C4's overlapping lower bounds each fit within their own index set and within k."""
    # --- act ---------------------------------------------
    problem = BenchmarkProblemFactory.construct_problem(name, n=n, diversity_metric=METRIC)

    # --- assert ------------------------------------------
    assert len(problem.constraints) == problem.m
    for constraint in problem.constraints:
        assert constraint.min_count <= constraint.max_count <= problem.k
        assert constraint.min_count <= len(constraint.int_set)
