"""Shared fixtures: small deterministic problems for harness tests."""

import numpy as np
import pytest

from max_div._core.constraints import Constraint
from max_div.metrics import DiversityMetric
from max_div.problem import MaxDivProblem, VectorMaxDivProblem


@pytest.fixture
def small_problem() -> VectorMaxDivProblem:
    """Unconstrained problem: 30 random 3-D vectors, select 5."""
    rng = np.random.default_rng(42)
    return MaxDivProblem.new(vectors=rng.random((30, 3)).astype(np.float32), k=5)


@pytest.fixture
def small_constrained_problem() -> VectorMaxDivProblem:
    """Constrained problem: 30 vectors, two disjoint groups, select 6."""
    rng = np.random.default_rng(43)
    return MaxDivProblem.new(
        vectors=rng.random((30, 3)).astype(np.float32),
        k=6,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        constraints=[
            Constraint(int_set=set(range(15)), min_count=2, max_count=3),
            Constraint(int_set=set(range(15, 30)), min_count=2, max_count=4),
        ],
    )
