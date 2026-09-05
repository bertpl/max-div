import urllib.error

import numpy as np
import pytest

from benchmarks.adapters import (
    ApricotFacilityLocation,
    CodeFdmFairFlow,
    FpsampleFPS,
    KMedoidsFasterPAM,
    QcSelectorMaxMin,
    QcSelectorMaxSum,
    RandomBaseline,
    RdkitMaxMin,
    SkmatterFPS,
)
from benchmarks.adapters.code_fdm import _constraints_to_colors
from max_div._core.constraints import Constraint
from max_div.problem import MaxDivProblem

UNCONSTRAINED_ADAPTERS = [
    RandomBaseline(),
    FpsampleFPS(),
    SkmatterFPS(),
    RdkitMaxMin(),
    ApricotFacilityLocation(),
    KMedoidsFasterPAM(),
    QcSelectorMaxMin(),
    QcSelectorMaxSum(),
]


@pytest.mark.parametrize("adapter", UNCONSTRAINED_ADAPTERS, ids=lambda a: a.name)
def test_adapter_returns_valid_selection(small_problem, adapter):
    # --- act ---------------------------------------------
    indices, measured_sec = adapter.timed_select(small_problem, seed=0)

    # --- assert ------------------------------------------
    assert len(indices) == small_problem.k
    assert len(np.unique(indices)) == small_problem.k
    assert indices.min() >= 0
    assert indices.max() < small_problem.n
    assert measured_sec > 0.0


@pytest.mark.parametrize("adapter", UNCONSTRAINED_ADAPTERS, ids=lambda a: a.name)
def test_adapter_is_deterministic_given_seed(small_problem, adapter):
    # --- act ---------------------------------------------
    first = adapter.select(small_problem, seed=3)
    second = adapter.select(small_problem, seed=3)

    # --- assert ------------------------------------------
    assert np.array_equal(first, second)


def test_code_fdm_adapter_satisfies_constraints(small_constrained_problem):
    # fetched research code: robustness over blocking — network trouble skips, never fails
    # --- act ---------------------------------------------
    try:
        indices, _ = CodeFdmFairFlow().timed_select(small_constrained_problem, seed=0)
    except (urllib.error.URLError, OSError) as e:  # pragma: no cover -- network-dependent
        pytest.skip(f"code-FDM fetch failed: {e}")

    # --- assert ------------------------------------------
    from benchmarks.common import n_constraints_satisfied

    assert len(np.unique(indices)) == small_constrained_problem.k
    assert n_constraints_satisfied(small_constrained_problem, indices) == small_constrained_problem.m


def test_code_fdm_rejects_overlapping_constraints():
    # pure mapping logic, no fetch involved
    # --- arrange -----------------------------------------
    rng = np.random.default_rng(0)
    problem = MaxDivProblem.new(
        vectors=rng.random((20, 2)).astype(np.float32),
        k=4,
        constraints=[
            Constraint(int_set={0, 1, 2, 3}, min_count=1, max_count=2),
            Constraint(int_set={3, 4, 5, 6}, min_count=1, max_count=2),  # overlaps on 3
        ],
    )

    # --- act / assert ------------------------------------
    with pytest.raises(ValueError, match="overlap"):
        _constraints_to_colors(problem)
