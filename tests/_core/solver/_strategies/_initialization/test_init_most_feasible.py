import numpy as np
import pytest

from max_div._core.constraints import Constraint
from max_div._core.constraints.feasibility import FeasibilityStatus
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import DistanceStore, compute_pdist
from max_div._core.solver._solver_state import SolverState
from max_div._core.solver._strategies import InitializationStrategy
from max_div._core.solver._strategies._initialization._init_most_feasible import InitMostFeasible


# =================================================================================================
#  Helpers
# =================================================================================================
def _state(constraints: list[Constraint], n: int = 20, k: int = 8) -> SolverState:
    """Build a solver state over n random 3-d vectors with the given constraints."""
    vectors = np.random.default_rng(42).random((n, 3)).astype(np.float32)
    return SolverState.new(
        n=n,
        store=DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.L2_EUCLIDEAN), n=n),
        k=k,
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        diversity_tie_breakers=[],
        constraints=constraints,
    )


def _satisfies(selection: np.ndarray, constraints: list[Constraint]) -> bool:
    """Check a selection against every constraint's [min_count, max_count]."""
    chosen = {int(j) for j in selection}
    return all(con.min_count <= len(con.int_set & chosen) <= con.max_count for con in constraints)


# 3 + 5 == k, so the two quotas are jointly satisfiable
FEASIBLE_CONS = [
    Constraint(int_set=set(range(10)), min_count=3, max_count=3),
    Constraint(int_set=set(range(10, 20)), min_count=5, max_count=5),
]

# every item is a member, so at most 5 can be selected while k is 8 -- floor 3
INFEASIBLE_CONS = [Constraint(int_set=set(range(20)), min_count=0, max_count=5)]

# two disjoint 5-cycles with a min-1 cover per edge: LP-feasible at 1/2 per item (no certificate
# exists), while covering each odd cycle integrally needs 3 items -- 6 > k, so no witness either
UNKNOWN_CONS = [
    Constraint(int_set={cycle * 5 + i, cycle * 5 + (i + 1) % 5}, min_count=1, max_count=2)
    for cycle in range(2)
    for i in range(5)
]


# =================================================================================================
#  Verdict dispatch
# =================================================================================================
def test_most_feasible_starts_from_a_witness():
    """A satisfiable constrained problem is initialized with a selection meeting every constraint."""
    # --- arrange -----------------------------------------
    state = _state(FEASIBLE_CONS)
    strategy = InitializationStrategy.most_feasible()

    # --- act ---------------------------------------------
    selection = strategy.get_next_samples(state, state.k)

    # --- assert ------------------------------------------
    assert len(selection) == state.k
    assert len(set(selection.tolist())) == state.k
    assert _satisfies(selection, FEASIBLE_CONS)
    assert strategy.get_debug_info() == FeasibilityStatus.FEASIBLE.name


def test_most_feasible_starts_from_the_least_infeasible_selection_when_infeasibility_is_proven():
    """A certified-infeasible problem still yields a full selection, graded against the floor."""
    # --- arrange -----------------------------------------
    state = _state(INFEASIBLE_CONS)
    strategy = InitializationStrategy.most_feasible()

    # --- act ---------------------------------------------
    selection = strategy.get_next_samples(state, state.k)

    # --- assert ------------------------------------------
    assert len(selection) == state.k
    assert len(set(selection.tolist())) == state.k
    assert strategy.get_debug_info() == FeasibilityStatus.INFEASIBLE.name


def test_most_feasible_hands_over_to_the_fallback_on_unknown():
    """An inconclusive verdict must leave the selection exactly as the fallback alone would make it."""
    # --- arrange -----------------------------------------
    strategy = InitializationStrategy.most_feasible(max_iter=300, fallback=InitializationStrategy.random_one_shot())
    reference = InitializationStrategy.random_one_shot()
    strategy.set_seed(7)
    reference.set_seed(7)

    # --- act ---------------------------------------------
    selection = strategy.get_next_samples(_state(UNKNOWN_CONS, n=10, k=5), 5)
    expected = reference.get_next_samples(_state(UNKNOWN_CONS, n=10, k=5), 5)

    # --- assert ------------------------------------------
    assert strategy.get_debug_info() == FeasibilityStatus.UNKNOWN.name
    assert np.array_equal(selection, expected)


def test_most_feasible_bypasses_the_pipeline_without_constraints():
    """With nothing to satisfy, the fallback initializes and the pipeline never runs."""
    # --- arrange -----------------------------------------
    strategy = InitializationStrategy.most_feasible(fallback=InitializationStrategy.random_one_shot())
    reference = InitializationStrategy.random_one_shot()
    strategy.set_seed(11)
    reference.set_seed(11)

    # --- act ---------------------------------------------
    selection = strategy.get_next_samples(_state([]), 8)
    expected = reference.get_next_samples(_state([]), 8)

    # --- assert ------------------------------------------
    assert np.array_equal(selection, expected)
    assert strategy.get_debug_info() == "/"  # the pipeline never ran, so there is no verdict


# =================================================================================================
#  Diversity tilt
# =================================================================================================
def test_most_feasible_diversity_tilt_moves_the_witness_without_losing_feasibility():
    """A nonzero beta selects a different witness, and it is still a witness."""
    # --- arrange -----------------------------------------
    plain = InitializationStrategy.most_feasible()
    tilted = InitializationStrategy.most_feasible(beta=5.0)

    # --- act ---------------------------------------------
    plain_selection = plain.get_next_samples(_state(FEASIBLE_CONS), 8)
    tilted_selection = tilted.get_next_samples(_state(FEASIBLE_CONS), 8)

    # --- assert ------------------------------------------
    assert _satisfies(plain_selection, FEASIBLE_CONS)
    assert _satisfies(tilted_selection, FEASIBLE_CONS)
    assert not np.array_equal(plain_selection, tilted_selection)


# =================================================================================================
#  Configuration & contract
# =================================================================================================
@pytest.mark.parametrize(
    "kwargs,match",
    [({"max_iter": 0}, "max_iter"), ({"beta": -0.1}, "beta")],
    ids=["max_iter_below_one", "negative_beta"],
)
def test_most_feasible_rejects_invalid_arguments(kwargs: dict, match: str):
    # --- act & assert ------------------------------------
    with pytest.raises(ValueError, match=match):
        InitMostFeasible(**kwargs)


def test_most_feasible_seeds_the_fallback_it_delegates_to():
    """One seed drives both paths, so a fall-through stays reproducible."""
    # --- arrange -----------------------------------------
    fallback = InitializationStrategy.random_one_shot()
    strategy = InitMostFeasible(fallback=fallback)

    # --- act ---------------------------------------------
    strategy.set_seed(123)

    # --- assert ------------------------------------------
    assert fallback.seed == strategy.seed


def test_most_feasible_needs_an_empty_state():
    """The pipeline reads the constraint bounds as problem-level values, which holds only while empty."""
    # --- arrange -----------------------------------------
    state = _state(FEASIBLE_CONS)
    state.add_many(np.array([0], dtype=np.int32))
    strategy = InitializationStrategy.most_feasible()

    # --- act & assert ------------------------------------
    with pytest.raises(RuntimeError, match="empty state"):
        strategy.get_next_samples(state, state.k - 1)
