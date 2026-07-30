import numpy as np
import pytest
from numpy import random

from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import compute_pdist
from max_div._core.solver._diversity_contribution import MeanDistanceTracker, SeparationTracker
from max_div._core.solver._solver_state import Savepoint, SolverState, _build_con_membership


# =================================================================================================
#  Fixtures
# =================================================================================================
@pytest.fixture
def new_solver_state() -> SolverState:
    vectors = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    return SolverState.new(
        n=vectors.shape[0],
        pdist=compute_pdist(vectors, DistanceMetric.L1_MANHATTAN),
        k=3,
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        diversity_tie_breakers=[DiversityMetric.NON_ZERO_SEPARATION_FRAC],
        constraints=[
            Constraint(int_set={0, 1, 2, 3}, min_count=1, max_count=2),
            Constraint(int_set={2, 3, 4, 5}, min_count=1, max_count=2),
        ],
    )


@pytest.fixture
def new_solver_state_unconstrained() -> SolverState:
    vectors = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    return SolverState.new(
        n=vectors.shape[0],
        pdist=compute_pdist(vectors, DistanceMetric.L1_MANHATTAN),
        k=3,
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        diversity_tie_breakers=[DiversityMetric.NON_ZERO_SEPARATION_FRAC],
        constraints=[],
    )


# =================================================================================================
#  Tests
# =================================================================================================
def test_solver_state_properties(new_solver_state, new_solver_state_unconstrained):
    # with constraints
    assert new_solver_state.has_constraints
    assert new_solver_state.k == 3
    assert new_solver_state.m == 2
    assert new_solver_state.n == 6

    assert new_solver_state.score.constraints < 1.0  # constraints present and not all satisfied --> <1.0

    # without constraints
    assert not new_solver_state_unconstrained.has_constraints
    assert new_solver_state_unconstrained.k == 3
    assert new_solver_state_unconstrained.m == 0
    assert new_solver_state_unconstrained.n == 6

    assert new_solver_state_unconstrained.score.constraints == 1.0  # no constraints -> perfect score


def test_solver_state_add_remove_validation(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        state.remove(3)  # never added

    with pytest.raises(ValueError):
        state.remove_many(np.array([3, 4], dtype=np.int32))  # never added

    state.add(0)
    with pytest.raises(ValueError):
        state.add(0)  # already selected

    with pytest.raises(ValueError):
        state.add_many(np.array([0, 1], dtype=np.int32))  # 0 is already selected

    state.add(1)  # this should be possible, since validation errors invalidate the entire batch

    state.remove(0)
    with pytest.raises(ValueError):
        state.remove(0)  # already not selected


def test_solver_state_end_to_end(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state

    # --- assert 1 ----------------------------------------
    assert state.selected_index_array.size == 0
    assert state.not_selected_index_array.size == 6
    assert state.score.size < 1.0  # insufficient items selected
    assert state.score.constraints < 1.0  # constraints not satisfied
    assert np.array_equal(state.con_values, state._con_values)
    assert np.array_equal(state.con_indices, state._con_indices)
    assert np.array_equal(state.global_contribution_array, state._contribution_tracker.contribution_wrt_dataset)
    assert state.n_selected == 0
    assert state.n_not_selected == 6

    # --- act 1 -------------------------------------------
    state.add(0)
    state.add(2)
    state.add(5)

    # --- assert 2 ----------------------------------------
    assert np.array_equal(state.selected_index_array, [0, 2, 5])
    assert np.array_equal(state.not_selected_index_array, [1, 3, 4])
    assert np.allclose(state.selected_contribution_array, [2, 2, 3])
    assert state.score.size == 1.0  # correct number of items selected
    assert state.score.constraints == 1.0  # all constraints satisfied
    assert state.score.diversity == pytest.approx((2 * 2 * 3) ** (1 / 3))  # geomean of separations 2, 2, 3
    assert state.n_selected == 3
    assert state.n_not_selected == 3

    # --- act 2 -------------------------------------------
    state.remove(5)
    state.add(4)

    # --- assert 3 ----------------------------------------
    assert np.array_equal(state.selected_index_array, [0, 2, 4])
    assert np.array_equal(state.not_selected_index_array, [1, 3, 5])
    assert np.allclose(state.selected_contribution_array, [2, 2, 2])
    assert np.allclose(state.not_selected_contribution_array, [1, 1, 1])
    assert state.score.size == 1.0  # correct number of items selected
    assert state.score.constraints == 1.0  # all constraints satisfied
    assert state.score.diversity == pytest.approx(2.0)  # geomean of separations 2, 2, 2
    assert state.n_selected == 3
    assert state.n_not_selected == 3


def test_savepoint_rolls_back_by_default(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state
    state.add(0)
    state.add(2)

    # current state so we can compare with state after
    orig_selected_array = state.selected_index_array.copy()
    orig_not_selected_array = state.not_selected_index_array.copy()
    orig_separation_array = state.selected_contribution_array.copy()
    orig_con_values = state._con_values.copy()

    # --- act ---------------------------------------------
    # the below should be a no-op: the scope is never kept
    with state.savepoint():
        state.add(5)

    # --- assert ------------------------------------------
    assert np.array_equal(state.selected_index_array, orig_selected_array)
    assert np.array_equal(state.not_selected_index_array, orig_not_selected_array)
    assert np.allclose(state.selected_contribution_array, orig_separation_array)
    assert np.array_equal(state.con_values, orig_con_values)


def test_savepoint_keep(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state
    state.add(0)

    # --- act ---------------------------------------------
    with state.savepoint() as sp:
        state.add(5)
        sp.keep()

    # --- assert ------------------------------------------
    assert np.array_equal(state.selected_index_array, [0, 5])


def test_savepoint_nesting(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state
    state.add(0)

    # --- act ---------------------------------------------
    # kept outer scope, with one rolled-back and one kept scope nested inside it
    with state.savepoint() as outer:
        state.add(2)
        with state.savepoint():
            state.add(4)  # rolled back
        with state.savepoint() as inner:
            state.add(5)
            inner.keep()
        outer.keep()

    # --- assert ------------------------------------------
    assert np.array_equal(state.selected_index_array, [0, 2, 5])


def test_savepoint_rolled_back_outer_discards_kept_inner(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state
    state.add(0)

    # --- act ---------------------------------------------
    with state.savepoint(), state.savepoint() as inner:
        state.add(5)
        inner.keep()

    # --- assert ------------------------------------------
    # keeping a scope only survives up to its parent; the rolled-back outer scope undoes it all
    assert np.array_equal(state.selected_index_array, [0])


def test_savepoint_exception_rolls_back_and_propagates(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state
    state.add(0)

    def trial_that_raises() -> None:
        with state.savepoint() as sp:
            state.add(5)
            sp.keep()  # an exception rolls back even a kept scope
            raise RuntimeError("boom")

    # --- act ---------------------------------------------
    with pytest.raises(RuntimeError, match="boom"):
        trial_that_raises()

    # --- assert ------------------------------------------
    assert np.array_equal(state.selected_index_array, [0])


def test_savepoint_restores_cached_score(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state
    state.add(0)
    score_before = state.score  # cached, clean

    # --- act ---------------------------------------------
    with state.savepoint():
        state.add(5)
        _ = state.score

    # --- assert ------------------------------------------
    # rollback reinstates the score cached at scope entry (by reference; Score is immutable), clean
    assert state._score is score_before
    assert not state._score_dirty


@pytest.mark.parametrize("seed", list(range(1, 100)))
def test_solver_state_consistency_stress_test(new_solver_state, seed: int):
    """Check solver state consistency after a large series of add/remove operations."""

    # --- arrange -----------------------------------------
    state = new_solver_state
    state_ref = new_solver_state.copy()  # we'll leave this untouched until the end
    n_iters = 100

    # --- act ---------------------------------------------
    random.seed(seed)
    for it in range(n_iters):
        # every iteration's changes are provisional; roughly half the scopes are kept
        with state.savepoint() as sp:
            # add random number of items
            n_to_add = random.randint(0, len(state.not_selected_index_array) + 1)
            indices_to_select = state.not_selected_index_array.copy()
            random.shuffle(indices_to_select)
            if it % 2 == 0:
                # use add()
                for idx in indices_to_select[:n_to_add]:
                    state.add(idx)
            else:
                # use add_many()
                state.add_many(indices_to_select[:n_to_add])

            # remove random number of items
            n_to_remove = random.randint(0, len(state.selected_index_array) + 1)
            indices_to_remove = state.selected_index_array.copy()
            random.shuffle(indices_to_remove)
            if it % 2 == 0:
                # use remove()
                for idx in indices_to_remove[:n_to_remove]:
                    state.remove(idx)
            else:
                # use remove_many()
                state.remove_many(indices_to_remove[:n_to_remove])

            # keep the scope with some probability
            if random.rand() >= 0.5:
                sp.keep()

    # --- assert ------------------------------------------

    # double check state_ref was not changed
    assert len(state_ref.selected_index_array) == 0
    assert len(state_ref.not_selected_index_array) == state.n

    # sync state_ref with state
    for idx in state.selected_index_array:
        state_ref.add(idx)

    # check if they're the same
    assert np.array_equal(state.selected_index_array, state_ref.selected_index_array)
    assert np.array_equal(state.not_selected_index_array, state_ref.not_selected_index_array)
    assert np.array_equal(state.global_contribution_array, state_ref.global_contribution_array)
    assert np.array_equal(state.not_selected_contribution_array, state_ref.not_selected_contribution_array)
    assert np.array_equal(state.selected_contribution_array, state_ref.selected_contribution_array)
    assert np.array_equal(state.con_values, state_ref.con_values)
    assert np.array_equal(state.con_indices, state_ref.con_indices)

    assert state.score == state_ref.score

    assert state.n_selected == state_ref.n_selected
    assert state.n_not_selected == state_ref.n_not_selected


def test_solver_state_tracker_set_dormancy(new_solver_state):
    """Separation-only metric configurations construct exactly one tracker: dormancy by absence."""
    # --- act ---------------------------------------------
    trackers = new_solver_state._contribution_trackers._trackers

    # --- assert ------------------------------------------
    assert len(trackers) == 1
    assert type(trackers[0]) is SeparationTracker
    assert new_solver_state._contribution_tracker is trackers[0]


def test_solver_state_tracker_set_mean_distance(new_solver_state):
    """A mean-distance main metric constructs only a MeanDistanceTracker; mixed metrics construct both."""
    # --- arrange -----------------------------------------
    vectors = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    pdist = compute_pdist(vectors, DistanceMetric.L1_MANHATTAN)

    # --- act ---------------------------------------------
    state_pure = SolverState.new(
        n=4,
        pdist=pdist,
        k=2,
        diversity_metric=DiversityMetric.MEAN_PAIRWISE_DISTANCE,
        diversity_tie_breakers=[],
        constraints=[],
    )
    state_mixed = SolverState.new(
        n=4,
        pdist=pdist,
        k=2,
        diversity_metric=DiversityMetric.MEAN_PAIRWISE_DISTANCE,
        diversity_tie_breakers=[DiversityMetric.NON_ZERO_SEPARATION_FRAC],
        constraints=[],
    )

    # --- assert ------------------------------------------
    assert [type(t) for t in state_pure._contribution_trackers._trackers] == [MeanDistanceTracker]
    assert type(state_pure._contribution_tracker) is MeanDistanceTracker
    assert {type(t) for t in state_mixed._contribution_trackers._trackers} == {MeanDistanceTracker, SeparationTracker}
    assert type(state_mixed._contribution_tracker) is MeanDistanceTracker  # main metric's tracker faces the strategies


def test_solver_state_mean_pairwise_distance_score():
    """The diversity score under MEAN_PAIRWISE_DISTANCE equals the brute-force mean over selected pairs."""
    # --- arrange -----------------------------------------
    vectors = np.array([[0.0], [1.0], [3.0], [7.0]], dtype=np.float32)
    state = SolverState.new(
        n=4,
        pdist=compute_pdist(vectors, DistanceMetric.L1_MANHATTAN),
        k=3,
        diversity_metric=DiversityMetric.MEAN_PAIRWISE_DISTANCE,
        diversity_tie_breakers=[],
        constraints=[],
    )

    # --- act ---------------------------------------------
    state.add_many(np.array([0, 1, 3], dtype=np.int32))  # points 0, 1, 7 -> pair distances 1, 7, 6

    # --- assert ------------------------------------------
    assert state.score.diversity == pytest.approx((1 + 7 + 6) / 3)


def test_build_con_membership():
    # --- arrange -----------------------------------------
    cons = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
        Constraint(int_set={10, 11, 12, 13}, min_count=0, max_count=7),
        Constraint(int_set={3, 11}, min_count=2, max_count=2),
    ]
    m = np.int32(14)
    expected_membership = {
        0: [0],
        1: [0],
        2: [0],
        3: [0, 2],
        4: [0],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
        10: [1],
        11: [1, 2],
        12: [1],
        13: [1],
    }
    expected_membership = {k: np.array(v, dtype=np.int32) for k, v in expected_membership.items()}

    # --- act ---------------------------------------------
    con_membership = _build_con_membership(m, cons)

    # --- assert ------------------------------------------
    assert con_membership.keys() == expected_membership.keys()
    for key, value in con_membership.items():
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.int32
        assert np.array_equal(value, expected_membership[key])


# =================================================================================================
#  Consistency invariant
# =================================================================================================
def _make_reference_state() -> SolverState:
    """Build a fresh, empty-selection state on a fixed random problem with overlapping constraints."""
    rng = random.default_rng(seed=123)
    vectors = rng.random((30, 3)).astype(np.float32)
    return SolverState.new(
        n=vectors.shape[0],
        pdist=compute_pdist(vectors, DistanceMetric.L2_EUCLIDEAN),
        k=8,
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        diversity_tie_breakers=[DiversityMetric.NON_ZERO_SEPARATION_FRAC],
        constraints=[
            Constraint(int_set=set(range(12)), min_count=2, max_count=5),
            Constraint(int_set=set(range(8, 22)), min_count=1, max_count=6),
            Constraint(int_set=set(range(18, 30)), min_count=2, max_count=4),
        ],
    )


def _assert_state_matches_fresh_rebuild(state: SolverState) -> None:
    """Assert score & internals of 'state' exactly match a freshly built state with the same selection."""
    fresh = _make_reference_state()
    selection = state.selected_index_array
    if selection.size > 0:
        fresh.add_many(selection)

    assert state.score == fresh.score
    assert state._n_selected == fresh._n_selected
    assert np.array_equal(state._selected, fresh._selected)
    assert np.array_equal(state._contribution_tracker._sep_selected, fresh._contribution_tracker._sep_selected)
    assert np.array_equal(state._con_values, fresh._con_values)


def _apply_savepoint_operation(state: SolverState, operation: str, open_savepoints: list[Savepoint]) -> None:
    """Enter or exit a savepoint on 'state' per 'operation', maintaining the open-savepoint stack.

    Savepoints are entered and exited through their context-manager protocol directly: a random
    walk opens and closes scopes at arbitrary points, which lexical `with` blocks cannot express.
    """
    if operation == "enter_savepoint":
        open_savepoints.append(state.savepoint().__enter__())
    else:
        savepoint = open_savepoints.pop()
        if operation == "exit_keep":
            savepoint.keep()
        savepoint.__exit__(None, None, None)


def _apply_random_operation(state: SolverState, rng: random.Generator, open_savepoints: list[Savepoint]) -> None:
    """Apply one randomly chosen valid operation to 'state', maintaining the open-savepoint stack."""
    selected = state.selected_index_array
    not_selected = state.not_selected_index_array
    operations = ["enter_savepoint"]
    if open_savepoints:
        operations += ["exit_rollback", "exit_keep"]
    if not_selected.size > 0:
        operations += ["add", "add_many"]
    if selected.size > 0:
        operations += ["remove", "remove_many"]

    match rng.choice(operations):
        case ("enter_savepoint" | "exit_rollback" | "exit_keep") as operation:
            _apply_savepoint_operation(state, operation, open_savepoints)
        case "add":
            state.add(rng.choice(not_selected))
        case "add_many":
            n_add = int(rng.integers(1, min(4, not_selected.size) + 1))
            state.add_many(rng.choice(not_selected, size=n_add, replace=False).astype(np.int32))
        case "remove":
            state.remove(rng.choice(selected))
        case "remove_many":
            n_remove = int(rng.integers(1, min(4, selected.size) + 1))
            state.remove_many(rng.choice(selected, size=n_remove, replace=False).astype(np.int32))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_solver_state_consistency_invariant(seed: int):
    """Apply a random operation sequence; after EVERY op, state must equal a fresh rebuild bit-for-bit."""
    # --- arrange -----------------------------------------
    state = _make_reference_state()
    rng = random.default_rng(seed=seed)
    open_savepoints: list[Savepoint] = []

    # --- act & assert ------------------------------------
    for _ in range(60):
        _apply_random_operation(state, rng, open_savepoints)
        _assert_state_matches_fresh_rebuild(state)


def test_solver_state_score_lazy_recompute(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state
    state.add_many(np.array([0, 2, 4], dtype=np.int32))
    expected_score = state.score

    # --- act ---------------------------------------------
    # force the lazy branch: invalidate the cached score and mark it dirty
    state._score = None
    state._score_dirty = True
    observed_score = state.score

    # --- assert ------------------------------------------
    assert observed_score == expected_score  # recomputed on read
    assert not state._score_dirty  # flag cleared by the recompute
