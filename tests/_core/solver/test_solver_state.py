import numpy as np
import pytest
from numpy import random

from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import DistanceStore, compute_pdist
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
        store=DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.L1_MANHATTAN), n=vectors.shape[0]),
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
        store=DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.L1_MANHATTAN), n=vectors.shape[0]),
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
    assert new_solver_state_unconstrained.con_weights.shape == (0,)


def test_solver_state_con_weights_reach_the_state():
    """Constraint weights arrive as float64, in constraint order, and survive a copy."""
    # --- arrange -----------------------------------------
    vectors = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)

    # --- act ---------------------------------------------
    state = SolverState.new(
        n=vectors.shape[0],
        store=DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.L1_MANHATTAN), n=vectors.shape[0]),
        k=3,
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        diversity_tie_breakers=[],
        constraints=[
            Constraint(int_set={0, 1, 2, 3}, min_count=1, max_count=2, weight=2.5),
            Constraint(int_set={2, 3, 4, 5}, min_count=1, max_count=2),
        ],
    )

    # --- assert ------------------------------------------
    assert state.con_weights.dtype == np.float64
    assert np.array_equal(state.con_weights, [2.5, 1.0])
    assert np.array_equal(state.copy().con_weights, state.con_weights)


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


def test_savepoint_restores_by_default(new_solver_state):
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
    # kept outer scope, with one restored and one kept scope nested inside it
    with state.savepoint() as outer:
        state.add(2)
        with state.savepoint():
            state.add(4)  # dropped: this scope is not kept, so its exit restores {0, 2}
        with state.savepoint() as inner:
            state.add(5)
            inner.keep()
        outer.keep()

    # --- assert ------------------------------------------
    assert np.array_equal(state.selected_index_array, [0, 2, 5])


def test_savepoint_restoring_outer_discards_kept_inner(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state
    state.add(0)

    # --- act ---------------------------------------------
    with state.savepoint(), state.savepoint() as inner:
        state.add(5)
        inner.keep()

    # --- assert ------------------------------------------
    # keeping a scope only survives up to its parent; the outer exit restores the state from before it
    assert np.array_equal(state.selected_index_array, [0])


def test_savepoint_exception_restores_and_propagates(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state
    state.add(0)

    def trial_that_raises() -> None:
        with state.savepoint() as sp:
            state.add(5)
            sp.keep()  # an exception restores even a kept scope
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
    # the exit restores the score cached at scope entry (by reference; Score is immutable), clean
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
        store=DistanceStore.condensed(pdist, n=4),
        k=2,
        diversity_metric=DiversityMetric.MEAN_PAIRWISE_DISTANCE,
        diversity_tie_breakers=[],
        constraints=[],
    )
    state_mixed = SolverState.new(
        n=4,
        store=DistanceStore.condensed(pdist, n=4),
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
        store=DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.L1_MANHATTAN), n=4),
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
        store=DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.L2_EUCLIDEAN), n=vectors.shape[0]),
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
        operations += ["exit_restore", "exit_keep"]
    if not_selected.size > 0:
        operations += ["add", "add_many"]
    if selected.size > 0:
        operations += ["remove", "remove_many"]

    match rng.choice(operations):
        case ("enter_savepoint" | "exit_restore" | "exit_keep") as operation:
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


# =================================================================================================
#  Selected-index list
# =================================================================================================
def _assert_index_list_matches_mask(state: SolverState) -> None:
    """The maintained index list must always equal what deriving it from the mask would give."""
    expected = np.flatnonzero(state._selected).astype(np.int32)
    actual = state.selected_index_array
    assert actual.tolist() == expected.tolist()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_selected_index_list_survives_random_mutation_sequences(seed: int):
    """Long random sequences of every mutation, with nested savepoints kept and rolled back.

    The failure mode this guards is silent: a list that drifts from the mask yields wrong
    selections rather than an error, so the invariant is asserted after every single step
    rather than at the end.
    """
    # --- arrange ----------------------
    n = 40
    vectors = np.arange(n, dtype=np.float32).reshape(n, 1)
    rng = random.default_rng(seed)

    def fresh() -> SolverState:
        return SolverState.new(
            n=n,
            store=DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.L1_MANHATTAN), n=n),
            k=8,
            diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
            diversity_tie_breakers=[],
            constraints=[],
        )

    state = fresh()

    # --- act / assert -----------------
    for _ in range(300):
        selected = np.flatnonzero(state._selected).astype(np.int32)
        free = np.flatnonzero(~state._selected).astype(np.int32)
        action = rng.integers(0, 6)
        if action == 0 and free.size:
            state.add(np.int32(rng.choice(free)))
        elif action == 1 and selected.size:
            state.remove(np.int32(rng.choice(selected)))
        elif action == 2 and free.size >= 3:
            state.add_many(np.ascontiguousarray(rng.choice(free, size=3, replace=False), dtype=np.int32))
        elif action == 3 and selected.size >= 3:
            state.remove_many(np.ascontiguousarray(rng.choice(selected, size=3, replace=False), dtype=np.int32))
        elif action == 4 and free.size:
            # a rolled-back trial: the list must come back with the mask
            with state.savepoint():
                state.add(np.int32(rng.choice(free)))
                _assert_index_list_matches_mask(state)
        elif action == 5 and selected.size:
            # a nested trial that is kept at the inner level and discarded at the outer
            with state.savepoint():
                with state.savepoint() as inner:
                    state.remove(np.int32(rng.choice(selected)))
                    inner.keep()
                _assert_index_list_matches_mask(state)
        _assert_index_list_matches_mask(state)


def test_selected_index_array_is_ascending(new_solver_state_unconstrained):
    """Ascending order is what keeps position-based candidate draws unchanged."""
    # --- arrange / act ----------------
    for index in (4, 1, 5, 0):
        new_solver_state_unconstrained.add(np.int32(index))
    new_solver_state_unconstrained.remove(np.int32(1))

    # --- assert -----------------------
    indices = new_solver_state_unconstrained.selected_index_array
    assert indices.tolist() == sorted(indices.tolist())
    assert indices.tolist() == [0, 4, 5]


# =================================================================================================
#  Adoption
# =================================================================================================
def _make_adoption_state(
    diversity_metric: DiversityMetric, diversity_tie_breakers: list[DiversityMetric]
) -> SolverState:
    """Build a small constrained state for adoption tests."""
    vectors = np.array([[0.0], [1.0], [3.0], [6.0], [10.0], [15.0], [21.0], [28.0]], dtype=np.float32)
    return SolverState.new(
        n=vectors.shape[0],
        store=DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.L1_MANHATTAN), n=vectors.shape[0]),
        k=4,
        diversity_metric=diversity_metric,
        diversity_tie_breakers=diversity_tie_breakers,
        constraints=[
            Constraint(int_set={0, 1, 2, 3}, min_count=1, max_count=3),
            Constraint(int_set={4, 5, 6, 7}, min_count=1, max_count=3),
        ],
    )


def _assert_state_matches_reference(state: SolverState, reference: SolverState) -> None:
    """Assert `state` is indistinguishable from `reference`, a fresh state holding the same selection."""
    assert np.array_equal(state.selected_index_array, reference.selected_index_array)
    assert state.n_selected == reference.n_selected
    assert np.array_equal(state.con_values, reference.con_values)
    assert state.score.as_tuple() == pytest.approx(reference.score.as_tuple())
    np.testing.assert_allclose(state.full_contribution_array, reference.full_contribution_array, rtol=1e-6, atol=1e-6)


_METRIC_CONFIGS = [
    (DiversityMetric.GEOMEAN_SEPARATION, [DiversityMetric.NON_ZERO_SEPARATION_FRAC]),  # separation tracker only
    (DiversityMetric.MEAN_PAIRWISE_DISTANCE, []),  # mean-distance tracker only
    (DiversityMetric.MIN_SEPARATION, [DiversityMetric.MEAN_PAIRWISE_DISTANCE]),  # both tracker families
]


def test_reset_returns_the_state_to_empty():
    """A reset state is indistinguishable from a freshly built one, and stays fully usable."""
    # --- arrange -----------------------------------------
    state = _make_adoption_state(DiversityMetric.MIN_SEPARATION, [DiversityMetric.MEAN_PAIRWISE_DISTANCE])
    state.add_many(np.array([0, 2, 4, 6], dtype=np.int32))
    reference = _make_adoption_state(DiversityMetric.MIN_SEPARATION, [DiversityMetric.MEAN_PAIRWISE_DISTANCE])

    # --- act ---------------------------------------------
    state.reset()

    # --- assert ------------------------------------------
    _assert_state_matches_reference(state, reference)
    state.add(np.int32(1))  # the reset state accepts mutations as usual
    assert state.selected_index_array.tolist() == [1]


def test_reset_inside_a_savepoint_is_rejected():
    """A reset cannot be provisional, so an open savepoint rejects it."""
    # --- arrange -----------------------------------------
    state = _make_adoption_state(DiversityMetric.GEOMEAN_SEPARATION, [])
    state.add_many(np.array([0, 1], dtype=np.int32))

    # --- act & assert ------------------------------------
    with state.savepoint(), pytest.raises(RuntimeError):
        state.reset()
    assert state.selected_index_array.tolist() == [0, 1]


@pytest.mark.parametrize("diversity_metric, tie_breakers", _METRIC_CONFIGS)
@pytest.mark.parametrize(
    "start, target",
    [
        ([0, 2, 4, 6], [0, 2, 4, 7]),  # small diff -> diff route
        ([0, 1, 2, 3], [4, 5, 6, 7]),  # disjoint -> rebuild route
        ([], [1, 3, 5, 7]),  # from empty -> diff route (adds only)
        ([0, 2, 4, 6], []),  # to empty -> rebuild route
        ([1, 3, 5, 7], [1, 3, 5, 7]),  # identical -> no-op diff
        ([0, 1, 2], [2, 4, 5, 6, 7]),  # different sizes, low overlap -> rebuild route
    ],
)
def test_adopt_selection_matches_fresh_state(diversity_metric, tie_breakers, start, target):
    """Adopting a selection must be indistinguishable from having built it directly, on either route."""
    # --- arrange -----------------------------------------
    state = _make_adoption_state(diversity_metric, tie_breakers)
    state.add_many(np.array(start, dtype=np.int32))
    reference = _make_adoption_state(diversity_metric, tie_breakers)
    reference.add_many(np.array(target, dtype=np.int32))

    # --- act ---------------------------------------------
    state.adopt_selection(np.array(target, dtype=np.int32))

    # --- assert ------------------------------------------
    _assert_state_matches_reference(state, reference)


def test_adopt_selection_accepts_unordered_input():
    """Adoption sorts its input itself; the caller's ordering carries no meaning."""
    # --- arrange -----------------------------------------
    state = _make_adoption_state(DiversityMetric.GEOMEAN_SEPARATION, [])
    reference = _make_adoption_state(DiversityMetric.GEOMEAN_SEPARATION, [])
    reference.add_many(np.array([1, 4, 6], dtype=np.int32))

    # --- act ---------------------------------------------
    state.adopt_selection(np.array([6, 1, 4], dtype=np.int32))

    # --- assert ------------------------------------------
    _assert_state_matches_reference(state, reference)


def test_adopt_selection_state_remains_fully_usable():
    """After adoption the state supports mutators and savepoints as usual."""
    # --- arrange -----------------------------------------
    state = _make_adoption_state(DiversityMetric.MIN_SEPARATION, [DiversityMetric.MEAN_PAIRWISE_DISTANCE])
    state.add_many(np.array([0, 1, 2, 3], dtype=np.int32))

    # --- act ---------------------------------------------
    state.adopt_selection(np.array([4, 5, 6, 7], dtype=np.int32))  # rebuild route
    score_after_adopt = state.score.as_tuple()
    with state.savepoint():
        state.remove(np.int32(4))
        state.add(np.int32(0))
    state.adopt_selection(np.array([4, 5, 6, 0], dtype=np.int32))  # diff route

    # --- assert ------------------------------------------
    assert score_after_adopt != state.score.as_tuple()
    assert state.selected_index_array.tolist() == [0, 4, 5, 6]


def test_adopt_selection_validation():
    """Duplicate, out-of-range, and savepoint-open calls are rejected without touching the state."""
    # --- arrange -----------------------------------------
    state = _make_adoption_state(DiversityMetric.GEOMEAN_SEPARATION, [])
    state.add_many(np.array([0, 1], dtype=np.int32))

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        state.adopt_selection(np.array([1, 1, 2], dtype=np.int32))  # duplicates
    with pytest.raises(ValueError):
        state.adopt_selection(np.array([-1, 2], dtype=np.int32))  # negative index
    with pytest.raises(ValueError):
        state.adopt_selection(np.array([2, 8], dtype=np.int32))  # index >= n
    with state.savepoint(), pytest.raises(RuntimeError):
        state.adopt_selection(np.array([2, 3], dtype=np.int32))  # open savepoint
    assert state.selected_index_array.tolist() == [0, 1]  # untouched by the rejected calls
