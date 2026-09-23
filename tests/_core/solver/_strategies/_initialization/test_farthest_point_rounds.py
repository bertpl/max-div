import numpy as np
import pytest

from max_div._core._random import new_rng_state
from max_div._core.metrics import DistanceMetric, DiversityObjectiveSimple
from max_div._core.metrics._distance import DistanceStore
from max_div._core.solver._solver_step import InitializationStep
from max_div._core.solver._step_identity import SolverStepIdentity
from max_div._core.solver._strategies import InitializationStrategy
from max_div._core.solver._strategies._initialization._farthest_point_rounds import _draw_round
from max_div._core.solver._strategies._initialization._init_farthest_point import InitFarthestPoint
from max_div.metrics import DiversityMetric

from ._helpers import new_solver_state, new_solver_state_unconstrained

# each step records its checkpoints under this identity when run on its own in these tests
_STEP_IDENTITY = SolverStepIdentity(1, "test")


def _farthest_point_in_rounds(**kwargs) -> InitFarthestPoint:
    """Return a farthest-point strategy bound to a separation objective, so that it draws in rounds."""
    strategy = InitializationStrategy.farthest_point(**kwargs)
    strategy.bind_objective(DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION))
    return strategy


@pytest.mark.parametrize("top_k", [1, 8])
def test_rounds_completes_selection(top_k: int):
    """Drawing in rounds selects exactly k distinct items and reaches full size."""
    # --- arrange ----------------------
    state = new_solver_state(has_constraints=False)
    step = InitializationStep(_farthest_point_in_rounds(top_k=top_k))

    # --- act --------------------------
    step.run(state, _STEP_IDENTITY)

    # --- assert -----------------------
    assert state.score.size == 1.0
    selection = state.selected_index_array
    assert len(np.unique(selection)) == state.k


def test_rounds_is_deterministic_per_seed():
    """The same seed reproduces the same selection; a different seed varies it."""
    # --- arrange ----------------------
    selections = []
    for seed in (7, 7, 8):
        state = new_solver_state(has_constraints=False)
        step = InitializationStep(_farthest_point_in_rounds())
        step.set_seed(seed)

        # --- act ----------------------
        step.run(state, _STEP_IDENTITY)
        selections.append(np.sort(state.selected_index_array).copy())

    # --- assert -----------------------
    np.testing.assert_array_equal(selections[0], selections[1])
    assert not np.array_equal(selections[0], selections[2])


def test_rounds_quality_near_one_item_at_a_time():
    """The min-separation reached in rounds is near the one reached one item at a time."""
    # --- arrange ----------------------
    results = {}
    for label, strategy in (
        ("one_at_a_time", InitializationStrategy.farthest_point(top_k=8, batch_size=None)),
        ("rounds", _farthest_point_in_rounds()),
    ):
        state = new_solver_state(has_constraints=False)
        step = InitializationStep(strategy)
        step.set_seed(42)

        # --- act ----------------------
        step.run(state, _STEP_IDENTITY)
        results[label] = state.score.diversity

    # --- assert -----------------------
    assert results["rounds"] >= 0.8 * results["one_at_a_time"]


@pytest.mark.parametrize(
    "metric, batch_size, is_drawing_rounds",
    [
        (DiversityMetric.MIN_SEPARATION, 16, True),
        (DiversityMetric.MIN_SEPARATION, None, False),
        (DiversityMetric.MEAN_PAIRWISE_DISTANCE, 16, False),
    ],
)
def test_rounds_only_for_a_separation_objective_and_a_batch_size(
    metric: DiversityMetric, batch_size: int | None, is_drawing_rounds: bool
):
    """A strategy draws several items per call only when bound to a separation objective with a batch size."""
    # --- arrange ----------------------
    state = new_solver_state_unconstrained()
    strategy = InitializationStrategy.farthest_point(batch_size=batch_size)
    strategy.bind_objective(DiversityObjectiveSimple(metric))

    # --- act --------------------------
    batch_sizes = []
    while state.n_selected < state.k:
        batch = strategy.get_next_samples(state, np.int32(state.k - state.n_selected))
        batch_sizes.append(len(batch))
        state.add_many(batch)

    # --- assert -----------------------
    assert (max(batch_sizes) > 1) is is_drawing_rounds


@pytest.mark.parametrize(
    "kwargs",
    [{"top_k": 0}, {"batch_size": 0}, {"top_k": 8, "batch_size": 4}],
)
def test_rounds_rejects_invalid_parameters(kwargs: dict):
    """The constructor rejects `top_k` below 1 and `batch_size` below `top_k`."""
    with pytest.raises(ValueError):
        InitFarthestPoint(**kwargs)


def test_rounds_batches_respect_the_contract():
    """Every returned batch is duplicate-free, in range, and not yet selected."""
    # --- arrange ----------------------
    state = new_solver_state(has_constraints=False)
    strategy = _farthest_point_in_rounds(batch_size=16)

    # --- arrange / act / assert -------
    while state.n_selected < state.k:
        batch = strategy.get_next_samples(state, np.int32(state.k - state.n_selected))
        assert 1 <= len(batch) <= state.k - state.n_selected
        assert len(np.unique(batch)) == len(batch)
        assert batch.min() >= 0
        assert batch.max() < state.n
        assert not np.isin(batch, state.selected_index_array).any()
        state.add_many(batch)


def test_draw_round_ends_once_the_pool_can_no_longer_be_shown_to_hold_the_best():
    """A refresh that pushes the live pool below `threshold` ends the round instead of drawing on."""
    # --- arrange ----------------------
    # items 1, 2, 3 sit far from item 0 but close to each other, so drawing one drops the
    # separations of the others far below `threshold`
    vectors = np.array([[0.0], [100.0], [101.0], [102.0]], dtype=np.float32)
    store = DistanceStore.lazy(vectors, DistanceMetric.l2_euclidean())
    cand_idx = np.array([1, 2, 3], dtype=np.int32)
    cand_val = np.array([100.0, 101.0, 102.0], dtype=np.float32)  # separations wrt a selection of {0}
    threshold = np.float32(100.0)  # the pool's lowest value at round start
    out_batch = np.empty(3, dtype=np.int32)

    # --- act --------------------------
    n_drawn = _draw_round(
        cand_idx,
        cand_val,
        np.int32(1),
        threshold,
        np.int64(3),
        store,
        new_rng_state(np.int64(1)),
        out_batch,
        np.empty(1, dtype=np.int32),
    )

    # --- assert -----------------------
    assert n_drawn == 1
    assert out_batch[0] == 3  # the farthest item; its neighbors then fall below `threshold`


def test_draw_round_draws_while_the_pool_still_holds_the_best():
    """With every candidate far apart, no refresh drops below `threshold` and the round runs on."""
    # --- arrange ----------------------
    vectors = np.array([[0.0], [100.0], [200.0], [300.0]], dtype=np.float32)
    store = DistanceStore.lazy(vectors, DistanceMetric.l2_euclidean())
    cand_idx = np.array([1, 2, 3], dtype=np.int32)
    cand_val = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    out_batch = np.empty(3, dtype=np.int32)

    # --- act --------------------------
    n_drawn = _draw_round(
        cand_idx,
        cand_val,
        np.int32(1),
        np.float32(100.0),
        np.int64(3),
        store,
        new_rng_state(np.int64(1)),
        out_batch,
        np.empty(1, dtype=np.int32),
    )

    # --- assert -----------------------
    assert n_drawn >= 2


def test_draw_round_ends_when_fewer_than_top_k_candidates_remain():
    """With a finite threshold, a draw needs top_k live candidates to be shown to range over the dataset's best."""
    # --- arrange ----------------------
    vectors = np.array([[0.0], [100.0], [200.0], [300.0]], dtype=np.float32)
    store = DistanceStore.lazy(vectors, DistanceMetric.l2_euclidean())
    cand_idx = np.array([1, 2, 3], dtype=np.int32)
    cand_val = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    out_batch = np.empty(3, dtype=np.int32)

    # --- act --------------------------
    n_drawn = _draw_round(
        cand_idx,
        cand_val,
        np.int32(2),
        np.float32(50.0),
        np.int64(3),
        store,
        new_rng_state(np.int64(1)),
        out_batch,
        np.empty(2, dtype=np.int32),
    )

    # --- assert -----------------------
    assert n_drawn == 2  # the third draw would have only 1 live candidate for a top_k of 2


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_draw_is_among_the_top_k_contributions(seed: int):
    """Each draw lands among the top_k highest contributions over all not-selected items."""
    # --- arrange ----------------------
    top_k = 4
    state = new_solver_state_unconstrained()
    strategy = _farthest_point_in_rounds(top_k=top_k, batch_size=16)
    strategy.set_seed(seed)

    # --- arrange / act / assert -------
    while state.n_selected < state.k:
        batch = strategy.get_next_samples(state, np.int32(state.k - state.n_selected))
        assert len(batch) >= 1, "a round must always draw at least one item"
        for item in batch:
            # the comparison values are read from SolverState, not from the strategy's candidate
            # pool, so a candidate pool that is out of date produces a draw below the true top-k
            contributions = state.full_contribution_array
            selected = np.zeros(state.n, dtype=bool)
            selected[state.selected_index_array] = True
            available = contributions[~selected]
            kth_best = np.sort(available)[-top_k] if len(available) >= top_k else available.min()
            assert contributions[item] >= kth_best
            state.add(item)  # one at a time, so the contributions update after every draw in the batch


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_top_k_one_reproduces_the_per_pick_construction_exactly(seed: int):
    """With `top_k=1`, both constructions take the same greedy pick and agree item for item."""
    # --- arrange ----------------------
    states = [new_solver_state_unconstrained() for _ in range(2)]
    steps = [
        InitializationStep(InitializationStrategy.farthest_point(top_k=1, batch_size=None)),
        InitializationStep(_farthest_point_in_rounds(top_k=1)),
    ]

    # --- act --------------------------
    for step, state in zip(steps, states, strict=True):
        step.set_seed(seed)
        step.run(state, _STEP_IDENTITY)

    # --- assert -----------------------
    np.testing.assert_array_equal(states[0].selected_index_array, states[1].selected_index_array)
