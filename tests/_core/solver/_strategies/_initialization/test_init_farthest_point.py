import numpy as np
import pytest

from max_div._core._random import new_rng_state
from max_div._core.metrics import DistanceMetric, DiversityObjectiveSimple
from max_div._core.metrics._distance import DistanceStore
from max_div._core.solver._solver_step import InitializationStep
from max_div._core.solver._step_identity import SolverStepIdentity
from max_div._core.solver._strategies import InitializationStrategy
from max_div._core.solver._strategies._initialization._init_farthest_point import InitFarthestPoint, _draw_from_pool
from max_div.metrics import DiversityMetric
from tests.helpers import hybrid_objective

from ._helpers import new_solver_state, new_solver_state_unconstrained

# each step records its checkpoints under this identity when run on its own in these tests
_STEP_IDENTITY = SolverStepIdentity(1, "test")

L1 = DistanceMetric.l1_manhattan()
L2 = DistanceMetric.l2_euclidean()


@pytest.mark.parametrize("problem_has_constraints", [True, False])
def test_init_farthest_point(problem_has_constraints: bool):
    """Initialization reaches the full selection size, with and without constraints."""
    # --- arrange ----------------------
    solver_state = new_solver_state(problem_has_constraints)
    strategy = InitializationStrategy.farthest_point()
    init_step = InitializationStep(strategy)

    # --- act --------------------------
    init_step.run(solver_state, _STEP_IDENTITY)
    score = solver_state.score

    # --- assert -----------------------
    assert score.size == 1.0, "Selection size should be equal to k after initialization"


def test_init_farthest_point_first_pick_is_seeded():
    """The start item comes from the strategy's seeded RNG, so different seeds can differ."""
    # --- arrange / act ----------------
    first_items = []
    for seed in (0, 1, 2, 3):
        solver_state = new_solver_state(has_constraints=False)
        strategy = InitializationStrategy.farthest_point()
        strategy.set_seed(seed)
        first_items.append(int(strategy.get_next_samples(solver_state, solver_state.k)[0]))

    # --- assert -----------------------
    assert len(set(first_items)) > 1, "Different seeds should be able to produce different start items"


def test_init_farthest_point_picks_are_greedy():
    """With top_k=1, every pick after the first is the argmax of contribution wrt the current selection."""
    # --- arrange ----------------------
    solver_state = new_solver_state(has_constraints=False)
    strategy = InitializationStrategy.farthest_point(top_k=1, candidate_pool_size=None)
    solver_state.add(np.int32(0))

    # --- act --------------------------
    picked = int(strategy.get_next_samples(solver_state, solver_state.k)[0])

    # --- assert -----------------------
    contributions = solver_state.not_selected_contribution_array
    expected = int(solver_state.not_selected_index_array[np.argmax(contributions)])
    assert picked == expected, "Pick should be the highest-contribution not-selected item"
    assert picked != 0, "The already-selected item should never be picked again"


def test_init_farthest_point_beats_random_init():
    """The greedy construction should start from a better diversity score than a random draw."""
    # --- arrange ----------------------
    state_fps = new_solver_state(has_constraints=False)
    state_random = new_solver_state(has_constraints=False)

    # --- act --------------------------
    InitializationStep(InitializationStrategy.farthest_point()).run(state_fps, _STEP_IDENTITY)
    InitializationStep(InitializationStrategy.random_selection()).run(state_random, _STEP_IDENTITY)

    # --- assert -----------------------
    assert state_fps.score.diversity > state_random.score.diversity


def test_init_farthest_point_name():
    """Test that the strategy name is generated as expected."""
    # --- arrange / act ----------------
    strategy = InitializationStrategy.farthest_point()

    # --- assert -----------------------
    assert strategy.name == "InitFarthestPoint"


@pytest.mark.parametrize(
    "kwargs",
    [{"top_k": 0}, {"top_k": -1}, {"candidate_pool_size": 0}, {"top_k": 8, "candidate_pool_size": 4}],
)
def test_init_farthest_point_rejects_invalid_parameters(kwargs: dict):
    """The constructor rejects `top_k` below 1 and `candidate_pool_size` below `top_k`."""
    # --- act & assert -----------------
    with pytest.raises(ValueError):
        InitFarthestPoint(**kwargs)


def test_farthest_point_factory_passes_top_k_through():
    """The public factory's top_k reaches the strategy: same seed, same pick as direct construction."""
    # --- arrange ----------------------
    solver_state = new_solver_state(has_constraints=False)
    solver_state.add(np.int32(0))

    # --- act --------------------------
    picks = {}
    for name, strategy in [
        ("factory", InitializationStrategy.farthest_point(top_k=5)),
        ("direct", InitFarthestPoint(top_k=5)),
    ]:
        strategy.set_seed(3)
        picks[name] = int(strategy.get_next_samples(solver_state, solver_state.k)[0])

    # --- assert -----------------------
    assert picks["factory"] == picks["direct"]


@pytest.mark.parametrize("top_k", [0, -1])
def test_farthest_point_factory_rejects_top_k_below_one(top_k: int):
    """The strategy's top_k validation raises through the public factory."""
    # --- act & assert -----------------
    with pytest.raises(ValueError, match="top_k"):
        InitializationStrategy.farthest_point(top_k=top_k)


def test_init_farthest_point_default_pick_varies_by_seed():
    """The default top_k samples among several top candidates, so the pick after the start item varies by seed."""
    # --- arrange ----------------------
    solver_state = new_solver_state(has_constraints=False)
    solver_state.add(np.int32(0))

    # --- act --------------------------
    picks = set()
    for seed in range(20):
        strategy = InitializationStrategy.farthest_point(candidate_pool_size=None)
        strategy.set_seed(seed)
        picks.add(int(strategy.get_next_samples(solver_state, solver_state.k)[0]))

    # --- assert -----------------------
    assert len(picks) > 1


def test_init_farthest_point_top_k_draws_from_the_top_set():
    """Every top_k>1 greedy pick comes from the top_k highest-contribution items, and the draw varies by seed."""
    # --- arrange ----------------------
    solver_state = new_solver_state(has_constraints=False)
    solver_state.add(np.int32(0))
    contributions = solver_state.not_selected_contribution_array
    index_array = solver_state.not_selected_index_array
    top_k = 5
    threshold = np.sort(contributions)[-top_k]  # k-th largest contribution

    # --- act --------------------------
    picks = []
    for seed in range(20):
        strategy = InitFarthestPoint(top_k=top_k, candidate_pool_size=None)
        strategy.set_seed(seed)
        picks.append(int(strategy.get_next_samples(solver_state, solver_state.k)[0]))

    # --- assert -----------------------
    for pick in picks:
        pos = int(np.where(index_array == pick)[0][0])
        assert contributions[pos] >= threshold  # each pick is among the top_k by contribution
    assert len(set(picks)) > 1  # the uniform draw varies across seeds


# =================================================================================================
#  Drawing in rounds
# =================================================================================================
@pytest.mark.parametrize("top_k", [1, 8])
def test_rounds_completes_selection(top_k: int):
    """Drawing in rounds selects exactly k distinct items and reaches full size."""
    # --- arrange ----------------------
    state = new_solver_state(has_constraints=False)
    step = InitializationStep(InitializationStrategy.farthest_point(top_k=top_k))

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
        step = InitializationStep(InitializationStrategy.farthest_point())
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
        ("one_at_a_time", InitializationStrategy.farthest_point(top_k=8, candidate_pool_size=None)),
        ("rounds", InitializationStrategy.farthest_point()),
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
    "metric, candidate_pool_size, is_drawing_rounds",
    [
        (DiversityMetric.MIN_SEPARATION, 16, True),
        (DiversityMetric.MIN_SEPARATION, None, False),
        (DiversityMetric.MEAN_PAIRWISE_DISTANCE, 16, False),
    ],
)
def test_rounds_only_for_a_separation_objective_and_a_candidate_pool_size(
    metric: DiversityMetric, candidate_pool_size: int | None, is_drawing_rounds: bool
):
    """A strategy draws several items per call only for a separation objective and a pool size."""
    # --- arrange ----------------------
    state = new_solver_state_unconstrained(metric=metric)
    strategy = InitializationStrategy.farthest_point(candidate_pool_size=candidate_pool_size)

    # --- act --------------------------
    batch_sizes = []
    while state.n_selected < state.k:
        batch = strategy.get_next_samples(state, np.int32(state.k - state.n_selected))
        batch_sizes.append(len(batch))
        state.add_many(batch)

    # --- assert -----------------------
    assert (max(batch_sizes) > 1) is is_drawing_rounds


def test_rounds_batches_respect_the_contract():
    """Every returned batch is duplicate-free, in range, and not yet selected."""
    # --- arrange ----------------------
    state = new_solver_state(has_constraints=False)
    strategy = InitializationStrategy.farthest_point(candidate_pool_size=16)

    # --- arrange / act / assert -------
    while state.n_selected < state.k:
        batch = strategy.get_next_samples(state, np.int32(state.k - state.n_selected))
        assert 1 <= len(batch) <= state.k - state.n_selected
        assert len(np.unique(batch)) == len(batch)
        assert batch.min() >= 0
        assert batch.max() < state.n
        assert not np.isin(batch, state.selected_index_array).any()
        state.add_many(batch)


def test_draw_from_pool_ends_once_the_pool_can_no_longer_be_shown_to_hold_the_best():
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
    n_drawn = _draw_from_pool(
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


def test_draw_from_pool_draws_while_the_pool_still_holds_the_best():
    """With every candidate far apart, no refresh drops below `threshold` and the round runs on."""
    # --- arrange ----------------------
    vectors = np.array([[0.0], [100.0], [200.0], [300.0]], dtype=np.float32)
    store = DistanceStore.lazy(vectors, DistanceMetric.l2_euclidean())
    cand_idx = np.array([1, 2, 3], dtype=np.int32)
    cand_val = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    out_batch = np.empty(3, dtype=np.int32)

    # --- act --------------------------
    n_drawn = _draw_from_pool(
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


def test_draw_from_pool_ends_when_fewer_than_top_k_candidates_remain():
    """With a finite threshold, the round ends once fewer than top_k undrawn pool candidates remain."""
    # --- arrange ----------------------
    vectors = np.array([[0.0], [100.0], [200.0], [300.0]], dtype=np.float32)
    store = DistanceStore.lazy(vectors, DistanceMetric.l2_euclidean())
    cand_idx = np.array([1, 2, 3], dtype=np.int32)
    cand_val = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    out_batch = np.empty(3, dtype=np.int32)

    # --- act --------------------------
    n_drawn = _draw_from_pool(
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
    strategy = InitializationStrategy.farthest_point(top_k=top_k, candidate_pool_size=16)
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
def test_top_k_one_reproduces_the_one_item_at_a_time_construction_exactly(seed: int):
    """With `top_k=1`, both constructions take the same greedy pick and agree item for item."""
    # --- arrange ----------------------
    states = [new_solver_state_unconstrained() for _ in range(2)]
    steps = [
        InitializationStep(InitializationStrategy.farthest_point(top_k=1, candidate_pool_size=None)),
        InitializationStep(InitializationStrategy.farthest_point(top_k=1)),
    ]

    # --- act --------------------------
    for step, state in zip(steps, states, strict=True):
        step.set_seed(seed)
        step.run(state, _STEP_IDENTITY)

    # --- assert -----------------------
    np.testing.assert_array_equal(states[0].selected_index_array, states[1].selected_index_array)


@pytest.mark.parametrize(
    "objective, expected",
    [
        (DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION), True),
        (DiversityObjectiveSimple(DiversityMetric.MEAN_PAIRWISE_DISTANCE), False),  # mean-distance family
        (
            hybrid_objective(
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L2),
            ),
            False,  # two distinct specs
        ),
        (
            hybrid_objective(
                DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, L1),
                DiversityObjectiveSimple(DiversityMetric.GEOMEAN_SEPARATION, L1),
            ),
            True,  # two terms over one separation spec
        ),
    ],
)
def test_are_rounds_supported(objective, expected) -> None:
    """One distinct separation spec supports farthest-point rounds; a second spec or another family does not."""
    # --- act / assert -----------------
    assert InitFarthestPoint._are_rounds_supported(objective) is expected
