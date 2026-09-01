import numpy as np
import pytest

from max_div._core.solver._solver_step import InitializationStep
from max_div._core.solver._strategies import InitializationStrategy
from max_div._core.solver._strategies._initialization._init_farthest_point_batched import InitFarthestPointBatched
from max_div.metrics import DiversityMetric

from ._helpers import new_solver_state


@pytest.mark.parametrize("top_k", [1, 8])
def test_init_farthest_point_batched_completes_selection(top_k: int):
    """The batched init selects exactly k distinct items and reaches full size."""
    # --- arrange ----------------------
    state = new_solver_state(has_constraints=False)
    step = InitializationStep(InitializationStrategy.farthest_point_batched(top_k=top_k))

    # --- act --------------------------
    step.run(state)

    # --- assert -----------------------
    assert state.score.size == 1.0
    selection = state.selected_index_array
    assert len(np.unique(selection)) == state.k


def test_init_farthest_point_batched_is_deterministic_per_seed():
    """The same seed reproduces the same selection; a different seed varies it."""
    # --- arrange ----------------------
    selections = []
    for seed in (7, 7, 8):
        state = new_solver_state(has_constraints=False)
        step = InitializationStep(InitializationStrategy.farthest_point_batched())
        step.set_seed(seed)

        # --- act ----------------------
        step.run(state)
        selections.append(np.sort(state.selected_index_array).copy())

    # --- assert -----------------------
    np.testing.assert_array_equal(selections[0], selections[1])
    assert not np.array_equal(selections[0], selections[2])


def test_init_farthest_point_batched_quality_near_exact_sibling():
    """The batched construction's min-separation lands near the exact per-pick sibling's."""
    # --- arrange ----------------------
    results = {}
    for label, strategy in (
        ("exact", InitializationStrategy.farthest_point(top_k=8)),
        ("batched", InitializationStrategy.farthest_point_batched()),
    ):
        state = new_solver_state(has_constraints=False)
        step = InitializationStep(strategy)
        step.set_seed(42)

        # --- act ----------------------
        step.run(state)
        results[label] = state.score.diversity

    # --- assert -----------------------
    assert results["batched"] >= 0.8 * results["exact"]


def test_init_farthest_point_batched_rejects_mean_family_metric():
    """A mean-family main metric is refused: the round heuristics are separation-tailored."""
    # --- arrange ----------------------
    strategy = InitializationStrategy.farthest_point_batched()

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="separation-based diversity metrics"):
        strategy.validate_diversity_metric(DiversityMetric.MEAN_PAIRWISE_DISTANCE)
    strategy.validate_diversity_metric(DiversityMetric.MIN_SEPARATION)  # accepted: no raise


@pytest.mark.parametrize(
    "kwargs",
    [{"top_k": 0}, {"alpha": 0.0}, {"alpha": 1.5}, {"batch_max": 0}],
)
def test_init_farthest_point_batched_rejects_invalid_parameters(kwargs: dict):
    """Constructor bounds: top_k >= 1, alpha in (0, 1], batch_max >= 1."""
    with pytest.raises(ValueError):
        InitFarthestPointBatched(**kwargs)


def test_init_farthest_point_batched_batches_respect_the_contract():
    """Every returned batch is duplicate-free, in range, and not yet selected."""
    # --- arrange ----------------------
    state = new_solver_state(has_constraints=False)
    strategy = InitializationStrategy.farthest_point_batched(batch_max=16)

    # --- act / assert -----------------
    while state.n_selected < state.k:
        batch = strategy.get_next_samples(state, np.int32(state.k - state.n_selected))
        assert 1 <= len(batch) <= state.k - state.n_selected
        assert len(np.unique(batch)) == len(batch)
        assert batch.min() >= 0 and batch.max() < state.n
        assert not np.isin(batch, state.selected_index_array).any()
        state.add_many(batch)
