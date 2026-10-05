import numpy as np
import pytest

from max_div._core.metrics._diversity._numba import gpq_separation, harmonic_mean_separation, min_separation


@pytest.mark.parametrize(
    "sep",
    [
        pytest.param([0.75, 2.5, 0.125, 9.0], id="finite_values"),
        pytest.param([np.inf, 2.5, np.inf, 0.75, np.inf], id="inf_sentinels_are_skipped"),
        pytest.param([np.inf, np.inf, np.inf], id="all_inf_gives_inf"),
        pytest.param([3.0, 0.0, 1.0], id="a_zero_wins"),
        pytest.param([1e-40, 1.0, 1e-39], id="denormals_keep_their_order"),
        pytest.param([1.0, 1.4e-45], id="the_smallest_denormal_wins"),
        pytest.param([1e30, 3.4e38, 1e20], id="large_magnitudes"),
        pytest.param([0.5], id="single_item"),
        pytest.param(
            [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125], id="min_last_past_a_vector_width"
        ),
    ],
)
def test_min_separation_matches_the_float_minimum(sep: list[float]):
    """The integer minimum over the bit patterns picks the same value as the float minimum on every input it sees."""
    # --- arrange ----------------------
    sep_array = np.array(sep, dtype=np.float32)

    # --- act / assert -----------------
    assert min_separation(sep_array) == np.min(sep_array)


def test_min_separation_on_random_separations():
    """On random separations with +inf entries the reducer agrees with the float minimum at every size."""
    # --- arrange ----------------------
    rng = np.random.default_rng(0)

    for n in (1, 7, 8, 9, 100, 1003):
        sep = rng.random(n).astype(np.float32)
        sep[rng.random(n) < 0.1] = np.inf

        # --- act / assert -------------
        assert min_separation(sep) == np.min(sep)


@pytest.mark.parametrize(
    "sep, expected",
    [
        pytest.param([np.inf, 2.0], 4.0, id="an_inf_entry_adds_nothing_to_the_reciprocal_sum"),
        pytest.param([np.inf, np.inf], np.inf, id="all_inf_entries_give_inf"),
        pytest.param([0.0, np.inf], 0.0, id="a_zero_entry_wins_over_an_inf_entry"),
    ],
)
def test_harmonic_mean_separation_zero_and_inf_limits(sep: list[float], expected: float):
    """A zero separation makes the harmonic mean zero, +inf entries add nothing, and all-+inf gives +inf."""
    # --- act / assert -----------------
    assert harmonic_mean_separation(np.array(sep, dtype=np.float32)) == expected


@pytest.mark.parametrize(
    "sep, rank_weights, expected",
    [
        pytest.param([4.0, 1.0, 2.0], [1.0, 1.0, 1.0], 2.0, id="unit_weights_give_the_geometric_mean"),
        pytest.param([4.0, 1.0, 2.0], [1.0, 0.0, 0.0], 1.0, id="weight_on_the_smallest_gives_the_minimum"),
        pytest.param([4.0, 1.0, 16.0], [1.0, 1.0, 0.0], 2.0, id="a_zero_weight_skips_its_separation"),
        pytest.param([0.5, 0.0, 2.0], [1.0, 0.5, 0.25], 0.0, id="a_weighted_zero_gives_zero"),
    ],
)
def test_gpq_separation_weights_the_separations_by_ascending_rank(
    sep: list[float], rank_weights: list[float], expected: float
):
    """The i-th smallest separation gets the i-th weight, whatever order the separations come in."""
    # --- arrange ----------------------
    sep_array = np.array(sep, dtype=np.float32)
    weights = np.array(rank_weights, dtype=np.float32)

    # --- act --------------------------
    result = gpq_separation(sep_array, weights)

    # --- assert -----------------------
    assert result == pytest.approx(expected, rel=1e-6)
    np.testing.assert_array_equal(sep_array, np.array(sep, dtype=np.float32))  # the input stays unsorted
