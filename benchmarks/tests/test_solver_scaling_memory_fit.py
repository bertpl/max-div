import pytest

from benchmarks.solver_scaling.grid import MEMORY_CAP_BYTES, operational_bound, size_grid
from benchmarks.solver_scaling.memory_fit import trust_conditions_met, fit_series


def test_quadratic_fit_recovers_a_known_crossing():
    # --- arrange ----------------------
    c0, c1, c2 = 1.0e8, 100.0, 50.0
    sizes_peaks = {n: c0 + c1 * n + c2 * n**2 for n in (1000, 2000, 5000, 10000)}

    # --- act --------------------------
    fit = fit_series(sizes_peaks)

    # --- assert -----------------------
    assert fit.max_n == max(n for n in size_grid(operational_bound()) if c0 + c1 * n + c2 * n**2 <= MEMORY_CAP_BYTES)
    assert fit.r2 == pytest.approx(1.0)


def test_the_c1_lower_bound_keeps_flat_baseline_dominated_data_well_posed():
    # Peaks barely rise with n (interpreter footprint dominates). Without the c1 >= 8 lower bound the
    # fit would read no growth and never cross; the bound makes the crossing the input-array limit.
    # --- arrange ----------------------
    sizes_peaks = {1000: 5.0e8, 2000: 5.0e8 + 1000}

    # --- act --------------------------
    fit = fit_series(sizes_peaks)

    # --- assert -----------------------
    assert fit.max_n is not None
    assert fit.max_n >= 1_000_000_000
    assert fit.coef is not None and fit.coef[1] == pytest.approx(8.0)


def test_an_implausibly_small_quadratic_term_falls_back_to_linear():
    """A fitted c2 below one byte per k*n entry cannot be a real allocation, so the fit is linear."""
    # --- arrange ----------------------
    # linear growth plus a curvature far below the plausibility threshold of 0.1 bytes per n^2
    sizes_peaks = {n: 1.6e8 + 40.0 * n + 1e-5 * n**2 for n in (50_000, 100_000, 200_000, 500_000, 1_000_000)}

    # --- act --------------------------
    fit = fit_series(sizes_peaks)

    # --- assert -----------------------
    assert fit.coef is not None and len(fit.coef) == 2
    assert fit.reason == "linear fit over 5 sizes"


def test_conditions_require_a_2gb_footprint_model_quality_and_enough_sizes():
    """The sweep may stop only with >= 5 sizes, one footprint at 2 GB, and a model that explains them."""
    # --- arrange ----------------------
    low = {n: 1.6e8 + 40.0 * n for n in (200_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000)}  # tops out at 560 MB
    four = {n: 1.6e8 + 40.0 * n for n in (5_000_000, 10_000_000, 20_000_000, 50_000_000)}  # reaches 2 GB, only 4 sizes
    grown = {n: 1.6e8 + 40.0 * n for n in (2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000)}  # 5 sizes, 2 GB

    # --- act / assert -----------------
    assert not trust_conditions_met(low, fit_series(low))  # no footprint at 2 GB
    assert not trust_conditions_met(four, fit_series(four))  # only 4 sizes
    assert trust_conditions_met(grown, fit_series(grown))


def test_one_footprint_far_off_the_trend_does_not_move_the_median_fit():
    """One footprint far off the trend leaves the fitted coefficients on the trend."""
    # --- arrange ----------------------
    sizes_peaks = {n: 1.6e8 + 40.0 * n for n in (100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000)}
    sizes_peaks[1_000_000] *= 3  # one run three times its trend value

    # --- act --------------------------
    fit = fit_series(sizes_peaks)

    # --- assert -----------------------
    assert fit.coef is not None
    assert fit.coef[0] == pytest.approx(1.6e8, rel=1e-6)
    assert fit.coef[1] == pytest.approx(40.0, rel=1e-6)
