import pytest

from benchmarks.solver_scaling.grid import MEMORY_CAP_BYTES, operational_bound, size_grid
from benchmarks.solver_scaling.memory_fit import conditions_met, fit_series


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


def test_conditions_require_span_and_model_quality():
    """The sweep may stop only once the footprints span 2x and the model explains them."""
    # --- arrange ----------------------
    flat = {n: 1.6e8 + 8.0 * n for n in (100, 200, 500)}  # true growth, but span far below 2x
    grown = {n: 1.6e8 + 40.0 * n for n in (1_000_000, 2_000_000, 5_000_000, 10_000_000)}  # spans > 2x

    # --- act / assert -----------------
    assert not conditions_met(flat, fit_series(flat))
    assert conditions_met(grown, fit_series(grown))
    assert not conditions_met({100: 1.0e8, 200: 3.0e8}, fit_series({100: 1.0e8, 200: 3.0e8}))  # two points never do
