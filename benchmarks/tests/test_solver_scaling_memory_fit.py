import pytest

from benchmarks.solver_scaling.grid import MEMORY_CAP_BYTES, operational_bound, size_grid
from benchmarks.solver_scaling.memory_fit import fit_all, fit_memory_limit
from benchmarks.solver_scaling.outcome import Outcome
from benchmarks.solver_scaling.records import ScalingRunRecord


def test_quadratic_fit_recovers_a_known_crossing():
    # --- arrange ----------------------
    c0, c1, c2 = 1.0e8, 100.0, 50.0
    sizes_peaks = {n: c0 + c1 * n + c2 * n**2 for n in (1000, 2000, 5000, 10000)}

    # --- act --------------------------
    fit = fit_memory_limit(sizes_peaks, Outcome.TIMEOUT)

    # --- assert -----------------------
    n_star = ((MEMORY_CAP_BYTES - c0 - c1 * 0) / c2) ** 0.5  # coarse: the c2*n^2 term dominates
    assert fit.max_n == max(n for n in size_grid(operational_bound()) if c0 + c1 * n + c2 * n**2 <= MEMORY_CAP_BYTES)
    assert fit.max_n is not None and fit.max_n < n_star


def test_the_c1_lower_bound_keeps_flat_baseline_dominated_data_well_posed():
    # Peaks barely rise with n (interpreter footprint dominates). Without the c1 >= 8 lower bound the
    # fit would read no growth and never cross; the bound makes the crossing the input-array limit.
    # --- arrange ----------------------
    sizes_peaks = {1000: 5.0e8, 2000: 5.0e8 + 1000}

    # --- act --------------------------
    fit = fit_memory_limit(sizes_peaks, Outcome.TIMEOUT)

    # --- assert -----------------------
    assert fit.max_n is not None
    assert fit.max_n >= 1_000_000_000
    assert fit.coef is not None and fit.coef[1] == pytest.approx(8.0)


def test_memory_exceeded_brackets_at_the_last_completed_size():
    # --- act --------------------------
    fit = fit_memory_limit({100: 1e9, 200: 2e9, 500: 4e9}, Outcome.MEMORY)

    # --- assert -----------------------
    assert fit.max_n == 500
    assert fit.coef is None


def test_a_single_completed_size_cannot_extrapolate():
    # --- act --------------------------
    fit = fit_memory_limit({100: 1e9}, Outcome.TIMEOUT)

    # --- assert -----------------------
    assert fit.max_n == 100


def test_no_completed_runs_yields_no_limit():
    # --- act --------------------------
    fit = fit_memory_limit({}, Outcome.SCALING_FAILURE)

    # --- assert -----------------------
    assert fit.max_n is None


def _completed(tool, config, n, peak):
    return ScalingRunRecord(tool, config, n, n // 10, 0, 60.0, True, None, 1.0, peak, 0.3)


def test_fit_all_groups_by_tool_and_config():
    # --- arrange ----------------------
    records = [
        _completed("max-div", "lean", 1000, int(1e9)),
        _completed("max-div", "lean", 2000, int(2e9)),
        _completed("max-div", "lean", 5000, int(4e9)),
    ]

    # --- act --------------------------
    fits = fit_all(records)

    # --- assert -----------------------
    assert "max-div/lean" in fits
    assert "rdkit/default" in fits  # every smoke config gets an entry, even with no records
    assert fits["max-div/lean"].max_n is not None
    assert fits["rdkit/default"].max_n is None


def test_an_implausibly_small_quadratic_term_falls_back_to_linear():
    """A fitted c2 below one byte per k*n entry cannot be a real allocation, so the fit is linear."""
    # --- arrange ----------------------
    # linear growth plus a curvature far below the plausibility threshold of 0.1 bytes per n^2
    sizes_peaks = {n: 1.6e8 + 40.0 * n + 1e-5 * n**2 for n in (50_000, 100_000, 200_000, 500_000, 1_000_000)}

    # --- act --------------------------
    fit = fit_memory_limit(sizes_peaks, Outcome.TIMEOUT)

    # --- assert -----------------------
    assert fit.coef is not None and len(fit.coef) == 2
    assert fit.reason == "linear fit over 5 sizes"


def test_a_config_with_worker_processes_is_excluded_from_extrapolation():
    """Per-process footprints miss the workers, so only a bracket is valid for such a config."""
    # --- arrange ----------------------
    sizes_peaks = {n: 1.6e8 + 40.0 * n for n in (1000, 2000, 5000)}

    # --- act --------------------------
    excluded = fit_memory_limit(sizes_peaks, Outcome.TIMEOUT, spawned=True)
    bracketed = fit_memory_limit(sizes_peaks, Outcome.MEMORY, spawned=True)

    # --- assert -----------------------
    assert excluded.max_n is None and excluded.coef is None
    assert excluded.reason == "excluded from extrapolation: spawns worker processes"
    assert bracketed.max_n == 5000  # the machine-level cap kill brackets any process tree
