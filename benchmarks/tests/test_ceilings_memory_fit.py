"""Guards for the memory-model fits behind the memory ceilings."""

from benchmarks.ceilings.grid import MEMORY_CAP_BYTES
from benchmarks.ceilings.memory_fit import fit_memory_ceilings
from benchmarks.ceilings.records import CeilingRunRecord


def _record(tool: str, n: int, peak: int, mode: str = "fastest_valid", completed: bool = True) -> CeilingRunRecord:
    """A minimal record carrying only what the fit reads."""
    return CeilingRunRecord(
        tool=tool,
        mode=mode,
        n=n,
        k=n // 10,
        seed=0,
        budget_sec=60.0,
        completed=completed,
        reason=None if completed else "timeout",
        measured_sec=1.0 if completed else None,
        peak_rss_bytes=peak if completed else None,
        min_separation=0.1 if completed else None,
    )


def test_a_quadratic_tool_crosses_where_its_model_says() -> None:
    """Peaks following 100 MB + 8 * n^2 bytes must cross 32 GB just past n = 65k -> grid 50000."""
    # --- arrange ----------------------
    base, coef = 100 * 2**20, 8.0
    records = [_record("dppy", n, int(base + coef * n**2)) for n in (1000, 2000, 5000)]

    # --- act --------------------------
    fits = fit_memory_ceilings(records)

    # --- assert -----------------------
    assert fits["dppy"]["ceiling"] == 50000


def test_a_flat_footprint_extends_to_the_operational_bound() -> None:
    """A tool whose growth term fits to zero is bounded only by the input's own size."""
    # --- arrange ----------------------
    records = [_record("fpsample", n, 200 * 2**20) for n in (1000, 2000, 5000)]

    # --- act --------------------------
    fits = fit_memory_ceilings(records)

    # --- assert -----------------------
    assert fits["fpsample"]["ceiling"] == 2_000_000_000  # the last grid value under the operational bound


def test_too_few_completed_sizes_yields_no_ceiling() -> None:
    """One point cannot pin a two-parameter model; the fit must say so, not guess."""
    # --- arrange ----------------------
    records = [_record("scip", 100, 500 * 2**20), _record("scip", 200, 0, completed=False)]

    # --- act --------------------------
    fits = fit_memory_ceilings(records)

    # --- assert -----------------------
    assert fits["scip"]["ceiling"] is None
    assert "1 completed" in fits["scip"]["reason"]


def test_memory_optimal_records_take_precedence_over_fastest_valid() -> None:
    """Where dedicated memory-optimal runs exist, the fit must use them, not the fast ones."""
    # --- arrange ----------------------
    fast = [_record("max-div", n, int(1e12)) for n in (1000, 2000, 5000)]  # absurd, must be ignored
    optimal = [_record("max-div", n, 100 * 2**20 + 40 * n, mode="memory_optimal") for n in (1000, 2000, 5000)]

    # --- act --------------------------
    fits = fit_memory_ceilings(fast + optimal)

    # --- assert -----------------------
    assert fits["max-div"]["fit_sizes"] == [1000, 2000, 5000]
    assert fits["max-div"]["ceiling"] is not None
    assert 100 * 2**20 + 40 * fits["max-div"]["ceiling"] < MEMORY_CAP_BYTES
