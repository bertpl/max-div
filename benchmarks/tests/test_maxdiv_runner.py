import numpy as np
import pytest

from benchmarks.runners import run_maxdiv_budget_series
from benchmarks.runners.maxdiv_runner import maxdiv_tool_label, memory_bound_concurrency


def test_single_worker_series_records_one_row_per_budget_and_seed(small_problem):
    """One record per (budget, seed), labeled by the preset, timed end to end."""
    # --- act --------------------------
    records = run_maxdiv_budget_series(
        small_problem, problem_name="U1", size=30, time_budgets_sec=[0.001, 0.002], seeds=(0, 1)
    )

    # --- assert -----------------------
    assert len(records) == 4
    assert {r.tool for r in records} == {"max-div[DEFAULT]"}
    assert {r.budget for r in records} == {"time:0.001s", "time:0.002s"}
    assert all(r.measured_sec > 0 and len(np.unique(r.quality)) > 0 for r in records)


def test_multi_worker_series_uses_the_worker_count_in_its_label(small_problem):
    """A multi-worker series is a separate tool label, so both series can share one chart."""
    # --- act --------------------------
    records = run_maxdiv_budget_series(
        small_problem, problem_name="U1", size=30, time_budgets_sec=[0.5], seeds=(0,), n_workers=2
    )

    # --- assert -----------------------
    assert [r.tool for r in records] == [maxdiv_tool_label(n_workers=2)] == ["max-div[DEFAULT, 2 workers]"]
    assert records[0].measured_sec > 0.5  # end to end: spawning the workers is inside the measured time


def test_concurrent_single_worker_runs_return_the_same_records_shape(small_problem):
    """Packing solves across processes changes nothing about the records, only the wall clock."""
    # --- act --------------------------
    records = run_maxdiv_budget_series(
        small_problem, problem_name="U1", size=30, time_budgets_sec=[0.001], seeds=(0, 1, 2), concurrency=3
    )

    # --- assert -----------------------
    assert sorted(r.seed for r in records) == [0, 1, 2]


@pytest.mark.parametrize(
    ("total_memory_bytes", "expected"),
    [
        (None, 12),  # RAM unknown: the request stands
        (64 * 2**30, 12),  # 12 full-matrix stores of 30 items are nothing next to 64 GiB
        (4 * 3600 * 3, 6),  # AUTO resolves to a 3600-byte full matrix; half of 43200 bytes holds 6 of them
        (3600 * 3, 1),  # half of 10800 bytes holds one full store
        (5 * 900, 12),  # AUTO resolves to lazy: nothing is stored, so the request stands
    ],
)
def test_memory_bound_concurrency_caps_side_by_side_solves_by_their_stores(small_problem, total_memory_bytes, expected):
    """The side-by-side solve count follows the resolved store's bytes against `_PACKED_STORES_MEMORY_FRACTION` of the given RAM, never below one."""
    assert memory_bound_concurrency(small_problem, 12, total_memory_bytes) == expected
