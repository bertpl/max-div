import numpy as np

from benchmarks.runners import run_maxdiv_budget_series
from benchmarks.runners.maxdiv_runner import maxdiv_tool_label


def test_single_worker_series_records_one_row_per_budget_and_seed(small_problem):
    """One record per (budget, seed), labelled by the preset, timed end to end."""
    # --- act ---------------------------------------------
    records = run_maxdiv_budget_series(
        small_problem, problem_name="U1", size=30, time_budgets_sec=[0.001, 0.002], seeds=(0, 1)
    )

    # --- assert ------------------------------------------
    assert len(records) == 4
    assert {r.tool for r in records} == {"max-div[DEFAULT]"}
    assert {r.budget for r in records} == {"time:0.001s", "time:0.002s"}
    assert all(r.measured_sec > 0 and len(np.unique(r.quality)) > 0 for r in records)


def test_multi_worker_series_uses_the_worker_count_in_its_label(small_problem):
    """A multi-worker series is a separate tool label, so both series can share one chart."""
    # --- act ---------------------------------------------
    records = run_maxdiv_budget_series(
        small_problem, problem_name="U1", size=30, time_budgets_sec=[0.5], seeds=(0,), n_workers=2
    )

    # --- assert ------------------------------------------
    assert [r.tool for r in records] == [maxdiv_tool_label(n_workers=2)] == ["max-div[DEFAULT, 2 workers]"]
    assert records[0].measured_sec > 0.5  # end to end: spawning the workers is inside the measured time


def test_concurrent_single_worker_runs_return_the_same_records_shape(small_problem):
    """Packing solves across processes changes nothing about the records, only the wall clock."""
    # --- act ---------------------------------------------
    records = run_maxdiv_budget_series(
        small_problem, problem_name="U1", size=30, time_budgets_sec=[0.001], seeds=(0, 1, 2), concurrency=3
    )

    # --- assert ------------------------------------------
    assert sorted(r.seed for r in records) == [0, 1, 2]
