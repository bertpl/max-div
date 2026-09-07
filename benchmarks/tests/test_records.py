from pathlib import Path

import pytest

from benchmarks.common import RunRecord, load_records, save_records
from benchmarks.common.records import budget_sec, budget_tag, iteration_count, iteration_tag, within_budget_tolerance


def test_records_round_trip(tmp_path: Path):
    # --- arrange -----------------------------------------
    records = [
        RunRecord(
            tool="max-div[SMART]",
            problem="U1",
            size=2,
            n=200,
            k=20,
            diversity_metric="GEOMEAN_SEPARATION",
            seed=1,
            budget="time:0.004s",
            measured_sec=0.0051,
            n_iterations=123,
            quality={"MIN_SEPARATION": 0.5, "GEOMEAN_SEPARATION": 0.9},
            n_constraints=2,
            n_constraints_satisfied=2,
        ),
        RunRecord(
            tool="fpsample[FPS]",
            problem="U1",
            size=2,
            n=200,
            k=20,
            diversity_metric="GEOMEAN_SEPARATION",
            seed=1,
            budget="single-shot",
            measured_sec=0.0002,
            n_iterations=None,
            quality={"MIN_SEPARATION": 0.4},
            proven_optimal=None,
        ),
    ]

    # --- act ---------------------------------------------
    path = tmp_path / "sub" / "records.jsonl"
    save_records(records, path)
    loaded = load_records(path)

    # --- assert ------------------------------------------
    assert loaded == records


def _record(budget: str, measured_sec: float) -> RunRecord:
    """Build a record with only the fields the tolerance filter reads."""
    return RunRecord("max-div[DEFAULT]", "U1", 20, 20, 2, "MIN_SEPARATION", 0, budget, measured_sec, None, {})


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("time:0.004s", 0.004), ("time:60.0s", 60.0), ("iterations:1280", None), ("single-shot", None)],
)
def test_budget_sec_reads_wall_clock_tags_only(tag: str, expected: float | None):
    """Only a wall-clock tag names a budget in seconds; `budget_tag` writes the tag `budget_sec` reads."""
    # --- act / assert -----------------
    assert budget_sec(tag) == expected
    if expected is not None:
        assert budget_tag(expected) == tag


@pytest.mark.parametrize(("tag", "expected"), [("iterations:1280", 1280), ("time:0.004s", None), ("single-shot", None)])
def test_iteration_count_reads_iteration_tags_only(tag: str, expected: int | None):
    """Only an iteration tag names a count; `iteration_tag` writes the tag `iteration_count` reads."""
    # --- act / assert -----------------
    assert iteration_count(tag) == expected
    if expected is not None:
        assert iteration_tag(expected) == tag


def test_within_budget_tolerance_drops_solves_that_missed_their_budget():
    """A solve is kept unless it overshoots its budget by more than the tolerance; untimed budgets are always kept."""
    # --- arrange ----------------------
    records = [
        _record("time:1.0s", 1.05),  # within
        _record("time:1.0s", 0.5),  # finished early: a measurement under its budget
        _record("time:1.0s", 1.2),  # past the budget
        _record("time:0.001s", 0.03),  # set-up alone exceeded the budget
        _record("iterations:100", 5.0),
        _record("single-shot", 5.0),
    ]

    # --- act --------------------------
    kept = within_budget_tolerance(records, tolerance=0.1)

    # --- assert -----------------------
    assert [(r.budget, r.measured_sec) for r in kept] == [
        ("time:1.0s", 1.05),
        ("time:1.0s", 0.5),
        ("iterations:100", 5.0),
        ("single-shot", 5.0),
    ]
