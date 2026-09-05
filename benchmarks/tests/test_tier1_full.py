import json
from pathlib import Path

from benchmarks.common import load_records
from benchmarks.tier1 import full


def _exact_row(problem: str, objective: str, solver: str, n: int, proven: bool) -> dict:
    """Build one exact-solver row as the drivers write it."""
    return {
        "problem": problem,
        "objective": objective,
        "solver": solver,
        "n": n,
        "k": n // 10,
        "m": 0,
        "optimum": 0.5,
        "proven_optimal": proven,
        "measured_sec": 1.0,
    }


def test_certification_stops_at_the_first_size_not_certified(tmp_path: Path, monkeypatch):
    """A solver stops at its first uncertified size, and every attempt is on file."""
    # --- arrange ----------------------
    calls = []

    def fake_certify(solver_key, problem_name, objective, n):
        calls.append(n)
        return full.CertifiedOptimum(0.5, n < 100, 0.1)

    monkeypatch.setattr(full, "build_problem", lambda name, n, diversity_metric: type("P", (), {"n": n, "k": n // 10, "m": 0})())
    rows: list[dict] = []
    path = tmp_path / "exact.json"

    # --- act --------------------------
    full._certify_increasing_sizes(rows, path, "U1", full.DiversityMetric.MIN_SEPARATION, "scip", fake_certify)

    # --- assert -----------------------
    assert calls == [20, 50, 100]
    assert [r["proven_optimal"] for r in json.loads(path.read_text())] == [True, True, False]


def test_certification_resumes_after_the_rows_on_file(tmp_path: Path, monkeypatch):
    """Recorded sizes are not re-solved; the ascent continues from the first missing size."""
    # --- arrange ----------------------
    calls = []

    def fake_certify(solver_key, problem_name, objective, n):
        calls.append(n)
        return full.CertifiedOptimum(0.5, False, 0.1)

    monkeypatch.setattr(full, "build_problem", lambda name, n, diversity_metric: type("P", (), {"n": n, "k": n // 10, "m": 0})())
    rows = [_exact_row("U1", "MIN_SEPARATION", "scip", 20, True), _exact_row("U1", "MIN_SEPARATION", "scip", 50, True)]

    # --- act --------------------------
    full._certify_increasing_sizes(rows, tmp_path / "exact.json", "U1", full.DiversityMetric.MIN_SEPARATION, "scip", fake_certify)

    # --- assert -----------------------
    assert calls == [100]


def test_certified_sizes_collects_every_size_some_solver_certified():
    """A size counts once any solver certified it, whichever others failed there."""
    # --- arrange ----------------------
    rows = [
        _exact_row("U1", "MIN_SEPARATION", "scip", 20, True),
        _exact_row("U1", "MIN_SEPARATION", "scip", 50, False),
        _exact_row("U1", "MIN_SEPARATION", "ortools-cpsat", 50, True),
        _exact_row("C1", "MEAN_SEPARATION", "ortools-cpsat", 20, True),
    ]

    # --- act / assert -----------------
    assert full.certified_sizes(rows) == {("U1", "MIN_SEPARATION"): [20, 50], ("C1", "MEAN_SEPARATION"): [20]}


def test_run_maxdiv_writes_both_series_per_certified_cell(tmp_path: Path):
    """Both series run on every certified cell, with the protocol's labels, and a rerun adds nothing."""
    # --- arrange ----------------------
    rows = [_exact_row("U1", "MIN_SEPARATION", "scip", 20, True)]

    # --- act --------------------------
    for _ in range(2):
        full.run_maxdiv(
            rows, seeds=(0,), single_budgets_sec=[0.001], multi_budgets_sec=[0.2], n_workers=2, records_dir=tmp_path
        )

    # --- assert -----------------------
    records = load_records(tmp_path / "maxdiv_min_separation.jsonl")
    assert sorted(r.tool for r in records) == ["max-div[DEFAULT, 2 workers]", "max-div[DEFAULT]"]
    assert {r.n for r in records} == {20}


def test_certify_isolated_runs_a_real_certifier_in_a_child_process():
    """The isolated run returns the child's certified optimum for a tiny problem."""
    # --- act --------------------------
    outcome = full.certify_isolated("scip", "U1", "MIN_SEPARATION", 20)

    # --- assert -----------------------
    assert outcome.proven_optimal
    assert outcome.optimum is not None and outcome.optimum > 0
    assert outcome.note == ""


def test_certify_isolated_reports_a_crashed_child_as_not_certified():
    """A child that dies leaves a failed certification with the exit code in its note, not an exception."""
    # --- act --------------------------
    outcome = full.certify_isolated("benchmarks.tests.crash_certifier:crash", "U1", "MIN_SEPARATION", 20)

    # --- assert -----------------------
    assert not outcome.proven_optimal
    assert outcome.optimum is None
    assert outcome.note == "process exited with code 3"
