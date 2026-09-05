from pathlib import Path

from benchmarks.adapters import FpsampleFPS, RandomBaseline, RdkitMaxMin
from benchmarks.common import load_records
from benchmarks.tier2 import full


def test_runs_at_size_reads_the_tools_scaling_time_limit():
    """A tool enters a size when its best configuration's largest n within the time budget covers it."""
    # --- arrange ----------------------
    limits = {("fpsample", "vanilla"): 500_000, ("fpsample", "kdline"): 5_000_000, ("rdkit", "default"): 20_000}

    # --- act / assert -----------------
    assert full.runs_at_size(FpsampleFPS(), 1_000_000, limits)
    assert full.runs_at_size(RdkitMaxMin(), 20_000, limits)
    assert not full.runs_at_size(RdkitMaxMin(), 100_000, limits)
    assert full.runs_at_size(RandomBaseline(), 100_000, limits)


def test_run_entrants_skips_tools_outside_their_limit_and_resumes(tmp_path: Path):
    """Entrants outside their scaling limit are not run, and a rerun adds no records."""
    # --- arrange ----------------------
    out_path = tmp_path / "third_party.jsonl"
    limits = {("fpsample", "vanilla"): 100, ("rdkit", "default"): 5_000}

    # --- act --------------------------
    for _ in range(2):
        records = full.run_entrants(
            sizes=(200,), seeds=(0,), adapters=[FpsampleFPS(), RdkitMaxMin()], limits=limits, out_path=out_path
        )

    # --- assert -----------------------
    assert [r.tool for r in records] == ["RDKit[MaxMinPicker]"]
    assert load_records(out_path) == records


def test_run_maxdiv_writes_both_series(tmp_path: Path):
    """Both max-div series run at each size, on min separation."""
    # --- act --------------------------
    records = full.run_maxdiv(
        sizes=(200,), seeds=(0,), single_budgets_sec=[0.001], multi_budgets_sec=[0.2], n_workers=2,
        out_path=tmp_path / "maxdiv.jsonl",
    )

    # --- assert -----------------------
    assert sorted(r.tool for r in records) == ["max-div[DEFAULT, 2 workers]", "max-div[DEFAULT]"]
    assert {r.diversity_metric for r in records} == {"MIN_SEPARATION"}
