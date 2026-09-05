from pathlib import Path

from benchmarks.adapters import FpsampleFPS, RandomBaseline, RdkitMaxMin
from benchmarks.common import load_records
from benchmarks.tier2 import full


def test_enters_reads_the_tools_scaling_time_limit():
    """A tool enters a size when its best configuration's largest n within the time budget covers it."""
    # --- arrange -----------------------------------------
    limits = {("fpsample", "vanilla"): 500_000, ("fpsample", "kdline"): 5_000_000, ("rdkit", "default"): 20_000}

    # --- act / assert ------------------------------------
    assert full.enters(FpsampleFPS(), 1_000_000, limits)
    assert full.enters(RdkitMaxMin(), 20_000, limits)
    assert not full.enters(RdkitMaxMin(), 100_000, limits)
    assert full.enters(RandomBaseline(), 100_000, limits)


def test_run_competitors_skips_tools_outside_their_limit_and_resumes(tmp_path: Path):
    """Entrants outside their scaling limit are not run, and a rerun adds no records."""
    # --- arrange -----------------------------------------
    out_path = tmp_path / "third_party.jsonl"
    limits = {("fpsample", "vanilla"): 100, ("rdkit", "default"): 5_000}

    # --- act ---------------------------------------------
    for _ in range(2):
        records = full.run_competitors(
            sizes=(200,), seeds=(0,), adapters=[FpsampleFPS(), RdkitMaxMin()], limits=limits, out_path=out_path
        )

    # --- assert ------------------------------------------
    assert [r.tool for r in records] == ["RDKit[MaxMinPicker]"]
    assert load_records(out_path) == records


def test_run_maxdiv_writes_both_series(tmp_path: Path):
    # --- act ---------------------------------------------
    records = full.run_maxdiv(
        sizes=(200,), seeds=(0,), single_budgets_sec=[0.001], multi_budgets_sec=[0.2], n_workers=2,
        out_path=tmp_path / "maxdiv.jsonl",
    )

    # --- assert ------------------------------------------
    assert sorted(r.tool for r in records) == ["max-div[DEFAULT, 2 workers]", "max-div[DEFAULT]"]
    assert {r.diversity_metric for r in records} == {"MIN_SEPARATION"}
