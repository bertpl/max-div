from pathlib import Path

from benchmarks.common import load_records
from benchmarks.tier1.full import run_incumbent_maxdiv, run_maxmin_maxdiv


def test_run_maxmin_maxdiv_smoke(tmp_path: Path):
    # --- arrange -----------------------------------------
    out_path = tmp_path / "maxmin_records.jsonl"

    # --- act ---------------------------------------------
    run_maxmin_maxdiv(
        problems=("U1",),
        sizes=(1,),
        time_budgets_sec=[0.001],
        seeds=(0,),
        out_path=out_path,
    )

    # --- assert ------------------------------------------
    records = load_records(out_path)
    assert len(records) == 1
    assert records[0].tool == "max-div[DEFAULT]"
    assert records[0].diversity_metric == "MIN_SEPARATION"


def test_run_incumbent_maxdiv_smoke(tmp_path: Path):
    # --- arrange -----------------------------------------
    out_path = tmp_path / "incumbent_records.jsonl"

    # --- act ---------------------------------------------
    run_incumbent_maxdiv(
        cases=(("U3", 1, 1.0),),
        time_budgets_sec=[0.001],
        seeds=(0,),
        out_path=out_path,
    )

    # --- assert ------------------------------------------
    records = load_records(out_path)
    assert len(records) == 1
    assert records[0].diversity_metric == "GEOMEAN_SEPARATION"
