from pathlib import Path

from benchmarks.common import load_records
from benchmarks.tier2.full import EVALUATED_DIVERSITY_METRICS, run_maxdiv_constrained, run_maxdiv_unconstrained


def test_run_maxdiv_unconstrained_smoke(tmp_path: Path):
    # --- arrange -----------------------------------------
    out_path = tmp_path / "maxdiv_unconstrained.jsonl"

    # --- act ---------------------------------------------
    records = run_maxdiv_unconstrained(
        problems=("U1",),
        sizes=(1,),
        time_budgets_sec=[0.001, 0.002],
        seeds=(0,),
        out_path=out_path,
    )

    # --- assert ------------------------------------------
    assert load_records(out_path) == records
    assert {r.tool for r in records} == {"max-div[DEFAULT]"}
    assert {r.diversity_metric for r in records} == {m.name for m in EVALUATED_DIVERSITY_METRICS}
    assert {r.budget for r in records} == {"time:0.001s", "time:0.002s"}


def test_run_maxdiv_constrained_smoke(tmp_path: Path):
    # --- arrange -----------------------------------------
    out_path = tmp_path / "maxdiv_constrained.jsonl"

    # --- act ---------------------------------------------
    records = run_maxdiv_constrained(
        problems=("C1",),
        sizes=(1,),
        time_budgets_sec=[0.001],
        seeds=(0,),
        out_path=out_path,
    )

    # --- assert ------------------------------------------
    assert load_records(out_path) == records
    assert {r.tool for r in records} == {"max-div[DEFAULT]"}
    assert {r.diversity_metric for r in records} == {"MIN_SEPARATION"}
    assert all(r.n_constraints > 0 for r in records)
