from benchmarks.common.records import RunRecord
from benchmarks.mdplib.best_known import BestKnown, load_best_known
from benchmarks.tier3.report import (
    best_of_seeds,
    best_overall,
    build_geo_gap_table,
    build_match_table,
    gap_pct,
)


def _record(instance: str, k: int, budget: str, min_separation: float, seed: int = 0) -> RunRecord:
    """Minimal tier-3 budget-series record (problem = instance file name, size = k)."""
    return RunRecord(
        tool="max-div[SMART]",
        problem=instance,
        size=k,
        n=500,
        k=k,
        diversity_metric="MIN_SEPARATION",
        seed=seed,
        budget=budget,
        measured_sec=1.0,
        n_iterations=None,
        quality={"MIN_SEPARATION": min_separation},
    )


def test_load_best_known_vendored_table():
    # --- act ---------------------------------------------
    rows = load_best_known()

    # --- assert ------------------------------------------
    assert len(rows) == 195  # 75 Glover pairings + 60 Geo + 60 Ran
    assert {r.family for r in rows} == {"Geo", "Glover", "Ran"}
    geo1 = next(r for r in rows if r.instance == "Geo 100 1.txt")
    assert (geo1.n, geo1.k, geo1.best_known) == (100, 10, 89.3701)


def test_best_of_seeds_and_gap():
    # --- arrange -----------------------------------------
    records = [
        _record("Geo 500 1.txt", 50, "time:1.024s", 90.0, seed=0),
        _record("Geo 500 1.txt", 50, "time:1.024s", 95.0, seed=1),
    ]

    # --- act / assert ------------------------------------
    assert best_of_seeds(records, 1.024) == 95.0
    assert best_of_seeds(records, 16.384) is None
    assert gap_pct(95.0, 100.0) == 5.0


def test_best_overall_spans_seeds_and_budgets():
    # a deeper budget on the same instance must win over a lower rung's best seed
    # --- arrange -----------------------------------------
    records = [
        _record("Ran 500 1.txt", 50, "time:1.024s", 54.0, seed=0),
        _record("Ran 500 1.txt", 50, "time:16.384s", 55.0, seed=1),
    ]

    # --- act / assert ------------------------------------
    assert best_overall(records) == 55.0


def test_build_geo_gap_table_groups_by_nk():
    # --- arrange -----------------------------------------
    rows = [BestKnown(family="Geo", instance="Geo 500 1.txt", n=500, k=50, best_known=100.0)]
    records = [_record("Geo 500 1.txt", 50, f"time:{b}s", 95.0) for b in (0.128, 1.024, 16.384)]

    # --- act ---------------------------------------------
    table = build_geo_gap_table(rows, records)

    # --- assert ------------------------------------------
    assert "| 500 | 50 | 1 | 5.0% / 5.0% | 5.0% / 5.0% | 5.0% / 5.0% |" in table


def test_build_match_table_classifies_exceeded_matched_below():
    # --- arrange -----------------------------------------
    rows = [
        BestKnown(family="Ran", instance="Ran 500 1.txt", n=500, k=50, best_known=54.0),
        BestKnown(family="Ran", instance="Ran 500 2.txt", n=500, k=50, best_known=55.0),
        BestKnown(family="Ran", instance="Ran 500 3.txt", n=500, k=50, best_known=55.0),
    ]
    records = [
        _record("Ran 500 1.txt", 50, "time:16.384s", 55.0),  # exceeded (deep budget)
        _record("Ran 500 2.txt", 50, "time:1.024s", 55.0),  # matched
        _record("Ran 500 3.txt", 50, "time:1.024s", 54.0),  # below
    ]

    # --- act ---------------------------------------------
    table = build_match_table(rows, records, "Ran")

    # --- assert ------------------------------------------
    assert "| 500 | 50 | 3 | 1 | 1 | 1 |" in table
