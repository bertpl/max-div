from pathlib import Path

import pytest

from benchmarks.common.records import RunRecord, save_records
from benchmarks.mdplib.best_known import BestKnown, load_best_known
from benchmarks.tier3 import report


def _record(instance: str, k: int, tool: str, budget: str, min_separation: float, seed: int = 0) -> RunRecord:
    """Build a minimal tier-3 record (problem = instance file name, size = k)."""
    return RunRecord(
        tool=tool,
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


def _row(instance: str, best_known: float, family: str = "Ran", n: int = 500, k: int = 50) -> BestKnown:
    """Build one best-known row."""
    return BestKnown(family, instance, n, k, best_known, best_known, "RMGD2010", False)


def test_load_best_known_vendored_table():
    """The table covers every published pairing, carries provenance, and improves on the 2010 values where later work did."""
    # --- act --------------------------
    rows = load_best_known()

    # --- assert -----------------------
    assert len(rows) == 195  # 75 Glover pairings + 60 Geo + 60 Ran
    assert {r.family for r in rows} == {"Geo", "Glover", "Ran"}
    geo1 = next(r for r in rows if r.instance == "Geo 100 1.txt")
    assert (geo1.n, geo1.k, geo1.best_known, geo1.source, geo1.proven_optimal) == (100, 10, 89.3701, "RMGD2010", True)
    ran = next(r for r in rows if r.instance == "Ran 500 3.txt")
    assert (ran.best_known, ran.best_known_2010, ran.source) == (56.0, 55.0, "PHG2011")
    assert all(r.best_known >= r.best_known_2010 for r in rows)


def test_gap_records_re_express_quality_as_gap_percent():
    """Records of known pairings become gap records; others are dropped."""
    # --- arrange ----------------------
    references = {("Ran 500 1.txt", 50): _row("Ran 500 1.txt", 50.0)}
    records = [_record("Ran 500 1.txt", 50, "max-div[DEFAULT]", "time:1.0s", 45.0), _record("other.txt", 50, "x", "s", 1.0)]

    # --- act --------------------------
    gaps = report.gap_records(records, references)

    # --- assert -----------------------
    assert len(gaps) == 1
    assert gaps[0].quality == {report.GAP_METRIC: pytest.approx(10.0)}


def test_classify_matched_exceeded_below():
    """Values above, at, and below the reference classify as exceeded, matched, and below."""
    # --- act / assert -----------------
    assert report.classify(55.0, 55.0) == "matched"
    assert report.classify(56.0, 55.0) == "exceeded"
    assert report.classify(54.0, 55.0) == "below"
    assert report.classify(None, 55.0) is None


def test_count_table_uses_the_best_over_seeds_and_series_at_t_max():
    """The counts take the best value over seeds and both series at T_max, and nothing from other budgets."""
    # --- arrange ----------------------
    rows = [_row("Ran 500 1.txt", 54.0), _row("Ran 500 2.txt", 55.0), _row("Ran 500 3.txt", 55.0)]
    records = [
        _record("Ran 500 1.txt", 50, "max-div[DEFAULT]", "time:60.0s", 55.0),  # exceeded
        _record("Ran 500 2.txt", 50, "max-div[DEFAULT]", "time:60.0s", 54.0),
        _record("Ran 500 2.txt", 50, "max-div[DEFAULT, 12 workers]", "time:60.0s", 55.0),  # matched by the other series
        _record("Ran 500 3.txt", 50, "max-div[DEFAULT]", "time:60.0s", 54.0),  # below
        _record("Ran 500 3.txt", 50, "max-div[DEFAULT]", "time:1.0s", 56.0),  # not at T_max: ignored
    ]

    # --- act --------------------------
    table = report.build_count_table(records, report.group_pairings(rows))

    # --- assert -----------------------
    assert "| Ran | 500 | 50 | 3 | 1 | 1 | 1 |" in table


def test_glover_sentence_counts_reached_pairings():
    """The Glover sentence counts matched and exceeded pairings among those measured."""
    # --- arrange ----------------------
    rows = [_row("Glover (n 10) 1.txt", 10.0, "Glover", 10, 2), _row("Glover (n 10) 2.txt", 10.0, "Glover", 10, 2)]
    records = [_record("Glover (n 10) 1.txt", 2, "max-div[DEFAULT]", "time:0.001s", 10.0)]

    # --- act / assert -----------------
    assert report.glover_sentence(records, rows).startswith("On the Glover set (n ≤ 30), max-div reaches the published value on 1 of the 1 measured")


def test_main_emits_charts_and_tables(tmp_path: Path):
    """The report writes one chart per charted group with data, the chart lists, and every table."""
    # --- arrange ----------------------
    data_dir, records_dir, docs_dir = tmp_path / "data", tmp_path / "records", tmp_path / "docs"
    rows = load_best_known()
    ran_500_50 = [r for r in rows if r.family == "Ran" and r.n == 500 and r.k == 50]
    maxdiv = [_record(r.instance, 50, "max-div[DEFAULT]", f"time:{b}s", r.best_known - 1) for r in ran_500_50 for b in (1.0, 60.0)]
    entrants = [_record(r.instance, 50, "qc-selector[MaxMin]", "single-shot", r.best_known - 5) for r in ran_500_50]
    save_records(maxdiv, records_dir / report.MAXDIV_FILE)
    save_records(entrants, data_dir / report.ENTRANT_FILE)

    # --- act --------------------------
    report.main(records_dir=records_dir, docs_dir=docs_dir, data_dir=data_dir)

    # --- assert -----------------------
    assert (docs_dir / "images" / "tier3_ran_500_50.webp").exists()
    assert "tier3_ran_500_50" in (docs_dir / "results" / "tier3_charts_ran.md").read_text()
    assert (docs_dir / "results" / "tier3_charts_geo.md").read_text() == "\n"
    assert "| Ran | 500 | 50 | 10 |" in (docs_dir / "results" / "tier3_gaps.md").read_text()
    assert "qc-selector[MaxMin]" in (docs_dir / "results" / "tier3_entrants.md").read_text()
    assert "| Ran | Ran 500 3 | 500 | 50 | 56 | 55 | PHG2011 |" in (docs_dir / "results" / "tier3_best_known.md").read_text()
