import pytest

from benchmarks.solver_scaling import quality_stage
from benchmarks.solver_scaling.configs import resolve
from benchmarks.solver_scaling.grid import DEFAULT_SEED, REFERENCE_BUDGET_SEC
from benchmarks.solver_scaling.quality_stage import (
    QUALITY_SEEDS,
    best_known_pool,
    compute_q_random,
    median_qualities,
    quality_limits,
    seeds_for,
    time_limits,
    tool_quality_limits,
)
from benchmarks.solver_scaling.records import ScalingRunRecord, save_scaling_records


def _record(n, *, tool="rdkit", config="default", seed=DEFAULT_SEED, completed=True, min_separation=0.2, sec=1.0):
    """Build one quality-stage run record; overrides drive the pool and verdict cases."""
    return ScalingRunRecord(
        tool, config, n, n // 10, seed, REFERENCE_BUDGET_SEC, completed, None, sec, 1000, min_separation
    )


def test_seeds_for_seed_influenceable_and_fixed() -> None:
    """A configuration whose seed can influence the result runs every quality seed; the rest keep the fixed seed."""
    # --- act / assert -----------------
    assert seeds_for(resolve("fpsample", "vanilla")) == QUALITY_SEEDS
    assert seeds_for(resolve("code-fdm", "default")) == (DEFAULT_SEED,)


def test_time_limits_takes_each_configs_largest_passing_size(tmp_path) -> None:
    """The per-configuration bound is the largest passing size, and over-budget runs do not count."""
    # --- arrange ----------------------
    path = tmp_path / "time.jsonl"
    save_scaling_records(
        [
            _record(20),
            _record(50),
            _record(100, sec=REFERENCE_BUDGET_SEC + 5.0),  # the run completed, but over budget, so it must not count
            _record(20, tool="dppy"),
        ],
        path,
    )

    # --- act --------------------------
    limits = time_limits(path)

    # --- assert -----------------------
    assert limits == {("rdkit", "default"): 50, ("dppy", "default"): 20}


def test_sweep_runs_every_seed_and_skips_recorded_runs(monkeypatch, tmp_path) -> None:
    """The sweep covers size x seed, resumes past recorded runs, and continues past a failure."""
    # --- arrange ----------------------
    calls: list[tuple[int, int]] = []

    def fake_run(tool, config, n, k, seed, budget_sec):
        calls.append((n, seed))
        return _record(n, tool="fpsample", config="vanilla", seed=seed, completed=(n != 50))

    monkeypatch.setattr(quality_stage, "run_measurement", fake_run)
    already = _record(20, tool="fpsample", config="vanilla", seed=1)
    done = {("fpsample", "vanilla", 20, 1): already}

    # --- act --------------------------
    quality_stage._sweep(resolve("fpsample", "vanilla"), done, tmp_path / "runs.jsonl", 100, REFERENCE_BUDGET_SEC)

    # --- assert -----------------------
    # n=20 seed=1 was recorded, so it is not re-run; the n=50 failures do not stop the sweep
    expected = [(n, seed) for n in (20, 50, 100) for seed in QUALITY_SEEDS if (n, seed) != (20, 1)]
    assert calls == expected


def test_compute_q_random_is_deterministic_and_resumes(monkeypatch, tmp_path) -> None:
    """Q_random reproduces across calls, and a persisted size is not recomputed."""
    # --- arrange ----------------------
    path = tmp_path / "q_random.json"

    # --- act --------------------------
    first = compute_q_random([20], path)
    monkeypatch.setattr(quality_stage, "_q_random", lambda n: pytest.fail("recomputed a persisted size"))
    second = compute_q_random([20], path)

    # --- assert -----------------------
    assert first == second
    assert 0.0 < first[20] < 1.0


def test_best_known_pool_takes_per_seed_maximum_over_both_stages() -> None:
    """The pool keeps the best completed quality per size, across quality seeds and extended runs."""
    # --- arrange ----------------------
    quality = [
        _record(20, tool="fpsample", config="vanilla", seed=1, min_separation=0.30),
        _record(20, tool="fpsample", config="vanilla", seed=2, min_separation=0.10),
        _record(20, completed=False, min_separation=None),
    ]
    extended = [_record(20, min_separation=0.25), _record(50, min_separation=0.40)]

    # --- act / assert -----------------
    assert best_known_pool(quality, extended) == {20: 0.30, 50: 0.40}


def test_quality_limits_judges_the_median_against_the_pooled_threshold() -> None:
    """One seed's high quality raises the shared threshold but cannot make its own configuration pass."""
    # --- arrange ----------------------
    # fpsample's seed 1 sets Q_best_known = 1.0, so the threshold is 0.1*0.0 + 0.9*1.0 = 0.9;
    # its own median over seeds (0.5) fails it, while rdkit's single run (0.95) passes
    quality = [
        _record(20, tool="fpsample", config="vanilla", seed=1, min_separation=1.0),
        _record(20, tool="fpsample", config="vanilla", seed=2, min_separation=0.5),
        _record(20, tool="fpsample", config="vanilla", seed=3, min_separation=0.4),
        _record(20, min_separation=0.95),
    ]

    # --- act --------------------------
    limits = quality_limits(quality, [], {20: 0.0}, gap_closure=0.9)

    # --- assert -----------------------
    assert limits == {"fpsample/vanilla": None, "rdkit/default": 20}


def test_quality_limits_stops_at_the_first_failing_size() -> None:
    """A size that misses the threshold ends the passing range; a larger passing size is not reported."""
    # --- arrange ----------------------
    # the reference config sets Q_best_known = 1.0 at every size; rdkit passes at 20 and 100
    # but misses at 50, so its limit is 20 — the pass at 100 must not lift it
    quality = [
        _record(20, min_separation=1.0),
        _record(50, min_separation=0.5),
        _record(100, min_separation=1.0),
        _record(20, tool="fpsample", config="vanilla", min_separation=1.0),
        _record(50, tool="fpsample", config="vanilla", min_separation=1.0),
        _record(100, tool="fpsample", config="vanilla", min_separation=1.0),
    ]

    # --- act --------------------------
    limits = quality_limits(quality, [], {20: 0.0, 50: 0.0, 100: 0.0}, gap_closure=0.9)

    # --- assert -----------------------
    assert limits == {"rdkit/default": 20, "fpsample/vanilla": 100}


def test_quality_limits_judges_each_fraction_separately() -> None:
    """A configuration closing 60% of the gap passes the 0.5 fraction but not the 0.9 fraction."""
    # --- arrange ----------------------
    quality = [
        _record(20, min_separation=0.6),
        _record(20, tool="fpsample", config="vanilla", min_separation=1.0),
    ]
    q_random = {20: 0.0}

    # --- act / assert -----------------
    assert quality_limits(quality, [], q_random, gap_closure=0.5)["rdkit/default"] == 20
    assert quality_limits(quality, [], q_random, gap_closure=0.9)["rdkit/default"] is None


def test_tool_quality_limits_judges_the_best_configuration_per_size() -> None:
    """Configurations covering each other's failing sizes extend the tool's range beyond either one's."""
    # --- arrange ----------------------
    # fpsample sets Q_best_known = 1.0 at every size; rdkit's configs each fail one size
    # (default at 50, alt at 20 and 100) but their per-size best passes everywhere
    quality = [
        _record(20, min_separation=1.0),
        _record(50, min_separation=0.5),
        _record(100, min_separation=1.0),
        _record(20, config="alt", min_separation=0.5),
        _record(50, config="alt", min_separation=1.0),
        _record(100, config="alt", min_separation=0.5),
        _record(20, tool="fpsample", config="vanilla", min_separation=1.0),
        _record(50, tool="fpsample", config="vanilla", min_separation=1.0),
        _record(100, tool="fpsample", config="vanilla", min_separation=1.0),
    ]
    q_random = {20: 0.0, 50: 0.0, 100: 0.0}

    # --- act --------------------------
    tool_limits = tool_quality_limits(quality, [], q_random, gap_closure=0.9)
    config_limits = quality_limits(quality, [], q_random, gap_closure=0.9)

    # --- assert -----------------------
    assert tool_limits == {"rdkit": 100, "fpsample": 100}
    assert config_limits == {"rdkit/default": 20, "rdkit/alt": None, "fpsample/vanilla": 100}


def test_median_qualities_ignores_failed_runs() -> None:
    """A failed seed contributes nothing; the median covers the completed seeds only."""
    # --- arrange ----------------------
    records = [
        _record(20, seed=1, min_separation=0.2),
        _record(20, seed=2, min_separation=0.4),
        _record(20, seed=3, completed=False, min_separation=None),
    ]

    # --- act / assert -----------------
    assert median_qualities(records) == {("rdkit", "default"): {20: pytest.approx(0.3)}}


def test_run_quality_stage_skips_a_config_without_a_passing_time_size(monkeypatch, tmp_path, capsys) -> None:
    """A configuration absent from the time-stage limits is skipped, not run."""
    # --- arrange ----------------------
    monkeypatch.setattr(quality_stage, "time_limits", lambda: {})
    monkeypatch.setattr(
        quality_stage, "run_measurement", lambda *args: pytest.fail("ran a config with no passing size")
    )

    # --- act --------------------------
    quality_stage.run_quality_stage([resolve("rdkit", "default")], tmp_path / "runs.jsonl")

    # --- assert -----------------------
    assert "skipped" in capsys.readouterr().out
