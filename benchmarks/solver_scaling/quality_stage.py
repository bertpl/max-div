"""Quality stage: every configuration re-runs its passing sizes at the reference budget, once per seed.

For each configuration: run it at every grid size up to its own largest n within the time budget
(`time_stage`), once per quality seed — `QUALITY_SEEDS` for a stochastic configuration, the
protocol's fixed seed for a deterministic one.

A configuration's verdict at a size compares its median quality over seeds against the threshold
`(1 - gap_closure) * Q_random + gap_closure * Q_best_known`, once per value in
`GAP_CLOSURE_FRACTIONS`; `gap_closure`, wherever it appears below, is the fraction of the
random-to-best gap that must be closed. The best-known pool combines the extended runs
(`best_known_stage`) with every per-seed quality run: one seed's high quality raises the
threshold for every configuration, but each configuration is judged on its median, so that seed
alone cannot make its own configuration pass. Per fraction, the reported limit is the largest n
up to which every judged size passes — a failing size ends the range (see `quality_limits`).
The measurement protocol's section IV.D carries the full rationale.

`Q_random` — the per-size median quality of `N_RANDOM_DRAWS` random selections — is computed
in-process (no solver involved) and persisted beside the run records.

Records are appended after every run to the tracked data file, so an interrupted stage resumes by
rerunning: already-recorded runs are skipped.

Usage: python -m benchmarks.solver_scaling.quality_stage [tool | tool/config ...]  # default: all
"""

import json
import statistics
import sys
from pathlib import Path

import numpy as np

from .best_known_stage import DATA_PATH as BEST_KNOWN_DATA_PATH
from .configs import CONFIGS, ScalingConfig
from .grid import DEFAULT_SEED, GRID_MIN, REFERENCE_BUDGET_SEC, WARMUP_BUDGET_SEC, size_grid
from .records import ScalingRunRecord, append_scaling_record, load_scaling_records
from .runner import run_measurement
from .time_stage import DATA_PATH as TIME_DATA_PATH
from .time_stage import passes_time

DATA_PATH = Path(__file__).resolve().parent / "data" / "quality_stage.jsonl"
Q_RANDOM_PATH = Path(__file__).resolve().parent / "data" / "q_random.json"

QUALITY_SEEDS = (1, 2, 3, 4, 5)
N_RANDOM_DRAWS = 31  # each Q_random value is the median over this many random selections
GAP_CLOSURE_FRACTIONS = (0.5, 0.9)  # least strict first, matching the capability columns' tightening order


def seeds_for(config: ScalingConfig) -> tuple[int, ...]:
    """Return the seeds the quality stage runs a configuration under."""
    return QUALITY_SEEDS if config.stochastic else (DEFAULT_SEED,)


def time_limits(time_data_path: Path = TIME_DATA_PATH) -> dict[tuple[str, str], int]:
    """Return, per (tool, config), the largest n that passed within the time budget.

    A configuration with no passing size is absent from the result.

    Raises:
        FileNotFoundError: If the time stage has not run yet — the quality stage needs its
            result to bound each configuration's grid.
    """
    limits: dict[tuple[str, str], int] = {}
    for record in load_scaling_records(time_data_path):
        if passes_time(record):
            key = (record.tool, record.config)
            limits[key] = max(limits.get(key, 0), record.n)
    return limits


# ==================================================================================================
#  Solver runs
# ==================================================================================================
def run_quality_stage(
    configs: list[ScalingConfig] | None = None,
    data_path: Path = DATA_PATH,
    budget_sec: float = REFERENCE_BUDGET_SEC,
) -> None:
    """Run every configuration's quality runs; a configuration with no passing time-stage size is skipped."""
    configs = list(CONFIGS) if configs is None else configs
    limits = time_limits()
    done = {(r.tool, r.config, r.n, r.seed): r for r in (load_scaling_records(data_path) if data_path.exists() else [])}
    for config in configs:
        n_bound = limits.get((config.tool, config.name))
        if n_bound is None:
            print(f"{config.tool}/{config.name}: skipped — no size passed within the time budget")
            continue
        _sweep(config, done, data_path, n_bound, budget_sec)


def _sweep(config: ScalingConfig, done: dict, data_path: Path, n_bound: int, budget_sec: float) -> None:
    """Run one configuration over its bounded grid, every seed at every size.

    Unlike the time and best-known sweeps there is no stopping rule: the grid is already bounded
    by the configuration's proven sizes, so a failed run is recorded and the sweep continues. A
    configuration with no recorded runs first gets one discarded warm-up run at the smallest grid
    size (see `WARMUP_BUDGET_SEC`).
    """
    if not any(key[0] == config.tool and key[1] == config.name for key in done):
        run_measurement(config.tool, config.name, GRID_MIN, GRID_MIN // 10, DEFAULT_SEED, WARMUP_BUDGET_SEC)
    for n in size_grid(n_bound):
        for seed in seeds_for(config):
            record = done.get((config.tool, config.name, n, seed))
            if record is None:
                record = run_measurement(config.tool, config.name, n, n // 10, seed, budget_sec)
                append_scaling_record(record, data_path)
            print(f"  {config.tool}/{config.name} n={n} seed={seed}: {'ok' if record.completed else record.reason}")


# ==================================================================================================
#  Q_random
# ==================================================================================================
def compute_q_random(sizes: list[int], path: Path = Q_RANDOM_PATH) -> dict[int, float]:
    """Return the per-size median quality of `N_RANDOM_DRAWS` random selections, computing missing sizes.

    The result is persisted after every computed size, so an interrupted computation resumes.
    """
    values: dict[str, float] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for n in sizes:
        if str(n) in values:
            continue
        values[str(n)] = _q_random(n)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(sorted(values.items(), key=lambda kv: int(kv[0]))), indent=2) + "\n")
        print(f"Q_random n={n}: {values[str(n)]:.4f}")
    return {int(n): value for n, value in values.items()}


def _q_random(n: int) -> float:
    """Compute one size's Q_random on the reference problem, drawn with the protocol's fixed seed."""
    from benchmarks.common.problems import build_problem
    from benchmarks.common.quality import min_separation_nn
    from max_div.metrics import DiversityMetric

    problem = build_problem("U1", n=n, diversity_metric=DiversityMetric.MIN_SEPARATION)
    vectors = np.ascontiguousarray(problem.vectors)
    rng = np.random.default_rng(DEFAULT_SEED)
    draws = [min_separation_nn(vectors, rng.choice(n, size=n // 10, replace=False)) for _ in range(N_RANDOM_DRAWS)]
    return statistics.median(draws)


# ==================================================================================================
#  Verdicts
# ==================================================================================================
def best_known_pool(
    quality_records: list[ScalingRunRecord], best_known_records: list[ScalingRunRecord]
) -> dict[int, float]:
    """Return `Q_best_known` per size: the best quality over the extended runs and every per-seed quality run."""
    pool: dict[int, float] = {}
    for record in [*quality_records, *best_known_records]:
        if record.completed and record.min_separation is not None:
            pool[record.n] = max(pool.get(record.n, record.min_separation), record.min_separation)
    return pool


def median_qualities(quality_records: list[ScalingRunRecord]) -> dict[tuple[str, str], dict[int, float]]:
    """Return, per (tool, config) and size, the median quality over that size's completed seeds."""
    grouped: dict[tuple[str, str], dict[int, list[float]]] = {}
    for record in quality_records:
        if record.completed and record.min_separation is not None:
            grouped.setdefault((record.tool, record.config), {}).setdefault(record.n, []).append(record.min_separation)
    return {
        key: {n: statistics.median(draws) for n, draws in sorted(sizes.items())} for key, sizes in grouped.items()
    }


def quality_limits(
    quality_records: list[ScalingRunRecord],
    best_known_records: list[ScalingRunRecord],
    q_random: dict[int, float],
    gap_closure: float,
) -> dict[str, int | None]:
    """Return each configuration's quality-limit size for one gap-closure fraction.

    A failing size ends the passing range even when larger sizes pass again (rationale:
    protocol section IV.D.4).

    Returns:
        `tool/config` -> the largest n up to which every judged size reaches the threshold, or
        None when the smallest judged size already misses it.
        Only configurations with at least one completed quality run appear.
    """
    pool = best_known_pool(quality_records, best_known_records)
    return {
        f"{tool}/{config}": _largest_passing_prefix(medians, q_random, pool, gap_closure)
        for (tool, config), medians in median_qualities(quality_records).items()
    }


def tool_quality_limits(
    quality_records: list[ScalingRunRecord],
    best_known_records: list[ScalingRunRecord],
    q_random: dict[int, float],
    gap_closure: float,
) -> dict[str, int | None]:
    """Return each tool's quality-limit size, judging its best configuration per size.

    At each size the tool's quality is the best per-configuration median — the protocol's
    best-result-across-configurations rule (section III). The passing range still ends at the
    first failing size, so this can reach beyond every single configuration's limit when the
    configurations cover each other's failing sizes.

    Returns:
        tool -> the largest n up to which every judged size reaches the threshold, or None when
        the smallest judged size already misses it.
    """
    pool = best_known_pool(quality_records, best_known_records)
    tool_medians: dict[str, dict[int, float]] = {}
    for (tool, _), medians in median_qualities(quality_records).items():
        merged = tool_medians.setdefault(tool, {})
        for n, median in medians.items():
            merged[n] = max(median, merged.get(n, median))
    return {
        tool: _largest_passing_prefix(medians, q_random, pool, gap_closure) for tool, medians in tool_medians.items()
    }


def _largest_passing_prefix(
    medians: dict[int, float], q_random: dict[int, float], pool: dict[int, float], gap_closure: float
) -> int | None:
    """Return the largest n up to which every judged size's median meets the required gap closure."""
    limit: int | None = None
    for n, median in sorted(medians.items()):
        if median < (1.0 - gap_closure) * q_random[n] + gap_closure * pool[n]:
            break
        limit = n
    return limit


if __name__ == "__main__":
    selected = sys.argv[1:]
    chosen = [c for c in CONFIGS if c.tool in selected or f"{c.tool}/{c.name}" in selected] if selected else list(CONFIGS)
    run_quality_stage(chosen)
    grid_sizes = size_grid(max(time_limits().values()))
    q_random_values = compute_q_random(grid_sizes)
    records = load_scaling_records(DATA_PATH)
    extended = load_scaling_records(BEST_KNOWN_DATA_PATH)
    for gap_closure in GAP_CLOSURE_FRACTIONS:
        for name, limit in quality_limits(records, extended, q_random_values, gap_closure).items():
            print(f"{name}: largest n closing {gap_closure:.0%} of the gap = {limit}")
        for tool, limit in tool_quality_limits(records, extended, q_random_values, gap_closure).items():
            print(f"{tool} (best configuration per size): largest n closing {gap_closure:.0%} of the gap = {limit}")
