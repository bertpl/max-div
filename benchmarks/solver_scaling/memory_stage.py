"""Largest-n-within-memory sweep: every configuration walks the size grid under observation windows.

Independent of the time sweep — a run here does not need to complete, it needs to allocate. Each
size runs for up to the reference budget while the runner records the footprint reached; a run
that outlives its window is killed and still yields a footprint. Every footprint feeds the fit
equally; each run also carries a `memory_settled` diagnostic (whether its footprint had stopped
growing by the kill), recorded for later inspection but not used to weight or exclude points. The
sweep walks upward until one of:

* the machine-level memory cap kills a run — the previous size **brackets** the result;
* the solver fails outright — the previous size brackets the result, with the failure disclosed;
* the recorded footprints span a `2x` range and the fitted model explains them — the crossing is
  read off the fit at the cap (`memory_fit` owns the fit and the trust conditions);
* the grid is exhausted — the fit is published with that noted.

A configuration observed spawning worker processes is not measured: the recorded per-process
footprints would miss the workers (the measurement protocol, section III, argues why nothing is
lost).

Records are appended after every run to the tracked data file, so an interrupted sweep resumes
by rerunning: already-recorded runs are skipped.

Usage: python -m benchmarks.solver_scaling.memory_stage [tool | tool/config ...]   # default: all
"""

import json
import sys
from pathlib import Path

from .configs import CONFIGS, ScalingConfig
from .grid import DEFAULT_SEED, GRID_MIN, REFERENCE_BUDGET_SEC, WARMUP_BUDGET_SEC, operational_bound, size_grid
from .memory_fit import FIT_PATH, MemoryFit, trust_conditions_met, fit_series
from .outcome import Outcome, classify
from .records import append_scaling_record, load_scaling_records
from .runner import run_measurement

DATA_PATH = Path(__file__).resolve().parent / "data" / "memory_stage.jsonl"


def run_memory_stage(configs: list[ScalingConfig] | None = None, data_path: Path = DATA_PATH) -> dict[str, MemoryFit]:
    """Run the sweep for the given configurations, persist the fits, and return them per `tool/config`."""
    configs = list(CONFIGS) if configs is None else configs
    done = {(r.tool, r.config, r.n, r.seed): r for r in (load_scaling_records(data_path) if data_path.exists() else [])}
    fits: dict[str, MemoryFit] = {}
    for config in configs:
        fits[f"{config.tool}/{config.name}"] = _sweep(config, done, data_path)
        print(f"{config.tool}/{config.name}: largest n within memory = "
              f"{fits[f'{config.tool}/{config.name}'].max_n}  ({fits[f'{config.tool}/{config.name}'].reason})")
    _write_fits(fits)
    return fits


def _sweep(config: ScalingConfig, done: dict, data_path: Path) -> MemoryFit:
    """Walk one configuration up the grid and return its memory result.

    A configuration with no recorded runs first gets one discarded warm-up run at the smallest
    grid size (see `WARMUP_BUDGET_SEC`).
    """
    if not any(key[0] == config.tool and key[1] == config.name for key in done):
        run_measurement(config.tool, config.name, GRID_MIN, GRID_MIN // 10, DEFAULT_SEED, WARMUP_BUDGET_SEC)
    footprints: dict[int, float] = {}
    last_under_cap: int | None = None
    fit = MemoryFit(None, None, "no runs")
    for n in size_grid(operational_bound()):
        record = done.get((config.tool, config.name, n, DEFAULT_SEED))
        if record is None:
            record = run_measurement(config.tool, config.name, n, n // 10, DEFAULT_SEED, REFERENCE_BUDGET_SEC)
            append_scaling_record(record, data_path)
        if record.spawned_processes:
            return MemoryFit(None, None, "not measured: spawns worker processes")
        outcome = classify(record.completed, record.reason)
        print(f"  {config.tool}/{config.name} n={n}: {outcome.value}"
              f"{'' if record.memory_settled else ' (footprint not settled)'}")
        if outcome is Outcome.MEMORY:
            return MemoryFit(last_under_cap, None, "bracketed: the memory cap was reached at the next size")
        if outcome is Outcome.SCALING_FAILURE:
            return MemoryFit(last_under_cap, None, f"bracketed: fails at the next size (`{record.reason}`)")
        last_under_cap = n
        if record.peak_memory_bytes:
            footprints[n] = max(footprints.get(n, 0.0), float(record.peak_memory_bytes))
            fit = fit_series(footprints)
            if trust_conditions_met(footprints, fit):
                return fit
    return MemoryFit(fit.max_n, fit.coef, fit.reason + "; grid exhausted before the trust conditions held", fit.r2)


def _write_fits(fits: dict[str, MemoryFit]) -> None:
    """Persist the fits as the JSON file the results-page generator reads."""
    FIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIT_PATH.write_text(
        json.dumps(
            {key: {"max_n": f.max_n, "coef": f.coef, "reason": f.reason, "r2": f.r2} for key, f in fits.items()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    selected = sys.argv[1:]
    chosen = [c for c in CONFIGS if c.tool in selected or f"{c.tool}/{c.name}" in selected] if selected else list(CONFIGS)
    run_memory_stage(chosen)
