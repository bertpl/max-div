"""Largest-n-within-memory fits: per configuration, turn recorded footprints into a memory-cap crossing.

No runs happen here — the time stage recorded every run's peak solver-process RSS, and this
reads those back.
Per the measurement protocol, a configuration whose sweep ended by crossing the memory cap is
bracketed directly (its largest completed size is the answer); one that ended any other way is
extrapolated: a constrained least-squares fit of `mem = c0 + c1*n [+ c2*n^2]` over its largest
completed sizes, read off at the cap. Configurations observed spawning worker processes are
excluded from extrapolation — the recorded per-process footprints miss the workers.

The fit is bound-constrained (`c0 >= 0`, `c1 >= 8`, `c2 >= 0`): the `c1 >= 8` lower bound is the
input-array cost — every solver holds at least the n x d float32 vectors, 8 bytes per item at d=2
— which keeps the extrapolation well-posed even when the recorded footprints are dominated by the
child interpreter's fixed cost. Three or more completed sizes admit a quadratic term, but it is
kept only when physically plausible (`_C2_MIN_BYTES`); two sizes fit linear.

Usage: python -m benchmarks.solver_scaling.memory_fit
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from .configs import CONFIGS
from .grid import MEMORY_CAP_BYTES, operational_bound, size_grid
from .outcome import Outcome, classify
from .records import ScalingRunRecord, load_scaling_records
from .time_stage import DATA_PATH

# Only a configuration's largest completed sizes feed the fit: the smallest sizes sit below the
# interpreter's fixed footprint and would drag the growth term toward zero.
N_FIT_SIZES = 5
_INPUT_MIN_BYTES = 8.0  # 4 bytes x d=2: the raw float32 vectors, the linear coefficient's lower bound

# Physical-plausibility threshold for the fitted quadratic coefficient: the smallest real
# quadratically growing structure is one byte per k x n entry, i.e. 0.1 bytes per n^2 at k = n/10.
# A fitted c2 below this cannot be an allocation and is measurement noise amplified by the long
# extrapolation from time-limited (hence small-n, small-RSS) runs to the cap — refit linear.
_C2_MIN_BYTES = 0.1
FIT_PATH = Path(__file__).resolve().parent / "data" / "memory_fits.json"


@dataclass(frozen=True)
class MemoryFit:
    """One configuration's memory result: the largest n within the cap, and how it was found.

    `coef` and `r2` are set only when the value comes from a fit — a bracketed or single-size
    result has neither.
    """

    max_n: int | None
    coef: tuple[float, ...] | None
    reason: str
    r2: float | None = None


def fit_memory_limit(sizes_peaks: dict[int, float], terminal: Outcome, spawned: bool = False) -> MemoryFit:
    """Return the largest-n-within-memory result for one configuration.

    Args:
        sizes_peaks: per completed size, the peak memory footprint observed there.
        terminal: the outcome that ended the size sweep — `MEMORY` brackets directly, any other
            outcome extrapolates from the recorded peaks.
        spawned: whether the configuration was ever observed with worker processes. Bracketing
            still applies (the cap kill is machine-level, so a bracket is honest for any process
            tree), but extrapolation does not: the per-process footprints miss the workers. No
            information is lost — worker processes only ever add memory, so a solver's
            memory-bound size is reached by its single-process configurations.
    """
    sizes = sorted(sizes_peaks)
    if not sizes:
        return MemoryFit(None, None, "no completed runs")
    if terminal is Outcome.MEMORY:
        return MemoryFit(sizes[-1], None, "bracketed: the memory cap was reached at the next size")
    if spawned:
        return MemoryFit(None, None, "excluded from extrapolation: spawns worker processes")
    fit_sizes = sizes[-N_FIT_SIZES:]
    if len(fit_sizes) == 1:
        return MemoryFit(fit_sizes[0], None, "single completed size; no growth term to extrapolate")
    ns = np.asarray(fit_sizes, dtype=np.float64)
    peaks = np.asarray([sizes_peaks[n] for n in fit_sizes], dtype=np.float64)
    model = "linear"
    coef = _fit_linear(ns, peaks)
    if len(fit_sizes) >= 3:
        quadratic = _fit_quadratic(ns, peaks)
        if quadratic[2] >= _C2_MIN_BYTES:
            model, coef = "quadratic", quadratic
    return MemoryFit(_crossing(coef), coef, f"{model} fit over {len(fit_sizes)} sizes", _r_squared(ns, peaks, coef))


def _r_squared(ns: np.ndarray, peaks: np.ndarray, coef: tuple[float, ...]) -> float:
    """Return the coefficient of determination of the fitted model over the fit window."""
    predicted = sum(c * ns**p for p, c in enumerate(coef))
    ss_res = float(np.sum((peaks - predicted) ** 2))
    ss_tot = float(np.sum((peaks - peaks.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 1.0


def fit_all(records: list[ScalingRunRecord]) -> dict[str, MemoryFit]:
    """Fit every smoke configuration and key the results by ``tool/config``."""
    fits: dict[str, MemoryFit] = {}
    for config in CONFIGS:
        rows = [r for r in records if r.tool == config.tool and r.config == config.name]
        spawned = any(row.spawned_processes for row in rows)
        fits[f"{config.tool}/{config.name}"] = fit_memory_limit(_peaks_by_size(rows), _terminal_outcome(rows), spawned)
    return fits


def _peaks_by_size(rows: list[ScalingRunRecord]) -> dict[int, float]:
    """Return, per size with a completed run carrying a footprint, the largest one recorded there."""
    per_size: dict[int, float] = {}
    for row in rows:
        if row.completed and row.peak_memory_bytes:
            per_size[row.n] = max(per_size.get(row.n, 0.0), float(row.peak_memory_bytes))
    return per_size


def _terminal_outcome(rows: list[ScalingRunRecord]) -> Outcome:
    """Return the outcome at the largest attempted size — what ended the sweep.

    The time stage runs one measurement per size, so the largest attempted size carries a single
    outcome; SUCCESS when no run was recorded (nothing bounded the sweep).
    """
    if not rows:
        return Outcome.SUCCESS
    largest = max(rows, key=lambda row: row.n)
    return classify(largest.completed, largest.reason)


def _fit_quadratic(ns: np.ndarray, peaks: np.ndarray) -> tuple[float, float, float]:
    """Fit `peaks = c0 + c1*n + c2*n^2` with `c0 >= 0`, `c1 >= 8`, `c2 >= 0`."""
    design = np.column_stack([np.ones_like(ns), ns, ns**2])
    result = lsq_linear(design, peaks, bounds=([0.0, _INPUT_MIN_BYTES, 0.0], np.inf))
    return float(result.x[0]), float(result.x[1]), float(result.x[2])


def _fit_linear(ns: np.ndarray, peaks: np.ndarray) -> tuple[float, float]:
    """Fit `peaks = c0 + c1*n` with `c0 >= 0`, `c1 >= 8`."""
    design = np.column_stack([np.ones_like(ns), ns])
    result = lsq_linear(design, peaks, bounds=([0.0, _INPUT_MIN_BYTES], np.inf))
    return float(result.x[0]), float(result.x[1])


def _crossing(coef: tuple[float, ...]) -> int | None:
    """Return the largest grid size whose predicted peak stays within the memory cap."""
    powers = np.arange(len(coef))
    grid = size_grid(operational_bound())
    predicted = [sum(c * n**p for c, p in zip(coef, powers, strict=True)) for n in grid]
    passing = [n for n, peak in zip(grid, predicted, strict=True) if peak <= MEMORY_CAP_BYTES]
    return passing[-1] if passing else None


if __name__ == "__main__":
    all_fits = fit_all(load_scaling_records(DATA_PATH))
    FIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIT_PATH.write_text(
        json.dumps(
            {key: {"max_n": f.max_n, "coef": f.coef, "reason": f.reason, "r2": f.r2} for key, f in all_fits.items()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for key, fit in all_fits.items():
        print(f"{key}: largest n within memory = {fit.max_n}  ({fit.reason})")
    sys.exit(0)
