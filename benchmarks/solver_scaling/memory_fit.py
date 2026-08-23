"""Largest-n-within-memory fits: per configuration, turn recorded peaks into a memory-cap crossing.

No runs happen here — the time stage recorded peak RSS on every run, and this reads those back.
Per the measurement protocol, a configuration whose sweep ended by crossing the memory cap is
bracketed directly (its largest completed size is the answer); one that ended any other way is
extrapolated: a constrained least-squares fit of `rss = c0 + c1*n [+ c2*n^2]` over its largest
completed sizes, read off at the cap.

The fit is bound-constrained (`c0 >= 0`, `c1 >= 8`, `c2 >= 0`): the `c1 >= 8` lower bound is the
input-array cost — every solver holds at least the n x d float32 vectors, 8 bytes per item at d=2
— which keeps the extrapolation well-posed even when the recorded peaks are dominated by the
interpreter's fixed footprint. Degree follows the completed-size count (three or more → quadratic,
two → linear).

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
FIT_PATH = Path(__file__).resolve().parent / "data" / "memory_fits.json"


@dataclass(frozen=True)
class MemoryFit:
    """One configuration's memory result: the largest n within the cap, and how it was found."""

    max_n: int | None
    coef: tuple[float, ...] | None
    reason: str


def fit_memory_limit(sizes_peaks: dict[int, float], terminal: Outcome) -> MemoryFit:
    """Return the largest-n-within-memory result for one configuration.

    Args:
        sizes_peaks: per completed size, the peak RSS observed there.
        terminal: the outcome that ended the size sweep — `MEMORY` brackets directly, any other
            outcome extrapolates from the recorded peaks.
    """
    sizes = sorted(sizes_peaks)
    if not sizes:
        return MemoryFit(None, None, "no completed runs")
    if terminal is Outcome.MEMORY:
        return MemoryFit(sizes[-1], None, "bracketed: the memory cap was reached at the next size")
    fit_sizes = sizes[-N_FIT_SIZES:]
    if len(fit_sizes) == 1:
        return MemoryFit(fit_sizes[0], None, "single completed size; no growth term to extrapolate")
    ns = np.asarray(fit_sizes, dtype=np.float64)
    peaks = np.asarray([sizes_peaks[n] for n in fit_sizes], dtype=np.float64)
    coef = _fit_quadratic(ns, peaks) if len(fit_sizes) >= 3 else _fit_linear(ns, peaks)
    return MemoryFit(_crossing(coef), coef, f"fit over {len(fit_sizes)} sizes")


def fit_all(records: list[ScalingRunRecord]) -> dict[str, MemoryFit]:
    """Fit every smoke configuration and key the results by ``tool/config``."""
    fits: dict[str, MemoryFit] = {}
    for config in CONFIGS:
        rows = [r for r in records if r.tool == config.tool and r.config == config.name]
        fits[f"{config.tool}/{config.name}"] = fit_memory_limit(_peaks_by_size(rows), _terminal_outcome(rows))
    return fits


def _peaks_by_size(rows: list[ScalingRunRecord]) -> dict[int, float]:
    """Return, per size with a completed run carrying a peak, the largest peak recorded there."""
    per_size: dict[int, float] = {}
    for row in rows:
        if row.completed and row.peak_rss_bytes:
            per_size[row.n] = max(per_size.get(row.n, 0.0), float(row.peak_rss_bytes))
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
            {key: {"max_n": f.max_n, "coef": f.coef, "reason": f.reason} for key, f in all_fits.items()}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    for key, fit in all_fits.items():
        print(f"{key}: largest n within memory = {fit.max_n}  ({fit.reason})")
    sys.exit(0)
