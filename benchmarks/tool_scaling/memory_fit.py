"""Memory-limit fits: per tool, extrapolate recorded peaks to the memory-cap crossing.

No runs happen here. The time stage recorded peak memory on every run; this fits, per
tool, the model ``rss = c + a * n^p`` — with p fixed to the tool's documented memory
exponent from the configuration registry — over the completed runs at that tool's largest
sizes, and reads off the largest candidate size whose predicted peak stays within the
cap.

The fit is least squares on the two free parameters; fixing p makes an extrapolation
from a handful of sizes defensible.

Tools whose memory-optimal configuration differs from the fastest-valid one contribute
their dedicated memory-optimal records instead, when present.

Usage: python -m benchmarks.tool_scaling.memory_fit
"""

import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.tool_scaling.configs import TOOLS, Mode
from benchmarks.tool_scaling.grid import MEMORY_CAP_BYTES, operational_bound, size_grid
from benchmarks.tool_scaling.records import ScalingRunRecord, load_scaling_records
from benchmarks.tool_scaling.time_stage import DATA_PATH

# Dedicated memory-optimal runs land here — the exception path for tools whose
# memory-optimal configuration differs from their fastest-valid one.
MEMORY_DATA_PATH = Path(__file__).resolve().parent / "data" / "memory_stage.jsonl"
FIT_PATH = Path(__file__).resolve().parent / "data" / "memory_fits.json"

# Only the tool's N_FIT_SIZES largest completed sizes feed the fit. Small sizes are dominated
# by the interpreter's fixed footprint and would drag the growth term toward zero.
N_FIT_SIZES = 3


def fit_memory_limits(records: list[ScalingRunRecord]) -> dict[str, dict]:
    """Fit every tool's memory model and return its limit with the fit's parameters."""
    fits: dict[str, dict] = {}
    for tool, entry in TOOLS.items():
        points = _fit_points(records, tool)
        if len(points) < 2:
            fits[tool] = {"max_n": None, "reason": f"only {len(points)} completed size(s) to fit on"}
            continue
        sizes = np.asarray(sorted(points), dtype=np.float64)
        peaks = np.asarray([points[int(n)] for n in sizes], dtype=np.float64)
        a, c = _least_squares(sizes**entry.memory_exponent, peaks)
        if entry.min_growth_bytes is not None:
            # Every completed size can sit below where the growth term shows above the
            # interpreter baseline; the analytic bound then carries the extrapolation.
            a = max(a, entry.min_growth_bytes)
        max_n = _crossing(a, c, entry.memory_exponent)
        fits[tool] = {
            "max_n": max_n,
            "exponent": entry.memory_exponent,
            "coef": a,
            "offset": c,
            "fit_sizes": [int(n) for n in sizes],
        }
    return fits


def _fit_points(records: list[ScalingRunRecord], tool: str) -> dict[int, float]:
    """Return the tool's fit points: per size, the largest completed peak, memory-optimal runs first."""
    for mode in (Mode.MEMORY_OPTIMAL, Mode.FASTEST_VALID):
        rows = [r for r in records if r.tool == tool and r.mode == mode.value and r.completed and r.peak_rss_bytes]
        if rows:
            break
    per_size: dict[int, float] = {}
    for row in rows:
        per_size[row.n] = max(per_size.get(row.n, 0.0), float(row.peak_rss_bytes))
    largest = sorted(per_size)[-N_FIT_SIZES:]
    return {n: per_size[n] for n in largest}


def _least_squares(growth: np.ndarray, peaks: np.ndarray) -> tuple[float, float]:
    """Fit peaks = a * growth + c; a is clamped non-negative (memory does not shrink with n)."""
    design = np.column_stack([growth, np.ones_like(growth)])
    (a, c), *_ = np.linalg.lstsq(design, peaks, rcond=None)
    return max(float(a), 0.0), float(c)


def _crossing(a: float, c: float, exponent: int) -> int | None:
    """Return the largest grid size whose predicted peak stays within the cap."""
    if c >= MEMORY_CAP_BYTES:
        return None
    bound = operational_bound()
    if a <= 0.0:
        return size_grid(bound)[-1]
    n_star = ((MEMORY_CAP_BYTES - c) / a) ** (1.0 / exponent)
    passing = [n for n in size_grid(bound) if n <= n_star]
    return passing[-1] if passing else None


if __name__ == "__main__":
    records = load_scaling_records(DATA_PATH)
    if MEMORY_DATA_PATH.exists():
        records += load_scaling_records(MEMORY_DATA_PATH)
    all_fits = fit_memory_limits(records)
    FIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIT_PATH.write_text(json.dumps(all_fits, indent=2) + "\n", encoding="utf-8")
    for tool_key, fit in all_fits.items():
        print(f"{tool_key}: largest n within memory = {fit.get('max_n')}")
    sys.exit(0)
