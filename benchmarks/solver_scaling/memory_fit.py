"""Memory-fit arithmetic: turn a series of recorded footprints into a memory-cap crossing.

The memory sweep (`memory_stage`) collects the footprints and decides when to stop; this module
owns the fit itself. The fit is bound-constrained (`c0 >= 0`, `c1 >= 8`, `c2 >= 0`): the
`c1 >= 8` lower bound is the input-array cost — every solver holds at least the n x d float32
vectors, 8 bytes per item at d=2. A quadratic term is kept only when physically plausible
(`_C2_MIN_BYTES`); otherwise the fit is linear.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from .grid import MEMORY_CAP_BYTES, operational_bound, size_grid

_INPUT_MIN_BYTES = 8.0  # 4 bytes x d=2: the raw float32 vectors, the linear coefficient's lower bound

# Physical-plausibility threshold for the fitted quadratic coefficient: the smallest real
# quadratically growing structure is one byte per k x n entry, i.e. 0.1 bytes per n^2 at k = n/10.
# A fitted c2 below this cannot be an allocation and is measurement noise amplified by the long
# extrapolation to the cap — refit linear.
_C2_MIN_BYTES = 0.1

# The trust conditions (measurement protocol, IV.B.1): the recorded footprints must span this
# range factor, and the fitted model must reach this R^2.
_SPAN_FACTOR = 2.0
_R2_MIN = 0.95

FIT_PATH = Path(__file__).resolve().parent / "data" / "memory_fits.json"


@dataclass(frozen=True)
class MemoryFit:
    """One configuration's memory result: the largest n within the cap, and how it was found.

    `coef` and `r2` are set only when the value comes from a fit — a value measured directly
    from the runs, or an unmeasured configuration, has neither.
    """

    max_n: int | None
    coef: tuple[float, ...] | None
    reason: str
    r2: float | None = None


def fit_series(sizes_peaks: dict[int, float]) -> MemoryFit:
    """Fit the given footprints and return the crossing read off at the memory cap."""
    ns = np.asarray(sorted(sizes_peaks), dtype=np.float64)
    peaks = np.asarray([sizes_peaks[n] for n in sorted(sizes_peaks)], dtype=np.float64)
    model = "linear"
    coef = _fit_linear(ns, peaks)
    if len(ns) >= 3:
        quadratic = _fit_quadratic(ns, peaks)
        if quadratic[2] >= _C2_MIN_BYTES:
            model, coef = "quadratic", quadratic
    return MemoryFit(_crossing(coef), coef, f"{model} fit over {len(ns)} sizes", _r_squared(ns, peaks, coef))


def trust_conditions_met(sizes_peaks: dict[int, float], fit: MemoryFit) -> bool:
    """Return whether the fitted crossing is trustworthy enough to end the memory sweep."""
    if len(sizes_peaks) < 3 or fit.r2 is None:
        return False
    return max(sizes_peaks.values()) >= _SPAN_FACTOR * min(sizes_peaks.values()) and fit.r2 >= _R2_MIN


def _r_squared(ns: np.ndarray, peaks: np.ndarray, coef: tuple[float, ...]) -> float:
    """Return the coefficient of determination of the fitted model over the series."""
    predicted = sum(c * ns**p for p, c in enumerate(coef))
    ss_res = float(np.sum((peaks - predicted) ** 2))
    ss_tot = float(np.sum((peaks - peaks.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 1.0


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
