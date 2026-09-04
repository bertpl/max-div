"""Memory-fit arithmetic: turn a series of recorded footprints into a memory-cap crossing.

The memory sweep (`memory_stage`) collects the footprints and decides when to stop; this module
owns the fit itself. The fit is a median fit — it minimizes the sum of absolute residuals, so a
single run's bump does not pull the curve — under coefficient bounds (`c0 >= 0`, `c1 >= 8`,
`c2 >= 0`): the `c1 >= 8` lower bound is the input-array cost — every solver holds at least the
n x d float32 vectors, 8 bytes per item at d=2. A quadratic term is kept only when physically
plausible (`_C2_MIN_BYTES`); otherwise the fit is linear.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from .grid import MEMORY_CAP_BYTES, operational_bound, size_grid

_INPUT_MIN_BYTES = 8.0  # 4 bytes x d=2: the raw float32 vectors, the linear coefficient's lower bound

# Physical-plausibility threshold for the fitted quadratic coefficient: the smallest real
# quadratically growing structure is one byte per k x n entry, i.e. 0.1 bytes per n^2 at k = n/10.
# A fitted c2 below this cannot be an allocation and is measurement noise amplified by the long
# extrapolation to the cap — refit linear.
_C2_MIN_BYTES = 0.1

# The trust conditions (measurement protocol, IV.B.1): one recorded footprint must reach this
# size, the fitted model must reach this R^2, and there must be at least this many distinct
# sizes. The footprint threshold keeps the extrapolation to the cap within a factor 16 in memory
# and puts the growth term well above any solver's fixed baseline; a high-R^2 fit over only a
# few points near the baseline extrapolates to the cap on too little evidence to trust.
_TRUST_MIN_BYTES = 2 * 2**30
_R2_MIN = 0.95
_MIN_TRUST_SIZES = 5

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
    if len(sizes_peaks) < _MIN_TRUST_SIZES or fit.r2 is None:
        return False
    return max(sizes_peaks.values()) >= _TRUST_MIN_BYTES and fit.r2 >= _R2_MIN


def _r_squared(ns: np.ndarray, peaks: np.ndarray, coef: tuple[float, ...]) -> float:
    """Return the coefficient of determination of the fitted model over the series."""
    predicted = sum(c * ns**p for p, c in enumerate(coef))
    ss_res = float(np.sum((peaks - predicted) ** 2))
    ss_tot = float(np.sum((peaks - peaks.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 1.0


def _fit_quadratic(ns: np.ndarray, peaks: np.ndarray) -> tuple[float, float, float]:
    """Median-fit `peaks = c0 + c1*n + c2*n^2` with `c0 >= 0`, `c1 >= 8`, `c2 >= 0`."""
    design = np.column_stack([np.ones_like(ns), ns, ns**2])
    c0, c1, c2 = _fit_median(design, peaks, (0.0, _INPUT_MIN_BYTES, 0.0))
    return c0, c1, c2


def _fit_linear(ns: np.ndarray, peaks: np.ndarray) -> tuple[float, float]:
    """Median-fit `peaks = c0 + c1*n` with `c0 >= 0`, `c1 >= 8`."""
    design = np.column_stack([np.ones_like(ns), ns])
    c0, c1 = _fit_median(design, peaks, (0.0, _INPUT_MIN_BYTES))
    return c0, c1


def _fit_median(design: np.ndarray, targets: np.ndarray, lower_bounds: tuple[float, ...]) -> tuple[float, ...]:
    """Return the coefficients minimizing `sum |targets - design @ coef|` subject to `coef >= lower_bounds`.

    The problem is a linear program: each residual splits into a non-negative positive part `u`
    and negative part `v` with `design @ coef + u - v = targets`, and the objective is `sum(u + v)`.
    The design columns are scaled to unit maximum before solving, since `n^2` reaches 1e18 while
    the intercept column is 1, a spread the solver's tolerances cannot handle; the coefficients
    are scaled back afterwards.

    Args:
        design: One row per observation, one column per coefficient.
        targets: The observed values, one per row of `design`.
        lower_bounds: One lower bound per coefficient; no upper bounds apply.
    """
    n_obs, n_coef = design.shape
    column_scale = np.abs(design).max(axis=0)
    scaled_design = design / column_scale
    cost = np.concatenate([np.zeros(n_coef), np.ones(2 * n_obs)])
    equality = np.hstack([scaled_design, np.eye(n_obs), -np.eye(n_obs)])
    bounds = [(lb * scale, None) for lb, scale in zip(lower_bounds, column_scale, strict=True)] + [(0.0, None)] * (
        2 * n_obs
    )
    result = linprog(cost, A_eq=equality, b_eq=targets, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"median fit failed: {result.message}")
    return tuple(float(c / scale) for c, scale in zip(result.x[:n_coef], column_scale, strict=True))


def _crossing(coef: tuple[float, ...]) -> int | None:
    """Return the largest grid size whose predicted peak stays within the memory cap."""
    powers = np.arange(len(coef))
    grid = size_grid(operational_bound())
    predicted = [sum(c * n**p for c, p in zip(coef, powers, strict=True)) for n in grid]
    passing = [n for n, peak in zip(grid, predicted, strict=True) if peak <= MEMORY_CAP_BYTES]
    return passing[-1] if passing else None
