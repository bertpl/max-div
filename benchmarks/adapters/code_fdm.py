"""code-FDM adapter: fair max-min diversification with per-group counts (Wang et al., Entropy 2023).

code-FDM (github.com/yhwang1990/code-FDM) is unpackaged research code without a license
grant, so it is never vendored or redistributed: the two modules it consists of are
fetched at run time from a pinned commit into a local cache and imported from there.
Scenarios must treat this adapter as optional (network + upstream availability).
"""

import importlib.util
import sys
import urllib.request
from pathlib import Path
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from max_div.problem import MaxDivProblem

from ._inputs import problem_vectors
from .base import SelectionAdapter

_REPO_RAW = "https://raw.githubusercontent.com/yhwang1990/code-FDM"
_PINNED_COMMIT = "d18758a93f9173e1e888a886e2e0ea10c5639022"
_MODULES = ("utils.py", "algorithms_offline.py")
_CACHE_DIR = Path("reports/benchmarks/vendor/code_fdm")


class CodeFdmFairFlow(SelectionAdapter):
    """Fair max-min selection via code-FDM's FairFlow algorithm.

    code-FDM's fairness model is disjoint color groups with an exact per-group count;
    the adapter maps max-div constraints onto that model (validating that groups are disjoint and
    deriving per-group counts within each constraint's min/max bounds) and treats
    items outside every constraint as one remainder group.
    """

    @property
    def name(self) -> str:
        """Tool name as it appears in records and figures."""
        return "code-FDM[FairFlow]"

    @property
    def supports_constraints(self) -> bool:
        """code-FDM is the one heuristic competitor that honors group constraints."""
        return True

    def select(self, problem: MaxDivProblem, seed: int) -> NDArray[np.int64]:
        """Run FairFlow with colors and counts derived from the problem's constraints."""
        algorithms, utils = _load_code_fdm()
        vectors = problem_vectors(problem)
        colors, counts = _constraints_to_colors(problem)

        elements = [utils.Element(idx=i, color=int(colors[i]), features=vectors[i].tolist()) for i in range(problem.n)]
        selected, _diversity, _t = algorithms.FairFlow(elements, len(counts), counts, utils.euclidean_dist)
        return np.asarray(sorted(selected), dtype=np.int64)


class CodeFdmSingleColor(SelectionAdapter):
    """FairFlow on an unconstrained problem: one color for every item, count = k.

    code-FDM's model is fairness-colored max-min selection; with a single color it reduces to
    plain max-min diverse selection, which is how the solver-scaling benchmarks run it on the
    unconstrained reference problem.
    """

    @property
    def name(self) -> str:
        """Return the tool name as it appears in records and figures."""
        return "code-FDM[single-color]"

    def select(self, problem: MaxDivProblem, seed: int) -> NDArray[np.int64]:
        """Run FairFlow with one color spanning all items.

        FairFlow returns `None` for the selection on some degenerate tiny instances (observed at
        n=20, k=2); that is surfaced as a clear failure rather than an opaque unpacking error.
        """
        algorithms, utils = _load_code_fdm()
        vectors = problem_vectors(problem)
        elements = [utils.Element(idx=i, color=0, features=vectors[i].tolist()) for i in range(problem.n)]
        selected, _diversity, _t = algorithms.FairFlow(elements, 1, [problem.k], utils.euclidean_dist)
        if selected is None:
            raise RuntimeError("code-FDM (FairFlow) produced no selection")
        return np.asarray(sorted(selected), dtype=np.int64)


def _constraints_to_colors(problem: MaxDivProblem) -> tuple[NDArray[np.int64], list[int]]:
    """Map the problem's constraints onto code-FDM's disjoint-color model.

    Each constraint becomes one color; items outside every constraint share a remainder
    color. Per-color counts start at each constraint's min_count and the remaining
    budget is distributed greedily within max_count bounds (largest slack first).

    Returns:
        Tuple of (per-item color array, per-color count list summing to k).

    Raises:
        ValueError: If constraints overlap (code-FDM cannot express that) or no
            count assignment within the bounds sums to k.
    """
    if problem.m == 0:
        raise ValueError("code-FDM comparison requires a constrained problem.")

    colors = np.full(problem.n, problem.m, dtype=np.int64)  # default: remainder color
    for color, con in enumerate(problem.constraints):
        indices = np.fromiter(con.int_set, dtype=np.int64)
        if (colors[indices] != problem.m).any():
            raise ValueError("Constraints overlap; code-FDM's disjoint-color model cannot express them.")
        colors[indices] = color

    counts = [con.min_count for con in problem.constraints]
    n_remainder = int((colors == problem.m).sum())
    has_remainder = n_remainder > 0
    if has_remainder:
        counts.append(0)
    max_counts = [con.max_count for con in problem.constraints] + ([n_remainder] if has_remainder else [])

    budget = problem.k - sum(counts)
    if budget < 0:
        raise ValueError("Sum of constraint min_counts exceeds k; no feasible code-FDM count assignment.")
    slack_order = sorted(range(len(counts)), key=lambda i: max_counts[i] - counts[i], reverse=True)
    for i in slack_order:
        take = min(budget, max_counts[i] - counts[i])
        counts[i] += take
        budget -= take
    if budget > 0:
        raise ValueError("Constraint max_counts leave no room to reach k; cannot map onto code-FDM.")
    return colors, counts


def _load_code_fdm() -> tuple[ModuleType, ModuleType]:
    """Fetch (if needed) and import code-FDM's two modules from the pinned commit."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for filename in _MODULES:
        target = _CACHE_DIR / filename
        if not target.exists():
            url = f"{_REPO_RAW}/{_PINNED_COMMIT}/{filename}"
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 -- pinned https URL
                target.write_bytes(response.read())

    modules = []
    for filename in _MODULES:
        module_name = f"code_fdm_{filename.removesuffix('.py')}"
        if module_name in sys.modules:
            modules.append(sys.modules[module_name])
            continue
        # code-FDM's modules cross-import each other by their original names
        alias = filename.removesuffix(".py")
        spec = importlib.util.spec_from_file_location(alias, _CACHE_DIR / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    utils, algorithms = modules[0], modules[1]
    return algorithms, utils
