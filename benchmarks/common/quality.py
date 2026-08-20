"""Subset-quality evaluation: score any tool's selection under max-div's own diversity metrics.

Every compared tool is judged with the functions here, so quality numbers are comparable
across tools by construction. Evaluation only ever touches the k x k distances among the
selected items — never the full n^2 pairwise matrix — so scoring stays cheap even at the
largest benchmark sizes: vector problems compute the k x k block directly from the selected
vectors (respecting the problem's distance metric), and precomputed-distance problems gather
it from their stored condensed vector.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import squareform

# The library-internal pdist kernel is used on purpose: selections must be scored under
# exactly the distance semantics the solver itself uses (incl. float32 behavior), and the
# public API only exposes distances via whole problems.
from max_div._core.metrics._distance import compute_pdist
from max_div.metrics import DiversityMetric
from max_div.problem import MaxDivProblem, VectorMaxDivProblem

# The four canonical metrics every selection is scored under (APPROX_GEOMEAN is a speed
# variant of GEOMEAN, not a distinct objective, so it is not evaluated separately).
EVALUATED_METRICS: tuple[DiversityMetric, ...] = (
    DiversityMetric.MIN_SEPARATION,
    DiversityMetric.MEAN_SEPARATION,
    DiversityMetric.GEOMEAN_SEPARATION,
    DiversityMetric.MEAN_PAIRWISE_DISTANCE,
)


def evaluate_selection(problem: MaxDivProblem, i_selected: NDArray[np.integer]) -> dict[str, float]:
    """Score a selection under all canonical diversity metrics.

    Args:
        problem: The problem the selection was made for.
        i_selected: Indices of the selected items (length k, unique).

    Returns:
        Mapping of diversity-metric name to its value for this selection.
    """
    dist = _selection_distance_matrix(problem, i_selected)
    k = dist.shape[0]
    off_diag = ~np.eye(k, dtype=bool)
    separations = np.where(off_diag, dist, np.inf).min(axis=1).astype(np.float32)
    mean_dists = (dist.sum(axis=1) / (k - 1)).astype(np.float32)

    values: dict[str, float] = {}
    for metric in EVALUATED_METRICS:
        contributions = mean_dists if metric == DiversityMetric.MEAN_PAIRWISE_DISTANCE else separations
        values[metric.name] = float(metric.compute(contributions))
    return values


def n_constraints_satisfied(problem: MaxDivProblem, i_selected: NDArray[np.integer]) -> int:
    """Count how many of the problem's constraints the selection satisfies."""
    selected = {int(i) for i in i_selected}
    n_ok = 0
    for con in problem.constraints:
        count = len(con.int_set & selected)
        if con.min_count <= count <= con.max_count:
            n_ok += 1
    return n_ok


def _selection_distance_matrix(problem: MaxDivProblem, i_selected: NDArray[np.integer]) -> NDArray[np.float64]:
    """Build the k x k distance matrix among selected items, never materializing all n^2 distances.

    Vector problems compute pairwise distances over just the selected vectors (their
    ``condensed_distances()`` would recompute the full n^2 pdist on every call — at the
    largest benchmark sizes that costs seconds and ~1 GB per evaluation). Distance problems
    already store the condensed vector, so gathering the k x k block from it is cheap.
    """
    idx = np.asarray(i_selected, dtype=np.int64)
    k = idx.shape[0]
    if k < 2:
        raise ValueError("A selection needs at least 2 items to evaluate diversity.")
    if len(np.unique(idx)) != k:
        raise ValueError("Selection contains duplicate indices.")
    if isinstance(problem, VectorMaxDivProblem):
        return squareform(compute_pdist(problem.vectors[idx], problem.distance_metric)).astype(np.float64)
    n = problem.n
    condensed = problem.condensed_distances()
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    lo, hi = np.minimum(ii, jj), np.maximum(ii, jj)
    condensed_pos = (lo * (2 * n - lo - 1)) // 2 + (hi - lo - 1)
    dist = np.zeros((k, k), dtype=np.float64)
    off_diag = ~np.eye(k, dtype=bool)
    dist[off_diag] = condensed[condensed_pos[off_diag]]
    return dist

def min_separation_nn(vectors: NDArray[np.floating], i_selected: NDArray[np.integer]) -> float:
    """Score a selection's minimum separation via nearest-neighbor queries, in O(k log k).

    The k x k matrix behind `evaluate_selection` costs O(k^2) memory, which at the tool-scaling
    benchmarks' largest sizes would dwarf the tool being measured — the minimum separation
    equals the minimum over each selected point's nearest selected neighbor, and a KD-tree
    answers that without materializing any pairwise block.
    """
    from scipy.spatial import cKDTree

    selected = np.ascontiguousarray(vectors[np.asarray(i_selected, dtype=np.int64)], dtype=np.float64)
    if selected.shape[0] < 2:
        raise ValueError("A selection needs at least 2 items to evaluate diversity.")
    tree = cKDTree(selected)
    distances, _ = tree.query(selected, k=2)
    return float(distances[:, 1].min())
