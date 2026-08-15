import math

from max_div._core.benchmark_problems._registry import BenchmarkProblem
from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.problem import VectorMaxDivProblem

from ._helpers import make_banded_vectors_and_bands


# =================================================================================================
#  C1 - Exact stratified quotas - Constrained gateway
# =================================================================================================
class BenchmarkProblem_C1(BenchmarkProblem):
    """C1 is the fixed-d=2 constrained reference problem for cross-tool comparisons.

    Non-overlapping bands partition the population and each band carries an exact selection
    quota — the constraint form restricted third-party tools support.
    """

    @classmethod
    def name(cls) -> str:
        return "C1"

    @classmethod
    def description(cls) -> str:
        return "Constrained 2D problem with exact per-band quotas (non-overlapping, partition)"

    @classmethod
    def _get_problem_dimensions(cls, n: int) -> tuple[int, int, int, int, int]:
        d = 2
        k = math.ceil(n / 10)
        m = math.ceil(k / 5)  # tied to k so the exact quotas below can sum to k structurally
        n_con_indices = n
        return d, n, k, m, n_con_indices

    @classmethod
    def _create_problem_instance(cls, n: int, diversity_metric: DiversityMetric) -> VectorMaxDivProblem:
        _d, _, k, m, _ = cls._get_problem_dimensions(n)
        vectors, bands = make_banded_vectors_and_bands(n, m)

        # exact quota of 5 per band; the last band takes the remainder so the quotas sum to k
        constraints: list[Constraint] = []
        for i, band in enumerate(bands):
            quota = 5 if i < m - 1 else k - 5 * (m - 1)
            constraints.append(Constraint(int_set=set(band), min_count=quota, max_count=quota))

        return VectorMaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.L2_EUCLIDEAN,
            diversity_metric=diversity_metric,
            constraints=constraints,
        )
