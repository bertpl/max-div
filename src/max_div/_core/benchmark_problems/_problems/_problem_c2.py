import math

from max_div._core.benchmark_problems._registry import BenchmarkProblem
from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.problem import VectorMaxDivProblem

from ._helpers import make_banded_vectors_and_bands


# =================================================================================================
#  C2 - Soft band lower bounds - Constrained
# =================================================================================================
class BenchmarkProblem_C2(BenchmarkProblem):
    """C2 uses C1's vectors and band partition, with the exact quotas relaxed to lower bounds."""

    @classmethod
    def name(cls) -> str:
        return "C2"

    @classmethod
    def description(cls) -> str:
        return "Constrained 2D problem with per-band lower bounds (non-overlapping, partition)"

    @classmethod
    def _get_problem_dimensions(cls, n: int) -> tuple[int, int, int, int, int]:
        d = 2
        k = math.ceil(n / 10)
        m = math.ceil(k / 5)  # same band structure as C1
        n_con_indices = n
        return d, n, k, m, n_con_indices

    @classmethod
    def _create_problem_instance(cls, n: int, diversity_metric: DiversityMetric) -> VectorMaxDivProblem:
        _d, _, k, m, _ = cls._get_problem_dimensions(n)
        vectors, bands = make_banded_vectors_and_bands(n, m)

        # at least 4 picks per band; capped by k // m so the m lower bounds always sum to <= k
        # (m = ceil(k/5) can exceed k/5 when 5 does not divide k, and 4 * ceil(k/5) can then exceed k)
        min_count = min(4, k // m)
        constraints: list[Constraint] = [
            Constraint(int_set=set(band), min_count=min_count, max_count=k) for band in bands
        ]

        return VectorMaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.L2_EUCLIDEAN,
            diversity_metric=diversity_metric,
            constraints=constraints,
        )
