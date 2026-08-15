import math

import numpy as np

from max_div._core.benchmark_problems._registry import BenchmarkProblem
from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.problem import VectorMaxDivProblem

from ._helpers import sort_vectors


# =================================================================================================
#  C4 - Gaussian - Coupled constraints
# =================================================================================================
class BenchmarkProblem_C4(BenchmarkProblem):
    """C3's sign pairs plus a band constraint per dimension, so the constraints couple strongly."""

    @classmethod
    def name(cls) -> str:
        return "C4"

    @classmethod
    def description(cls) -> str:
        return "Problem with non-uniform vector density (gaussian distribution) and complex constraints"

    @classmethod
    def _get_problem_dimensions(cls, n: int) -> tuple[int, int, int, int, int]:
        d = math.ceil(n / 150)
        k = math.ceil(n / 15)
        m = 3 * d  # constraints are generated per dimension: a sign pair plus a band per dimension
        n_con_indices = round(d * (n * 1.62))  # each dimension has 3 constraints, with expected 1.62*n total indices
        return d, n, k, m, n_con_indices

    @classmethod
    def _create_problem_instance(cls, n: int, diversity_metric: DiversityMetric) -> VectorMaxDivProblem:
        d, _, k, _m, _ = cls._get_problem_dimensions(n)

        # Generate gaussian random vectors, such that in each dimension...
        #   ~31% of values are <=0
        #   ~69% of values are >=0
        #   ~62% of values are in [-1, +1]
        np.random.seed(42)
        vectors = np.random.randn(n, d).astype(np.float32) + 0.5  # shift by 0.5
        vectors = sort_vectors(vectors)  # sort by increasing L2 norm of rows

        # Generate constraints
        constraints: list[Constraint] = []
        for i in range(d):
            # at least 40% of the k samples should have positive or 0 value in dimension i
            indices_positive = [idx for idx in range(n) if vectors[idx, i] >= 0.0]
            constraints.append(
                Constraint(
                    int_set=set(indices_positive),
                    min_count=int(0.4 * k),
                    max_count=k,
                )
            )

            # at least 40% of the k samples should have negative or 0 value in dimension i
            indices_negative = [idx for idx in range(n) if vectors[idx, i] <= 0.0]
            constraints.append(
                Constraint(
                    int_set=set(indices_negative),
                    min_count=int(0.4 * k),
                    max_count=k,
                )
            )

            # at least 70% of the k samples should have value in [-1, +1] in dimension i
            indices_in_range = [idx for idx in range(n) if -1.0 <= vectors[idx, i] <= 1.0]
            constraints.append(
                Constraint(
                    int_set=set(indices_in_range),
                    min_count=int(0.7 * k),
                    max_count=k,
                )
            )

        return VectorMaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.L2_EUCLIDEAN,
            diversity_metric=diversity_metric,
            constraints=constraints,
        )
