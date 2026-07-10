from typing import Any

import numpy as np

from max_div._core.benchmark_problems._registry import BenchmarkProblem
from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.problem import MaxDivProblem

from ._helpers import sort_vectors


# =================================================================================================
#  C4 - Gaussian - Hard constraints
# =================================================================================================
class BenchmarkProblem_C4(BenchmarkProblem):
    @classmethod
    def name(cls) -> str:
        return "C4"

    @classmethod
    def description(cls) -> str:
        return "Problem with non-uniform vector density (gaussian distribution) and complex constraints"

    @classmethod
    def supported_params(cls) -> dict[str, str]:
        return {
            "size": "(int) value in [1, ...].  Problem size, with d=size, n=100*size, k=10*size, m=3*size",
            "diversity_metric": "(DiversityMetric) diversity metric to be maximized",
        }

    @classmethod
    def get_example_parameters(cls) -> dict[str, Any]:
        return {
            "size": 1,
            "diversity_metric": DiversityMetric.APPROX_GEOMEAN_SEPARATION,
        }

    @classmethod
    def get_problem_dimensions(cls, **kwargs: Any) -> tuple[int, int, int, int, int]:  # noqa: ANN401 -- heterogeneous per-problem parameters
        size: int = kwargs["size"]  # required parameter, see supported_params()
        d = size
        n = 150 * size
        k = 10 * size
        m = 3 * size
        n_con_indices = round(d * (n * 1.62))  # each dimension has 3 constraints, with expected 1.62*n total indices
        return d, n, k, m, n_con_indices

    @classmethod
    def _create_problem_instance(  # ty: ignore[invalid-method-override] -- factory always dispatches with matching named kwargs
        cls,
        size: int,
        diversity_metric: DiversityMetric,
        **kwargs: Any,  # noqa: ANN401 -- heterogeneous per-problem parameters
    ) -> MaxDivProblem:
        d, n, k, _m, _ = cls.get_problem_dimensions(size=size)

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
            # at least half of k samples should have positive or 0 value in dimension i
            indices_positive = [idx for idx in range(n) if vectors[idx, i] >= 0.0]
            constraints.append(
                Constraint(
                    int_set=set(indices_positive),
                    min_count=int(0.4 * k),
                    max_count=k,
                )
            )

            # at least half of k samples should have negative or 0 value in dimension i
            indices_negative = [idx for idx in range(n) if vectors[idx, i] <= 0.0]
            constraints.append(
                Constraint(
                    int_set=set(indices_negative),
                    min_count=int(0.4 * k),
                    max_count=k,
                )
            )

            # exact half of k samples should have value in [-1, +1] in dimension i
            indices_in_range = [idx for idx in range(n) if -1.0 <= vectors[idx, i] <= 1.0]
            constraints.append(
                Constraint(
                    int_set=set(indices_in_range),
                    min_count=int(0.7 * k),
                    max_count=k,
                )
            )

        return MaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.L2_EUCLIDEAN,
            diversity_metric=diversity_metric,
            constraints=constraints,
        )
