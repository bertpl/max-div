from typing import Any

import numpy as np

from max_div._core.benchmark_problems._registry import BenchmarkProblem
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.problem import MaxDivProblem

from ._helpers import sort_vectors


# =================================================================================================
#  U2 - Gaussian - Unconstrained
# =================================================================================================
class BenchmarkProblem_U2(BenchmarkProblem):
    @classmethod
    def name(cls) -> str:
        return "U2"

    @classmethod
    def description(cls) -> str:
        return "Unconstrained problem with non-uniform vector density (gaussian distribution)"

    @classmethod
    def supported_params(cls) -> dict[str, str]:
        return {
            "size": "(int) value in [1, ...].  Problem size, with d=size, n=100*size, k=10*size",
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
        n = 100 * size
        k = 10 * size
        m = 0
        n_con_indices = 0
        return d, n, k, m, n_con_indices

    @classmethod
    def _create_problem_instance(  # ty: ignore[invalid-method-override] -- factory always dispatches with matching named kwargs
        cls,
        size: int,
        diversity_metric: DiversityMetric,
        **kwargs: Any,  # noqa: ANN401 -- heterogeneous per-problem parameters
    ) -> MaxDivProblem:
        d, n, k, _, _ = cls.get_problem_dimensions(size=size)

        # Generate gaussian random vectors
        np.random.seed(42)
        vectors = np.random.randn(n, d).astype(np.float32)
        vectors = sort_vectors(vectors)  # sort by increasing L2 norm of rows

        return MaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.L2_EUCLIDEAN,
            diversity_metric=diversity_metric,
            constraints=[],
        )
