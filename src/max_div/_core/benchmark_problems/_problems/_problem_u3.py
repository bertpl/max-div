from typing import Any

import numpy as np

from max_div._core.benchmark_problems._registry import BenchmarkProblem
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.problem import VectorMaxDivProblem

from ._helpers import sort_vectors


# =================================================================================================
#  U3 - Log-distributed - Unconstrained
# =================================================================================================
class BenchmarkProblem_U3(BenchmarkProblem):
    @classmethod
    def name(cls) -> str:
        return "U3"

    @classmethod
    def description(cls) -> str:
        return "Unconstrained problem with non-uniform vector density (exponentially spaced)"

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
    ) -> VectorMaxDivProblem:
        d, n, k, _, _ = cls.get_problem_dimensions(size=size)

        # Generate log-uniform random vectors in [0.1, 10] along each axis
        np.random.seed(42)
        vectors = np.random.random_sample(size=(n, d)).astype(np.float32)
        vectors = np.power(10.0, (vectors * 2.0) - 1.0)  # map [0,1] to [0.1,10]
        vectors = sort_vectors(vectors)  # sort by increasing L2 norm of rows

        return VectorMaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.L2_EUCLIDEAN,
            diversity_metric=diversity_metric,
            constraints=[],
        )
