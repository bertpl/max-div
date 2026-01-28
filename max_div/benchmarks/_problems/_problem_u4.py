from typing import Any

import numpy as np

from max_div.benchmarks._registry import BenchmarkProblem
from max_div.solver import DistanceMetric, DiversityMetric, MaxDivProblem

from ._helpers import sort_vectors


# =================================================================================================
#  U4 - Conic - Unconstrained
# =================================================================================================
class BenchmarkProblem_U4(BenchmarkProblem):
    @classmethod
    def name(cls) -> str:
        return "U4"

    @classmethod
    def description(cls) -> str:
        return "Unconstrained problem with non-uniform vector density (conic)"

    @classmethod
    def supported_params(cls) -> dict[str, str]:
        return dict(
            size="(int) value in [1, ...].  Problem size, with d=size, n=100*size, k=10*size",
            diversity_metric="(DiversityMetric) diversity metric to be maximized",
        )

    @classmethod
    def get_example_parameters(cls) -> dict[str, Any]:
        return dict(
            size=1,
            diversity_metric=DiversityMetric.approx_geomean_separation(),
        )

    @classmethod
    def get_problem_dimensions(cls, **kwargs) -> tuple[int, int, int, int, int]:
        size = kwargs.get("size")
        d = size
        n = 100 * size
        k = 10 * size
        m = 0
        n_con_indices = 0
        return d, n, k, m, n_con_indices

    @classmethod
    def _create_problem_instance(cls, size: int, diversity_metric: DiversityMetric, **kwargs) -> MaxDivProblem:
        """
        We will generate vectors in d-dim. space as follows:
          - [a, a, a, ..., a] + r * [x1, x2, ..., xd]

        - a is sampled uniformly in [0,1]
        - x-values are sampled uniformly in the hyper-box [-1, +1]^d and then rescaled to have L2 norm = r
        - r = 0.1 * sqrt(d), such that the perceived angle of the cone from the origin remains constant as d increases
        """

        d, n, k, _, _ = cls.get_problem_dimensions(size=size)
        r = 0.1 * np.sqrt(d)

        # step 1 - generate n x (d+1) matrix of random values in [0,1]
        np.random.seed(42)
        random_data = np.random.random_sample(size=(n, d + 1)).astype(np.float32)

        # step 2 - generate vectors as described above
        vectors = np.empty(shape=(n, d), dtype=np.float32)
        for i in range(n):
            a = random_data[i, 0]
            x_vals = (random_data[i, 1:] * 2.0) - 1.0  # map [0,1] to [-1,+1]
            x_norm = np.linalg.norm(x_vals, ord=2)
            if x_norm > 0:
                x_vals = (x_vals / x_norm) * r  # rescale to have L2 norm = r
            vectors[i, :] = a + x_vals

        # step 3 - sort vectors by increasing L2 norm of rows & return problem instance
        vectors = sort_vectors(vectors)  # sort by increasing L2 norm of rows
        return MaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.L2_EUCLIDEAN,
            diversity_metric=diversity_metric,
            constraints=[],
        )
