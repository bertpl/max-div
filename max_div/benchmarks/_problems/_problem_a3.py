from typing import Any

import numpy as np

from max_div.benchmarks._registry import BenchmarkProblem
from max_div.solver import Constraint, DistanceMetric, DiversityMetric, MaxDivProblem


# =================================================================================================
#  A3 - Non-Uniform - Simple constraints
# =================================================================================================
class BenchmarkProblem_A3(BenchmarkProblem):
    @classmethod
    def name(cls) -> str:
        return "A3"

    @classmethod
    def description(cls) -> str:
        return "Problem with semi-non-uniform vector density and simple constraints"

    @classmethod
    def supported_params(cls) -> dict[str, str]:
        return dict(
            size="(int) value in [1, ...].  Problem size, with d=2, n=100*size, k=10*size, m=2*size",
            diversity_metric="(DiversityMetric) diversity metric to be maximized",
        )

    @classmethod
    def get_example_parameters(cls) -> dict[str, Any]:
        return dict(
            size=1,
            diversity_metric=DiversityMetric.approx_geomean_separation(),
        )

    @classmethod
    def _create_problem_instance(cls, size: int, diversity_metric: DiversityMetric, **kwargs) -> MaxDivProblem:
        n = 100 * size
        k = 10 * size
        m = 2 * size

        # Generate semi-non-uniform random vectors (uniform + gaussian)
        np.random.seed(42)
        uniform_col = np.random.rand(n, 1)
        gaussian_col = np.random.randn(n, 1)
        vectors = np.concatenate((uniform_col, gaussian_col), axis=1).astype(np.float32)

        # Generate constraints
        constraints: list[Constraint] = []
        for i in range(m):
            # generate m bands [v_min, v_max] spanning dimension 0   (total range [0,1])
            # add specify constraint that at least 4 samples should be taken from each band
            # (k=5*m and n=50*m, so this should always be feasible)
            v_min, v_max = i / m, (i + 1) / m  # range of values in dimension 0
            indices_in_range = [idx for idx in range(n) if v_min <= vectors[idx, 0] <= v_max]
            constraints.append(Constraint(int_set=set(indices_in_range), min_count=4, max_count=k))

        return MaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.L2_EUCLIDEAN,
            diversity_metric=diversity_metric,
            constraints=constraints,
        )
