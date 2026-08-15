import math

import numpy as np

from max_div._core.benchmark_problems._registry import BenchmarkProblem
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.problem import VectorMaxDivProblem

from ._helpers import sort_vectors


# =================================================================================================
#  U1 - Clustered 2D gateway - Unconstrained
# =================================================================================================
class BenchmarkProblem_U1(BenchmarkProblem):
    """U1 is the fixed-d=2 reference problem for cross-tool comparisons.

    Its clustered geometry is chosen so solver-class quality differences are clearly visible, and
    its component proportions are fixed fractions of n, so the density structure is n-invariant:
    growing n yields the same picture, denser.
    """

    @classmethod
    def name(cls) -> str:
        return "U1"

    @classmethod
    def description(cls) -> str:
        return "Unconstrained 2D problem with clustered density, uniform background and outlier halo"

    @classmethod
    def _get_problem_dimensions(cls, n: int) -> tuple[int, int, int, int, int]:
        d = 2
        k = math.ceil(n / 10)
        m = 0
        n_con_indices = 0
        return d, n, k, m, n_con_indices

    @classmethod
    def _create_problem_instance(cls, n: int, diversity_metric: DiversityMetric) -> VectorMaxDivProblem:
        """Generate the clustered gateway geometry in the unit square.

        Components (fractions of n): three gaussian clusters with equal point counts and spreads in
        ratio ~1:4:9 (75% of mass, volumetric density spanning ~two orders of magnitude), a uniform
        background (20%), and a sparse ring of far outliers (5%).
        """
        # component counts: three equal clusters, background, and the remainder as halo
        n_cluster = int(n * 0.75) // 3
        n_background = int(n * 0.20)
        n_halo = n - 3 * n_cluster - n_background

        np.random.seed(42)
        parts = []

        # step 1 - three gaussian clusters with equal counts and strongly different spreads
        centers = [(0.25, 0.7), (0.7, 0.65), (0.55, 0.25)]
        sigmas = [0.012, 0.045, 0.11]
        for center, sigma in zip(centers, sigmas):
            parts.append(np.random.normal(loc=center, scale=sigma, size=(n_cluster, 2)))

        # step 2 - uniform background over the unit square
        parts.append(np.random.random_sample(size=(n_background, 2)))

        # step 3 - outlier halo: a sparse ring well outside the unit square's core
        angle = np.random.random_sample(size=n_halo) * 2.0 * np.pi
        radius = 0.72 + 0.10 * np.random.random_sample(size=n_halo)
        parts.append(np.stack([0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)], axis=1))

        # step 4 - sort vectors by increasing L2 norm of rows & return problem instance
        vectors = np.concatenate(parts, axis=0).astype(np.float32)
        vectors = sort_vectors(vectors)
        _, _, k, _, _ = cls._get_problem_dimensions(n)
        return VectorMaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.L2_EUCLIDEAN,
            diversity_metric=diversity_metric,
            constraints=[],
        )
