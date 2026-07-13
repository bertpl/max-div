from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric, validate_cosine_vectors
from max_div._core.metrics._distance import compute_pdist


# =================================================================================================
#  MaxDivProblem (base)
# =================================================================================================
@dataclass(frozen=True, slots=True, kw_only=True)
class MaxDivProblem(ABC):
    """Immutable definition of a Maximum Diversity Problem.

    A problem consists of ``n`` items of which ``k`` must be selected, a diversity metric,
    and optionally a list of fairness constraints. Two flavors exist, differing in how item
    dissimilarity is defined:

      - [`VectorMaxDivProblem`][max_div.problem.VectorMaxDivProblem] — items are vectors and
        distances are computed with a chosen distance metric; created via `new`.
      - [`DistanceMaxDivProblem`][max_div.problem.DistanceMaxDivProblem] — pairwise distances
        are supplied directly, for custom or non-Euclidean metrics; created via `from_distances`.

    Use the `new` / `from_distances` factory methods to create instances with validation.
    """

    # --- primary fields ----------------------------------
    k: int
    diversity_metric: DiversityMetric
    constraints: list[Constraint]

    # --- flavor-specific ---------------------------------
    @property
    @abstractmethod
    def n(self) -> int:
        """Number of items in the problem."""

    @abstractmethod
    def condensed_distances(self) -> NDArray[np.float32]:
        """Return the condensed pairwise-distance vector (scipy layout), computing it if needed."""

    # --- computed fields ---------------------------------
    @property
    def m(self) -> int:
        return len(self.constraints)

    @property
    def n_constraint_indices(self) -> int:
        return sum([len(con.int_set) for con in self.constraints])

    # --- factory methods ---------------------------------
    @classmethod
    def new(
        cls,
        vectors: np.ndarray,
        k: int,
        distance_metric: DistanceMetric = DistanceMetric.L2_EUCLIDEAN,
        diversity_metric: DiversityMetric = DiversityMetric.GEOMEAN_SEPARATION,
        constraints: list[Constraint] | None = None,
    ) -> "VectorMaxDivProblem":
        """Create a new VectorMaxDivProblem with validation.

        :param vectors: 2D numpy array of shape ``(n, d)`` with at least 3 rows.
                        Converted to ``float32`` automatically if needed.
        :param k: Number of vectors to select (must satisfy ``2 <= k <= n``).
        :param distance_metric: Distance metric for pairwise distances.
        :param diversity_metric: Diversity metric to maximize.
        :param constraints: Optional list of fairness constraints.
        """
        # --- validate ----------------
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D numpy array.")
        if vectors.shape[0] < 3:
            raise ValueError("At least 3 vectors are required to formulate a max-div problem.")
        if vectors.shape[1] == 0:
            raise ValueError("Vectors must have at least one dimension.")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if distance_metric == DistanceMetric.COSINE:
            validate_cosine_vectors(vectors)  # fail fast: zero vectors have no defined angle

        _validate_k(k, vectors.shape[0])

        if constraints is None:
            constraints = []

        # --- build -------------------
        return VectorMaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=distance_metric,
            diversity_metric=diversity_metric,
            constraints=constraints,
        )

    @classmethod
    def from_distances(
        cls,
        distances: np.ndarray,
        k: int,
        diversity_metric: DiversityMetric = DiversityMetric.GEOMEAN_SEPARATION,
        constraints: list[Constraint] | None = None,
    ) -> "DistanceMaxDivProblem":
        """Create a new DistanceMaxDivProblem from precomputed pairwise distances, with validation.

        Accepts either a square symmetric ``(n, n)`` distance matrix or a condensed distance
        vector of length ``n*(n-1)/2`` (scipy layout, as produced by ``scipy.spatial.distance.pdist``).
        Distances are converted to ``float32`` internally.

        :param distances: Square symmetric ``(n, n)`` matrix with zero diagonal, or condensed
                          1D vector of length ``n*(n-1)/2``, with at least 3 items.
                          All values must be finite and non-negative.
        :param k: Number of items to select (must satisfy ``2 <= k <= n``).
        :param diversity_metric: Diversity metric to maximize.
        :param constraints: Optional list of fairness constraints.
        """
        # --- validate & condense -----
        distances = np.asarray(distances)
        if distances.ndim == 2:
            pdist = _condense_square_distances(distances)
        elif distances.ndim == 1:
            pdist = _validate_condensed_distances(distances)
        else:
            raise ValueError(f"Distances must be a square (n, n) matrix or condensed 1D vector; got {distances.ndim}D.")

        if not np.all(np.isfinite(pdist)):
            raise ValueError("Distances must all be finite (no NaN or inf).")
        if np.any(pdist < 0.0):
            raise ValueError("Distances must all be non-negative.")

        n = _n_from_condensed_size(pdist.size)
        _validate_k(k, n)

        if constraints is None:
            constraints = []

        # --- build -------------------
        return DistanceMaxDivProblem(
            pdist=pdist,
            k=k,
            diversity_metric=diversity_metric,
            constraints=constraints,
        )


# =================================================================================================
#  Flavors
# =================================================================================================
@dataclass(frozen=True, slots=True, kw_only=True)
class VectorMaxDivProblem(MaxDivProblem):
    """MaxDivProblem flavor defined by ``n`` vectors in ``d`` dimensions plus a distance metric.

    Use `MaxDivProblem.new` to create instances with validation.
    """

    # --- primary fields ----------------------------------
    vectors: NDArray[np.float32]
    distance_metric: DistanceMetric

    # --- flavor-specific ---------------------------------
    @property
    def n(self) -> int:
        return self.vectors.shape[0]

    @property
    def d(self) -> int:
        return self.vectors.shape[1]

    def condensed_distances(self) -> NDArray[np.float32]:
        return compute_pdist(self.vectors, self.distance_metric)


@dataclass(frozen=True, slots=True, kw_only=True)
class DistanceMaxDivProblem(MaxDivProblem):
    """MaxDivProblem flavor defined directly by precomputed pairwise distances.

    Use `MaxDivProblem.from_distances` to create instances with validation.
    """

    # --- primary fields ----------------------------------
    pdist: NDArray[np.float32]

    # --- flavor-specific ---------------------------------
    @property
    def n(self) -> int:
        return _n_from_condensed_size(self.pdist.size)

    def condensed_distances(self) -> NDArray[np.float32]:
        return self.pdist


# =================================================================================================
#  Helpers
# =================================================================================================
def _validate_k(k: int, n: int) -> None:
    """Raise ValueError unless 2 <= k <= n."""
    if not (2 <= k <= n):
        raise ValueError(f"k must be in range [2, number of items (={n})]; here: {k}.")


def _n_from_condensed_size(size: int) -> int:
    """Return n such that n*(n-1)/2 == size, raising ValueError if no such integer exists."""
    n = round((1 + np.sqrt(1 + 8 * size)) / 2)
    if (n * (n - 1)) // 2 != size:
        raise ValueError(f"Condensed distance vector has invalid length {size}: not a triangular number n*(n-1)/2.")
    return n


def _condense_square_distances(distances: np.ndarray) -> NDArray[np.float32]:
    """Validate a square symmetric distance matrix and return its condensed float32 form."""
    n = distances.shape[0]
    if distances.shape[0] != distances.shape[1]:
        raise ValueError(f"Square distance matrix must be (n, n); got {distances.shape}.")
    if n < 3:
        raise ValueError("At least 3 items are required to formulate a max-div problem.")
    distances = distances.astype(np.float32)
    if not np.allclose(np.diag(distances), 0.0, atol=1e-6):
        raise ValueError("Square distance matrix must have a zero diagonal.")
    if not np.allclose(distances, distances.T, rtol=1e-5, atol=1e-6, equal_nan=True):
        raise ValueError("Square distance matrix must be symmetric.")
    i_upper, j_upper = np.triu_indices(n, k=1)
    return np.ascontiguousarray(distances[i_upper, j_upper])


def _validate_condensed_distances(distances: np.ndarray) -> NDArray[np.float32]:
    """Validate a condensed distance vector and return it as contiguous float32."""
    n = _n_from_condensed_size(distances.size)
    if n < 3:
        raise ValueError("At least 3 items are required to formulate a max-div problem.")
    return np.ascontiguousarray(distances, dtype=np.float32)
