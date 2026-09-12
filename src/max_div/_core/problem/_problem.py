import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from max_div._core.constraints import Constraint, ConstraintList
from max_div._core.feasibility import (
    FeasibilityResult,
    find_feasible,
)
from max_div._core.metrics import (
    DistanceMetric,
    DiversityMetric,
    DiversityObjectiveSimple,
    validate_axis_within_dimensions,
    validate_cosine_distance_vectors,
)
from max_div._core.metrics._distance import compute_full_matrix, expand_condensed

from ._validate_distances import _n_from_condensed_size, validated_condensed_distances, validated_square_distances


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

    # --- primary fields -------------------------
    k: int
    diversity_metric: DiversityMetric
    constraints: list[Constraint]

    # --- flavor-specific ------------------------
    @property
    @abstractmethod
    def n(self) -> int:
        """Number of items in the problem."""

    @property
    @abstractmethod
    def has_full_matrix(self) -> bool:
        """Return True when the problem already holds its distances as a full matrix, so `full_matrix` is zero-copy."""

    @property
    @abstractmethod
    def default_distance_metric(self) -> DistanceMetric | None:
        """The problem's default distance, if the flavor defines one; None otherwise."""

    @abstractmethod
    def full_matrix(self) -> NDArray[np.float32]:
        """Return the full (n, n) pairwise-distance matrix under the problem's own distance.

        The matrix is computed from the vectors, returned as given, or expanded from a condensed
        input, whichever the flavor holds; a problem that already holds a full matrix returns it
        without copying.  The solver does not read this matrix; it builds its own stores.
        """

    # --- computed fields ------------------------
    @property
    def diversity_objectives(self) -> tuple[DiversityObjectiveSimple, ...]:
        """Return the simple objectives parsed from the user's input.

        Today there is one, the diversity metric over the problem's own distance (`distance_metric` is
        `None`).
        """
        return (DiversityObjectiveSimple(self.diversity_metric),)

    @property
    def m(self) -> int:
        return len(self.constraints)

    @property
    def n_constraint_indices(self) -> int:
        return sum([len(con.int_set) for con in self.constraints])

    # --- feasibility ----------------------------
    def check_feasibility(self, thorough: bool = False, max_iter: int | None = None) -> FeasibilityResult:
        """Report whether `k` items can be selected such that every constraint holds.

        Deciding feasibility is NP-complete in general, so the verdict is three-valued; see
        `FeasibilityStatus` for what each value claims.  No solver path calls `check_feasibility`.
        The certified violation floor is exact: the underlying relaxation is solved to optimality.

        Args:
            thorough: Spend more rounding attempts on the returned selection.  Verdicts are
                unaffected; on problems where neither proof exists, the extra attempts can lower
                the violation of the selection returned.
            max_iter: Deprecated and ignored; the exact relaxation solve has no iteration
                budget to set.
        """
        if max_iter is not None:
            warnings.warn(
                "check_feasibility(max_iter=...) is deprecated and ignored: the relaxation is "
                "solved exactly by an interior-point method, which needs no iteration budget.",
                DeprecationWarning,
                stacklevel=2,
            )
        con_values, con_indices = ConstraintList(self.constraints).to_numpy()
        return find_feasible(
            con_values=con_values,
            con_indices=con_indices,
            con_weights=np.array([con.weight for con in self.constraints], dtype=np.float64),
            n=self.n,
            k=self.k,
            thorough=thorough,
        )

    # --- factory methods ------------------------
    @classmethod
    def new(
        cls,
        vectors: np.ndarray,
        k: int,
        distance_metric: DistanceMetric = DistanceMetric.l2_euclidean(),  # noqa: B008 -- immutable NamedTuple, safe as a default
        diversity_metric: DiversityMetric = DiversityMetric.GEOMEAN_SEPARATION,
        constraints: list[Constraint] | None = None,
    ) -> "VectorMaxDivProblem":
        """Create a new VectorMaxDivProblem with validation.

        Args:
            vectors: 2D numpy array of shape ``(n, d)`` with at least 3 rows.
                Converted to ``float32`` and C-contiguous layout automatically if needed.
            k: Number of items to select (must satisfy ``2 <= k <= n``).
            distance_metric: Distance metric for pairwise distances.
            diversity_metric: Diversity metric to maximize.
            constraints: Optional list of fairness constraints.
        """
        # --- validate ---------------------------
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D numpy array.")
        if vectors.shape[0] < 3:
            raise ValueError("At least 3 vectors are required to formulate a max-div problem.")
        if vectors.shape[1] == 0:
            raise ValueError("Vectors must have at least one dimension.")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)  # the form every distance function expects
        if distance_metric == DistanceMetric.cosine():
            validate_cosine_distance_vectors(vectors)  # fail fast: zero vectors have no defined angle
        validate_axis_within_dimensions(distance_metric, vectors.shape[1])

        _validate_k(k, vectors.shape[0])

        if constraints is None:
            constraints = []
        _validate_constraints(constraints, vectors.shape[0])

        # --- build ------------------------------
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
        Distances are converted to ``float32`` internally.  The solver reads from a full matrix, so
        a condensed input is expanded when the solver builds its store; see `DistanceStorageType` for
        the cost.

        Args:
            distances: Square symmetric ``(n, n)`` matrix with zero diagonal, or condensed
                1D vector of length ``n*(n-1)/2``, with at least 3 items.
                All values must be finite and non-negative.
            k: Number of items to select (must satisfy ``2 <= k <= n``).
            diversity_metric: Diversity metric to maximize.
            constraints: Optional list of fairness constraints.
        """
        # --- validate, keeping the provided format ---
        distances = np.asarray(distances)
        if distances.ndim == 2:
            validated = validated_square_distances(distances)
            n = validated.shape[0]
        elif distances.ndim == 1:
            validated = validated_condensed_distances(distances)
            n = _n_from_condensed_size(validated.size)
        else:
            raise ValueError(f"Distances must be a square (n, n) matrix or condensed 1D vector; got {distances.ndim}D.")

        _validate_k(k, n)

        if constraints is None:
            constraints = []
        _validate_constraints(constraints, n)

        # --- build ------------------------------
        return DistanceMaxDivProblem(
            distances=validated,
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

    # --- primary fields -------------------------
    vectors: NDArray[np.float32]
    distance_metric: DistanceMetric

    # --- flavor-specific ------------------------
    @property
    def n(self) -> int:
        return self.vectors.shape[0]

    @property
    def d(self) -> int:
        return self.vectors.shape[1]

    @property
    def default_distance_metric(self) -> DistanceMetric | None:
        return self.distance_metric

    @property
    def has_full_matrix(self) -> bool:
        return False

    def full_matrix(self) -> NDArray[np.float32]:
        return compute_full_matrix(self.vectors, self.distance_metric)


@dataclass(frozen=True, slots=True, kw_only=True)
class DistanceMaxDivProblem(MaxDivProblem):
    """MaxDivProblem flavor defined directly by precomputed pairwise distances.

    Use `MaxDivProblem.from_distances` to create instances with validation.
    """

    # --- primary fields -------------------------
    distances: NDArray[np.float32]  # as provided: (n, n) square matrix or condensed 1D vector

    # --- flavor-specific ------------------------
    @property
    def n(self) -> int:
        if self.has_full_matrix:
            return self.distances.shape[0]
        return _n_from_condensed_size(self.distances.size)

    @property
    def default_distance_metric(self) -> DistanceMetric | None:
        return None

    @property
    def has_full_matrix(self) -> bool:
        return self.distances.ndim == 2

    def full_matrix(self) -> NDArray[np.float32]:
        if self.has_full_matrix:
            return self.distances
        return expand_condensed(self.distances, self.n)


# =================================================================================================
#  Helpers
# =================================================================================================
def _validate_k(k: int, n: int) -> None:
    """Raise ValueError unless 2 <= k <= n.

    `k == n` is allowed: the selection is then forced to every item (`MaxDivSolver.solve` adopts
    that selection directly).
    """
    if not (2 <= k <= n):
        raise ValueError(f"k must be in range [2, number of items (={n})]; here: {k}.")


def _validate_constraints(constraints: list[Constraint], n: int) -> None:
    """Raise ValueError when a constraint references an item index outside the problem's `[0, n)`.

    `Constraint.__post_init__` owns every check that needs no problem context; the index-vs-`n`
    check is the one that does.  An out-of-range index would reach compiled code with bounds
    checking off, where it is a memory error, not an exception.  `min_count` or `max_count`
    above `k` stay legal: such a constraint is unsatisfiable but meaningful, and `find_feasible`
    reports it as infeasible with its exact violation.
    """
    for i, con in enumerate(constraints):
        largest = max(con.int_set)
        if largest >= n:
            raise ValueError(f"Constraint {i} references item index {largest}, outside the problem's [0, {n}) items.")
