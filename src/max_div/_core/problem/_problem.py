import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from max_div._core.constraints import Constraint, to_numpy_constraints
from max_div._core.distance_storage import (
    USER_MATRIX_ID,
    AdoptingDataMatrixProducer,
    ComputingDataMatrixProducer,
    DataMatrixProducer,
)
from max_div._core.feasibility import (
    FeasibilityResult,
    find_feasible,
)
from max_div._core.metrics import (
    DistanceMetric,
    DiversityMetric,
    DiversityObjective,
    DiversityObjectiveSimple,
    HybridDiversityMetric,
)
from max_div._core.metrics._distance import (
    DistanceSpec,
    FullMatrixDistanceSpec,
    VectorDistanceSpec,
    compute_full_matrix,
    expand_condensed,
)

from ._validate_distances import _n_from_condensed_size, validated_condensed_distances, validated_square_distances


# ==================================================================================================
#  MaxDivProblem (base)
# ==================================================================================================
@dataclass(frozen=True, slots=True, kw_only=True)
class MaxDivProblem(ABC):
    """Immutable definition of a Maximum Diversity Problem.

    A problem consists of ``n`` items of which ``k`` must be selected, a diversity metric (a
    `DiversityMetric` over the problem's own distance, or a `HybridDiversityMetric` over several),
    and optionally a list of fairness constraints. Two flavors exist, differing in how item
    dissimilarity is defined:

      - [`VectorMaxDivProblem`][max_div.problem.VectorMaxDivProblem] — items are vectors and
        distances are computed with a chosen distance metric; created via `new`.
      - [`DistanceMaxDivProblem`][max_div.problem.DistanceMaxDivProblem] — pairwise distances
        are supplied directly, for custom or non-Euclidean metrics; created via `from_distances`.

    Every construction converts and validates the fields, whether through the `new` / `from_distances`
    factory methods or through a flavor's own constructor.
    """

    # --- primary fields -------------------------
    k: int
    diversity_metric: DiversityMetric | HybridDiversityMetric
    constraints: Sequence[Constraint]  # stored as a tuple

    # --- validation -----------------------------
    def __post_init__(self) -> None:
        """Convert and validate every field, each flavor supplying its own steps through 2 hook methods.

        A flavor supplies hooks and does not override `__post_init__`, because zero-argument `super()` raises
        inside a `slots=True` dataclass on Python 3.12.
        """
        self._convert_and_validate_data()
        # k is checked before the metrics, so a k out of range is reported as such, not as a mismatch with a metric's k
        self._validate_k(self.k, self.n)
        self._validate_metrics()
        object.__setattr__(self, "constraints", tuple(self.constraints))  # the dataclass is frozen
        self._validate_constraints(self.constraints, self.n)

    # --- flavor-specific ------------------------
    @abstractmethod
    def _convert_and_validate_data(self) -> None:
        """Store the flavor's data in the form that the solver reads, raising ValueError when it is malformed."""

    @abstractmethod
    def _validate_metrics(self) -> None:
        """Raise ValueError when a distance or diversity metric cannot be used with the flavor's data and `k`."""

    @property
    @abstractmethod
    def n(self) -> int:
        """Number of items in the problem."""

    @property
    @abstractmethod
    def has_full_matrix(self) -> bool:
        """Return True when the problem already holds its distances as a full matrix, so `full_matrix` is zero-copy."""

    @abstractmethod
    def full_matrix(self) -> NDArray[np.float32]:
        """Return the full (n, n) pairwise-distance matrix under the problem's own distance.

        The matrix is computed from the vectors, returned as given, or expanded from a condensed
        input, whichever the flavor holds; a problem that already holds a full matrix returns it
        without copying.  The solver does not read this matrix; it builds its own stores.
        """

    @abstractmethod
    def _distance_spec_of(self, distance_metric: DistanceMetric | None) -> DistanceSpec:
        """Return the distance spec of the distances that a diversity term over `distance_metric` reads.

        `None` stands for a bare `DiversityMetric`, or a hybrid term that names no distance metric; this
        method is the only code that decides what such a term reads.
        """

    @abstractmethod
    def _user_data_matrix_producer(self) -> DataMatrixProducer:
        """Return the producer of the user's data matrix: the problem's vectors or its given distances."""

    # --- computed fields ------------------------
    @property
    def diversity_objective(self) -> DiversityObjective:
        """Return the objective that the solver maximizes, declared over the user's data matrix.

        The returned objective's vector distance specs name the vectors as given, so
        `DistanceStoragePlan.decide` must resolve them before a distance store can be built over them.
        """
        if isinstance(self.diversity_metric, DiversityMetric):
            return DiversityObjectiveSimple(self.diversity_metric, self._distance_spec_of(None))
        else:
            return self.diversity_metric._to_objective(self._distance_spec_of)  # noqa: SLF001 -- kept off the public API; the problem is its intended caller

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
        con_values, con_indices = to_numpy_constraints(self.constraints, self.n)
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
        distance_metric: DistanceMetric = DistanceMetric.l2_euclidean(),  # noqa: B008 -- immutable frozen dataclass, safe as a default
        diversity_metric: DiversityMetric | HybridDiversityMetric = DiversityMetric.geomean_separation(),  # noqa: B008 -- immutable frozen dataclass, safe as a default
        constraints: Sequence[Constraint] | None = None,
    ) -> "VectorMaxDivProblem":
        """Create a VectorMaxDivProblem.

        Args:
            vectors: 2D numpy array of shape ``(n, d)`` with at least 3 rows.
                Converted to ``float32`` and C-contiguous layout automatically if needed.
            k: Number of items to select (must satisfy ``2 <= k <= n``).
            distance_metric: Distance metric for pairwise distances.
            diversity_metric: Diversity metric to maximize; a `HybridDiversityMetric` term over its own
                distance metric reads that metric, not `distance_metric`.
            constraints: Optional list of fairness constraints.
        """
        return VectorMaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=distance_metric,
            diversity_metric=diversity_metric,
            constraints=constraints if constraints is not None else (),
        )

    @classmethod
    def from_distances(
        cls,
        distances: np.ndarray,
        k: int,
        diversity_metric: DiversityMetric | HybridDiversityMetric = DiversityMetric.geomean_separation(),  # noqa: B008 -- immutable frozen dataclass, safe as a default
        constraints: Sequence[Constraint] | None = None,
    ) -> "DistanceMaxDivProblem":
        """Create a DistanceMaxDivProblem from precomputed pairwise distances.

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
            diversity_metric: Diversity metric to maximize; a `HybridDiversityMetric` term may not name a
                distance metric of its own, as there are no vectors to compute one from.
            constraints: Optional list of fairness constraints.
        """
        return DistanceMaxDivProblem(
            distances=distances,
            k=k,
            diversity_metric=diversity_metric,
            constraints=constraints if constraints is not None else (),
        )

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _validate_k(k: int, n: int) -> None:
        """Raise ValueError unless 2 <= k <= n.

        `k == n` is allowed: the selection is then forced to every item (`MaxDivSolver.solve` adopts
        that selection directly).
        """
        if not (2 <= k <= n):
            raise ValueError(f"k must be in range [2, number of items (={n})]; here: {k}.")

    @staticmethod
    def _validate_constraints(constraints: Sequence[Constraint], n: int) -> None:
        """Raise ValueError when a constraint references an item index outside the problem's `[0, n)`.

        `Constraint.__post_init__` owns every check that needs no problem context; the index-vs-`n`
        check is the one that does.  It runs here so that the error is raised when the problem is built,
        before any solve.  A `min_count` above `k` stays legal: such a constraint is unsatisfiable but
        can be intentional, and `find_feasible` reports it as infeasible with its exact violation.
        """
        for i, con in enumerate(constraints):
            largest = max(con.int_set)
            if largest >= n:
                raise ValueError(
                    f"Constraint {i} references item index {largest}, outside the problem's [0, {n}) items."
                )


# ==================================================================================================
#  Flavors
# ==================================================================================================
@dataclass(frozen=True, slots=True, kw_only=True)
class VectorMaxDivProblem(MaxDivProblem):
    """MaxDivProblem flavor defined by ``n`` vectors in ``d`` dimensions plus a distance metric.

    `MaxDivProblem.new` creates one with default metrics; the constructor converts and validates the same way.
    """

    # --- primary fields -------------------------
    vectors: NDArray[np.float32]
    distance_metric: DistanceMetric

    # --- flavor-specific ------------------------
    def _convert_and_validate_data(self) -> None:
        """Require at least 3 vectors with at least 1 dimension, and store them float32 C-contiguous."""
        vectors = np.asarray(self.vectors)
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D numpy array.")
        if vectors.shape[0] < 3:
            raise ValueError("At least 3 vectors are required to formulate a max-div problem.")
        if vectors.shape[1] == 0:
            raise ValueError("Vectors must have at least one dimension.")
        # the form every distance function expects; the dataclass is frozen
        object.__setattr__(self, "vectors", np.ascontiguousarray(vectors, dtype=np.float32))

    def _validate_metrics(self) -> None:
        """Validate every distance metric against the vectors and `k`.

        These are the problem's own distance metric, which `full_matrix()` reads, and every one that a hybrid term
        names.
        """
        distance_metrics = [self.distance_metric]
        if isinstance(self.diversity_metric, HybridDiversityMetric):
            distance_metrics.extend(self.diversity_metric.named_distance_metrics)
        for metric in distance_metrics:
            metric.validate(self.vectors, self.k)  # fail fast, before any distance store is built

    @property
    def n(self) -> int:
        return self.vectors.shape[0]

    @property
    def d(self) -> int:
        return self.vectors.shape[1]

    @property
    def has_full_matrix(self) -> bool:
        return False

    def full_matrix(self) -> NDArray[np.float32]:
        return compute_full_matrix(self.vectors, self.distance_metric)

    def _distance_spec_of(self, distance_metric: DistanceMetric | None) -> DistanceSpec:
        """Return the spec of the distances under `distance_metric`, over the vectors as given.

        `None` reads the problem's own `distance_metric`, so a term need not repeat the problem's metric.
        """
        metric = self.distance_metric if distance_metric is None else distance_metric
        return VectorDistanceSpec(matrix_id=USER_MATRIX_ID, metric=metric, is_matrix_preprocessed=False)

    def _user_data_matrix_producer(self) -> DataMatrixProducer:
        """Return the producer that adopts the vectors."""
        return AdoptingDataMatrixProducer(self.vectors)


@dataclass(frozen=True, slots=True, kw_only=True)
class DistanceMaxDivProblem(MaxDivProblem):
    """MaxDivProblem flavor defined directly by precomputed pairwise distances.

    `MaxDivProblem.from_distances` creates one with a default metric; the constructor converts and validates the
    same way.
    """

    # The given distances carry this label, because no distance metric names them.
    _USER_DISTANCES_LABEL: ClassVar[str] = "user distances"

    # --- primary fields -------------------------
    distances: NDArray[np.float32]  # as provided: (n, n) square matrix or condensed 1D vector

    # --- flavor-specific ------------------------
    def _convert_and_validate_data(self) -> None:
        """Validate the distances in the format provided, and store them float32 C-contiguous."""
        distances = np.asarray(self.distances)
        if distances.ndim == 2:
            validated = validated_square_distances(distances)
        elif distances.ndim == 1:
            validated = validated_condensed_distances(distances)
        else:
            raise ValueError(f"Distances must be a square (n, n) matrix or condensed 1D vector; got {distances.ndim}D.")
        object.__setattr__(self, "distances", validated)  # the dataclass is frozen

    def _validate_metrics(self) -> None:
        """Reject a hybrid term that names a distance metric of its own."""
        # reading `diversity_objective` calls `_distance_spec_of`, which raises for such a term
        _ = self.diversity_objective

    @property
    def n(self) -> int:
        if self.has_full_matrix:
            return self.distances.shape[0]
        return _n_from_condensed_size(self.distances.size)

    @property
    def has_full_matrix(self) -> bool:
        return self.distances.ndim == 2

    def full_matrix(self) -> NDArray[np.float32]:
        if self.has_full_matrix:
            return self.distances
        return expand_condensed(self.distances, self.n)

    def _distance_spec_of(self, distance_metric: DistanceMetric | None) -> DistanceSpec:
        """Return the full-matrix spec of the given distances.

        A term that names no distance metric reads these distances.

        Raises:
            ValueError: If `distance_metric` is not None: the problem has no vectors to compute it from.
        """
        if distance_metric is not None:
            raise ValueError(
                "A problem defined by its distances has no vectors, so a hybrid term cannot read a distance "
                f"metric of its own; got {distance_metric!r}."
            )
        return FullMatrixDistanceSpec(matrix_id=USER_MATRIX_ID, label=self._USER_DISTANCES_LABEL)

    def _user_data_matrix_producer(self) -> DataMatrixProducer:
        """Return the producer of the given distances as a full matrix; a condensed input is expanded into it."""
        if self.has_full_matrix:
            return AdoptingDataMatrixProducer(self.distances)
        else:
            n = self.n
            return ComputingDataMatrixProducer((n, n), partial(expand_condensed, self.distances, n))
