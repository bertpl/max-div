"""The distance store factory builds the distance stores for one solve.

Its input is the problem and the list of distance metrics that the diversity metric uses; its
output is one distance store per distance metric, in that order.  The factory owns three things:

- the policy that picks a storage type (full matrix or lazy) for each distance metric;
- one distance spec per distance store, which names the data matrix that the store reads, together
  with one source per data matrix; `DataMatrixRegistry` produces the data matrices from those
  sources, in this process or in shared memory;
- the rule for which distance stores share one data matrix: every lazy distance store whose metric
  does not preprocess the vectors reads the data matrix of the user's vectors, so those stores share one
  array and one shared-memory segment; a metric that preprocesses gets a data matrix of its own.

Preprocessing for a lazy distance store happens here; `compute_full_matrix` preprocesses the
vectors itself.
"""

import itertools
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from typing import ClassVar

from max_div._core.metrics._distance import (
    DistanceMetric,
    DistanceSpec,
    DistanceStore,
    FullMatrixDistanceSpec,
    VectorDistanceSpec,
    compute_full_matrix,
    expand_condensed,
)
from max_div._core.problem import DistanceMaxDivProblem, MaxDivProblem, VectorMaxDivProblem

from .data_matrix_registry import DataMatrixRegistry
from .data_matrix_source import ComputedDataMatrixSource, DataMatrixSource, ExistingDataMatrixSource
from .memory_budget import AUTO_MEMORY_FRACTION, check_fits_physical_memory, full_matrix_bytes
from .shared_memory import PublishedDistanceStoresRecord
from .storage import DistanceStorageType, DistanceStorageTypes


# ==================================================================================================
#  DistanceStoreFactory
# ==================================================================================================
class DistanceStoreFactory:
    """The distance store factory builds the distance stores that one solve reads.

    The memory probe is injected, so that the storage-type policy is a pure function of its
    arguments and can be tested without the machine's RAM.

    "Storage type" names a `DistanceStorageType` value throughout; "kind" is reserved for
    `DistanceStore.kind`, the compiled selector that a distance store carries.
    """

    # the problem's own array, its vectors or its given distances, has this matrix id
    _USER_MATRIX_ID: ClassVar[int] = 0
    # the distance spec over a problem's given distances carries this label, because no metric names those distances
    _USER_DISTANCES_LABEL: ClassVar[str] = "user distances"

    # --------------------------------------------------------------------------
    #  Construction
    # --------------------------------------------------------------------------
    def __init__(
        self,
        problem: MaxDivProblem,
        distances: Sequence[DistanceMetric | None],
        storage_type: DistanceStorageType,
        total_memory_bytes: int | None,
    ) -> None:
        """Bind the factory to the problem and the distances that it builds distance stores for.

        Args:
            problem: the problem whose vectors, or given distances, the distance stores hold.
            distances: one entry per distance store: a distance metric over the problem's vectors,
                or None for the given distances.  A distance-input problem accepts None only; a
                vector problem accepts distance metrics only.
            storage_type: the user's choice of storage type, possibly AUTO.
            total_memory_bytes: the total physical RAM of the machine, or None when it is unknown.

        Raises:
            ValueError: If an entry does not fit the problem flavor, or the list is empty.
        """
        if not distances:
            raise ValueError("A factory needs at least one distance.")
        is_vector_problem = isinstance(problem, VectorMaxDivProblem)
        for distance in distances:
            if not is_vector_problem and distance is not None:
                raise ValueError(
                    "A distance-input problem has no vectors; every entry must be None (its given distances)."
                )
        self._problem = problem
        self._distances = tuple(distances)
        self._storage_type = storage_type
        self._total_memory_bytes = total_memory_bytes

    @property
    def distances(self) -> tuple[DistanceMetric | None, ...]:
        """Return the distances this factory builds stores for, in store order (as the objectives declare them)."""
        return self._distances

    # --------------------------------------------------------------------------
    #  Policy
    # --------------------------------------------------------------------------
    def determine_storage_types(self) -> list[DistanceStorageType]:
        """Return the storage type for each distance; an explicit choice passes through, AUTO is decided here.

        AUTO is decided differently for the two problem flavors, deliberately:

        - For a vector problem the distances are an internal artifact that the user never sees, so
          AUTO gives full matrices to as many distances as there is room for in `AUTO_MEMORY_FRACTION`
          of the total RAM, and computes the rest on demand.
            - A full matrix is faster to read than any distance is to compute, so the matrices go to the
              distances that are most expensive to compute (`DistanceMetric.estimated_lazy_cost_ns`).
            - Among equal estimates, the distance earlier in the list gets the full matrix.
        - For a distance-input problem the distances exist already, so AUTO stores them as a full
          matrix.

        When the total RAM is unknown, AUTO picks lazy, the one storage type that cannot page.
        """
        count = len(self._distances)
        if self._storage_type != DistanceStorageType.AUTO:
            return [self._storage_type] * count
        elif not isinstance(self._problem, VectorMaxDivProblem):
            return [DistanceStorageType.FULL_MATRIX] * count
        elif self._total_memory_bytes is None:
            return [DistanceStorageType.LAZY] * count
        else:
            budget_bytes = self._total_memory_bytes * AUTO_MEMORY_FRACTION
            n_full_matrices = min(count, int(budget_bytes // full_matrix_bytes(self._problem.n)))
            costs = []
            for distance in self._resolved_distances():
                assert distance is not None  # noqa: S101 -- a vector problem always resolves to a metric
                costs.append(distance.estimated_lazy_cost_ns(self._problem.d))
            # sorted() is stable, so among equal estimates the distance earlier in the list comes first
            indices_most_expensive_first = sorted(range(count), key=lambda i: -costs[i])
            full_matrix_indices = set(indices_most_expensive_first[:n_full_matrices])
            return [
                DistanceStorageType.FULL_MATRIX if i in full_matrix_indices else DistanceStorageType.LAZY
                for i in range(count)
            ]

    def resolved_storage(self) -> DistanceStorageTypes:
        """Return each store's distance paired with its resolved storage type, in store order.

        A `None` distance (the problem's own distance) is reported as the metric it resolves to, so
        the reported distance is a metric name rather than blank.
        """
        return DistanceStorageTypes(tuple(zip(self._resolved_distances(), self.determine_storage_types(), strict=True)))

    def _resolved_distances(self) -> tuple[DistanceMetric | None, ...]:
        """Return each store's distance with a vector problem's `None` (its own distance) replaced by its metric.

        A distance-input problem's `None` stays `None`: its default distance is `None` (it holds given
        distances, not a metric).
        """
        return tuple(distance or self._problem.default_distance_metric for distance in self._distances)

    # --------------------------------------------------------------------------
    #  Construction of the stores
    # --------------------------------------------------------------------------
    def create_stores(self) -> list[DistanceStore]:
        """Build the distance stores in this process, one per distance, in store order.

        Raises:
            ValueError: For the LAZY storage type on a distance-input problem, which has no vectors
                to compute distances from, or when the full matrices cannot fit in physical memory
                at all.
        """
        sources, distance_specs = self._data_matrix_sources_and_distance_specs()
        data_matrix_reader = DataMatrixRegistry.in_process(sources)
        return [spec.build_distance_store(data_matrix_reader) for spec in distance_specs]

    def create_stores_by_distance(self) -> dict[DistanceMetric | None, DistanceStore]:
        """Build the stores in this process, keyed by the distance each was built for (`None` = the problem's own)."""
        return stores_by_distance(self._distances, self.create_stores())

    @contextmanager
    def publish_distance_stores(self) -> Iterator[PublishedDistanceStoresRecord]:
        """Build the data matrices in shared memory and yield the published distance stores, for the block's duration.

        Inside the block the shared-memory segments exist, and a worker process builds the distance
        stores with `PublishedDistanceStoresRecord.attached_distance_stores`.  On exit the segments are
        destroyed, so leave the block only after every worker is done.

        Raises:
            ValueError: as `create_stores`.
        """
        sources, distance_specs = self._data_matrix_sources_and_distance_specs()
        with DataMatrixRegistry.publish_to_shared_memory(sources) as published_matrix_records:
            yield PublishedDistanceStoresRecord(published_matrix_records, distance_specs)

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    def _data_matrix_sources_and_distance_specs(self) -> tuple[dict[int, DataMatrixSource], tuple[DistanceSpec, ...]]:
        """Return the source of each data matrix that a store reads, by matrix id, and each store's distance spec.

        Matrix id `_USER_MATRIX_ID` is the problem's own array.  Every other data matrix is derived from
        the vectors, one per store that needs one, numbered in store order: a full distance matrix, or
        the vectors preprocessed for a lazy store's metric.

        Only a data matrix that some store reads has a source, and the memory check runs here, before
        any data matrix is produced.

        Raises:
            ValueError: as `create_stores`.
        """
        store_types = self.determine_storage_types()
        problem = self._problem
        n = problem.n

        # --- given distances --------------------
        if isinstance(problem, DistanceMaxDivProblem):
            if DistanceStorageType.LAZY in store_types:
                raise ValueError(
                    "Lazy distance storage computes distances from vectors, which a distance-input "
                    "problem does not have; choose FULL_MATRIX, or construct the problem from vectors."
                )
            if not problem.has_full_matrix:
                check_fits_physical_memory(len(store_types) * full_matrix_bytes(n), lazy_available=False)
            spec = FullMatrixDistanceSpec(matrix_id=self._USER_MATRIX_ID, label=self._USER_DISTANCES_LABEL)
            return {self._USER_MATRIX_ID: self._given_distances_source(problem)}, (spec,) * len(store_types)
        if not isinstance(problem, VectorMaxDivProblem):  # pragma: no cover -- the two flavors above are the only ones
            raise TypeError(f"Unknown problem flavor {type(problem).__name__}.")

        # --- vectors ----------------------------
        n_full = sum(store_type == DistanceStorageType.FULL_MATRIX for store_type in store_types)
        if n_full:
            check_fits_physical_memory(n_full * full_matrix_bytes(n), lazy_available=True)
        sources: dict[int, DataMatrixSource] = {}
        distance_specs: list[DistanceSpec] = []
        derived_matrix_ids = itertools.count(self._USER_MATRIX_ID + 1)
        for distance, store_type in zip(self._resolved_distances(), store_types, strict=True):
            assert distance is not None  # noqa: S101 -- a vector problem always resolves to a metric
            if store_type == DistanceStorageType.FULL_MATRIX:
                matrix_id = next(derived_matrix_ids)
                sources[matrix_id] = ComputedDataMatrixSource(
                    (n, n), partial(compute_full_matrix, problem.vectors, distance)
                )
                distance_specs.append(FullMatrixDistanceSpec(matrix_id=matrix_id, label=distance.label))
            else:
                if distance.needs_preprocessed_vectors:
                    matrix_id = next(derived_matrix_ids)
                    sources[matrix_id] = ExistingDataMatrixSource(distance.preprocess(problem.vectors))
                else:
                    matrix_id = self._USER_MATRIX_ID
                    sources[matrix_id] = ExistingDataMatrixSource(problem.vectors)
                # every data matrix of a lazy store is already preprocessed for its metric: the factory
                # preprocessed it above, or the metric does not preprocess and reads the user's vectors as they are
                distance_specs.append(
                    VectorDistanceSpec(matrix_id=matrix_id, metric=distance, is_matrix_preprocessed=True)
                )
        return sources, tuple(distance_specs)

    @staticmethod
    def _given_distances_source(problem: DistanceMaxDivProblem) -> DataMatrixSource:
        """Return the source of the given distances as a full matrix; a condensed input is expanded into it."""
        if problem.has_full_matrix:
            return ExistingDataMatrixSource(problem.distances)
        else:
            n = problem.n
            return ComputedDataMatrixSource((n, n), partial(expand_condensed, problem.distances, n))


# ==================================================================================================
#  Helpers
# ==================================================================================================
def stores_by_distance(
    distances: Sequence[DistanceMetric | None], stores: Sequence[DistanceStore]
) -> dict[DistanceMetric | None, DistanceStore]:
    """Pair each distance with its store, both in the bindings' store order.

    The publisher and a worker both call this over that one order (`DiversityObjectiveBindings`), so
    the mapping a worker rebuilds from its attached stores matches the one the publisher built.
    """
    return dict(zip(distances, stores, strict=True))
