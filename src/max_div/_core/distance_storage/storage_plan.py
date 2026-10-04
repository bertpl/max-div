"""A distance storage plan decides, before a solve starts, how each of its distances is stored.

A problem declares its diversity objectives over distance specs that all name the user's data
matrix, `USER_MATRIX_ID`: the problem's vectors, or the distances that it was given.
`DistanceStoragePlan.decide` turns those declared objectives into a plan:

- the resolved objectives, whose distance specs each name the data matrix that their distance
  store reads: the user's data matrix, a full distance matrix computed from the vectors, or the
  vectors preprocessed for a metric;
- the producer of every data matrix that a resolved spec reads, by matrix id;
- the storage type of each distance store, which the solution reports.

The plan produces no data matrix itself: an `InProcessDataMatrixReader`, or a
`SharedMemoryDataMatrixPublisher` for a parallel solve, runs the producers when the solve starts.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
from numpy.typing import NDArray

from max_div._core.metrics import DiversityObjective
from max_div._core.metrics._distance import (
    DistanceSpec,
    FullMatrixDistanceSpec,
    VectorDistanceSpec,
    compute_full_matrix,
)

from .data_matrix_producer import AdoptingDataMatrixProducer, ComputingDataMatrixProducer, DataMatrixProducer
from .memory_budget import AUTO_MEMORY_FRACTION, check_fits_physical_memory, full_matrix_bytes
from .storage import DistanceStorageType, DistanceStorageTypes

# The user's data matrix, the vectors or the given distances of a problem, has this matrix id.
USER_MATRIX_ID = 0


# ==================================================================================================
#  DistanceStoragePlan
# ==================================================================================================
@dataclass(frozen=True)
class DistanceStoragePlan:
    """A distance storage plan holds a solve's resolved objectives and the producers of the data matrices they read.

    "Storage type" names a `DistanceStorageType` value throughout; "kind" is reserved for
    `DistanceStore.kind`, the compiled selector that a distance store carries.
    """

    # the primary objective first, then the tie-breakers; each distance spec names the data matrix
    # that its distance store reads
    diversity_objectives: list[DiversityObjective]
    # the producer of every data matrix that a distance spec of `diversity_objectives` reads, by matrix id
    data_matrix_producers: dict[int, DataMatrixProducer]
    # the label and storage type of each distinct distance spec, in store order
    distance_storage_types: DistanceStorageTypes

    def __post_init__(self) -> None:
        """Reject an objective over a vector distance spec that names the user's vectors as given.

        Raises:
            ValueError: If a vector distance spec of an objective is not preprocessed: no distance store
                can be built over it.
        """
        for objective in self.diversity_objectives:
            for spec in objective.distinct_distance_specs():
                if isinstance(spec, VectorDistanceSpec) and not spec.is_matrix_preprocessed:
                    raise ValueError(f"A distance storage plan holds only resolved distance specs; got {spec!r}.")

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def decide(
        cls,
        declared_objectives: Sequence[DiversityObjective],
        user_data_matrix_producer: DataMatrixProducer,
        storage_type: DistanceStorageType,
        total_memory_bytes: int | None,
        are_adopted_arrays_copied: bool,
    ) -> "DistanceStoragePlan":
        """Decide how each distance of the declared objectives is stored, and return the plan.

        The distinct distance specs of the objectives, in first-seen order with the primary objective
        first, are the distance stores of the solve, in store order.  Each spec is decided and
        resolved in that order, so the matrix ids of the data matrices that the plan adds are
        deterministic:

        - a full-matrix spec stays as it is;
        - a vector spec stored as a full matrix becomes a full-matrix spec over a new data matrix,
          computed from the vectors;
        - a lazy vector spec over a metric that preprocesses gets a new data matrix holding the
          preprocessed vectors;
        - any other lazy vector spec reads the user's vectors as they are.

        The memory check runs last, over the data matrices that some resolved spec reads, which are
        the only ones the plan keeps.

        Args:
            declared_objectives: the primary objective first, then the tie-breakers, as the problem
                declares them: each distance spec names the user's data matrix, and no vector
                distance spec is preprocessed.
            user_data_matrix_producer: the producer of the user's data matrix.
            storage_type: the user's choice of storage type, possibly AUTO.
            total_memory_bytes: the total physical RAM of the machine, or None when it is unknown.
            are_adopted_arrays_copied: whether the data matrices are produced through an allocator
                that copies the arrays it adopts, as a parallel solve's shared memory does; the
                memory check then counts those copies.

        Raises:
            ValueError: For the LAZY storage type without a vector distance spec, or when the data
                matrices cannot fit in physical memory at all.
        """
        # --- decide a storage type per distinct spec --
        declared_specs = tuple(
            dict.fromkeys(spec for objective in declared_objectives for spec in objective.distinct_distance_specs())
        )
        storage_types = cls._decide_storage_types(
            declared_specs, user_data_matrix_producer.shape, storage_type, total_memory_bytes
        )

        # --- resolve each spec, adding derived matrices ---
        producers: dict[int, DataMatrixProducer] = {USER_MATRIX_ID: user_data_matrix_producer}
        derived_matrix_ids = itertools.count(USER_MATRIX_ID + 1)
        resolved_specs: dict[DistanceSpec, DistanceSpec] = {}
        for spec, spec_storage_type in zip(declared_specs, storage_types, strict=True):
            if isinstance(spec, VectorDistanceSpec):
                vectors = cls._vectors_of(producers[spec.matrix_id])
                n, n_dims = vectors.shape
                if spec_storage_type == DistanceStorageType.FULL_MATRIX:
                    matrix_id = next(derived_matrix_ids)
                    producers[matrix_id] = ComputingDataMatrixProducer(
                        (n, n), partial(compute_full_matrix, vectors, spec.metric)
                    )
                    resolved_specs[spec] = FullMatrixDistanceSpec(matrix_id=matrix_id, label=spec.label)
                elif spec.metric.needs_preprocessed_vectors:
                    matrix_id = next(derived_matrix_ids)
                    producers[matrix_id] = ComputingDataMatrixProducer(
                        (n, spec.metric.preprocessed_n_dims(n_dims)), partial(spec.metric.preprocess_into, vectors)
                    )
                    resolved_specs[spec] = VectorDistanceSpec(
                        matrix_id=matrix_id, metric=spec.metric, is_matrix_preprocessed=True
                    )
                else:
                    # the metric reads the user's vectors as they are, so they already are its preprocessed vectors
                    resolved_specs[spec] = replace(spec, is_matrix_preprocessed=True)
            else:
                resolved_specs[spec] = spec

        # --- keep the referenced data matrices ----
        referenced_matrix_ids = {spec.matrix_id for spec in resolved_specs.values()}
        data_matrix_producers = {
            matrix_id: producer for matrix_id, producer in producers.items() if matrix_id in referenced_matrix_ids
        }

        # --- check that they fit in memory -------
        check_fits_physical_memory(
            sum(producer.bytes_allocated(are_adopted_arrays_copied) for producer in data_matrix_producers.values()),
            total_memory_bytes,
            lazy_available=any(isinstance(spec, VectorDistanceSpec) for spec in declared_specs),
        )

        return cls(
            diversity_objectives=[objective.with_distance_specs(resolved_specs) for objective in declared_objectives],
            data_matrix_producers=data_matrix_producers,
            distance_storage_types=DistanceStorageTypes(
                tuple(zip((spec.label for spec in resolved_specs.values()), storage_types, strict=True))
            ),
        )

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _decide_storage_types(
        declared_specs: Sequence[DistanceSpec],
        user_matrix_shape: tuple[int, ...],
        storage_type: DistanceStorageType,
        total_memory_bytes: int | None,
    ) -> list[DistanceStorageType]:
        """Return the storage type of each declared spec; an explicit choice passes through, AUTO is decided here.

        A full-matrix spec reads distances that exist already, so it is always stored as a full
        matrix.  A vector spec computes its distances, which the user never sees, so under AUTO:

        - the vector specs get full matrices for as many of them as there is room for in
          `AUTO_MEMORY_FRACTION` of the total RAM, and compute the rest on demand;
        - a full matrix is faster to read than any distance is to compute, so the full matrices go
          to the specs whose distances are most expensive to compute
          (`DistanceMetric.estimated_lazy_cost_ns`), and among equal estimates to the spec earlier
          in store order;
        - when the total RAM is unknown, every vector spec is lazy, the one storage type that cannot page.

        Raises:
            ValueError: For the LAZY storage type without a vector spec, which has no vectors to
                compute distances from.
        """
        vector_specs = {i: spec for i, spec in enumerate(declared_specs) if isinstance(spec, VectorDistanceSpec)}
        if storage_type == DistanceStorageType.LAZY and not vector_specs:
            raise ValueError(
                "Lazy distance storage computes distances from vectors, which a distance-input "
                "problem does not have; choose FULL_MATRIX, or construct the problem from vectors."
            )

        # --- how many vector specs get a full matrix --
        if storage_type == DistanceStorageType.FULL_MATRIX:
            n_full_matrices = len(vector_specs)
        elif storage_type == DistanceStorageType.AUTO and total_memory_bytes is not None:
            budget_bytes = total_memory_bytes * AUTO_MEMORY_FRACTION
            n_full_matrices = int(budget_bytes // full_matrix_bytes(user_matrix_shape[0]))
        else:
            n_full_matrices = 0

        # --- which ones --------------------------
        costs = {i: spec.metric.estimated_lazy_cost_ns(user_matrix_shape[1]) for i, spec in vector_specs.items()}
        # sorted() is stable, so among equal estimates the spec earlier in store order comes first
        full_matrix_indices = set(sorted(vector_specs, key=lambda i: -costs[i])[:n_full_matrices])
        lazy_indices = set(vector_specs) - full_matrix_indices
        return [
            DistanceStorageType.LAZY if i in lazy_indices else DistanceStorageType.FULL_MATRIX
            for i in range(len(declared_specs))
        ]

    @staticmethod
    def _vectors_of(producer: DataMatrixProducer) -> NDArray[np.float32]:
        """Return the vectors that a vector distance spec reads, which its producer holds as an existing array.

        Raises:
            TypeError: If the producer computes its matrix: no vectors exist to compute distances from.
        """
        if isinstance(producer, AdoptingDataMatrixProducer):
            return producer.array
        else:
            raise TypeError(f"A vector distance spec reads existing vectors; its data matrix comes from {producer!r}.")
