"""A distance storage plan decides, before a solve starts, how each distance of the solve is stored.

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

from max_div._core.metrics import DiversityObjective
from max_div._core.metrics._distance import (
    DistanceSpec,
    FullMatrixDistanceSpec,
    VectorDistanceSpec,
    compute_full_matrix,
)

from .data_matrix_producer import AdoptingDataMatrixProducer, ComputingDataMatrixProducer, DataMatrixProducer
from .memory_budget import AUTO_MEMORY_FRACTION, data_matrix_bytes
from .storage import DistanceStorageType, DistanceStorageTypes

# The user's data matrix, the vectors or the given distances of a problem, has this matrix id.
USER_MATRIX_ID = 0


# ==================================================================================================
#  DistanceStoragePlan
# ==================================================================================================
@dataclass(frozen=True)
class DistanceStoragePlan:
    """A distance storage plan holds a solve's resolved objectives and the producers of their data matrices."""

    # The primary objective comes first, then the tie-breakers; each distance spec names the data
    # matrix that its distance store reads.
    diversity_objectives: list[DiversityObjective]
    # Every data matrix that a distance spec of `diversity_objectives` reads has its producer here, by matrix id.
    data_matrix_producers: dict[int, DataMatrixProducer]
    # Each distinct distance spec has its label and storage type here, in first-seen order with the
    # primary objective first.
    distance_storage: DistanceStorageTypes

    def __post_init__(self) -> None:
        """Reject an objective with a vector distance spec whose data matrix is not marked as preprocessed.

        Raises:
            ValueError: If a vector distance spec of an objective is not preprocessed: no distance store
                can be built over it.
        """
        for objective in self.diversity_objectives:
            for spec in objective.distinct_distance_specs():
                if isinstance(spec, VectorDistanceSpec) and not spec.is_matrix_preprocessed:
                    raise ValueError(f"A distance storage plan holds only resolved distance specs; got {spec!r}.")

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
        """Return the storage type of each declared spec; AUTO is decided here.

        An explicit choice applies to every vector spec.

        A full-matrix spec reads distances that exist already, so it is always stored as a full
        matrix.  The distances of a vector spec are computed from the vectors and never seen by the
        user, so under AUTO:

        - as many vector specs as fit in `AUTO_MEMORY_FRACTION` of the total RAM get a full matrix,
          and the others compute their distances on demand;
        - a full matrix is faster to read than any distance is to compute, so the full matrices go
          to the specs whose distances are most expensive to compute
          (`DistanceMetric.estimated_lazy_cost_ns`), and among equal estimates to the spec earlier
          in store order;
        - when the total RAM is unknown, every vector spec is lazy, because a lazy store allocates no
          full matrix that could exceed RAM.

        Raises:
            ValueError: For the LAZY storage type when no objective reads a vector distance spec (a
                distance-input problem), which leaves no vectors to compute distances from.
        """
        vector_spec_by_index = {
            i: spec for i, spec in enumerate(declared_specs) if isinstance(spec, VectorDistanceSpec)
        }
        if storage_type == DistanceStorageType.LAZY and not vector_spec_by_index:
            raise ValueError(
                "Lazy distance storage computes distances from vectors, which a distance-input "
                "problem does not have; choose FULL_MATRIX, or construct the problem from vectors."
            )

        # --- how many get a full matrix ---------
        if storage_type == DistanceStorageType.FULL_MATRIX:
            n_full_matrices = len(vector_spec_by_index)
        elif storage_type == DistanceStorageType.AUTO and total_memory_bytes is not None:
            budget_bytes = total_memory_bytes * AUTO_MEMORY_FRACTION
            n_full_matrices = int(budget_bytes // data_matrix_bytes((user_matrix_shape[0], user_matrix_shape[0])))
        else:
            n_full_matrices = 0

        # --- which ones -------------------------
        costs = {
            i: spec.metric.estimated_lazy_cost_ns(user_matrix_shape[1]) for i, spec in vector_spec_by_index.items()
        }
        # sorted() is stable, so among equal estimates the spec earlier in store order comes first
        full_matrix_indices = set(sorted(vector_spec_by_index, key=lambda i: -costs[i])[:n_full_matrices])
        lazy_indices = set(vector_spec_by_index) - full_matrix_indices
        return [
            DistanceStorageType.LAZY if i in lazy_indices else DistanceStorageType.FULL_MATRIX
            for i in range(len(declared_specs))
        ]

    @staticmethod
    def _check_fits_physical_memory(bytes_needed: int, total_memory_bytes: int | None, is_lazy_available: bool) -> None:
        """Raise early, with the remedy named, when a solve's data matrices cannot fit in physical RAM at all.

        Args:
            bytes_needed: the bytes that the allocations will claim together.
            total_memory_bytes: the total physical RAM of the machine, or None when it is unknown; None
                skips the check.
            is_lazy_available: whether the problem has vectors, so the lazy storage type can be named as the remedy.
        """
        if total_memory_bytes is not None and bytes_needed > total_memory_bytes:
            lazy_hint = " or DistanceStorageType.LAZY (no O(n²) memory)" if is_lazy_available else ""
            raise ValueError(
                f"Distance storage needs ~{bytes_needed / 2**30:.1f} GiB, but this machine "
                f"has {total_memory_bytes / 2**30:.1f} GiB of physical memory; choose a smaller problem{lazy_hint}."
            )

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
        first, are the distance stores of the solve, in store order.

        In that order, each spec gets a storage type and is then replaced by a resolved spec that names
        the data matrix that its store reads, so the matrix ids of the data matrices that the plan adds
        are deterministic:

        - a full-matrix spec stays as it is;
        - a vector spec stored as a full matrix becomes a full-matrix spec over a new data matrix,
          computed from the vectors;
        - a lazy vector spec over a metric that preprocesses gets a new data matrix holding the
          preprocessed vectors;
        - any other lazy vector spec reads the user's vectors as they are.

        Last, the plan checks that the data matrices that some resolved spec reads fit in physical
        memory; the plan keeps only those data matrices.

        Args:
            declared_objectives: the primary objective first, then the tie-breakers, as the problem
                declares them: each distance spec names the user's data matrix, and no vector
                distance spec is preprocessed.
            user_data_matrix_producer: the producer of the user's data matrix, `USER_MATRIX_ID`; for a
                vector problem, an `AdoptingDataMatrixProducer` of the vectors.
            storage_type: the user's choice of storage type, possibly AUTO.
            total_memory_bytes: the total physical RAM of the machine, or None when it is unknown.
            are_adopted_arrays_copied: whether the data matrices are produced through an allocator
                that copies the arrays that it adopts, as a parallel solve's shared memory does; the
                check that the data matrices fit in physical memory then counts those copies.

        Raises:
            ValueError: For the LAZY storage type when no objective reads a vector distance spec (a
                distance-input problem), or when the data matrices cannot fit in physical memory at all.
            TypeError: If a vector distance spec reads a data matrix that its producer computes, so no
                vectors exist to compute distances from.
        """
        # --- storage type per distinct spec -----
        declared_specs = tuple(
            dict.fromkeys(spec for objective in declared_objectives for spec in objective.distinct_distance_specs())
        )
        storage_types = cls._decide_storage_types(
            declared_specs, user_data_matrix_producer.shape, storage_type, total_memory_bytes
        )

        # --- resolve each spec ------------------
        producers: dict[int, DataMatrixProducer] = {USER_MATRIX_ID: user_data_matrix_producer}
        derived_matrix_ids = itertools.count(USER_MATRIX_ID + 1)
        resolved_spec_by_declared_spec: dict[DistanceSpec, DistanceSpec] = {}
        for spec, spec_storage_type in zip(declared_specs, storage_types, strict=True):
            if isinstance(spec, VectorDistanceSpec):
                vectors_producer = producers[spec.matrix_id]
                if not isinstance(vectors_producer, AdoptingDataMatrixProducer):
                    raise TypeError(
                        "A vector distance spec reads existing vectors; its data matrix comes from "
                        f"{vectors_producer!r}."
                    )
                vectors = vectors_producer.array
                n, n_dims = vectors.shape
                if spec_storage_type == DistanceStorageType.FULL_MATRIX:
                    matrix_id = next(derived_matrix_ids)
                    producers[matrix_id] = ComputingDataMatrixProducer(
                        (n, n), partial(compute_full_matrix, vectors, spec.metric)
                    )
                    resolved_spec_by_declared_spec[spec] = FullMatrixDistanceSpec(matrix_id=matrix_id, label=spec.label)
                elif spec.metric.needs_preprocessed_vectors:
                    matrix_id = next(derived_matrix_ids)
                    producers[matrix_id] = ComputingDataMatrixProducer(
                        (n, spec.metric.preprocessed_n_dims(n_dims)), partial(spec.metric.preprocess_into, vectors)
                    )
                    resolved_spec_by_declared_spec[spec] = VectorDistanceSpec(
                        matrix_id=matrix_id, metric=spec.metric, is_matrix_preprocessed=True
                    )
                else:
                    # the metric reads the user's vectors as they are, so they already are its preprocessed vectors
                    resolved_spec_by_declared_spec[spec] = replace(spec, is_matrix_preprocessed=True)
            else:
                resolved_spec_by_declared_spec[spec] = spec
        resolved_specs = resolved_spec_by_declared_spec.values()

        # --- keep the referenced data matrices --
        referenced_matrix_ids = {spec.matrix_id for spec in resolved_specs}
        data_matrix_producers = {
            matrix_id: producer for matrix_id, producer in producers.items() if matrix_id in referenced_matrix_ids
        }

        # --- check that they fit in memory ------
        cls._check_fits_physical_memory(
            sum(producer.bytes_to_allocate(are_adopted_arrays_copied) for producer in data_matrix_producers.values()),
            total_memory_bytes,
            is_lazy_available=any(isinstance(spec, VectorDistanceSpec) for spec in declared_specs),
        )

        return cls(
            diversity_objectives=[
                objective.with_distance_specs(resolved_spec_by_declared_spec) for objective in declared_objectives
            ],
            data_matrix_producers=data_matrix_producers,
            distance_storage=DistanceStorageTypes(
                tuple(zip((spec.label for spec in resolved_specs), storage_types, strict=True))
            ),
        )
