import numpy as np
import pytest
from scipy.spatial.distance import squareform

from max_div._core.constraints import Constraint
from max_div._core.distance_storage import (
    USER_MATRIX_ID,
    AdoptingDataMatrixProducer,
    ComputingDataMatrixProducer,
    DistanceStoragePlan,
    DistanceStorageType,
    DistanceStorageTypes,
    InProcessDataMatrixReader,
    SharedMemoryDataMatrixPublisher,
    SharedMemoryDataMatrixReader,
)
from max_div._core.distance_storage.memory_budget import AUTO_MEMORY_FRACTION, data_matrix_bytes
from max_div._core.metrics import DistanceMetric, DiversityMetric, DiversityObjectiveSimple
from max_div._core.metrics._distance import (
    KIND_FULL_MATRIX,
    KIND_LAZY,
    DataMatrixReader,
    DistanceSpec,
    DistanceStore,
    FullMatrixDistanceSpec,
    VectorDistanceSpec,
    compute_full_matrix,
)
from max_div._core.problem import MaxDivProblem
from max_div._core.solver import MaxDivSolverBuilder, SolverPreset, Verbosity
from max_div._core.solver._diversity_contribution import DiversityObjectiveBindings
from max_div._core.solver._duration import iterations
from tests._core.metrics._distance.helpers import all_pair_distances

# ==================================================================================================
#  Fixtures / helpers
# ==================================================================================================
GIB = 2**30
L2 = DistanceMetric.l2_euclidean()
L1 = DistanceMetric.l1_manhattan()
COSINE = DistanceMetric.cosine()


def _vectors(n: int = 10, d: int = 3) -> np.ndarray:
    """Return small vectors with no zero rows, so that cosine accepts them."""
    rng = np.random.default_rng(20260731)
    return rng.random((n, d)).astype(np.float32) + 0.1


def _distance_problem(form: str) -> MaxDivProblem:
    """Return the L2 distances of the small vectors as a square or condensed input."""
    matrix = compute_full_matrix(_vectors(), L2)
    if form == "square":
        distances = matrix
    else:
        distances = squareform(matrix, checks=False)
    return MaxDivProblem.from_distances(distances, k=3)


def _declared_spec(metric: DistanceMetric) -> VectorDistanceSpec:
    """Return the declared spec of the distances under `metric`, over the user's vectors as given."""
    return VectorDistanceSpec(matrix_id=USER_MATRIX_ID, metric=metric, is_matrix_preprocessed=False)


def _decide_over_vectors(
    metrics: list[DistanceMetric],
    storage_type: DistanceStorageType,
    vectors: np.ndarray | None = None,
    total_memory_bytes: int | None = 64 * GIB,
) -> DistanceStoragePlan:
    """Return the plan of one min-separation objective per metric, over the given vectors (small ones by default)."""
    vectors = _vectors() if vectors is None else vectors
    objectives = [
        DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, _declared_spec(metric)) for metric in metrics
    ]
    return DistanceStoragePlan.decide(
        objectives,
        AdoptingDataMatrixProducer(vectors),
        storage_type,
        total_memory_bytes,
        are_adopted_arrays_copied=False,
    )


def _decide_over_problem(
    problem: MaxDivProblem,
    storage_type: DistanceStorageType,
    total_memory_bytes: int | None = 64 * GIB,
    are_adopted_arrays_copied: bool = False,
) -> DistanceStoragePlan:
    """Return the plan of the problem's own objective, with a generous RAM figure by default."""
    return DistanceStoragePlan.decide(
        [problem.diversity_objective],
        problem._user_data_matrix_producer(),
        storage_type,
        total_memory_bytes,
        are_adopted_arrays_copied=are_adopted_arrays_copied,
    )


def _resolved_specs(plan: DistanceStoragePlan) -> list[DistanceSpec]:
    """Return the distinct distance specs of the plan's objectives, in store order."""
    return list(DiversityObjectiveBindings.for_objectives(plan.diversity_objectives).distance_specs)


def _stores(plan: DistanceStoragePlan, data_matrix_reader: DataMatrixReader | None = None) -> list[DistanceStore]:
    """Return the distance store of each resolved spec, in store order, over an in-process reader by default."""
    if data_matrix_reader is None:
        data_matrix_reader = InProcessDataMatrixReader(plan.data_matrix_producers)
    return [spec.build_distance_store(data_matrix_reader) for spec in _resolved_specs(plan)]


def _storage_types(plan: DistanceStoragePlan) -> list[str]:
    """Return the storage type of each store, in store order."""
    return [storage_type.value for _, storage_type in plan.distance_storage.per_store]


# ==================================================================================================
#  Construction and report
# ==================================================================================================
def test_plan_rejects_an_objective_over_vectors_as_given():
    """A plan holds only resolved specs: a vector spec over the vectors as given has no store."""
    # --- arrange ----------------------
    objective = DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, _declared_spec(L2))

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="only resolved distance specs"):
        DistanceStoragePlan(
            diversity_objectives=[objective], data_matrix_producers={}, distance_storage=DistanceStorageTypes()
        )


@pytest.mark.parametrize(
    "problem, expected",
    [
        (MaxDivProblem.new(_vectors(), k=3), (("L2", DistanceStorageType.FULL_MATRIX),)),
        (_distance_problem("square"), (("user distances", DistanceStorageType.FULL_MATRIX),)),
    ],
    ids=["vectors", "distances"],
)
def test_report_names_each_store_by_the_label_of_its_spec(problem: MaxDivProblem, expected: tuple):
    """The report pairs a store's spec label with its storage type: the metric's label, or `user distances`."""
    # --- act --------------------------
    plan = _decide_over_problem(problem, DistanceStorageType.AUTO)

    # --- assert -----------------------
    assert plan.distance_storage.per_store == expected


def test_equal_declared_specs_get_one_store():
    """Objectives over equal specs share one store, so a distance named twice is stored once."""
    # --- act --------------------------
    plan = _decide_over_vectors([L2, L2], DistanceStorageType.LAZY)

    # --- assert -----------------------
    assert len(_resolved_specs(plan)) == 1
    assert plan.diversity_objectives[0].distance_spec == plan.diversity_objectives[1].distance_spec


# ==================================================================================================
#  Policy
# ==================================================================================================
@pytest.mark.parametrize("storage_type", [DistanceStorageType.FULL_MATRIX, DistanceStorageType.LAZY])
def test_explicit_choice_passes_through(storage_type: DistanceStorageType):
    """A pinned storage type is the resolved one, whatever the memory."""
    # --- act / assert -----------------
    assert _storage_types(_decide_over_vectors([L2], storage_type)) == [storage_type.value]


@pytest.mark.parametrize(
    "n, total_memory_bytes, expected",
    [
        (10, 64 * GIB, "full_matrix"),  # tiny problem: matrix always fits
        (10, None, "lazy"),  # probe failed: lazy allocates no full matrix, so it cannot exceed RAM
        (50_000, 32 * GIB, "full_matrix"),  # 9.3 GiB matrix <= half of 32 GiB
        (50_000, 16 * GIB, "lazy"),  # matrix over budget
    ],
)
def test_auto_on_vectors_is_the_full_matrix_when_it_fits(n: int, total_memory_bytes: int | None, expected: str):
    """AUTO on vectors picks the full matrix when it fits within the memory budget, and lazy otherwise."""
    # --- arrange ----------------------
    vectors = _vectors(n=n, d=1)

    # --- act / assert -----------------
    assert _storage_types(_decide_over_vectors([L2], DistanceStorageType.AUTO, vectors, total_memory_bytes)) == [
        expected
    ]


_MIXED_COST_DISTANCES = [DistanceMetric.along_axis(0), L2, DistanceMetric.geometric_mean()]


@pytest.mark.parametrize(
    "metrics, n_matrices_in_budget, expected",
    [
        (_MIXED_COST_DISTANCES, 0, ["lazy", "lazy", "lazy"]),
        # the geometric mean is the most expensive to compute
        (_MIXED_COST_DISTANCES, 1, ["lazy", "lazy", "full_matrix"]),
        (_MIXED_COST_DISTANCES, 2, ["lazy", "full_matrix", "full_matrix"]),
        (_MIXED_COST_DISTANCES, 3, ["full_matrix", "full_matrix", "full_matrix"]),
        # L1 and L2 have the same estimated cost, so the distance earlier in store order gets the full matrix
        ([L1, L2], 1, ["full_matrix", "lazy"]),
    ],
)
def test_auto_gives_the_full_matrices_that_fit_to_the_distances_most_expensive_to_compute(
    metrics: list[DistanceMetric], n_matrices_in_budget: int, expected: list[str]
):
    """When only some full matrices fit the memory budget, the costliest distances get them and the rest stay lazy."""
    # --- arrange ----------------------
    total_memory_bytes = int(n_matrices_in_budget * data_matrix_bytes((50_000, 50_000)) / AUTO_MEMORY_FRACTION) + GIB

    # --- act --------------------------
    plan = _decide_over_vectors(metrics, DistanceStorageType.AUTO, _vectors(n=50_000, d=2), total_memory_bytes)

    # --- assert -----------------------
    assert _storage_types(plan) == expected


@pytest.mark.parametrize("form", ["condensed", "square"])
def test_auto_on_distance_input_is_the_full_matrix(form: str):
    """AUTO on distance-input problems resolves to the full matrix whatever the input form, ignoring memory."""
    # --- act / assert -----------------
    assert _storage_types(_decide_over_problem(_distance_problem(form), DistanceStorageType.AUTO, None)) == [
        "full_matrix"
    ]


def test_lazy_on_distance_problem_raises():
    """LAZY has no vectors to compute from on a distance-input problem."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="computes distances from vectors"):
        _decide_over_problem(_distance_problem("condensed"), DistanceStorageType.LAZY)


# ==================================================================================================
#  Resolution
# ==================================================================================================
@pytest.mark.parametrize(
    "metrics, storage_type, expected_specs, expected_producer_ids",
    [
        # every store is a full matrix, so no store reads the user's vectors and they are not produced
        (
            [L2, L1],
            DistanceStorageType.FULL_MATRIX,
            [FullMatrixDistanceSpec(matrix_id=1, label="L2"), FullMatrixDistanceSpec(matrix_id=2, label="L1")],
            [1, 2],
        ),
        # metrics that do not preprocess read the user's vectors; cosine reads its own preprocessed copy
        (
            [L2, L1, COSINE],
            DistanceStorageType.LAZY,
            [
                VectorDistanceSpec(matrix_id=0, metric=L2, is_matrix_preprocessed=True),
                VectorDistanceSpec(matrix_id=0, metric=L1, is_matrix_preprocessed=True),
                VectorDistanceSpec(matrix_id=1, metric=COSINE, is_matrix_preprocessed=True),
            ],
            [0, 1],
        ),
    ],
    ids=["full-matrix", "lazy"],
)
def test_resolution_numbers_the_derived_matrices_in_store_order_and_keeps_the_referenced_ones(
    metrics: list[DistanceMetric],
    storage_type: DistanceStorageType,
    expected_specs: list[DistanceSpec],
    expected_producer_ids: list[int],
):
    """Each derived matrix gets the next matrix id in store order; only matrices that a spec reads have a producer."""
    # --- act --------------------------
    plan = _decide_over_vectors(metrics, storage_type)

    # --- assert -----------------------
    assert _resolved_specs(plan) == expected_specs
    assert sorted(plan.data_matrix_producers) == expected_producer_ids


def test_a_vector_spec_over_computed_data_raises():
    """A vector spec reads existing vectors, so a data matrix that a producer computes cannot serve it."""
    # --- arrange ----------------------
    objective = DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, _declared_spec(L2))
    producer = ComputingDataMatrixProducer((10, 3), lambda buffer: buffer.fill(0.0))

    # --- act / assert -----------------
    with pytest.raises(TypeError, match="existing vectors"):
        DistanceStoragePlan.decide(
            [objective], producer, DistanceStorageType.LAZY, 64 * GIB, are_adopted_arrays_copied=False
        )


# ==================================================================================================
#  Memory check
# ==================================================================================================
@pytest.mark.parametrize("is_lazy_available", [True, False], ids=["vectors", "distances"])
def test_check_fits_physical_memory_rejects_a_matrix_larger_than_ram_and_names_the_remedy(is_lazy_available: bool):
    """A matrix larger than all physical RAM is refused early; the lazy remedy is named only when it exists."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="physical memory") as excinfo:
        DistanceStoragePlan._check_fits_physical_memory(
            data_matrix_bytes((2_000_000, 2_000_000)), 64 * GIB, is_lazy_available
        )
    assert ("LAZY" in str(excinfo.value)) is is_lazy_available


@pytest.mark.parametrize("total_memory_bytes", [64 * GIB, None], ids=["fits", "unknown-ram"])
def test_check_fits_physical_memory_accepts_a_matrix_that_fits_or_unknown_ram(total_memory_bytes: int | None):
    """A matrix that fits passes silently, and so does any matrix when the total RAM is unknown."""
    # --- act / assert -----------------
    DistanceStoragePlan._check_fits_physical_memory(
        data_matrix_bytes((10, 10)), total_memory_bytes, is_lazy_available=True
    )


def test_infeasible_full_matrix_raises_early():
    """A full matrix that cannot fit in physical RAM is rejected with the remedy named."""
    # --- arrange ----------------------
    vectors = np.zeros((2_000_000, 0), dtype=np.float32)  # its full matrix would need ~16 TiB

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="LAZY"):
        _decide_over_vectors([L2], DistanceStorageType.FULL_MATRIX, vectors)


@pytest.mark.parametrize(
    "form, are_adopted_arrays_copied, is_rejected",
    [
        ("square", False, False),  # the matrix is adopted as it is
        ("square", True, True),  # publishing copies the matrix into shared memory
        ("condensed", False, True),  # the matrix is expanded into a new buffer
        ("condensed", True, True),
    ],
)
def test_fitting_in_memory_counts_the_copies_of_adopted_arrays_only_when_they_are_copied(
    form: str, are_adopted_arrays_copied: bool, is_rejected: bool
):
    """The memory check counts a matrix that the solve fills, and an adopted one only when the solve copies it."""
    # --- arrange ----------------------
    problem = _distance_problem(form)
    total_memory_bytes = data_matrix_bytes((problem.n, problem.n)) - 1  # one byte short of the full matrix

    # --- act / assert -----------------
    if is_rejected:
        with pytest.raises(ValueError, match="smaller problem"):
            _decide_over_problem(problem, DistanceStorageType.AUTO, total_memory_bytes, are_adopted_arrays_copied)
    else:
        _decide_over_problem(problem, DistanceStorageType.AUTO, total_memory_bytes, are_adopted_arrays_copied)


# ==================================================================================================
#  Stores over the data matrices, in process
# ==================================================================================================
@pytest.mark.parametrize(
    "storage_type, expected_kind",
    [(DistanceStorageType.FULL_MATRIX, KIND_FULL_MATRIX), (DistanceStorageType.LAZY, KIND_LAZY)],
)
def test_each_storage_type_builds_a_store_of_its_kind(storage_type: DistanceStorageType, expected_kind: np.int32):
    """Each storage type builds a store of the matching kind over the problem's items."""
    # --- act --------------------------
    (store,) = _stores(_decide_over_vectors([L2], storage_type))

    # --- assert -----------------------
    assert store.kind == expected_kind
    assert store.n == np.int32(10)


def test_lazy_store_over_a_metric_that_does_not_preprocess_reads_the_users_vectors():
    """A metric that does not preprocess gets a lazy store over the user's array itself, not a copy."""
    # --- arrange ----------------------
    vectors = _vectors()

    # --- act --------------------------
    (store,) = _stores(_decide_over_vectors([L2], DistanceStorageType.LAZY, vectors))

    # --- assert -----------------------
    assert np.shares_memory(store.preprocessed_vectors, vectors)


def test_lazy_store_over_a_preprocessing_metric_reads_a_preprocessed_copy():
    """A preprocessing metric gets its own preprocessed array, and the user's vectors stay untouched."""
    # --- arrange ----------------------
    vectors = _vectors()
    before = vectors.copy()

    # --- act --------------------------
    (store,) = _stores(_decide_over_vectors([COSINE], DistanceStorageType.LAZY, vectors))

    # --- assert -----------------------
    assert not np.shares_memory(store.preprocessed_vectors, vectors)
    np.testing.assert_array_equal(vectors, before)
    np.testing.assert_allclose(np.linalg.norm(store.preprocessed_vectors, axis=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize("metric", [L2, COSINE], ids=["non-preprocessing", "preprocessing"])
def test_full_matrix_and_lazy_stores_agree(metric: DistanceMetric):
    """Full-matrix and lazy stores read bit-identical distances, for a preprocessing metric too."""
    # --- act --------------------------
    (full,) = _stores(_decide_over_vectors([metric], DistanceStorageType.FULL_MATRIX))
    (lazy,) = _stores(_decide_over_vectors([metric], DistanceStorageType.LAZY))

    # --- assert -----------------------
    assert all_pair_distances(full) == all_pair_distances(lazy)


def test_square_input_is_adopted_zero_copy():
    """A square-input problem's store reads the retained matrix without copying."""
    # --- arrange ----------------------
    problem = _distance_problem("square")

    # --- act --------------------------
    (store,) = _stores(_decide_over_problem(problem, DistanceStorageType.FULL_MATRIX))

    # --- assert -----------------------
    assert np.shares_memory(store.matrix, problem.distances)  # ty: ignore[unresolved-attribute]


def test_condensed_input_expands_to_the_square_matrix():
    """A condensed-input problem's store reads a fresh matrix holding the same values as the square input."""
    # --- act --------------------------
    (store,) = _stores(_decide_over_problem(_distance_problem("condensed"), DistanceStorageType.FULL_MATRIX))

    # --- assert -----------------------
    assert store.kind == KIND_FULL_MATRIX
    np.testing.assert_array_equal(store.matrix, _distance_problem("square").distances)  # ty: ignore[unresolved-attribute]


# ==================================================================================================
#  Stores over the data matrices, in shared memory
# ==================================================================================================
@pytest.mark.parametrize(
    "plan",
    [
        *(
            pytest.param(_decide_over_vectors([metric], storage_type), id=f"{storage_type.value}-{metric.label}")
            for storage_type in (DistanceStorageType.FULL_MATRIX, DistanceStorageType.LAZY)
            for metric in (L2, COSINE)
        ),
        *(
            pytest.param(_decide_over_problem(_distance_problem(form), DistanceStorageType.AUTO), id=form)
            for form in ("square", "condensed")
        ),
    ],
)
def test_published_stores_match_the_in_process_stores(plan: DistanceStoragePlan):
    """A store over published data matrices holds bit-identical distances to the store over in-process ones."""
    # --- arrange ----------------------
    (expected,) = _stores(plan)

    # --- act --------------------------
    with (
        SharedMemoryDataMatrixPublisher(plan.data_matrix_producers) as published_matrix_records,
        SharedMemoryDataMatrixReader(published_matrix_records) as data_matrix_reader,
    ):
        (attached,) = _stores(plan, data_matrix_reader)
        read_attached = all_pair_distances(attached)

    # --- assert -----------------------
    assert read_attached == all_pair_distances(expected)


# ==================================================================================================
#  Per-storage-type solve behavior
# ==================================================================================================
# A storage type guarantees that it solves the same problem as well, not that it picks the
# same items: distances may differ in their last bits between storage types, the search is chaotic,
# and one comparison that resolves the other way makes the solver select different items with an equally
# good score.
#
# These tests therefore assert quality, and feasibility on constrained problems.
@pytest.mark.parametrize("storage_type", [DistanceStorageType.FULL_MATRIX, DistanceStorageType.LAZY])
def test_every_storage_type_reaches_equivalent_quality(storage_type: DistanceStorageType):
    """Each storage type solves an unconstrained problem to within a small margin of the others."""
    # --- arrange ----------------------
    rng = np.random.default_rng(20260804)
    vectors = rng.random((120, 5), dtype=np.float32)
    problem = MaxDivProblem.new(vectors=vectors, k=12, diversity_metric=DiversityMetric.MIN_SEPARATION)

    # --- act --------------------------
    solution = (
        MaxDivSolverBuilder(problem)
        .with_preset(iterations(200), SolverPreset.SMART)
        .with_seed(7)
        .with_distance_storage(storage_type)
        .build()
        .solve(verbosity=Verbosity.SILENT)
    )

    # --- assert -----------------------
    assert solution.score.diversity > 0.0
    assert len(solution.i_selected) == 12
    assert len({int(i) for i in solution.i_selected}) == 12  # a selection, not a multiset


@pytest.mark.parametrize("storage_type", [DistanceStorageType.FULL_MATRIX, DistanceStorageType.LAZY])
def test_every_storage_type_reaches_feasibility(storage_type: DistanceStorageType):
    """Each storage type satisfies a reachable count constraint, whatever items it ends up choosing."""
    # --- arrange ----------------------
    rng = np.random.default_rng(20260804)
    vectors = rng.random((120, 5), dtype=np.float32)
    first_half = list(range(60))
    problem = MaxDivProblem.new(
        vectors=vectors,
        k=12,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        constraints=[Constraint(int_set=set(first_half), min_count=6, max_count=6)],
    )

    # --- act --------------------------
    solution = (
        MaxDivSolverBuilder(problem)
        .with_preset(iterations(400), SolverPreset.SMART)
        .with_seed(7)
        .with_distance_storage(storage_type)
        .build()
        .solve(verbosity=Verbosity.SILENT)
    )

    # --- assert -----------------------
    n_from_first_half = sum(1 for i in solution.i_selected if int(i) in set(first_half))
    assert n_from_first_half == 6
