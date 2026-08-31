import numpy as np
import pytest
from scipy.spatial.distance import squareform

from max_div._core._utils import stdout_to_file
from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.problem import MaxDivProblem
from max_div._core.solver import DistanceStorage, MaxDivSolution, MaxDivSolverBuilder, Verbosity
from max_div._core.solver._duration import Elapsed, iterations
from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._score import Score


# =================================================================================================
#  Helpers
# =================================================================================================
def assert_score_checkpoints_are_sane(score_checkpoints: list[tuple[str, Elapsed, Score]]):
    # --- non-empty ------------------------------
    assert len(score_checkpoints) >= 1, "score_checkpoints must contain at least one entry"

    # --- check step names -----------------------
    singular_step_names = []
    for step_name, _, _ in score_checkpoints:
        if (len(singular_step_names) == 0) or (step_name != singular_step_names[-1]):
            # only deduplicate consecutive identical step names
            singular_step_names.append(step_name)

    assert len(set(singular_step_names)) == len(singular_step_names), (
        "score_checkpoints contains duplicate non-consecutive step names"
    )

    for i, step_name in enumerate(singular_step_names):
        # e.g. if we have 4 steps reported...
        #  - first step is step 0/3 representing SolverState initialization
        #  - other steps are step 1/3, step 2/3, step 3/3, represent actual SolverSteps
        assert f"{i}/{len(singular_step_names) - 1}" in step_name

    # --- check iteration counts -----------------
    iter_values = [e.n_iterations for _, e, _ in score_checkpoints]
    assert min(iter_values) >= 0, "score_checkpoints contains negative iteration counts"
    assert len(iter_values) == len(set(iter_values)), "score_checkpoints contains duplicate iteration counts"
    assert iter_values == sorted(iter_values), "score_checkpoints iteration counts should be strictly increasing"

    # --- check elapsed times --------------------
    t_values = [e.t_elapsed_sec for _, e, _ in score_checkpoints]
    assert min(t_values) >= 0.0, "score_checkpoints contains negative elapsed times"
    # NOTE: duplicate time values can happen if iterations are very fast, so we don't assert uniqueness here
    assert t_values == sorted(t_values), "score_checkpoints elapsed times should be non-decreasing"


# =================================================================================================
#  Tests
# =================================================================================================
def test_solver_minimal(example_solver):
    # --- act --------------------------
    solution = example_solver.solve()

    # --- assert -----------------------
    assert isinstance(solution, MaxDivSolution)
    assert_score_checkpoints_are_sane(solution.score_checkpoints)
    assert solution.duration == sum(list(solution.step_durations.values()))
    assert solution.duration == solution.score_checkpoints[-1][1]
    assert solution.score == solution.score_checkpoints[-1][2]


def test_solver_solution_constraint_counts(example_solver):
    # --- act --------------------------
    solution = example_solver.solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    assert solution.n_constraints == 2
    assert solution.n_constraints_satisfied == 2


@pytest.mark.parametrize(
    "verbosity,error_expected",
    [
        (0, False),
        (10, False),
        (20, False),
        (21, False),
        (22, False),
        (23, False),
        (24, True),
        (25, False),
        (26, True),
        (42, True),
    ],
)
def test_solver_verbosity(example_solver, tmp_path, verbosity: int, error_expected: bool):
    # --- act & assert -----------------
    if not error_expected:
        # arrange
        output_file = tmp_path / "output.txt"

        # act
        with stdout_to_file(filename=output_file):
            _ = example_solver.solve(verbosity=verbosity)

        # assert
        output_content = output_file.read_text()
        if verbosity == 0:
            assert len(output_content) == 0, f"Expected no output for verbosity=0, but got: {output_content}"
        else:
            assert output_content != "", f"Expected output for verbosity={verbosity}, but file is empty"

    else:
        # act & assert
        with pytest.raises(ValueError):
            _ = example_solver.solve(verbosity=verbosity)


@pytest.mark.parametrize("form", ["condensed", "square"])
def test_solver_vector_and_distance_input_bit_identical(form: str):
    """Solving via from_distances with the vector flavor's own distances gives bit-identical solutions.

    Square input is retained as a full matrix, so this also exercises the full-matrix store on the
    distance-input path.
    """

    # --- arrange ----------------------
    rng = np.random.default_rng(20260713)
    vectors = rng.random((40, 4)).astype(np.float32)
    kwargs: dict = {"k": 8, "diversity_metric": DiversityMetric.GEOMEAN_SEPARATION}
    problem_vec = MaxDivProblem.new(vectors, distance_metric=DistanceMetric.l2_euclidean(), **kwargs)
    condensed = problem_vec.condensed_distances()
    distances = np.ascontiguousarray(squareform(condensed)) if form == "square" else condensed
    problem_dist = MaxDivProblem.from_distances(distances, **kwargs)

    # --- act --------------------------
    solutions = [
        MaxDivSolverBuilder(problem).with_preset(iterations(500)).with_seed(7).build().solve(verbosity=Verbosity.SILENT)
        for problem in (problem_vec, problem_dist)
    ]

    # --- assert -----------------------
    assert list(solutions[0].i_selected) == list(solutions[1].i_selected)
    assert solutions[0].score == solutions[1].score


def test_solver_deterministic_above_candidate_cap():
    """Same seed → identical selections on a problem large enough that swap candidates are subsampled."""

    # --- arrange ----------------------
    rng = np.random.default_rng(20260802)
    vectors = rng.random((600, 3)).astype(np.float32)  # pool of ~590 non-selected items exceeds the initial cap
    problem = MaxDivProblem.new(vectors, k=10, diversity_metric=DiversityMetric.GEOMEAN_SEPARATION)

    # --- act --------------------------
    solution_1 = (
        MaxDivSolverBuilder(problem).with_preset(iterations(100)).with_seed(7).build().solve(verbosity=Verbosity.SILENT)
    )
    solution_2 = (
        MaxDivSolverBuilder(problem).with_preset(iterations(100)).with_seed(7).build().solve(verbosity=Verbosity.SILENT)
    )

    # --- assert -----------------------
    assert list(solution_1.i_selected) == list(solution_2.i_selected)


def test_solver_repeated_solve_reproduces_the_first():
    """A second solve() on the same SMART solver selects what the first did; adaptive learning must not leak."""
    # --- arrange ----------------------
    rng = np.random.default_rng(20260822)
    vectors = rng.random((80, 3)).astype(np.float32)
    problem = MaxDivProblem.new(vectors, k=8)
    solver = MaxDivSolverBuilder(problem).with_preset(iterations(200), SolverPreset.SMART).with_seed(7).build()

    # --- act --------------------------
    first = solver.solve(verbosity=Verbosity.SILENT)
    second = solver.solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    assert list(first.i_selected) == list(second.i_selected)
    assert first.score == second.score


@pytest.mark.parametrize("backend", ["lazy", "full_matrix"])
@pytest.mark.parametrize("distance_metric", [DistanceMetric.l2_euclidean(), DistanceMetric.cosine()])
def test_solver_alternative_backend_bit_identical_selection(backend: str, distance_metric: DistanceMetric):
    """A solve on any alternative distance backend selects exactly what the condensed solve selects."""

    # --- arrange ----------------------
    rng = np.random.default_rng(20260731)
    vectors = rng.random((40, 4)).astype(np.float32)
    problem = MaxDivProblem.new(
        vectors, k=8, distance_metric=distance_metric, diversity_metric=DiversityMetric.GEOMEAN_SEPARATION
    )
    other_storage = DistanceStorage.LAZY if backend == "lazy" else DistanceStorage.FULL_MATRIX
    solver_condensed = (
        MaxDivSolverBuilder(problem)
        .with_preset(iterations(500))
        .with_seed(7)
        .with_distance_storage(DistanceStorage.CONDENSED)
        .build()
    )
    solver_other = (
        MaxDivSolverBuilder(problem)
        .with_preset(iterations(500))
        .with_seed(7)
        .with_distance_storage(other_storage)
        .build()
    )

    # --- act --------------------------
    solution_condensed = solver_condensed.solve(verbosity=Verbosity.SILENT)
    solution_other = solver_other.solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    assert list(solution_other.i_selected) == list(solution_condensed.i_selected)
    assert solution_other.score == solution_condensed.score


# =================================================================================================
#  MEAN_PAIRWISE_DISTANCE metric
# =================================================================================================
def _mean_pairwise_distance_of(vectors: np.ndarray, indices: np.ndarray) -> float:
    """Brute-force mean pairwise L2 distance among the vectors at 'indices'."""
    selected = vectors[indices].astype(np.float64)
    dists = [
        float(np.linalg.norm(selected[i] - selected[j]))
        for i in range(len(selected))
        for j in range(i + 1, len(selected))
    ]
    return float(np.mean(dists))


def _make_mean_pairwise_distance_problem(n: int = 60, k: int = 8) -> tuple[MaxDivProblem, np.ndarray]:
    rng = np.random.default_rng(seed=20260713)
    vectors = rng.random((n, 3)).astype(np.float32)
    problem = MaxDivProblem.new(
        vectors=vectors,
        k=k,
        distance_metric=DistanceMetric.l2_euclidean(),
        diversity_metric=DiversityMetric.MEAN_PAIRWISE_DISTANCE,
    )
    return problem, vectors


def test_solver_mean_pairwise_distance_end_to_end():
    # --- arrange ----------------------
    problem, vectors = _make_mean_pairwise_distance_problem()
    solver = MaxDivSolverBuilder(problem).with_preset(iterations(300)).with_seed(42).build()

    # --- act --------------------------
    solution = solver.solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    assert len(solution.i_selected) == problem.k
    assert len(set(solution.i_selected)) == problem.k
    # reported diversity equals brute-force mean pairwise distance of the returned selection
    expected = _mean_pairwise_distance_of(vectors, solution.i_selected)
    assert solution.score.diversity == pytest.approx(expected, rel=1e-5)


def test_solver_mean_pairwise_distance_deterministic():
    # --- arrange ----------------------
    problem, _ = _make_mean_pairwise_distance_problem()

    # --- act --------------------------
    solutions = [
        MaxDivSolverBuilder(problem).with_preset(iterations(300)).with_seed(7).build().solve(verbosity=Verbosity.SILENT)
        for _ in range(2)
    ]

    # --- assert -----------------------
    assert np.array_equal(solutions[0].i_selected, solutions[1].i_selected)
    assert solutions[0].score == solutions[1].score


def _greedy_max_sum_selection(vectors: np.ndarray, k: int) -> np.ndarray:
    """Classical greedy insertion for max-sum diversity (1/2-approximation baseline)."""
    v = vectors.astype(np.float64)
    d = np.sqrt(((v[:, None, :] - v[None, :, :]) ** 2).sum(axis=2))
    selection = list(np.unravel_index(np.argmax(d), d.shape))  # start from the farthest pair
    while len(selection) < k:
        sums = d[:, selection].sum(axis=1)
        sums[selection] = -np.inf  # already selected
        selection.append(int(np.argmax(sums)))
    return np.array(selection, dtype=np.int32)


def test_solver_mean_pairwise_distance_meets_greedy_baseline():
    """The solver must meet or beat the classical greedy max-sum baseline on a modest budget."""

    # --- arrange ----------------------
    problem, vectors = _make_mean_pairwise_distance_problem()
    greedy_score = _mean_pairwise_distance_of(vectors, _greedy_max_sum_selection(vectors, problem.k))

    # --- act --------------------------
    solution = (
        MaxDivSolverBuilder(problem)
        .with_preset(iterations(1500))
        .with_seed(3)
        .build()
        .solve(verbosity=Verbosity.SILENT)
    )

    # --- assert -----------------------
    assert solution.score.diversity >= greedy_score * (1.0 - 1e-6)


@pytest.mark.parametrize("preset", [SolverPreset.RANDOM, SolverPreset.GUIDED, SolverPreset.SMART])
@pytest.mark.parametrize("constrained", [False, True], ids=["unconstrained", "constrained"])
def test_solver_k_equals_n_returns_the_forced_full_selection(preset: SolverPreset, constrained: bool):
    """k == n adopts every item and skips the solver steps, whatever the preset and constraints."""
    # --- arrange ----------------------
    n = 20
    vectors = np.random.default_rng(0).random((n, 3)).astype(np.float32)
    constraints = [Constraint(int_set=set(range(8)), min_count=2, max_count=4)] if constrained else None
    problem = MaxDivProblem.new(vectors, k=n, constraints=constraints)

    # --- act --------------------------
    solution = MaxDivSolverBuilder(problem).with_preset(iterations(50), preset=preset).build().solve(verbosity=0)

    # --- assert -----------------------
    assert np.array_equal(np.sort(solution.i_selected), np.arange(n, dtype=np.int32))
    assert solution.score.size == 1.0
    assert solution.score.diversity > 0.0


@pytest.mark.parametrize("k", [2, 19, 20])
def test_solver_selection_is_valid_at_every_k_boundary(k: int):
    """Selections at the k boundaries (2, n-1, n) are in range, duplicate-free, and of size k."""
    # --- arrange ----------------------
    n = 20
    vectors = np.random.default_rng(1).random((n, 3)).astype(np.float32)
    problem = MaxDivProblem.new(vectors, k=k)

    # --- act --------------------------
    solution = MaxDivSolverBuilder(problem).with_preset(iterations(50)).build().solve(verbosity=0)

    # --- assert -----------------------
    selected = solution.i_selected
    assert selected.shape[0] == k
    assert len(set(selected.tolist())) == k
    assert selected.min() >= 0
    assert selected.max() < n
