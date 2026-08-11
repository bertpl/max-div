import numpy as np

from max_div._core.metrics import DiversityMetric
from max_div._core.problem import MaxDivProblem
from max_div._core.solver._builders import MaxDivSolverBuilder
from max_div._core.solver._distance_storage import DistanceStorage, build_distance_store
from max_div._core.solver._duration import iterations
from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._progress_reporting import Verbosity
from max_div._core.solver._solver import MaxDivSolver


def _builder() -> MaxDivSolverBuilder:
    """Return a builder over a small problem, configured enough to resolve."""
    rng = np.random.default_rng(1234)
    problem = MaxDivProblem.new(rng.random((40, 3)).astype(np.float32), k=4)
    return MaxDivSolverBuilder(problem).with_preset(iterations(20), SolverPreset.SMART)


def test_resolve_returns_the_backend_and_a_config_over_it():
    """Resolving hands back the chosen backend and a config carrying the builder's settings."""
    # --- arrange -----------------------------------------
    builder = _builder().with_seed(99)

    # --- act ---------------------------------------------
    resolved, config = builder.prepare_storage_and_config()

    # --- assert ------------------------------------------
    assert resolved != DistanceStorage.AUTO  # AUTO is resolved to something concrete
    assert config.seed == 99
    assert config.k == 4
    assert config.diversity_metric == DiversityMetric.GEOMEAN_SEPARATION  # the problem's own metric


def test_a_config_builds_a_solver_over_any_store():
    """A config plus a store is a solver."""
    # --- arrange -----------------------------------------
    builder = _builder()
    resolved, config = builder.prepare_storage_and_config()

    # --- act ---------------------------------------------
    solver = config.build_solver(build_distance_store(builder._problem, resolved))

    # --- assert ------------------------------------------
    assert isinstance(solver, MaxDivSolver)
    assert solver.solve(verbosity=Verbosity.SILENT).i_selected.size == 4


def test_with_seed_changes_only_the_seed():
    """Reseeding a config leaves every other setting alone, so workers differ in search only."""
    # --- arrange -----------------------------------------
    _, config = _builder().with_seed(1).prepare_storage_and_config()

    # --- act ---------------------------------------------
    reseeded = config.with_seed(2)

    # --- assert ------------------------------------------
    assert reseeded.seed == 2
    assert config.seed == 1  # the original is untouched
    assert reseeded.solver_steps is config.solver_steps
    assert reseeded.distance_storage_label == config.distance_storage_label


def test_build_produces_a_working_solver():
    """`build` produces a solver over a store it built itself."""
    # --- arrange / act -----------------------------------
    solution = _builder().with_seed(5).build().solve(verbosity=Verbosity.SILENT)

    # --- assert ------------------------------------------
    assert solution.i_selected.size == 4
    assert solution.score.diversity > 0.0
