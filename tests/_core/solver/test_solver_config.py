import numpy as np
import pytest

from max_div._core.distance_storage import DistanceStoragePlan, DistanceStorageType, InProcessDataMatrixReader
from max_div._core.metrics import DistanceMetric, DiversityMetric, DiversityObjectiveSimple
from max_div._core.metrics._distance import FullMatrixDistanceSpec
from max_div._core.problem import MaxDivProblem
from max_div._core.solver._builders import MaxDivSolverBuilder
from max_div._core.solver._duration import iterations
from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._progress_reporting import Verbosity
from max_div._core.solver._solver import MaxDivSolver


def _builder() -> MaxDivSolverBuilder:
    """Return a builder over a small problem, configured enough to resolve."""
    rng = np.random.default_rng(1234)
    problem = MaxDivProblem.new(rng.random((40, 3)).astype(np.float32), k=4)
    return MaxDivSolverBuilder(problem).with_preset(iterations(20), SolverPreset.SMART)


def test_prepare_returns_the_storage_plan_and_a_config_over_its_resolved_objectives():
    """Preparing hands back the storage plan and a config with the builder's settings and the resolved objectives."""
    # --- arrange ----------------------
    builder = _builder().with_seed(99).with_distance_storage(DistanceStorageType.FULL_MATRIX)

    # --- act --------------------------
    storage_plan, config = builder.prepare_storage_and_config()

    # --- assert -----------------------
    assert isinstance(storage_plan, DistanceStoragePlan)
    assert config.seed == 99
    assert config.k == 4
    assert config.diversity_objectives == storage_plan.diversity_objectives
    assert config.diversity_objectives[0] == DiversityObjectiveSimple(
        DiversityMetric.GEOMEAN_SEPARATION,
        FullMatrixDistanceSpec(matrix_id=1, label=DistanceMetric.l2_euclidean().label),
    )  # the problem's own metric, over the full matrix that the plan adds
    assert config.distance_storage == storage_plan.distance_storage_types


def test_a_config_builds_a_solver_over_a_data_matrix_reader():
    """A config plus a reader of its data matrices is a solver."""
    # --- arrange ----------------------
    builder = _builder()
    storage_plan, config = builder.prepare_storage_and_config()

    # --- act --------------------------
    solver = config.build_solver(data_matrix_reader=InProcessDataMatrixReader(storage_plan.data_matrix_producers))

    # --- assert -----------------------
    assert isinstance(solver, MaxDivSolver)
    assert solver.solve(verbosity=Verbosity.SILENT).i_selected.size == 4


def test_a_config_builds_a_solver_that_defers_its_data_matrices():
    """Given a provider instead of a reader, the provider is called at solve time, not before."""
    # --- arrange ----------------------
    builder = _builder()
    storage_plan, config = builder.prepare_storage_and_config()
    calls = 0

    def provide_reader():
        nonlocal calls
        calls += 1
        return InProcessDataMatrixReader(storage_plan.data_matrix_producers)

    # --- act --------------------------
    solver = config.build_solver(data_matrix_reader_provider=provide_reader)

    # --- assert -----------------------
    assert calls == 0  # nothing built until we solve
    assert solver.solve(verbosity=Verbosity.SILENT).i_selected.size == 4
    assert calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # neither
        {"data_matrix_reader": InProcessDataMatrixReader({}), "data_matrix_reader_provider": dict},  # both
    ],
)
def test_build_solver_requires_exactly_one_reader_source(kwargs):
    """Neither or both of data_matrix_reader / its provider is a caller error, not a silent fallback."""
    # --- arrange ----------------------
    _, config = _builder().prepare_storage_and_config()

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="exactly one"):
        config.build_solver(**kwargs)


def test_with_seed_changes_only_the_seed():
    """Reseeding a config leaves every other setting alone, so workers differ in search only."""
    # --- arrange ----------------------
    _, config = _builder().with_seed(1).prepare_storage_and_config()

    # --- act --------------------------
    reseeded = config.with_seed(2)

    # --- assert -----------------------
    assert reseeded.seed == 2
    assert config.seed == 1  # the original is untouched
    assert reseeded.solver_steps is config.solver_steps
    assert reseeded.distance_storage == config.distance_storage


def test_build_produces_a_working_solver():
    """`build` produces a solver over data matrices that it produces itself."""
    # --- arrange / act ----------------
    solution = _builder().with_seed(5).build().solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    assert solution.i_selected.size == 4
    assert solution.score.diversity > 0.0


def test_with_intermediate_selections_reaches_the_config():
    """The builder switch lands on the config, so a worker rebuilt from it records the selections too."""
    # --- arrange / act ----------------
    _, config_off = _builder().prepare_storage_and_config()
    _, config_on = _builder().with_intermediate_selections().prepare_storage_and_config()

    # --- assert -----------------------
    assert config_off.intermediate_selections_enabled is False
    assert config_on.intermediate_selections_enabled is True
