import numpy as np
import pytest

from max_div._core._cli.bm_solver_feasibility.main import (
    TURBO_N_SIZES,
    _build_report,
    _ceiling_cell,
    _constrained_problem_names,
    _verdict_cell,
    run_solver_feasibility_benchmark,
)
from max_div._core._cli.bm_solver_sizing import K_LADDER, determine_problem_size_for_k
from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.constraints import Constraint
from max_div._core.feasibility import FeasibilityResult, FeasibilityStatus
from max_div._core.problem import MaxDivProblem


def _result(status: FeasibilityStatus, bound: float = 0.0) -> FeasibilityResult:
    """Build a FeasibilityResult with the fields the table cells consume."""
    return FeasibilityResult(
        status=status,
        selection=np.arange(3, dtype=np.int64),
        violation=max(bound, 0.0),
        violation_per_constraint=np.zeros(1, dtype=np.int64),
        bound=bound,
        lam_min=np.zeros(1, dtype=np.float64),
        lam_max=np.zeros(1, dtype=np.float64),
    )


def test_constrained_problem_names_selects_exactly_the_problems_with_constraints():
    """The name list is derived from each problem's constraint count, not hard-coded."""
    # --- act --------------------------
    names = _constrained_problem_names()

    # --- assert -----------------------
    for name in BenchmarkProblemFactory.get_all_benchmark_names():
        probe_n = determine_problem_size_for_k(name, K_LADDER[0])
        m = BenchmarkProblemFactory.get_problem_dimensions(name, probe_n)[3]
        assert (name in names) == (m > 0)


@pytest.mark.parametrize(
    "status,expected_fragment",
    [
        (FeasibilityStatus.FEASIBLE, "**feasible** (proven)"),
        (FeasibilityStatus.UNKNOWN, "unknown"),
        (FeasibilityStatus.INFEASIBLE, "**infeasible** (proven)"),
    ],
    ids=["feasible", "unknown", "infeasible"],
)
def test_verdict_cell(status: FeasibilityStatus, expected_fragment: str):
    """Each verdict maps to its own cell text, with proofs highlighted."""
    assert expected_fragment in _verdict_cell(status)


def test_ceiling_cell_per_status():
    """Feasible rows state the exact 1.0, unknown rows show a dash, infeasible rows state the ceiling."""
    # --- arrange ----------------------
    constraints = [Constraint(int_set={0, 1, 2, 3}, min_count=4, max_count=4)]
    problem = MaxDivProblem.new(np.random.default_rng(0).random((10, 2)), k=4, constraints=constraints)

    # --- act & assert -----------------
    assert _ceiling_cell(problem, _result(FeasibilityStatus.FEASIBLE)) == "1.0"
    assert _ceiling_cell(problem, _result(FeasibilityStatus.UNKNOWN)) == "-"
    # worst case for this constraint is a violation of 4 (empty intersection at minimal selection),
    # so a certified floor of 2 maps to 1 - 2/(1+4) = 0.6 on the score scale
    assert _ceiling_cell(problem, _result(FeasibilityStatus.INFEASIBLE, bound=2.0)) == "0.60000"


def test_build_report_renders_one_row_per_size():
    """The report table carries the problem dimensions and a verdict for every requested size."""
    # --- arrange ----------------------
    sizes = [determine_problem_size_for_k("C1", k) for k in K_LADDER[:2]]

    # --- act --------------------------
    lines = _build_report("C1", sizes, max_iter=50).render(markdown=True)

    # --- assert -----------------------
    table_rows = [line for line in lines if line.startswith("|")]
    assert len(table_rows) == 2 + len(sizes)  # header + separator + one row per size
    for n in sizes:
        assert any(f"| {n} " in row for row in table_rows)


def test_run_writes_one_file_per_problem(tmp_path, monkeypatch):
    """File mode writes one markdown verdict file per constrained problem."""
    # --- arrange ----------------------
    monkeypatch.chdir(tmp_path)

    # --- act --------------------------
    run_solver_feasibility_benchmark(name="all", markdown=False, file=True, turbo=True)

    # --- assert -----------------------
    for name in _constrained_problem_names():
        content = (tmp_path / f"feasibility_verdicts_{name}.md").read_text()
        assert content.count("\n|") >= 2 + TURBO_N_SIZES


def test_run_rejects_a_problem_without_constraints():
    """An unconstrained or unknown problem name is rejected before any work happens."""
    with pytest.raises(ValueError, match="U1"):
        run_solver_feasibility_benchmark(name="U1", markdown=False, file=False, turbo=True)
