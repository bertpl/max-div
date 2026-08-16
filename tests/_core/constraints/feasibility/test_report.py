import numpy as np
import pytest

from max_div._core.constraints import Constraint, constraints_score_normalization
from max_div._core.constraints.feasibility import FeasibilityReport, FeasibilityResult, FeasibilityStatus


# =================================================================================================
#  Helpers
# =================================================================================================
def _result(status: FeasibilityStatus, bound: float = 0.0, violation: float = 0.0) -> FeasibilityResult:
    """Build a pipeline result with the fields the report reads."""
    return FeasibilityResult(
        status=status,
        selection=np.arange(8, dtype=np.int64),
        bound=bound,
        violation=violation,
        lam_min=np.zeros(1),
        lam_max=np.ones(1),
    )


# Every item is a member and k is 8, so at most 5 can be selected, leaving a worst-case violation
# of 3 and hence a score normalization of 1/(1+3).
CAP_BELOW_K = [Constraint(int_set=set(range(20)), min_count=0, max_count=5)]
CAP_BELOW_K_NORMALIZATION = constraints_score_normalization(CAP_BELOW_K, k=8)


# =================================================================================================
#  Floor and score conversion
# =================================================================================================
def test_report_converts_the_floor_onto_the_constraints_score_scale():
    """An infeasible verdict caps the score at 1 - normalization * floor."""
    # --- act ---------------------------------------------
    report = FeasibilityReport.from_result(
        _result(FeasibilityStatus.INFEASIBLE, bound=3.0, violation=3.0), CAP_BELOW_K_NORMALIZATION
    )

    # --- assert ------------------------------------------
    assert report.violation_floor == 3.0
    assert report.constraints_score_ceiling == pytest.approx(0.25)  # normalization 1/(1+3), so 1 - 3/4


@pytest.mark.parametrize(
    "status",
    [FeasibilityStatus.FEASIBLE, FeasibilityStatus.UNKNOWN],
    ids=["feasible", "unknown"],
)
def test_report_claims_no_floor_without_an_infeasibility_proof(status: FeasibilityStatus):
    """Only a proof of infeasibility may claim a floor; everything else leaves the ceiling at 1."""
    # --- act ---------------------------------------------
    report = FeasibilityReport.from_result(_result(status, bound=2.5, violation=1.0), CAP_BELOW_K_NORMALIZATION)

    # --- assert ------------------------------------------
    assert report.violation_floor == 0.0
    assert report.constraints_score_ceiling == 1.0


def test_report_clamps_a_negative_bound():
    """A dual bound below zero bounds nothing, so it must not become a negative floor."""
    # --- act ---------------------------------------------
    report = FeasibilityReport.from_result(_result(FeasibilityStatus.INFEASIBLE, bound=-1.0), CAP_BELOW_K_NORMALIZATION)

    # --- assert ------------------------------------------
    assert report.violation_floor == 0.0


# =================================================================================================
#  Rendering and certification
# =================================================================================================
@pytest.mark.parametrize(
    "status,certified",
    [(FeasibilityStatus.FEASIBLE, True), (FeasibilityStatus.INFEASIBLE, True), (FeasibilityStatus.UNKNOWN, False)],
    ids=["feasible", "infeasible", "unknown"],
)
def test_report_is_certified_only_for_the_two_proofs(status: FeasibilityStatus, certified: bool):
    """Both proofs count as certified; UNKNOWN does not."""
    # --- act ---------------------------------------------
    report = FeasibilityReport.from_result(_result(status, bound=1.0, violation=1.0), CAP_BELOW_K_NORMALIZATION)

    # --- assert ------------------------------------------
    assert report.is_certified is certified


@pytest.mark.parametrize(
    "status,expected_opening",
    [
        (FeasibilityStatus.FEASIBLE, "FEASIBLE:"),
        (FeasibilityStatus.INFEASIBLE, "INFEASIBLE:"),
        (FeasibilityStatus.UNKNOWN, "UNKNOWN:"),
    ],
    ids=["feasible", "infeasible", "unknown"],
)
def test_report_renders_each_verdict(status: FeasibilityStatus, expected_opening: str):
    """Each verdict renders a line naming itself, so a printed report is self-explaining."""
    # --- act ---------------------------------------------
    rendered = str(FeasibilityReport.from_result(_result(status, bound=3.0, violation=3.0), CAP_BELOW_K_NORMALIZATION))

    # --- assert ------------------------------------------
    assert rendered.startswith(expected_opening)


def test_report_rendering_disclaims_an_unknown_verdict():
    """UNKNOWN must read as 'nothing was learned', never as evidence against feasibility."""
    # --- act ---------------------------------------------
    rendered = str(
        FeasibilityReport.from_result(_result(FeasibilityStatus.UNKNOWN, violation=2.0), CAP_BELOW_K_NORMALIZATION)
    )

    # --- assert ------------------------------------------
    assert "says nothing about whether the constraints can be satisfied" in rendered
