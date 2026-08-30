import numpy as np
import pytest

from max_div._core.feasibility import FeasibilityResult, FeasibilityStatus


def _result(status: FeasibilityStatus, bound: float = 0.0, violation: float = 0.0) -> FeasibilityResult:
    """Build a result with the fields the derived members read."""
    return FeasibilityResult(
        status=status,
        selection=np.arange(8, dtype=np.int64),
        violation=violation,
        violation_per_constraint=np.zeros(1, dtype=np.int64),
        bound=bound,
        lam_min=np.zeros(1),
        lam_max=np.ones(1),
    )


@pytest.mark.parametrize(
    "status,expected_floor",
    [(FeasibilityStatus.INFEASIBLE, 3.0), (FeasibilityStatus.FEASIBLE, 0.0), (FeasibilityStatus.UNKNOWN, 0.0)],
    ids=["infeasible", "feasible", "unknown"],
)
def test_violation_floor_only_follows_from_a_proof(status: FeasibilityStatus, expected_floor: float):
    """Only an infeasibility proof bounds anything; the other verdicts certify no lower bound."""
    # --- act & assert -----------------
    assert _result(status, bound=3.0).violation_floor == expected_floor


def test_violation_floor_clamps_a_negative_bound():
    """A dual value below zero bounds nothing, so it must not surface as a negative violation floor."""
    # --- act & assert -----------------
    assert _result(FeasibilityStatus.INFEASIBLE, bound=-1.0).violation_floor == 0.0


@pytest.mark.parametrize(
    "status,certified",
    [(FeasibilityStatus.FEASIBLE, True), (FeasibilityStatus.INFEASIBLE, True), (FeasibilityStatus.UNKNOWN, False)],
    ids=["feasible", "infeasible", "unknown"],
)
def test_is_certified_covers_only_the_two_proofs(status: FeasibilityStatus, certified: bool):
    """Both proofs count as certified; UNKNOWN does not."""
    # --- act & assert -----------------
    assert _result(status).is_certified is certified


@pytest.mark.parametrize(
    "status,opening",
    [
        (FeasibilityStatus.FEASIBLE, "FEASIBLE:"),
        (FeasibilityStatus.INFEASIBLE, "INFEASIBLE:"),
        (FeasibilityStatus.UNKNOWN, "UNKNOWN:"),
    ],
    ids=["feasible", "infeasible", "unknown"],
)
def test_rendering_names_its_verdict(status: FeasibilityStatus, opening: str):
    """Each verdict renders a line naming itself, so a printed result is self-explaining."""
    # --- act & assert -----------------
    assert str(_result(status, bound=3.0, violation=3.0)).startswith(opening)


def test_rendering_disclaims_an_unknown_verdict():
    """UNKNOWN must read as 'nothing was learned', never as evidence against feasibility."""
    # --- act & assert -----------------
    assert "says nothing about whether the constraints can be satisfied" in str(
        _result(FeasibilityStatus.UNKNOWN, violation=2.0)
    )


def test_rendering_reports_the_floor_and_the_selection_violation():
    """An infeasible rendering states both the certified violation floor and what the best selection carries."""
    # --- act --------------------------
    rendered = str(_result(FeasibilityStatus.INFEASIBLE, bound=3.0, violation=5.0))

    # --- assert -----------------------
    assert "at least 3" in rendered  # the certified violation floor
    assert "carries 5" in rendered  # the best selection found
