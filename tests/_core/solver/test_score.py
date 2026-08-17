import numpy as np
import pytest

from max_div._core.constraints import Constraint
from max_div._core.metrics import DiversityMetric
from max_div._core.solver._score import Score, ScoreGenerator, _con_norm_constant

_NO_CONTRIBUTIONS = np.array([], dtype=np.float32)


def _as_contributions(separation_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wrap separation-family contribution values into a SelectedContributions tuple."""
    return (separation_values, _NO_CONTRIBUTIONS)


# =================================================================================================
#  Score
# =================================================================================================
def test_score_as_tuple():
    # --- arrange ----------------------
    score_1 = Score(size=0.8, constraints=0.9, diversity=0.95, div_tie_breakers=(0.7, 0.6))
    score_2 = Score(size=0.1, constraints=0.2, diversity=0.3, div_tie_breakers=())

    # --- act --------------------------
    score_tuple_1 = score_1.as_tuple()
    score_tuple_2 = score_2.as_tuple()

    # --- assert -----------------------
    assert score_tuple_1 == (0.8, 0.9, 0.95, 0.7, 0.6)
    assert score_tuple_2 == (0.1, 0.2, 0.3)


@pytest.mark.parametrize("soft", [0.0, 0.2, 0.66, 1.0])
def test_score_as_tuple_soft_constraints(soft: float):
    # --- arrange ----------------------
    score = Score(size=0.8, constraints=0.9, diversity=0.95, div_tie_breakers=(0.7, 0.6))
    expected_tuple = (0.8, (0.9 ** (1 - soft)) * (0.95**soft), 0.95, 0.7, 0.6)

    # --- act --------------------------
    score_tuple = score.as_tuple(soft=soft)

    # --- assert -----------------------
    assert np.allclose(score_tuple, expected_tuple)


@pytest.mark.parametrize(
    "con_score,div_score,soft,expected_soft_con_score",
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.5, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.9, 0.0, 0.0, 0.9),
        (0.9, 0.0, 0.5, 0.0),
        (0.9, 0.0, 1.0, 0.0),
        (0.0, 0.8, 0.0, 0.0),
        (0.0, 0.8, 0.5, 0.0),
        (0.0, 0.8, 1.0, 0.8),
    ],
)
def test_score_as_tuple_soft_constraints_corner_cases(
    con_score: float, div_score: float, soft: float, expected_soft_con_score: float
):
    """Check if we don't bump into 0^0 issues."""

    # --- arrange ----------------------
    score = Score(size=0.8, constraints=con_score, diversity=div_score, div_tie_breakers=(0.7, 0.6))
    expected_tuple = (0.8, expected_soft_con_score, div_score, 0.7, 0.6)

    # --- act --------------------------
    score_tuple = score.as_tuple(soft=soft)

    # --- assert -----------------------
    assert np.allclose(score_tuple, expected_tuple)


@pytest.mark.parametrize(
    "soft,ignore_infeasible_diversity,expected_feas_tuple,expected_infeas_tuple",
    [
        (0.0, False, (1.0, 0.8, 0.2, 0.7, 0.6), (1.0, 1.0, 4.0, 0.7, 0.6)),
        (0.5, False, (1.0, 0.4, 0.2, 0.7, 0.6), (1.0, 2.0, 4.0, 0.7, 0.6)),
        (1.0, False, (1.0, 0.2, 0.2, 0.7, 0.6), (1.0, 4.0, 4.0, 0.7, 0.6)),
        (0.0, True, (1.0, 0.8, 0.0, 0.0, 0.0), (1.0, 1.0, 4.0, 0.7, 0.6)),
        (0.5, True, (1.0, 0.8, 0.0, 0.0, 0.0), (1.0, 2.0, 4.0, 0.7, 0.6)),
        (1.0, True, (1.0, 0.8, 0.0, 0.0, 0.0), (1.0, 4.0, 4.0, 0.7, 0.6)),
    ],
)
def test_score_as_tuple_ignore_infeasible_diversity(
    soft: float, ignore_infeasible_diversity: bool, expected_feas_tuple: tuple, expected_infeas_tuple: tuple
):
    # --- arrange ----------------------
    score_feas = Score(size=1.0, constraints=0.8, diversity=0.2, div_tie_breakers=(0.7, 0.6))
    score_infeas = Score(size=1.0, constraints=1.0, diversity=4.0, div_tie_breakers=(0.7, 0.6))

    # --- act --------------------------
    tuple_feas = score_feas.as_tuple(soft=soft, ignore_infeasible_diversity=ignore_infeasible_diversity)
    tuple_infeas = score_infeas.as_tuple(soft=soft, ignore_infeasible_diversity=ignore_infeasible_diversity)

    # --- assert -----------------------
    assert np.allclose(tuple_feas, expected_feas_tuple)
    assert np.allclose(tuple_infeas, expected_infeas_tuple)


# =================================================================================================
#  ScoreGenerator
# =================================================================================================
def test_score_generator_size():
    # --- arrange ----------------------
    generator = ScoreGenerator(
        n=20,
        k=3,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        diversity_tie_breakers=[],
        constraints=[],
    )

    con_values = np.zeros((0, 2), dtype=np.int32)
    selected_contributions = _as_contributions(np.ones(4, dtype=np.float32))

    # --- act --------------------------
    size_score_0 = generator.compute_score(0, con_values, selected_contributions).size
    size_score_1 = generator.compute_score(1, con_values, selected_contributions).size
    size_score_2 = generator.compute_score(2, con_values, selected_contributions).size
    size_score_3 = generator.compute_score(3, con_values, selected_contributions).size
    size_score_4 = generator.compute_score(4, con_values, selected_contributions).size
    size_score_8 = generator.compute_score(8, con_values, selected_contributions).size
    size_score_12 = generator.compute_score(12, con_values, selected_contributions).size
    size_score_20 = generator.compute_score(20, con_values, selected_contributions).size

    # --- assert -----------------------
    assert 0.0 < size_score_0 < size_score_1 < size_score_2 < size_score_3 == 1.0
    assert 1.0 == size_score_3 > size_score_4 > size_score_8 > size_score_12 > size_score_20 > 0.0


def test_score_generator_constraints():
    # --- arrange ----------------------
    generator = ScoreGenerator(
        n=100,
        k=8,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        diversity_tie_breakers=[],
        constraints=[
            Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
            Constraint(int_set={5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, min_count=2, max_count=3),
        ],
    )

    sep = _as_contributions(np.ones(5, dtype=np.float32))

    # --- act --------------------------

    # scores if we haven't selected enough from the constraint sets
    con_score_0 = generator.compute_score(8, np.array([[2, 3], [2, 3]], dtype=np.int32), sep).constraints
    con_score_2 = generator.compute_score(8, np.array([[1, 2], [1, 2]], dtype=np.int32), sep).constraints

    # scores for selections that meet the constraint requirements
    con_score_4 = generator.compute_score(8, np.array([[0, 1], [0, 1]], dtype=np.int32), sep).constraints
    con_score_5 = generator.compute_score(8, np.array([[-1, 0], [0, 1]], dtype=np.int32), sep).constraints
    con_score_6 = generator.compute_score(8, np.array([[-1, 0], [-1, 0]], dtype=np.int32), sep).constraints

    # scores if we have selected too many from the constraint sets
    con_score_7 = generator.compute_score(8, np.array([[-2, -1], [-1, 0]], dtype=np.int32), sep).constraints
    con_score_8 = generator.compute_score(8, np.array([[-2, -1], [-2, -1]], dtype=np.int32), sep).constraints

    # --- assert -----------------------
    assert 0.0 < con_score_0 < con_score_2 < con_score_4
    assert con_score_4 == con_score_5 == con_score_6 == 1.0
    assert con_score_6 > con_score_7 > con_score_8 > 0.0


@pytest.mark.parametrize(
    "weights,quadratic,expected",
    [
        ([1.0, 1.0], False, 1 / 6),  # 1 / (1 + 2 + 3)
        ([1.0, 1.0], True, 1 / 14),  # 1 / (1 + 4 + 9)
        ([2.0, 0.5], False, 1 / 6.5),  # 1 / (1 + 2·2 + 0.5·3)
        ([2.0, 0.5], True, 1 / 13.5),  # 1 / (1 + 2·4 + 0.5·9)
    ],
)
def test_con_norm_constant(weights: list[float], quadratic: bool, expected: float):
    # --- arrange ----------------------
    max_violations = [2, 3]
    con_weights = np.array(weights, dtype=np.float32)

    # --- act --------------------------
    c = _con_norm_constant(max_violations, con_weights, quadratic)

    # --- assert -----------------------
    assert c == pytest.approx(expected)


@pytest.mark.parametrize(
    "violation,expected",
    [
        (0.0, 1.0),  # no violation -> perfect score
        (4.0, 1.0 - 4.0 / 8.0),  # worst-case violations are [2, 5] -> c = 1/8
        (7.0, 1.0 - 7.0 / 8.0),  # the worst case itself stays above 0
    ],
    ids=["zero", "partial", "worst_case"],
)
def test_constraints_score_for_violation(violation: float, expected: float):
    """A total weighted violation maps onto the same 0-1 scale compute_score uses."""
    # --- arrange ----------------------
    constraints = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),  # worst case 2
        Constraint(int_set=set(range(11)), min_count=2, max_count=3),  # worst case 5
    ]
    generator = ScoreGenerator(
        n=11,
        k=8,
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        diversity_tie_breakers=[],
        constraints=constraints,
    )

    # --- act --------------------------
    score = generator.constraints_score_for_violation(violation)

    # --- assert -----------------------
    assert score == pytest.approx(expected)


def test_constraints_score_for_violation_rejects_quadratic():
    """A scalar violation has no quadratic-scale conversion, so a quadratic generator raises."""
    # --- arrange ----------------------
    generator = ScoreGenerator(
        n=3,
        k=3,
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        diversity_tie_breakers=[],
        constraints=[Constraint(int_set={0, 1, 2}, min_count=2, max_count=3)],
        penalty_quadratic=True,
    )

    # --- act & assert -----------------
    with pytest.raises(ValueError, match="profile"):
        generator.constraints_score_for_violation(1.0)


def test_score_generator_constraints_linear_vs_quadratic():
    # --- arrange ----------------------
    # max_con_violations = [max(2, min(8,5)-3, 0), max(2, min(8,11)-3, 0)] = [2, 5]
    constraints = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
        Constraint(int_set=set(range(5, 16)), min_count=2, max_count=3),
    ]
    kwargs = {"n": 100, "k": 8, "diversity_metric": DiversityMetric.MIN_SEPARATION, "diversity_tie_breakers": []}
    gen_linear = ScoreGenerator(constraints=constraints, **kwargs)
    gen_quad = ScoreGenerator(constraints=constraints, penalty_quadratic=True, **kwargs)

    con_values = np.array([[2, 3], [2, 3]], dtype=np.int32)  # need 2 more from each -> v = [2, 2]
    sep = _as_contributions(np.ones(5, dtype=np.float32))

    # --- act --------------------------
    con_linear = gen_linear.compute_score(8, con_values, sep).constraints
    con_quad = gen_quad.compute_score(8, con_values, sep).constraints

    # --- assert -----------------------
    assert con_linear == 0.5  # 1 - (1/8)·(2 + 2)          - unchanged linear behavior, exact
    assert con_quad == pytest.approx(1 - 8 / 30)  # 1 - (1/30)·(2² + 2²)


def test_score_generator_constraints_weighted():
    # --- arrange ----------------------
    # max_con_violations = [2, 5], weights = [1, 2] -> _con_c = 1 / (1 + 1·2 + 2·5) = 1/13
    constraints = [
        Constraint(int_set={0, 1, 2, 3, 4}, min_count=2, max_count=3),
        Constraint(int_set=set(range(5, 16)), min_count=2, max_count=3, weight=2.0),
    ]
    gen = ScoreGenerator(
        n=100, k=8, diversity_metric=DiversityMetric.MIN_SEPARATION, diversity_tie_breakers=[], constraints=constraints
    )
    sep = _as_contributions(np.ones(5, dtype=np.float32))

    # --- act --------------------------
    con_violate_light = gen.compute_score(
        8, np.array([[1, 3], [0, 1]], dtype=np.int32), sep
    ).constraints  # con0 short 1
    con_violate_heavy = gen.compute_score(
        8, np.array([[0, 1], [1, 3]], dtype=np.int32), sep
    ).constraints  # con1 short 1

    # --- assert -----------------------
    assert con_violate_light == pytest.approx(1 - 1 / 13)
    assert con_violate_heavy == pytest.approx(1 - 2 / 13)
    assert con_violate_heavy < con_violate_light  # violating the higher-weight constraint hurts more


def test_score_generator_constraints_no_constraints():
    # --- arrange ----------------------
    generator = ScoreGenerator(
        n=100,
        k=8,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        diversity_tie_breakers=[],
        constraints=[],
    )

    # --- act --------------------------
    score = generator.compute_score(
        8, np.zeros((0, 2), dtype=np.int32), _as_contributions(np.ones(5, dtype=np.float32))
    )

    # --- assert -----------------------
    assert score.constraints == 1.0, "In case of no constraints, we expect a perfect 1.0 constraint score."


def test_score_generator_diversity_scores():
    # --- arrange ----------------------
    generator = ScoreGenerator(
        n=100,
        k=5,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        diversity_tie_breakers=[
            DiversityMetric.MEAN_SEPARATION,
            DiversityMetric.NON_ZERO_SEPARATION_FRAC,
        ],
        constraints=[],
    )

    con_values = np.zeros((0, 2), dtype=np.int32)
    sep = _as_contributions(np.array([0, 2, 3, 4, 6], dtype=np.float32))

    # --- act --------------------------
    score = generator.compute_score(5, con_values, sep)

    # --- assert -----------------------
    assert score.diversity == pytest.approx(0.0)
    assert len(score.div_tie_breakers) == 2
    assert score.div_tie_breakers[0] == pytest.approx(3.0)
    assert score.div_tie_breakers[1] == pytest.approx(0.8)


def test_score_comparison_happy_path():
    # --- arrange ----------------------
    score_1a = Score(size=0.8, constraints=0.9, diversity=0.95, div_tie_breakers=(0.7, 0.6))
    score_1b = Score(size=0.8, constraints=0.9, diversity=0.95, div_tie_breakers=(0.7, 0.6))
    score_2 = Score(size=0.8, constraints=0.9, diversity=0.95, div_tie_breakers=(0.7, 0.5))
    score_3 = Score(size=0.8, constraints=0.9, diversity=0.90, div_tie_breakers=(0.9, 0.9))
    score_4 = Score(size=0.7, constraints=1.0, diversity=1.0, div_tie_breakers=(1.0, 1.0))

    # --- act & assert -----------------
    assert score_1a == score_1b
    assert score_1a >= score_1b
    assert score_1a <= score_1b
    assert not score_1a < score_1b
    assert not score_1a > score_1b

    assert score_1a > score_2
    assert score_2 < score_1a

    assert score_1a > score_3
    assert score_3 < score_1a

    assert score_1a > score_4
    assert score_4 < score_1a


def test_score_comparison_invalid_types():
    # --- arrange ----------------------
    score = Score(size=0.8, constraints=0.9, diversity=0.95, div_tie_breakers=(0.7, 0.6))

    # --- act & assert -----------------
    _ = score == object()  # == is implemented in object()

    with pytest.raises(TypeError):
        _ = score < 42  # type: ignore[operator]

    with pytest.raises(TypeError):
        _ = score <= 42  # type: ignore[operator]

    with pytest.raises(TypeError):
        _ = score > 42  # type: ignore[operator]

    with pytest.raises(TypeError):
        _ = score >= 42  # type: ignore[operator]


@pytest.mark.parametrize(
    "score, expected_str",
    [
        (
            Score(size=1.0, constraints=1.0, diversity=0.7705, div_tie_breakers=(1.0,)),
            "size=1.0000 | constraints=1.0000 | diversity=0.7705",
        ),
        (
            Score(size=0.5, constraints=0.8, diversity=0.0, div_tie_breakers=()),
            "size=0.5000 | constraints=0.8000 | diversity=0.0000",
        ),
    ],
)
def test_score_str(score: Score, expected_str: str):
    # --- act --------------------------
    result = str(score)

    # --- assert -----------------------
    assert result == expected_str


def test_compute_score_binds_metrics_to_their_contribution_slot():
    """Separation-family metrics must read the separation slot, regardless of what else is passed."""

    # --- arrange ----------------------
    generator = ScoreGenerator(
        n=10,
        k=4,
        diversity_metric=DiversityMetric.MEAN_SEPARATION,
        diversity_tie_breakers=[DiversityMetric.MIN_SEPARATION],
        constraints=[],
    )
    separation_values = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    decoy_values = np.array([100.0, 100.0, 100.0], dtype=np.float32)

    # --- act --------------------------
    score = generator.compute_score(3, np.empty((0, 2), dtype=np.int32), (separation_values, decoy_values))

    # --- assert -----------------------
    assert score.diversity == pytest.approx(4.0)  # mean of the separation slot, not of the decoy
    assert score.div_tie_breakers[0] == pytest.approx(2.0)  # min of the separation slot, not of the decoy
