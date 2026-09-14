import numpy as np
import pytest

from max_div._core.constraints import Constraint
from max_div._core.metrics import (
    DistanceMetric,
    DiversityContributionFamily,
    DiversityMetric,
    DiversityObjectiveHybridFlattened,
    DiversityObjectiveHybridGeoMean,
    DiversityObjectiveSimple,
    DiversityTrackerSpec,
)
from max_div._core.solver._diversity_contribution import DiversityObjectiveBindings
from max_div._core.solver._score import Score, ScoreGenerator, _con_norm_constant

from .objectives import simple_objective, tie_breaker_objectives

SEPARATION = DiversityContributionFamily.SEPARATION
MEAN_DISTANCE = DiversityContributionFamily.MEAN_DISTANCE


def _as_contributions(separation_values: np.ndarray) -> list[np.ndarray]:
    """Wrap one separation-family contribution array as the per-spec list that `compute_score` reads."""
    return [separation_values]


def _score_generator(**kwargs) -> ScoreGenerator:
    """Build a generator with the bindings derived from its `diversity_objectives`."""
    return ScoreGenerator(bindings=DiversityObjectiveBindings.for_objectives(kwargs["diversity_objectives"]), **kwargs)


# =================================================================================================
#  Score
# =================================================================================================
def test_score_as_tuple():
    # --- arrange ----------------------
    score_1 = Score(size=0.8, constraints=0.9, diversities=(0.95, 0.7, 0.6))
    score_2 = Score(size=0.1, constraints=0.2, diversities=(0.3,))

    # --- act --------------------------
    score_tuple_1 = score_1.as_tuple()
    score_tuple_2 = score_2.as_tuple()

    # --- assert -----------------------
    assert score_tuple_1 == (0.8, 0.9, 0.95, 0.7, 0.6)
    assert score_tuple_2 == (0.1, 0.2, 0.3)


@pytest.mark.parametrize("soft", [0.0, 0.2, 0.66, 1.0])
def test_score_as_tuple_soft_constraints(soft: float):
    # --- arrange ----------------------
    score = Score(size=0.8, constraints=0.9, diversities=(0.95, 0.7, 0.6))
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
    score = Score(size=0.8, constraints=con_score, diversities=(div_score, 0.7, 0.6))
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
    score_feas = Score(size=1.0, constraints=0.8, diversities=(0.2, 0.7, 0.6))
    score_infeas = Score(size=1.0, constraints=1.0, diversities=(4.0, 0.7, 0.6))

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
    generator = _score_generator(
        n=20,
        k=3,
        diversity_objectives=[simple_objective(DiversityMetric.MIN_SEPARATION)],
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
    generator = _score_generator(
        n=100,
        k=8,
        diversity_objectives=[simple_objective(DiversityMetric.MIN_SEPARATION)],
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
    generator = _score_generator(
        n=11,
        k=8,
        diversity_objectives=[simple_objective(DiversityMetric.GEOMEAN_SEPARATION)],
        constraints=constraints,
    )

    # --- act --------------------------
    score = generator.constraints_score_for_violation(violation)

    # --- assert -----------------------
    assert score == pytest.approx(expected)


def test_constraints_score_for_violation_rejects_quadratic():
    """A scalar violation has no quadratic-scale conversion, so a quadratic generator raises."""
    # --- arrange ----------------------
    generator = _score_generator(
        n=3,
        k=3,
        diversity_objectives=[simple_objective(DiversityMetric.GEOMEAN_SEPARATION)],
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
    kwargs = {
        "n": 100,
        "k": 8,
        "diversity_objectives": [simple_objective(DiversityMetric.MIN_SEPARATION)],
    }
    gen_linear = _score_generator(constraints=constraints, **kwargs)
    gen_quad = _score_generator(constraints=constraints, penalty_quadratic=True, **kwargs)

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
    gen = _score_generator(
        n=100,
        k=8,
        diversity_objectives=[simple_objective(DiversityMetric.MIN_SEPARATION)],
        constraints=constraints,
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
    generator = _score_generator(
        n=100,
        k=8,
        diversity_objectives=[simple_objective(DiversityMetric.MIN_SEPARATION)],
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
    generator = _score_generator(
        n=100,
        k=5,
        diversity_objectives=[
            simple_objective(DiversityMetric.MIN_SEPARATION),
            *tie_breaker_objectives([DiversityMetric.MEAN_SEPARATION, DiversityMetric.NON_ZERO_SEPARATION_FRAC]),
        ],
        constraints=[],
    )

    con_values = np.zeros((0, 2), dtype=np.int32)
    sep = _as_contributions(np.array([0, 2, 3, 4, 6], dtype=np.float32))

    # --- act --------------------------
    score = generator.compute_score(5, con_values, sep)

    # --- assert -----------------------
    assert score.diversities == pytest.approx((0.0, 3.0, 0.8))  # main objective first, then the tie-breakers


def test_score_comparison_happy_path():
    # --- arrange ----------------------
    score_1a = Score(size=0.8, constraints=0.9, diversities=(0.95, 0.7, 0.6))
    score_1b = Score(size=0.8, constraints=0.9, diversities=(0.95, 0.7, 0.6))
    score_2 = Score(size=0.8, constraints=0.9, diversities=(0.95, 0.7, 0.5))
    score_3 = Score(size=0.8, constraints=0.9, diversities=(0.90, 0.9, 0.9))
    score_4 = Score(size=0.7, constraints=1.0, diversities=(1.0, 1.0, 1.0))

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
    score = Score(size=0.8, constraints=0.9, diversities=(0.95, 0.7, 0.6))

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
            Score(
                size=1.0,
                constraints=1.0,
                diversities=(
                    0.7705,
                    1.0,
                ),
            ),
            "size=1.0000 | constraints=1.0000 | diversity=0.7705",
        ),
        (
            Score(size=0.5, constraints=0.8, diversities=(0.0,)),
            "size=0.5000 | constraints=0.8000 | diversity=0.0000",
        ),
    ],
)
def test_score_str(score: Score, expected_str: str):
    # --- act --------------------------
    result = str(score)

    # --- assert -----------------------
    assert result == expected_str


_L1, _L2, _L3 = DistanceMetric.l1_manhattan(), DistanceMetric.l2_euclidean(), DistanceMetric.linf_chebyshev()


@pytest.mark.parametrize(
    "k, diversity_objective, tie_breaker, tracker_specs, contributions, expected_diversity, expected_tie_breaker",
    [
        pytest.param(
            3,
            simple_objective(DiversityMetric.MEAN_SEPARATION),
            DiversityObjectiveSimple(DiversityMetric.MEAN_PAIRWISE_DISTANCE),
            (DiversityTrackerSpec(None, SEPARATION), DiversityTrackerSpec(None, MEAN_DISTANCE)),
            [
                np.array([2.0, 4.0, 6.0], dtype=np.float32),  # the separation spec
                np.array([100.0, 100.0, 100.0], dtype=np.float32),  # the mean-distance spec
            ],
            4.0,  # mean of the separation array
            100.0,  # mean pairwise distance, from the second array
            id="one_array_per_objective",
        ),
        pytest.param(
            2,
            DiversityObjectiveHybridGeoMean(
                tuple(DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, metric) for metric in (_L1, _L2, _L3))
            ),
            DiversityObjectiveHybridFlattened(DiversityMetric.MIN_SEPARATION, (_L1, _L3)),
            tuple(DiversityTrackerSpec(metric, SEPARATION) for metric in (_L1, _L2, _L3)),
            [
                np.array([5.0, 9.0], dtype=np.float32),  # L1
                np.array(
                    [1.0, 1.0], dtype=np.float32
                ),  # L2: the smallest values, which the tie-breaker must not receive
                np.array([7.0, 8.0], dtype=np.float32),  # L3
            ],
            (5.0 * 1.0 * 7.0) ** (1.0 / 3.0),  # geomean of the three minima
            5.0,  # min over the L1 and L3 arrays only
            id="subset_of_tracked_arrays",
        ),
    ],
)
def test_compute_score_hands_each_objective_the_arrays_of_its_own_specs(
    k, diversity_objective, tie_breaker, tracker_specs, contributions, expected_diversity, expected_tie_breaker
):
    """Each objective reads its own specs' arrays by position; an unrelated tracked array is never passed to it."""
    # --- arrange ----------------------
    generator = _score_generator(
        n=10,
        k=k,
        diversity_objectives=[diversity_objective, tie_breaker],
        constraints=[],
    )

    # --- act --------------------------
    score = generator.compute_score(k, np.empty((0, 2), dtype=np.int32), contributions)

    # --- assert -----------------------
    assert DiversityObjectiveBindings.for_objectives([diversity_objective, tie_breaker]).tracker_specs == tracker_specs
    assert score.diversity == pytest.approx(expected_diversity, rel=1e-5)
    assert score.diversities[1] == pytest.approx(expected_tie_breaker)
