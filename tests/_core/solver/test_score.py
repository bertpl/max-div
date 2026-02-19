import numpy as np
import pytest

from max_div._core.constraints import Constraint
from max_div._core.metrics import DiversityMetric
from max_div._core.solver._score import Score, ScoreGenerator


# =================================================================================================
#  Score
# =================================================================================================
def test_score_as_tuple():
    # --- arrange -----------------------------------------
    score_1 = Score(size=0.8, constraints=0.9, diversity=0.95, div_tie_breakers=(0.7, 0.6))
    score_2 = Score(size=0.1, constraints=0.2, diversity=0.3, div_tie_breakers=tuple())

    # --- act ---------------------------------------------
    score_tuple_1 = score_1.as_tuple()
    score_tuple_2 = score_2.as_tuple()

    # --- assert ------------------------------------------
    assert score_tuple_1 == (0.8, 0.9, 0.95, 0.7, 0.6)
    assert score_tuple_2 == (0.1, 0.2, 0.3)


@pytest.mark.parametrize("soft", [0.0, 0.2, 0.66, 1.0])
def test_score_as_tuple_soft_constraints(soft: float):
    # --- arrange -----------------------------------------
    score = Score(size=0.8, constraints=0.9, diversity=0.95, div_tie_breakers=(0.7, 0.6))
    expected_tuple = (0.8, (0.9 ** (1 - soft)) * (0.95**soft), 0.95, 0.7, 0.6)

    # --- act ---------------------------------------------
    score_tuple = score.as_tuple(soft=soft)

    # --- assert ------------------------------------------
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

    # --- arrange -----------------------------------------
    score = Score(size=0.8, constraints=con_score, diversity=div_score, div_tie_breakers=(0.7, 0.6))
    expected_tuple = (0.8, expected_soft_con_score, div_score, 0.7, 0.6)

    # --- act ---------------------------------------------
    score_tuple = score.as_tuple(soft=soft)

    # --- assert ------------------------------------------
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
    # --- arrange -----------------------------------------
    score_feas = Score(size=1.0, constraints=0.8, diversity=0.2, div_tie_breakers=(0.7, 0.6))
    score_infeas = Score(size=1.0, constraints=1.0, diversity=4.0, div_tie_breakers=(0.7, 0.6))

    # --- act ---------------------------------------------
    tuple_feas = score_feas.as_tuple(soft=soft, ignore_infeasible_diversity=ignore_infeasible_diversity)
    tuple_infeas = score_infeas.as_tuple(soft=soft, ignore_infeasible_diversity=ignore_infeasible_diversity)

    # --- assert ------------------------------------------
    assert np.allclose(tuple_feas, expected_feas_tuple)
    assert np.allclose(tuple_infeas, expected_infeas_tuple)


# =================================================================================================
#  ScoreGenerator
# =================================================================================================
def test_score_generator_size():
    # --- arrange -----------------------------------------
    generator = ScoreGenerator(
        n=20,
        k=3,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        diversity_tie_breakers=[],
        constraints=[],
    )

    con_values = np.zeros((0, 2), dtype=np.int32)
    selected_separation_array = np.ones(4, dtype=np.float32)

    # --- act ---------------------------------------------
    size_score_0 = generator.compute_score(0, con_values, selected_separation_array).size
    size_score_1 = generator.compute_score(1, con_values, selected_separation_array).size
    size_score_2 = generator.compute_score(2, con_values, selected_separation_array).size
    size_score_3 = generator.compute_score(3, con_values, selected_separation_array).size
    size_score_4 = generator.compute_score(4, con_values, selected_separation_array).size
    size_score_8 = generator.compute_score(8, con_values, selected_separation_array).size
    size_score_12 = generator.compute_score(12, con_values, selected_separation_array).size
    size_score_20 = generator.compute_score(20, con_values, selected_separation_array).size

    # --- assert ------------------------------------------
    assert 0.0 < size_score_0 < size_score_1 < size_score_2 < size_score_3 == 1.0
    assert 1.0 == size_score_3 > size_score_4 > size_score_8 > size_score_12 > size_score_20 > 0.0


def test_score_generator_constraints():
    # --- arrange -----------------------------------------
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

    sep = np.ones(5, dtype=np.float32)

    # --- act ---------------------------------------------

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

    # --- assert ------------------------------------------
    assert 0.0 < con_score_0 < con_score_2 < con_score_4
    assert con_score_4 == con_score_5 == con_score_6 == 1.0
    assert con_score_6 > con_score_7 > con_score_8 > 0.0


def test_score_generator_constraints_no_constraints():
    # --- arrange -----------------------------------------
    generator = ScoreGenerator(
        n=100,
        k=8,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        diversity_tie_breakers=[],
        constraints=[],
    )

    # --- act ---------------------------------------------
    score = generator.compute_score(8, np.zeros((0, 2), dtype=np.int32), np.ones(5, dtype=np.float32))

    # --- assert ------------------------------------------
    assert score.constraints == 1.0, "In case of no constraints, we expect a perfect 1.0 constraint score."


def test_score_generator_diversity_scores():
    # --- arrange -----------------------------------------
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
    sep = np.array([0, 2, 3, 4, 6], dtype=np.float32)

    # --- act ---------------------------------------------
    score = generator.compute_score(5, con_values, sep)

    # --- assert ------------------------------------------
    assert score.diversity == pytest.approx(0.0)
    assert len(score.div_tie_breakers) == 2
    assert score.div_tie_breakers[0] == pytest.approx(3.0)
    assert score.div_tie_breakers[1] == pytest.approx(0.8)


def test_score_comparison_happy_path():
    # --- arrange -----------------------------------------
    score_1a = Score(size=0.8, constraints=0.9, diversity=0.95, div_tie_breakers=(0.7, 0.6))
    score_1b = Score(size=0.8, constraints=0.9, diversity=0.95, div_tie_breakers=(0.7, 0.6))
    score_2 = Score(size=0.8, constraints=0.9, diversity=0.95, div_tie_breakers=(0.7, 0.5))
    score_3 = Score(size=0.8, constraints=0.9, diversity=0.90, div_tie_breakers=(0.9, 0.9))
    score_4 = Score(size=0.7, constraints=1.0, diversity=1.0, div_tie_breakers=(1.0, 1.0))

    # --- act & assert ------------------------------------
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
    # --- arrange -----------------------------------------
    score = Score(size=0.8, constraints=0.9, diversity=0.95, div_tie_breakers=(0.7, 0.6))

    # --- act & assert ------------------------------------
    _ = score == object()  # == is implemented in object()

    with pytest.raises(TypeError):
        _ = score < 42  # type: ignore

    with pytest.raises(TypeError):
        _ = score <= 42  # type: ignore

    with pytest.raises(TypeError):
        _ = score > 42  # type: ignore

    with pytest.raises(TypeError):
        _ = score >= 42  # type: ignore
