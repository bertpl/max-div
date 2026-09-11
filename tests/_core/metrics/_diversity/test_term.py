from max_div._core.metrics import DistanceMetric, DiversityMetric, DiversityTerm


def test_term_distance_defaults_to_none() -> None:
    """A term over the problem's given distance carries None."""
    # --- act / assert -----------------
    assert DiversityTerm(DiversityMetric.GEOMEAN_SEPARATION).distance_metric is None


def test_term_carries_an_explicit_distance() -> None:
    """A term may name its own distance."""
    # --- arrange / act ----------------
    term = DiversityTerm(DiversityMetric.MEAN_SEPARATION, DistanceMetric.l1_manhattan())

    # --- assert -----------------------
    assert term.diversity_metric == DiversityMetric.MEAN_SEPARATION
    assert term.distance_metric == DistanceMetric.l1_manhattan()


def test_equal_terms_compare_equal() -> None:
    """Two terms with the same metric and distance are interchangeable."""
    # --- act / assert -----------------
    assert DiversityTerm(DiversityMetric.MIN_SEPARATION) == DiversityTerm(DiversityMetric.MIN_SEPARATION)
    assert DiversityTerm(DiversityMetric.MIN_SEPARATION) != DiversityTerm(DiversityMetric.MEAN_SEPARATION)
