import urllib.error

import numpy as np
import pytest

from benchmarks.mdplib import list_instances, load_instance
from max_div.problem import DistanceMaxDivProblem, VectorMaxDivProblem


@pytest.fixture(scope="module")
def mdplib_available() -> None:
    """Skip the module when the MDPLIB archive cannot be fetched (network robustness)."""
    try:
        list_instances("Glover")
    except (urllib.error.URLError, OSError) as e:  # pragma: no cover -- network-dependent
        pytest.skip(f"MDPLIB archive not fetchable: {e}")


def test_families_have_expected_instance_counts(mdplib_available):
    # --- act / assert ------------------------------------
    assert len(list_instances("Glover")) == 15
    assert len(list_instances("Geo")) == 60
    assert len(list_instances("Ran")) == 60


def test_coordinate_instance_loads_as_vector_problem(mdplib_available):
    # --- act ---------------------------------------------
    problem = load_instance("Geo", "Geo 100 1.txt", k=10)

    # --- assert ------------------------------------------
    assert isinstance(problem, VectorMaxDivProblem)
    assert problem.n == 100
    assert problem.k == 10


def test_edge_list_instance_loads_as_distance_problem(mdplib_available):
    # --- act ---------------------------------------------
    problem = load_instance("Ran", "Ran 100 1.txt", k=10)

    # --- assert ------------------------------------------
    assert isinstance(problem, DistanceMaxDivProblem)
    assert problem.n == 100
    condensed = problem.condensed_distances()
    assert condensed.shape == (100 * 99 // 2,)
    assert np.all(condensed > 0)


def test_unknown_family_is_rejected():
    # --- act / assert ------------------------------------
    with pytest.raises(ValueError, match="Unknown MMDP family"):
        list_instances("Nope")
