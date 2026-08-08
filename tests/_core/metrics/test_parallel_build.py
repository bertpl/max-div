import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import DistanceStore, compute_pdist
from max_div._core.metrics._distance._parallel_build import BUILD_TILE, parallel_build_enabled


@pytest.mark.parametrize(
    "env_value, expected",
    [
        (None, True),
        ("1", True),
        ("0", False),
    ],
)
def test_parallel_build_enabled(monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected: bool):
    """Parallel builds are on by default and disabled only by MAXDIV_PARALLEL_BUILD=0."""
    # --- arrange -----------------------------------------
    if env_value is None:
        monkeypatch.delenv("MAXDIV_PARALLEL_BUILD", raising=False)
    else:
        monkeypatch.setenv("MAXDIV_PARALLEL_BUILD", env_value)

    # --- act / assert ------------------------------------
    assert parallel_build_enabled() is expected


@pytest.mark.parametrize("metric", list(DistanceMetric))
@pytest.mark.parametrize("n", [30, BUILD_TILE * 2 + 2])  # below one tile, and across uneven tiles
def test_condensed_parallel_build_bit_identical(monkeypatch: pytest.MonkeyPatch, metric: DistanceMetric, n: int):
    """The parallel condensed build produces exactly the sequential build's values."""
    # --- arrange -----------------------------------------
    rng = np.random.default_rng(7)
    vectors = rng.random((n, 7), dtype=np.float32)

    # --- act ---------------------------------------------
    monkeypatch.setenv("MAXDIV_PARALLEL_BUILD", "1")
    parallel = compute_pdist(vectors, metric)
    monkeypatch.setenv("MAXDIV_PARALLEL_BUILD", "0")
    sequential = compute_pdist(vectors, metric)

    # --- assert ------------------------------------------
    assert np.array_equal(parallel, sequential)


@pytest.mark.parametrize("metric", list(DistanceMetric))
@pytest.mark.parametrize("n", [30, BUILD_TILE * 2 + 2])  # below one tile, and across uneven tiles
def test_full_matrix_parallel_build_bit_identical(monkeypatch: pytest.MonkeyPatch, metric: DistanceMetric, n: int):
    """The parallel full-matrix build produces exactly the sequential build's values."""
    # --- arrange -----------------------------------------
    rng = np.random.default_rng(7)
    vectors = rng.random((n, 7), dtype=np.float32)

    # --- act ---------------------------------------------
    monkeypatch.setenv("MAXDIV_PARALLEL_BUILD", "1")
    parallel = DistanceStore.full_matrix_from_vectors(vectors, metric)
    monkeypatch.setenv("MAXDIV_PARALLEL_BUILD", "0")
    sequential = DistanceStore.full_matrix_from_vectors(vectors, metric)

    # --- assert ------------------------------------------
    assert np.array_equal(parallel.matrix, sequential.matrix)
