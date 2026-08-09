import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance._build import BUILD_BLOCK_WIDTH, compute_pdist


@pytest.mark.parametrize("metric", list(DistanceMetric))
@pytest.mark.parametrize(
    "n", [30, BUILD_BLOCK_WIDTH, BUILD_BLOCK_WIDTH * 2, BUILD_BLOCK_WIDTH * 2 + 2]
)  # below one block / exact block multiples / uneven
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
def test_condensed_build_fills_a_supplied_buffer(metric: DistanceMetric):
    """A supplied buffer is filled and returned, holding what a freshly allocated one would."""
    # --- arrange -----------------------------------------
    vectors = np.random.default_rng(11).random((20, 4), dtype=np.float32)
    buffer = np.full((20 * 19) // 2, 7.0, dtype=np.float32)

    # --- act ---------------------------------------------
    filled = compute_pdist(vectors, metric, out=buffer)

    # --- assert ------------------------------------------
    assert filled is buffer
    assert np.array_equal(filled, compute_pdist(vectors, metric))


@pytest.mark.parametrize("metric", list(DistanceMetric))
def test_condensed_build_accepts_read_only_vectors(metric: DistanceMetric):
    """Vectors held read-only — as a DistanceStore holds them — are a valid input."""
    # --- arrange -----------------------------------------
    vectors = np.ascontiguousarray(np.random.default_rng(13).random((16, 3), dtype=np.float32))
    read_only = vectors.view()
    read_only.flags.writeable = False

    # --- act ---------------------------------------------
    from_read_only = compute_pdist(read_only, metric)

    # --- assert ------------------------------------------
    assert np.array_equal(from_read_only, compute_pdist(vectors, metric))
