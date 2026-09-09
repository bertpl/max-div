import numpy as np
import pytest
from scipy.spatial.distance import pdist

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance._build import (
    BUILD_BLOCK_WIDTH,
    compute_full_matrix,
    expand_condensed,
    parallel_build_enabled,
)


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
    # --- arrange ----------------------
    if env_value is None:
        monkeypatch.delenv("MAXDIV_PARALLEL_BUILD", raising=False)
    else:
        monkeypatch.setenv("MAXDIV_PARALLEL_BUILD", env_value)

    # --- act / assert -----------------
    assert parallel_build_enabled() is expected


@pytest.mark.parametrize(
    "n", [30, BUILD_BLOCK_WIDTH, BUILD_BLOCK_WIDTH * 2, BUILD_BLOCK_WIDTH * 2 + 2]
)  # below one block / exact block multiples / uneven
def test_full_matrix_parallel_build_bit_identical(monkeypatch: pytest.MonkeyPatch, metric: DistanceMetric, n: int):
    """The parallel full-matrix build produces exactly the sequential build's values."""
    # --- arrange ----------------------
    rng = np.random.default_rng(7)
    vectors = rng.random((n, 7), dtype=np.float32)

    # --- act --------------------------
    monkeypatch.setenv("MAXDIV_PARALLEL_BUILD", "1")
    parallel = compute_full_matrix(vectors, metric)
    monkeypatch.setenv("MAXDIV_PARALLEL_BUILD", "0")
    sequential = compute_full_matrix(vectors, metric)

    # --- assert -----------------------
    assert np.array_equal(parallel, sequential)


@pytest.mark.parametrize("parallel", [True, False])
def test_fills_leave_a_complete_matrix_in_a_dirty_buffer(monkeypatch: pytest.MonkeyPatch, parallel: bool):
    """A fill zeroes the diagonal it never computes, whatever the supplied buffer held before."""
    # --- arrange ----------------------
    monkeypatch.setenv("MAXDIV_PARALLEL_BUILD", "1" if parallel else "0")
    vectors = np.random.default_rng(3).random((40, 5), dtype=np.float32)
    dirty = np.full((40, 40), 7.0, dtype=np.float32)

    # --- act --------------------------
    matrix = compute_full_matrix(vectors, DistanceMetric.l2_euclidean(), out=dirty)

    # --- assert -----------------------
    assert matrix is dirty
    np.testing.assert_array_equal(np.diag(matrix), np.zeros(40, dtype=np.float32))
    np.testing.assert_array_equal(matrix, matrix.T)


def test_expanding_a_condensed_vector_into_a_dirty_buffer():
    """Expansion owns its diagonal and places every condensed value at scipy's (i, j) position."""
    # --- arrange ----------------------
    vectors = np.ascontiguousarray(np.random.default_rng(5).random((24, 4), dtype=np.float32))
    condensed = pdist(vectors).astype(np.float32)
    dirty = np.full((24, 24), -1.0, dtype=np.float32)

    # --- act --------------------------
    expanded = expand_condensed(condensed, 24, out=dirty)

    # --- assert -----------------------
    assert expanded is dirty
    i_upper, j_upper = np.triu_indices(24, k=1)
    np.testing.assert_array_equal(expanded[i_upper, j_upper], condensed)
    np.testing.assert_array_equal(expanded, expanded.T)
    np.testing.assert_array_equal(np.diag(expanded), np.zeros(24, dtype=np.float32))


def test_full_matrix_build_accepts_read_only_vectors():
    """Vectors held read-only — as a DistanceStore holds them — are a valid input."""
    # --- arrange ----------------------
    vectors = np.ascontiguousarray(np.random.default_rng(17).random((16, 3), dtype=np.float32))
    read_only = vectors.view()
    read_only.flags.writeable = False

    # --- act --------------------------
    from_read_only = compute_full_matrix(read_only, DistanceMetric.cosine())

    # --- assert -----------------------
    np.testing.assert_array_equal(from_read_only, compute_full_matrix(vectors, DistanceMetric.cosine()))
