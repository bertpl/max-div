import numpy as np
import pytest

from max_div.sampling.modified_power import modified_power_transform, sample_modified_power_distribution


@pytest.mark.parametrize("m", [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99])
def test_sample_modified_power_distribution(m: float):
    # --- act ---------------------------------------------
    samples = [sample_modified_power_distribution(np.float32(m), seed=np.int64(i)) for i in range(1000)]

    # --- assert ------------------------------------------
    assert 0.0 <= min(samples) <= 1.0
    assert 0.0 <= max(samples) <= 1.0

    assert len(set(samples)) == 1000, "samples should be unique"

    assert sum([1 for s in samples if s <= m]) > 400  # at least 40% of samples on either side of m
    assert sum([1 for s in samples if s >= m]) > 400  # at least 40% of samples on either side of m


def test_sample_symmetric_power_distribution_edge_cases():
    assert sample_modified_power_distribution(np.float32(0.0), seed=np.int64(42)) == 0.0
    assert sample_modified_power_distribution(np.float32(1.0), seed=np.int64(42)) == 1.0


@pytest.mark.parametrize("m", [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99])
def test_modified_power_transformation(m: float):
    # --- act ---------------------------------------------
    f0_0 = modified_power_transform(np.float32(0.0), np.float32(m))
    f0_5 = modified_power_transform(np.float32(0.5), np.float32(m))
    f1_0 = modified_power_transform(np.float32(1.0), np.float32(m))

    # --- assert ------------------------------------------
    assert f0_0 == 0.0
    assert f0_5 == pytest.approx(m)
    assert f1_0 == 1.0
