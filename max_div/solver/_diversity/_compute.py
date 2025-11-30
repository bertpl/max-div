import numpy as np
from numpy.typing import NDArray

from ._enum import DiversityMetric


def compute_diversity(separation: NDArray[np.float32], metric: DiversityMetric) -> np.float32:
    if separation.size == 0:
        return np.float32(np.inf)
    else:
        match metric:
            case DiversityMetric.MIN_SEPARATION:
                return np.min(separation)
            case DiversityMetric.MEAN_SEPARATION:
                return np.mean(separation)
            case DiversityMetric.GEOMEAN_SEPARATION:
                # To avoid issues with zero separation, we add a small epsilon before computing the geometric mean.
                epsilon = abs(1e-9 * np.max(separation))
                log_sep = np.log(np.maximum(separation, epsilon))
                geo_mean_log = np.mean(log_sep)
                geo_mean = np.exp(geo_mean_log) - epsilon
                return geo_mean.astype(np.float32)
