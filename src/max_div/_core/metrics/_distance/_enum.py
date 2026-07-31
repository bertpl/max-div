from enum import StrEnum


class DistanceMetric(StrEnum):
    """Enum for different distance metrics.

    Members
    -------

        - L1_MANHATTAN:            L1 (Manhattan) distance                   = sum_i |x_i - y_i|
        - L2_EUCLIDEAN:            L2 (Euclidean) distance                   = sqrt( sum_i (x_i - y_i)^2 )
        - L2S_EUCLIDEAN_SQUARED:   L2 squared (Euclidean squared) distance   = sum_i (x_i - y_i)^2
                                          --> avoids computing square root
                                          --> and produces identical solutions for GEOMEAN_SEPARATION diversity metric
        - LINF_CHEBYSHEV:          Linf (Chebyshev) distance                 = max_i |x_i - y_i|
                                          --> distance set by the single largest per-dimension gap
        - COSINE:                  cosine distance                           = 1 - (x . y) / (|x| |y|)
                                          --> range [0, 2]; angular, for embedding-style vectors
                                          --> undefined for zero vectors (rejected with an error)
    """

    L1_MANHATTAN = "L1_MANHATTAN"
    L2_EUCLIDEAN = "L2_EUCLIDEAN"
    L2S_EUCLIDEAN_SQUARED = "L2S_EUCLIDEAN_SQUARED"
    LINF_CHEBYSHEV = "LINF_CHEBYSHEV"
    COSINE = "COSINE"
