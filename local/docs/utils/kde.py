import numba
import numpy as np


# =================================================================================================
#  KDE - 1D - Adaptive
# =================================================================================================
def univariate_kde_adaptive(
    samples: np.ndarray,
    x_values: np.ndarray,
    n_centers_max: int = 1000,
    smoothness: float = 1.0,
    bw_q_clip: tuple = (0.1, 0.9),
) -> np.ndarray:

    # --- cap data size -----------------------------------
    if len(samples) > n_centers_max:
        samples = np.quantile(samples, np.linspace(0, 1, n_centers_max))

    # --- pilot KDE ---------------------------------------
    # pilot non-adaptive KDE, evaluated at 'samples' instead of 'x_values', so we get an estimate
    # of the density at each sample point
    pilot_kde = univariate_kde(samples, samples, n_centers_max, smoothness)

    # --- compute adapted bandwidths ----------------------
    bw_median = smoothness * bw_silverman(samples)
    d_median = np.median(pilot_kde)  # median density corresponds to Silverman's rule-of-thumb bandwidth
    bw_adaptive = bw_median * np.sqrt(d_median / pilot_kde)  # ~Abramson's square-root law for adaptive bandwidths

    bw_clip_lb, wb_clip_ub = np.quantile(bw_adaptive, bw_q_clip)
    bw_adaptive = np.clip(bw_adaptive, bw_clip_lb, wb_clip_ub)  # clip bandwidths to avoid extreme values

    # --- compute KDE with adaptive bandwidths ------------
    return _univariate_kde(
        mu=samples,
        sigma=bw_adaptive,
        x_values=x_values,
    )


# =================================================================================================
#  KDE - 1D - Non-Adaptive
# =================================================================================================
def univariate_kde(
    samples: np.ndarray, x_values: np.ndarray, n_centers_max: int = 1000, smoothness: float = 1.0
) -> np.ndarray:

    # --- cap data size -----------------------------------
    if len(samples) > n_centers_max:
        samples = np.quantile(samples, np.linspace(0, 1, n_centers_max))

    # --- compute bandwidth -------------------------------
    bw = smoothness * bw_silverman(samples)

    # --- compute KDE values ------------------------------
    return _univariate_kde(
        mu=samples,
        sigma=np.full_like(samples, bw),
        x_values=x_values,
    )


def bw_silverman(data: np.ndarray) -> float:
    """Computes Silverman's rule-of-thumb bandwidth for 1D data."""
    std_dev = np.std(data, ddof=1)
    q25, q75 = np.quantile(data, [0.25, 0.75])
    iqr = q75 - q25
    sigma = min(std_dev, iqr / 1.34) if iqr > 0 else std_dev
    return float(0.9 * sigma * (len(data) ** (-1 / 5)))


# =================================================================================================
#  KDE - 1D - Core
# =================================================================================================
@numba.njit(cache=True, fastmath=True, inline="always")
def _univariate_kde(mu: np.ndarray, sigma: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    """
    Evaluates a given KDE (defined by identically sized mu, sigma arrays) at the given x_values.
    """

    # --- prep ------------------------
    kde_values = np.zeros_like(x_values)
    main_const = 1.0 / (np.sqrt(2.0 * np.pi) * len(mu))

    # --- main loop -------------------
    for i in range(len(mu)):
        # prep inner loop
        mu_i = mu[i]
        sigma_i = sigma[i]
        sigma_i_inv = 1.0 / sigma_i
        norm_const = main_const * sigma_i_inv

        # inner loop
        for j in range(len(x_values)):
            z = sigma_i_inv * (x_values[j] - mu_i)
            kde_values[j] += norm_const * np.exp(-0.5 * z * z)

    return kde_values
