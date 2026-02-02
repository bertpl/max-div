import numpy as np
from scipy.stats import gaussian_kde, norm


def univariate_kde(samples: np.ndarray, x_values: np.ndarray, n_centers: int = 1000) -> np.ndarray:
    """
    Computes a KDE for the provided samples and evaluates it at the given x_values.
    """

    # convert to 1000 values
    samples = np.quantile(
        samples,
        np.linspace(0, 1, n_centers),
    )

    # KDE
    kde_obj = gaussian_kde(samples, bw_method="silverman")
    kde_values = kde_obj.evaluate(x_values)

    # return
    return kde_values


def univariate_kde_adaptive(
    samples: np.ndarray, x_values: np.ndarray, n_centers_max: int = 1000, smoothness: float = 1.0
) -> np.ndarray:

    # --- choose centers ----------------------------------
    n_centers = min(n_centers_max, len(samples))
    q_centers = np.linspace(0.5 / n_centers, 1 - (0.5 / n_centers), n_centers)
    x_centers = np.quantile(samples, q_centers)

    # --- compute local bandwidths -----------------------
    q_width = min(0.5, 0.3 * smoothness / (n_centers ** (1 / 5)))
    min_x_width = np.quantile(np.diff(sorted(set(x_centers))), 0.05)
    x_widths = []
    for q, x in zip(q_centers, x_centers):
        if q <= q_width:
            q_left = 0.0
            q_right = 2 * q_width
        elif q >= 1 - q_width:
            q_left = 1 - 2 * q_width
            q_right = 1.0
        else:
            q_left = q - q_width
            q_right = q + q_width

        w = np.quantile(samples, q_right) - np.quantile(samples, q_left)
        w = max(min_x_width, w)
        x_widths.append(w)

    # --- compute KDE values --------------------------------
    kde_values = np.zeros_like(x_values)
    for x_center, x_width in zip(x_centers, x_widths):
        kde_values += norm.pdf(x_values, loc=x_center, scale=x_width)
    kde_values /= len(x_centers)

    return kde_values
