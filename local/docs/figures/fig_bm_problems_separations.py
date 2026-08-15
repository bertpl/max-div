import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from tqdm import tqdm

from local.docs.figures.utils import save_fig
from local.docs.utils import univariate_kde_adaptive
from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.metrics import DiversityMetric
from max_div._core.metrics._distance import compute_pdist


# =================================================================================================
#  Main functionality
# =================================================================================================
def create_figures(target_folder: Path, show_plots: bool = True) -> None:
    """
    For all benchmark problems, create a figure showing the distribution of separations of the entire
    vector populate for problem sizes n = 200, 800, 3200, 12800.
    :param target_folder: Path to save the figures.
    :param show_plots: Whether to display the plots (blocking). Default is True.
    """

    problem_names = BenchmarkProblemFactory.get_all_benchmark_problems()
    for problem_name in tqdm(problem_names, desc="Creating benchmark problem figures"):
        # --- generate data ---------------------
        ns = [200, 800, 3200, 12800]
        sep_dict = {n: compute_separations(problem_name, n) for n in ns}

        # --- create figure ---------------------
        create_figure_for_problem(target_folder, problem_name, sep_dict)

    if show_plots:
        plt.show()


def create_figure_for_problem(target_folder: Path, problem_name: str, sep_dict: dict[int, np.ndarray]):

    # --- fig, ax -----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    # --- compute KDEs ------------------------------------
    x_min, x_max = determine_x_range(sep_dict)
    x_values = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
    kde_dict = {n: compute_kde(separations, x_values) for n, separations in sep_dict.items()}
    y_min, y_max = determine_y_range(kde_dict)

    # rug-related
    h_rug_rel = 0.05
    y_unit = h_rug_rel * y_max  # we will use this e.g. to determine vertical spacing of rug plots at the bottom

    # --- plot KDEs ---------------------------------------
    n_sizes = len(kde_dict)
    for i_size, n in enumerate(sorted(kde_dict.keys()), start=1):
        h = ax.plot(
            x_values,
            kde_dict[n],
            lw=0.5,
            label=f"$n={n}$",
        )

        # Extract the color from the Line2D object
        line_color = h[0].get_color()

        # plot filled area under the curve
        ax.fill_between(
            x_values,
            kde_dict[n],
            alpha=0.3,
            color=line_color,
        )

        # rug plot
        y_rug = -i_size * y_unit
        ax.plot([x_min, x_max], [y_rug, y_rug], lw=0.5, color=line_color)
        for sep in sep_dict[n]:
            ax.plot([sep, sep], [y_rug, y_rug + (0.5 * y_unit)], lw=0.25, color=line_color, alpha=0.5)

        # plot n (as text) slightly above the highest point of the curve
        max_kde_value = np.max(kde_dict[n])
        max_kde_x = x_values[np.argmax(kde_dict[n])]
        ax.text(
            max_kde_x,
            max_kde_value + 0.03 * y_max,
            f"$n={n}$",
            fontsize=9,
            ha="center",
            va="bottom",
        )

    # --- decorations -------------------------------------

    # axes
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$L_2 \ \text{distance}$")
    ax.set_ylabel("Probability density\n(adaptive KDE)")

    # disable clipping to allow lines outside the axis range to be visible
    ax.set_clip_on(False)
    for line in ax.get_lines():
        line.set_clip_on(False)

    # position spines: x-spine at negative y, y-spine at y=0 (creating a gap)
    ax.spines["bottom"].set_position(("axes", -(n_sizes + 0.5) * h_rug_rel))

    # grid & legend
    ax.grid()
    ax.legend(loc="upper left", title="Problem size")

    # title
    ax.set_title(f"Vector separations for problem {problem_name}", fontsize=14, fontweight="bold", pad=32)
    ax.text(
        0.0,
        1.05,
        "(full population; distances to nearest neighbor)",
        transform=ax.transAxes,
        fontsize=12,
        ha="left",
        va="bottom",
    )

    # --- save --------------------------------------------
    save_fig(fig, target_folder / f"problem_{problem_name}_separations.webp")


def determine_y_range(kde_dict: dict[int, np.ndarray]) -> tuple[float, float]:
    """
    Determine the y-axis range for the separations figure.
    """
    all_kde_values = np.concatenate(list(kde_dict.values()))
    max_kde_value = np.max(all_kde_values)
    return 0.0, 1.1 * max_kde_value


def determine_x_range(sep_dict: dict[int, np.ndarray]) -> tuple[float, float]:
    """
    Determine the x-axis range for the separations figure.
    """
    all_separations = np.concatenate(list(sep_dict.values()))
    min_sep = np.min(all_separations)
    max_sep = np.max(all_separations)
    log10_min_sep = np.log10(min_sep)
    log10_max_sep = np.log10(max_sep)

    x_margin = 0.1 * (log10_max_sep - log10_min_sep)
    return (
        10 ** float(log10_min_sep - x_margin),
        10 ** float(log10_max_sep + x_margin),
    )


def compute_kde(separations: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    """
    Computes a log-KDE:
      - convert separations and x_values to log10 scale
      - compute KDE and evaluate at x_values in log10 scale
      - return KDE
    """

    # log-transform
    log_separations = np.log10(separations)
    log_x_values = np.log10(x_values)

    # KDE
    return univariate_kde_adaptive(
        log_separations,
        log_x_values,
        smoothness=1.75,  # somewhat smoother curves, since we're more interested in overall shape than perfect 'fit'
        n_centers_max=1_000_000,
        bw_q_clip=(0.05, 0.95),
    )


# =================================================================================================
#  Generate data
# =================================================================================================
def compute_separations(problem_name: str, n: int) -> np.ndarray:
    """
    Compute the separations of the entire vector population for the given problem and size n.
    """
    problem = BenchmarkProblemFactory.construct_problem(
        problem_name,
        n=n,
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
    )

    # each vector's separation is its nearest-neighbor distance: reduce compute_pdist's
    # scipy-layout condensed distances to the per-row minimum over the other vectors
    square = squareform(compute_pdist(problem.vectors, problem.distance_metric))
    np.fill_diagonal(square, np.inf)
    return square.min(axis=1)


# =================================================================================================
#  Main Entrypoint
# =================================================================================================
if __name__ == "__main__":
    """
    Syntax: python fig_bm_problems_separations.py <target_folder> [--show-plots=true|false]
    """
    if len(sys.argv) < 2:
        print("Syntax: python fig_bm_problems_separations.py <target_folder> [--show-plots=true|false]")
    else:
        _target_folder = Path(sys.argv[1]).absolute()
        _show_plots = True
        for arg in sys.argv[2:]:
            if arg.startswith("--show-plots="):
                _show_plots = arg.split("=")[1].lower() in ("true", "yes", "1")
        create_figures(_target_folder, show_plots=_show_plots)
