from __future__ import annotations

import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from local.docs.figures.utils import save_fig
from local.docs.utils import UpperExponentialTransform
from local.docs.utils.regression import SplineBounds, SplineQuantileRegressor, SplineRegressor
from max_div._cli.bm_solver_presets._models import SolverPresetBenchmarkResult, results_from_json
from max_div.benchmarks import BenchmarkProblemFactory
from max_div.solver import SolverPreset


# =================================================================================================
#  Main functionality
# =================================================================================================
def create_figures(target_folder: Path):
    """
    For each (problem, size)-combination for which we have data we create the following figures:
      - constraint score vs iteration
      - constraint score vs elapsed time
      - diversity score vs iteration
      - diversity score vs elapsed time
    """

    plt.close("all")

    # --- scope -------------------------------------------
    problem_names = BenchmarkProblemFactory.get_all_benchmark_problems()
    sizes = [10, 100]

    problem_sizes: list[tuple[str, int]] = [
        (problem_name, size)
        for problem_name in problem_names
        for size in sizes
        if _get_data_file_name(problem_name, size).exists()
    ]

    # --- create all figures ------------------------------
    for problem, size in tqdm(problem_sizes, desc="Creating figures for benchmark problems"):
        # --- check problem props -----
        d, n, k, m, _ = BenchmarkProblemFactory.get_problem_dimensions(problem, size=size)
        has_constraints = m > 0

        # --- load data ---------------
        data_file_name = _get_data_file_name(problem, size)
        with open(data_file_name, "r") as f:
            json_str = f.read()

        result: list[SolverPresetBenchmarkResult] = results_from_json(json_str)

        # --- create figure -----------
        create_single_figure(
            target_folder=target_folder,
            problem_name=problem,
            size=size,
            results=result,
            show_constraints=has_constraints,
        )

    plt.show()


def create_single_figure(
    target_folder: Path,
    problem_name: str,
    size: int,
    results: list[SolverPresetBenchmarkResult],
    show_constraints: bool,
):

    # --- prep ---------------------------------------------
    all_diversity_scores = [result.score.diversity for result in results]
    all_constraint_scores = [result.score.constraints for result in results]

    for x_axis, y_axis in product(["iteration", "elapsed_sec"], ["constraint_score", "diversity_score"]):
        # --- x/y-axis settings ---------------------------
        if x_axis == "iteration":
            x_result_field = "n_iterations"
            x_label = "# of iterations"
            x_title = "Total Iterations"
        else:
            x_result_field = "t_elapsed_sec"
            x_label = "elapsed time (s)"
            x_title = "Total Time"

        if y_axis == "constraint_score":
            if not show_constraints:
                continue
            y_result_score_field = "constraints"
            y_label = "Constraint score\n(in [0,1])"
            y_title = "Constraint Score"
            y_transform = UpperExponentialTransform.from_y_values(all_constraint_scores)
            use_y_transform = max(all_constraint_scores) < 1
        else:
            y_result_score_field = "diversity"
            y_label = "Diversity score"
            y_title = "Diversity Score"
            y_transform = UpperExponentialTransform.from_y_values(all_diversity_scores)
            use_y_transform = max(all_constraint_scores) == 1

        # --- fig & ax ------------------------------------
        fig, ax = plt.subplots(figsize=(9, 6))

        # --- plot data per preset ------------------------
        colors = []
        for preset in SolverPreset.all_sorted():
            # --- collect data ---
            x: list[float] = []
            y: list[float] = []
            for result in results:
                if result.params.preset == preset:
                    x.append(getattr(result, x_result_field))
                    y.append(getattr(result.score, y_result_score_field))

            # --- plot data ---
            if use_y_transform:
                y_trans = y_transform.f(y)
                h = ax.plot(x, y_trans, label=preset.value, linestyle="None", marker="o")
            else:
                h = ax.plot(x, y, label=preset.value, linestyle="None", marker="o")
            color = h[0].get_color()

            # --- get uncertainty bounds & plot ---
            if use_y_transform:
                quantile_curves = QuantileCurves.from_data(
                    x_data=np.array(x),
                    y_data=np.array(y),
                    n_knots=3,
                    y_transform=y_transform,
                )
                x_q, q10, q50, q90 = quantile_curves.get_core_curves()  # in transformed space, if y_transform
                q10 = y_transform.f(q10)
                q50 = y_transform.f(q50)
                q90 = y_transform.f(q90)

                ax.fill_between(x_q, q10, q90, color=color, alpha=0.1)
                ax.plot(x_q, q50, color=color, linestyle="-", lw=0.8, alpha=0.8, label=f"q50")
                ax.plot(x_q, q10, color=color, linestyle="-", lw=0.5, alpha=0.5, label=f"q10, q90")
                ax.plot(x_q, q90, color=color, linestyle="-", lw=0.5, alpha=0.5)

        # --- decorations ---------------------------------

        # x-axis
        ax.set_xlabel(x_label)
        ax.set_xscale("log")

        # y-axis
        ax.set_ylabel(y_label)
        if use_y_transform:
            y_ticks, y_tick_labels = y_transform.get_ticks_and_labels(n_max=15)
            y_ticks = y_transform.f(y_ticks)  # conver to [0,1] transformed scale
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels)
            ax.set_ylim(-0.05, 1.05)

        # title
        ax.set_title(f"{y_title} vs {x_title}", fontsize=14, fontweight="bold", pad=32)
        ax.text(
            0.0,
            1.05,
            f"(problem {problem_name}, size={size})",
            transform=ax.transAxes,
            fontsize=12,
            ha="left",
            va="bottom",
        )

        # legend
        if y_axis == "diversity_score" and not use_y_transform:
            legend_loc = "upper right"
        else:
            legend_loc = "lower right"
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            n_rows = len(SolverPreset.all_sorted())
            n_cols = int(np.ceil(len(handles) / n_rows))
            # Matplotlib fills columns first; reorder so the visible layout reads row-wise.
            order = [r * n_cols + c for c in range(n_cols) for r in range(n_rows) if r * n_cols + c < len(handles)]
            handles = [handles[i] for i in order]
            labels = [labels[i] for i in order]
            ax.legend(handles, labels, title="Solver preset", loc=legend_loc, ncol=n_cols)
        else:
            ax.legend(title="Solver preset", loc=legend_loc, ncol=2)

        # misc
        ax.grid()
        fig.tight_layout()

        # --- save figure ---------------------------------
        save_fig(
            fig,
            target_folder / f"preset_results_{problem_name}_{str(size)}_{y_result_score_field}_vs_{x_result_field}.png",
        )


# =================================================================================================
#  File handling
# =================================================================================================
def _get_data_file_name(problem_name: str, size: int) -> Path:
    return Path(f"./local/docs/data/preset_results_{problem_name}_{size}.json")


def _get_figure_file_name(target_folder: Path, problem_name: str, size: int) -> Path:
    return target_folder / ""


# =================================================================================================
#  Uncertainty bounds
# =================================================================================================
class QuantileCurves:
    """
    Class modeling (q10, q50, q90)-quantile curves.  If y_transform is provided, spline regressions are performed
    in transformed y-space, but end results (quantiles, core curves) are returned in original y-space.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        q10: SplineRegressor,
        q50: SplineRegressor,
        q90: SplineRegressor,
        y_transform: UpperExponentialTransform | None = None,
    ):
        self._q10 = q10
        self._q50 = q50
        self._q90 = q90
        self._y_transform = y_transform

    # -------------------------------------------------------------------------
    #  Evaluation methods
    # -------------------------------------------------------------------------
    def get_core_curves(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(self._q50.knots[0], self._q50.knots[-1], 1000)  # transformed coordinates
        q10 = self._q10.predict(x)
        q50 = self._q50.predict(x)
        q90 = self._q90.predict(x)
        if self._y_transform is not None:
            q10 = self._y_transform.f_inv(q10)
            q50 = self._y_transform.f_inv(q50)
            q90 = self._y_transform.f_inv(q90)
        return self._transform_x_inv(x), q10, q50, q90

    # -------------------------------------------------------------------------
    #  Internal methods
    # -------------------------------------------------------------------------
    @staticmethod
    def _transform_x(x: np.ndarray | float) -> np.ndarray | float:
        if isinstance(x, float):
            return math.log(x)
        else:
            return np.log(x)

    @staticmethod
    def _transform_x_inv(x: np.ndarray | float) -> np.ndarray | float:
        if isinstance(x, float):
            return math.exp(x)
        else:
            return np.exp(x)

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def from_data(
        cls,
        x_data: np.ndarray,
        y_data: np.ndarray,
        n_knots: int = 3,
        y_transform: UpperExponentialTransform | None = None,
    ) -> QuantileCurves:
        """
        Create quantile curves for the provided data.

        We first transform the data
         - log transform for x_data
         - upper exponential transform for y_data

        Then we compute quantiles using quantile spline regression, which we post-process such that...
          - q10 <= q50 <= q90 for all x
          - q10, q50, q90 are monotone non-decreasing in x

        If y_transform is provided, the resulting QuantileCurves object will represent curves in transformed y-space.
        """

        # --- transform data ------------------------------
        x_transformed = cls._transform_x(x_data)
        if y_transform:
            y_transformed = y_transform.f(y_data)
        else:
            y_transformed = y_data

        # --- fit quantile splines ------------------------

        # derivative bounds
        dfdx = (max(y_transformed) - min(y_transformed)) / (max(x_transformed) - min(x_transformed))
        dfx_bounds = SplineBounds(
            x=x_transformed,
            lb=np.full_like(x_transformed, dfdx / 10),
        )

        # first fit q50
        q50 = SplineQuantileRegressor.cubic(x_transformed, y_transformed, n_knots, q=0.5, dfx_bounds=dfx_bounds)

        # fit q10 <= q50-margin
        q10 = SplineQuantileRegressor.cubic(
            x_transformed,
            y_transformed,
            n_knots,
            q=0.1,
            fx_bounds=SplineBounds(
                x=x_transformed,
                ub=q50.predict(x_transformed) - 0.01 * (max(y_transformed) - min(y_transformed)),
            ),
            dfx_bounds=dfx_bounds,
        )

        # fit q90 >= q50+margin
        q90 = SplineQuantileRegressor.cubic(
            x_transformed,
            y_transformed,
            n_knots,
            q=0.9,
            fx_bounds=SplineBounds(
                x=x_transformed,
                lb=q50.predict(x_transformed) + 0.01 * (max(y_transformed) - min(y_transformed)),
            ),
            dfx_bounds=dfx_bounds,
        )

        # --- return curves -------------------------------
        return QuantileCurves(q10, q50, q90, y_transform)


# =================================================================================================
#  Main Entrypoint
# =================================================================================================
if __name__ == "__main__":
    """
    Syntax: python fig_bm_problems_preset_results.py <target_folder>
    """
    if len(sys.argv) != 2:
        print("Syntax: python fig_bm_problems_preset_results.py <target_folder>")
    else:
        _target_folder = Path(sys.argv[1]).absolute()
        create_figures(_target_folder)
