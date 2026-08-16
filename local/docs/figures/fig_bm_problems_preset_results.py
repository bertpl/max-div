from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from local.docs.figures.utils import save_fig
from local.docs.utils import LogTransform, PresetQuantilesTable, QuantileCurves, UpperLogTransform
from max_div._core._cli.bm_solver_presets._models import SolverPresetBenchmarkResult, results_from_json
from max_div._core._utils import format_long_time_duration
from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.solver import SolverPreset

_DATA_FOLDER = Path("./local/docs/data")


# =================================================================================================
#  Main functionality
# =================================================================================================
def create_figures(target_fig_folder: Path, target_md_folder: Path, show_plots: bool = True):
    """
    For each (problem, n)-combination for which we have data we create the following figures:
      - constraint score vs iteration
      - constraint score vs elapsed time
      - diversity score vs iteration
      - diversity score vs elapsed time
    :param target_fig_folder: Path to save figure files.
    :param target_md_folder: Path to save markdown files.
    :param show_plots: Whether to display the plots (blocking). Default is True.
    """

    plt.close("all")

    # --- scope -------------------------------------------
    # The data folder is globbed rather than matched against a fixed list of `n` values, so the
    # figures reflect whatever the latest benchmark run produced.
    problem_ns: list[tuple[str, int]] = []
    for problem_name in BenchmarkProblemFactory.get_all_benchmark_problems():
        for data_file in sorted(_DATA_FOLDER.glob(f"preset_results_{problem_name}_*.json")):
            problem_ns.append((problem_name, int(data_file.stem.rsplit("_", 1)[1])))

    # --- create all figures ------------------------------
    for problem, n in tqdm(problem_ns, desc="Creating figures for benchmark problems"):
        # --- check problem props -----
        _d, _n, _k, m, _ = BenchmarkProblemFactory.get_problem_dimensions(problem, n=n)
        has_constraints = m > 0

        # --- load data ---------------
        json_str = _get_data_file_name(problem, n).read_text()
        result: list[SolverPresetBenchmarkResult] = results_from_json(json_str)

        # --- create figure -----------
        create_single_figure(
            target_fig_folder=target_fig_folder,
            target_md_folder=target_md_folder,
            problem_name=problem,
            n=n,
            results=result,
            show_constraints=has_constraints,
        )

    if show_plots:
        plt.show()


def create_single_figure(
    target_fig_folder: Path,
    target_md_folder: Path,
    problem_name: str,
    n: int,
    results: list[SolverPresetBenchmarkResult],
    show_constraints: bool,
):

    # --- prep ---------------------------------------------
    all_diversity_scores = [result.score.diversity for result in results]
    all_constraint_scores = [result.score.constraints for result in results]
    all_presets = SolverPreset.all_sorted()
    n_presets = len(all_presets)

    # --- create PresetQuantilesTable for markdown tables -
    quantiles_table = PresetQuantilesTable(problem_name, n, target_md_folder)

    # --- axis configurations -----------------------------
    x_axes = ["iteration", "elapsed_sec"]
    y_axes = ["constraint_score", "diversity_score"] if show_constraints else ["diversity_score"]

    n_rows = len(y_axes)
    n_cols = len(x_axes)

    # --- create 2x2 grid figure --------------------------
    # rows share y-axis, columns share x-axis
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 6 * n_rows),
        sharex="col",
        sharey="row",
        squeeze=False,
    )

    for row_idx, y_axis in enumerate(y_axes):
        for col_idx, x_axis in enumerate(x_axes):
            ax = axes[row_idx, col_idx]

            # --- x/y-axis settings ---------------------------
            if x_axis == "iteration":
                x_result_field = "n_iterations"
                x_label = "# of iterations"
                x_title = "Total Iterations"
            else:
                x_result_field = "t_elapsed_sec"
                x_label = "elapsed time"
                x_title = "Total Time"

            if y_axis == "constraint_score":
                y_result_score_field = "constraints"
                y_label = "Constraint score\n(in [0,1])"
                y_title = "Constraint Score"
                y_transform = UpperLogTransform.from_values(all_constraint_scores)
                use_y_transform = max(all_constraint_scores) < 1
            else:
                y_result_score_field = "diversity"
                y_label = "Diversity score"
                y_title = "Diversity Score"
                y_transform = UpperLogTransform.from_values(all_diversity_scores)
                use_y_transform = max(all_constraint_scores) == 1

            # --- plot data per preset ------------------------
            for preset in all_presets:
                # --- collect data ---
                # single-worker runs only; the parallel arm is a separate series (see the
                # "plot the parallel arm" block), so its records must not fold into the preset's
                # own scatter or quantile band
                x: list[float] = []
                y: list[float] = []
                for result in results:
                    if result.params.preset == preset and result.params.n_workers == 1:
                        x.append(getattr(result, x_result_field))
                        y.append(getattr(result.score, y_result_score_field))

                # a preset may be absent from a partial run; skip it rather than reducing over empties
                if not x:
                    continue

                # --- plot data ---
                if use_y_transform:
                    y_trans = y_transform.f(y)
                    h = ax.plot(x, y_trans, label=preset.value, linestyle="None", marker="o")
                else:
                    h = ax.plot(x, y, label=preset.value, linestyle="None", marker="o")
                    if y_axis == "constraint_score" and min(y) == 1.0:
                        ax.set_ylim([0.0, 1.1])
                color = h[0].get_color()

                # --- get uncertainty bounds & plot ---
                if use_y_transform:
                    quantile_curves = QuantileCurves.from_data(
                        x_data=np.array(x),
                        y_data=np.array(y),
                        n_knots=10,
                        x_transform=LogTransform(),
                        y_transform=UpperLogTransform.from_values(np.array(y)),  # specific to this curve
                        q50_reg=0.25,
                        q10_q90_reg=5.0,
                    )
                    # Add to quantiles table for markdown generation
                    quantiles_table.add_quantile_curves(
                        quantile_curves=quantile_curves,
                        preset=preset,
                        x_axis=x_axis,
                        y_axis=y_axis,
                    )
                    x_q, q10, q50, q90 = quantile_curves.get_full_curves()  # in transformed space, if y_transform
                    q10 = y_transform.f(q10)
                    q50 = y_transform.f(q50)
                    q90 = y_transform.f(q90)

                    ax.fill_between(x_q, q10, q90, color=color, alpha=0.1)
                    ax.plot(x_q, q50, color=color, linestyle="-", lw=0.8, alpha=0.8, label="q50")
                    ax.plot(x_q, q10, color=color, linestyle="-", lw=0.5, alpha=0.5, label="q10, q90")
                    ax.plot(x_q, q90, color=color, linestyle="-", lw=0.5, alpha=0.5)

            # --- plot the parallel arm -----------------------
            # The default parallel invocation (one preset, several cooperative workers) is shown as
            # a plain black-circle scatter with no quantile band: it is a single reference series,
            # not a preset sweep, and one run per budget point makes the scatter itself the spread.
            parallel_x: list[float] = []
            parallel_y: list[float] = []
            parallel_label = ""
            for result in results:
                if result.params.n_workers > 1:
                    parallel_x.append(getattr(result, x_result_field))
                    parallel_y.append(getattr(result.score, y_result_score_field))
                    parallel_label = result.params.column_label()
            if parallel_x:
                parallel_y_plot = y_transform.f(parallel_y) if use_y_transform else parallel_y
                # hollow black circles, so the parallel arm reads as distinct from the filled
                # preset dots
                ax.plot(
                    parallel_x,
                    parallel_y_plot,
                    label=parallel_label,
                    linestyle="None",
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor="black",
                )

            # --- decorations ---------------------------------

            # x-axis (only set label on bottom row)
            ax.set_xscale("log")
            if row_idx == n_rows - 1:
                ax.set_xlabel(x_label)

            # For the time axis, set human-readable tick labels (e.g. "1s", "1m", "1h")
            # using a sparser set of candidate values suitable for a log-scale axis.
            if x_axis == "elapsed_sec":
                # Collect all x values across all presets for this column to determine the plotted range
                all_x_vals: list[float] = []
                for result in results:
                    all_x_vals.append(result.t_elapsed_sec)
                if all_x_vals:
                    x_min_data, x_max_data = min(all_x_vals), max(all_x_vals)
                    # Sparse candidate ticks: fine steps up to 30 min, then fixed 1h increments
                    _m, _h = 60, 3600
                    candidate_ticks = [1, 2, 5, 10, 15, 30, 45]  # seconds
                    candidate_ticks += [1 * _m, 2 * _m, 5 * _m, 10 * _m, 30 * _m]  # minutes
                    candidate_ticks += [n * _h for n in range(1, 7)]  # 1h to 6h in 1h increments
                    ticks = [float(t) for t in candidate_ticks if x_min_data <= t <= x_max_data]
                    if ticks:
                        tick_labels = [format_long_time_duration(t, n_chars=3).strip() for t in ticks]
                        ax.set_xticks(ticks)
                        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
                        ax.set_xticks([], minor=True)

            # y-axis (only set label on left column)
            if col_idx == 0:
                ax.set_ylabel(y_label)
            if use_y_transform:
                y_ticks, y_tick_labels = y_transform.get_ticks_and_labels(n_max=15, add_missing_ticks=True)
                y_ticks = y_transform.f(y_ticks)  # convert to [0,1] transformed scale
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_tick_labels)
                ax.set_ylim(-0.05, 1.05)

            # subplot title
            ax.set_title(f"{y_title} vs {x_title}", fontsize=12, fontweight="bold")

            # grid
            ax.grid()

    # --- figure-level decorations ------------------------

    # suptitle (left-aligned, with subtitle on second line)
    fig.suptitle(
        "Solver Preset Results",
        fontsize=14,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.96 if len(y_axes) == 2 else 0.94,
        f"(problem {problem_name}, n={n})",
        fontsize=12,
        ha="left",
        va="top",
    )

    # show legend on 1 axes:  right-most col, where we have uncertainty bounds
    for i_row in range(n_rows):
        handles, labels = axes[i_row, 1].get_legend_handles_labels()
        if len(handles) > n_presets:
            # One row per series, three fixed columns [dot, q50, q10-q90]. Handles arrive grouped
            # per series (a preset contributes dot, q50, q10-q90; the parallel arm just a dot), so
            # group them, pad the parallel arm's row with blank cells, and emit column-first to
            # match matplotlib's column-major legend fill — keeping every series on its own row.
            band_labels = ("q50", "q10, q90")
            series: list[tuple[list, list]] = []
            for handle, lbl in zip(handles, labels):
                if lbl not in band_labels:
                    series.append(([handle], [lbl]))
                else:
                    series[-1][0].append(handle)
                    series[-1][1].append(lbl)
            n_cols = max(len(hs) for hs, _ in series)
            blank = Line2D([], [], linestyle="none", marker="none")
            grid_handles, grid_labels = [], []
            for col in range(n_cols):
                for hs, ls in series:
                    grid_handles.append(hs[col] if col < len(hs) else blank)
                    grid_labels.append(ls[col] if col < len(ls) else "")
            axes[i_row, 1].legend(
                grid_handles,
                grid_labels,
                title="Solver preset",
                loc="lower right",
                ncol=n_cols,
            )

    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.subplots_adjust(wspace=0.075, hspace=0.15)

    # --- save figure ---------------------------------
    # the default lossless webp resolution puts these dense scatter figures over the repo's
    # committed-file ceiling, so cap the resolution — still sharp at the page width they display at
    save_fig(fig, target_fig_folder / f"preset_results_{problem_name}_{n}.webp", tgt_pixels=6_000_000)

    # --- save markdown report ------------------------
    quantiles_table.generate_tables()


# =================================================================================================
#  File handling
# =================================================================================================
def _get_data_file_name(problem_name: str, n: int) -> Path:
    return _DATA_FOLDER / f"preset_results_{problem_name}_{n}.json"


# =================================================================================================
#  Main Entrypoint
# =================================================================================================
if __name__ == "__main__":
    """
    Syntax: python fig_bm_problems_preset_results.py <target_fig_folder> <target_md_folder> [--show-plots=true|false]
    """
    if len(sys.argv) < 3:
        print(
            "Syntax: python fig_bm_problems_preset_results.py <target_fig_folder> <target_md_folder> [--show-plots=true|false]"
        )
    else:
        _target_fig_folder = Path(sys.argv[1]).absolute()
        _target_md_folder = Path(sys.argv[2]).absolute()
        _show_plots = True
        for arg in sys.argv[3:]:
            if arg.startswith("--show-plots="):
                _show_plots = arg.split("=")[1].lower() in ("true", "yes", "1")
        create_figures(_target_fig_folder, _target_md_folder, show_plots=_show_plots)
