import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes
from tqdm import tqdm

from local.docs.figures.utils import save_fig
from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.metrics import DiversityMetric
from max_div._core.problem import MaxDivProblem
from max_div._core.solver import MaxDivSolution, MaxDivSolverBuilder, SolverPreset, iterations

# =================================================================================================
#  Config data
# =================================================================================================
_GREY_DOTS = dict(
    marker="o",
    color=(0.75, 0.75, 0.75),
    linestyle="",
    markersize=3,
)

_RED_DOTS = dict(
    marker="o",
    color=(1.0, 0.0, 0.0),
    linestyle="",
    markersize=5,
)

_GREY_SOLID_LINES = dict(
    color=(0.85, 0.85, 0.85),
    linestyle="-",
    linewidth=1,
    alpha=1.0,
    zorder=-100,
)

_GREY_DASHED_LINES = dict(
    color=(0.90, 0.90, 0.90),
    linestyle="--",
    linewidth=1,
    alpha=1.0,
    zorder=-105,
)

_CONSTRAINT_BOUNDARY = dict(
    color=(0.1, 0.2, 0.8),
    linestyle="-.",
    linewidth=0.75,
    alpha=1.0,
    zorder=-95,
)

_CONSTRAINT_TEXT = dict(
    color=(0.1, 0.2, 0.8),
    fontsize=9,
    zorder=10,
    va="center",
    ha="center",
)

_XY_LIMS = dict(
    U1=dict(x=[-0.1, 1.1], y=[-0.1, 1.15]),
    U2=dict(x=[-4.0, 4.0], y=[-3.0, 5.0]),
    U3=dict(x=[-0.5, 11], y=[-0.5, 11]),
    U4=dict(x=[-0.05, 1.25], y=[-0.05, 1.25]),
    C1=dict(x=[-0.1, 1.1], y=[-4.0, 5.5]),
    C2=dict(x=[-0.1, 1.1], y=[-4.0, 5.5]),
    C3=dict(x=[-5.0, 5.0], y=[-4.0, 6.0]),
    C4=dict(x=[-5.0, 5.0], y=[-4.0, 6.0]),
)

_LEGEND_POSITION = dict(
    U1="upper right",
    U2="upper right",
    U3="upper right",
    U4="upper left",
    C1="lower right",
    C2="lower right",
    C3="lower left",
    C4="lower left",
)


# =================================================================================================
#  Main functionality
# =================================================================================================
def create_figures(target_folder: Path, show_plots: bool = True) -> None:
    """
    For all benchmark problems, create a figure for the case d=2, showing the population of vectors,
    constraints & an example solution (using DEFAULT preset for 1000 iterations).
    :param target_folder: (str) The folder where the figures will be saved.
    :param show_plots: (bool) Whether to display the plots (blocking). Default is True.
    """

    problem_names = BenchmarkProblemFactory.get_all_benchmark_problems()
    for problem_name in tqdm(problem_names, desc="Creating benchmark problem figures"):
        # create problem for size==2  --> invariant for all problems: dimension==2 for size==2
        problem = BenchmarkProblemFactory.construct_problem(
            problem_name,
            size=2,
            diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        )

        # determine example solution
        solution = get_example_solution(problem)

        # create figure
        create_figure_for_problem(
            problem_name=problem_name,
            target_folder=target_folder,
            problem=problem,
            solution=solution,
        )

    if show_plots:
        plt.show()


def create_figure_for_problem(
    problem_name: str,
    target_folder: Path,
    problem: MaxDivProblem,
    solution: MaxDivSolution,
):
    # create fig, ax
    fig, ax = plt.subplots(figsize=(6, 6))

    # plot population
    pop_x = problem.vectors[:, 0].flatten()
    pop_y = problem.vectors[:, 1].flatten()
    ax.plot(pop_x, pop_y, **_GREY_DOTS, label="Population")

    # set x/y-limits
    ax.set_xlim(*_XY_LIMS[problem_name]["x"])
    ax.set_ylim(*_XY_LIMS[problem_name]["y"])

    # annotations
    annotate_population(ax, problem_name, problem)
    if problem.m > 0:
        annotate_constraints(ax, problem_name, problem)

    # title
    ax.set_title(f"Benchmark Problem {problem_name}", fontsize=14, fontweight="bold", pad=32)
    if problem.m > 0:
        sub_title = f"(select {problem.k} of {problem.n}; {problem.m} constraint groups)"
    else:
        sub_title = f"(select {problem.k} of {problem.n})"
    ax.text(0.0, 1.04, sub_title, transform=ax.transAxes, fontsize=12, ha="left", va="bottom")

    # final decoration
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc=_LEGEND_POSITION[problem_name])
    x_lim, y_lim = _XY_LIMS[problem_name]["x"], _XY_LIMS[problem_name]["y"]
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    if (x_lim[1] - x_lim[0]) > 2:
        ax.set_xticks(np.arange(round(x_lim[0] + 0.01), round(x_lim[1] - 0.01) + 1, 1))
    if (y_lim[1] - y_lim[0]) > 2:
        ax.set_yticks(np.arange(round(y_lim[0] + 0.01), round(y_lim[1] - 0.01) + 1, 1))
    fig.tight_layout()

    # save (without solution)
    save_fig(fig, target_folder / f"problem_{problem_name}.webp")

    # plot solution
    sol_x = problem.vectors[solution.i_selected, 0].flatten()
    sol_y = problem.vectors[solution.i_selected, 1].flatten()
    ax.plot(sol_x, sol_y, **_RED_DOTS, label="Example solution")
    ax.legend(loc=_LEGEND_POSITION[problem_name])

    # save (with solution)
    save_fig(fig, target_folder / f"problem_{problem_name}_with_solution.webp")


def get_example_solution(problem: MaxDivProblem) -> MaxDivSolution:
    solver = (
        MaxDivSolverBuilder(problem)
        .with_seed(42)
        .with_preset(
            target_duration=iterations(10_000),
            preset=SolverPreset.DEFAULT,
        )
        .build()
    )
    return solver.solve(verbosity=0)


# =================================================================================================
#  Annotations
# =================================================================================================
def annotate_population(ax: Axes, problem_name: str, problem: MaxDivProblem):
    # --- prep --------------------------------------------
    x_min, x_max = _XY_LIMS[problem_name]["x"]
    y_min, y_max = _XY_LIMS[problem_name]["y"]

    # --- PROBLEM U1 --------------------------------------
    if problem_name == "U1":
        ax.plot([x_min, x_max], [0.0, 0.0], **_GREY_DASHED_LINES)
        ax.plot([x_min, x_max], [1.0, 1.0], **_GREY_DASHED_LINES)
        ax.plot([0.0, 0.0], [y_min, y_max], **_GREY_DASHED_LINES)
        ax.plot([1.0, 1.0], [y_min, y_max], **_GREY_DASHED_LINES)
        ax.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], **_GREY_SOLID_LINES, label="Population boundary")

    # --- PROBLEM U2 --------------------------------------
    if problem_name == "U2":
        ax.plot([x_min, x_max], [0.0, 0.0], **_GREY_DASHED_LINES)
        ax.plot([0.0, 0.0], [y_min, y_max], **_GREY_DASHED_LINES)

    # --- PROBLEM U3 --------------------------------------
    if problem_name == "U3":
        ax.plot([x_min, x_max], [0.1, 0.1], **_GREY_DASHED_LINES)
        ax.plot([x_min, x_max], [10.0, 10.0], **_GREY_DASHED_LINES)
        ax.plot([0.1, 0.1], [y_min, y_max], **_GREY_DASHED_LINES)
        ax.plot([10.0, 10.0], [y_min, y_max], **_GREY_DASHED_LINES)
        ax.plot(
            [0.1, 0.1, 10.0, 10.0, 0.1],
            [0.1, 10.0, 10.0, 0.1, 0.1],
            **_GREY_SOLID_LINES,
            label="Population boundary",
        )

    # --- PROBLEM U4 --------------------------------------
    if problem_name == "U4":
        ax.plot([x_min, x_max], [0.0, 0.0], **_GREY_DASHED_LINES)
        ax.plot([x_min, x_max], [1.0, 1.0], **_GREY_DASHED_LINES)
        ax.plot([0.0, 0.0], [y_min, y_max], **_GREY_DASHED_LINES)
        ax.plot([1.0, 1.0], [y_min, y_max], **_GREY_DASHED_LINES)
        ax.plot([0.0, 1.0], [0.0, 1.0], **_GREY_DASHED_LINES)

        # Plot cone-shaped boundary: from (0,0) expanding to circle around (1,1)

        # Circle parameters
        center_x, center_y = 1.0, 1.0
        radius = 0.1 * np.sqrt(2)

        # Calculate tangent points from (0,0) to circle at (1,1)
        # Distance from origin to center
        d = np.sqrt(center_x**2 + center_y**2)

        # Angle from origin to center
        angle_to_center = np.arctan2(center_y, center_x)  # 45 degrees

        # Angle offset for tangent lines
        angle_offset = np.arcsin(radius / d)

        # Two tangent angles
        angle1 = angle_to_center + angle_offset
        angle2 = angle_to_center - angle_offset

        # Tangent points on the circle
        tangent1_x = center_x + radius * np.cos(angle1 + np.pi / 2)
        tangent1_y = center_y + radius * np.sin(angle1 + np.pi / 2)
        tangent2_x = center_x + radius * np.cos(angle2 - np.pi / 2)
        tangent2_y = center_y + radius * np.sin(angle2 - np.pi / 2)

        # Plot two tangent lines from origin to circle
        ax.plot([0.0, tangent1_x], [0.0, tangent1_y], **_GREY_SOLID_LINES)
        ax.plot([0.0, tangent2_x], [0.0, tangent2_y], **_GREY_SOLID_LINES)

        # Plot arc of circle between the two tangent points (top-right part)
        angle_tangent1 = np.arctan2(tangent1_y - center_y, tangent1_x - center_x)
        angle_tangent2 = np.arctan2(tangent2_y - center_y, tangent2_x - center_x)

        # Create arc from tangent1 to tangent2 going counter-clockwise
        theta = np.linspace(angle_tangent1, angle_tangent2, 1000)
        arc_x = center_x + radius * np.cos(theta)
        arc_y = center_y + radius * np.sin(theta)
        ax.plot(arc_x, arc_y, **_GREY_SOLID_LINES, label="Population boundary")

    # --- PROBLEM C1 / C2 ---------------------------------
    if problem_name in ["C1", "C2"]:
        ax.plot([x_min, x_max], [0.0, 0.0], **_GREY_DASHED_LINES)

    # --- PROBLEM C3 / C4 ---------------------------------
    if problem_name in ["C3", "C4"]:
        ax.plot([x_min, x_max], [0.5, 0.5], **_GREY_DASHED_LINES)
        ax.plot([0.5, 0.5], [y_min, y_max], **_GREY_DASHED_LINES)


def annotate_constraints(ax: Axes, problem_name: str, problem: MaxDivProblem):
    # --- prep --------------------------------------------
    x_min, x_max = _XY_LIMS[problem_name]["x"]
    y_min, y_max = _XY_LIMS[problem_name]["y"]

    # --- PROBLEM C1 / C2 ---------------------------------
    if problem_name in ["C1", "C2"]:
        x_edges = [float(x) for x in np.linspace(0.0, 1.0, problem.m + 1)]
        annotate_constraints_boundaries(ax, x_edges=x_edges, y_edges=[])

        for x_left, x_right in zip(x_edges[:-1], x_edges[1:]):
            txt = "select\nat least 4" if (problem_name == "C1") else "select\nexactly 5"
            annotate_constraint_text_hor(ax, x_left, x_right, 4.5, txt)

    # --- PROBLEM C3 --------------------------------------
    if problem_name == "C3":
        annotate_constraints_boundaries(ax, x_edges=[0.0], y_edges=[0.0])

        annotate_constraint_text_hor(ax, x_left=x_min, x_right=0.0, y=5.0, text="select\nat least 8")
        annotate_constraint_text_hor(ax, x_left=0.0, x_right=x_max, y=5.0, text="select\nat least 8")
        annotate_constraint_text_vert(ax, y_bot=y_min, y_top=0.0, x=4.0, text="select\nat least 8")
        annotate_constraint_text_vert(ax, y_bot=0.0, y_top=y_max, x=4.0, text="select\nat least 8")

    # --- PROBLEM C4 --------------------------------------
    if problem_name == "C4":
        annotate_constraints_boundaries(ax, x_edges=[-1, 0, 1], y_edges=[-1, 0, 1])

        annotate_constraint_text_hor(ax, x_left=x_min, x_right=0.0, y=5.0, text="select\nat least 8")
        annotate_constraint_text_hor(ax, x_left=-1, x_right=1, y=5.25, text="select\nat least 14")
        annotate_constraint_text_hor(ax, x_left=0.0, x_right=x_max, y=5.0, text="select\nat least 8")

        annotate_constraint_text_vert(ax, y_bot=y_min, y_top=0.0, x=4.0, text="select\nat least 8")
        annotate_constraint_text_vert(ax, y_bot=-1, y_top=1, x=4.25, text="select\nat least 14")
        annotate_constraint_text_vert(ax, y_bot=0.0, y_top=y_max, x=4.0, text="select\nat least 8")


def annotate_constraints_boundaries(ax: Axes, x_edges: list[float], y_edges: list[float]):
    y_lim = ax.get_ylim()
    x_lim = ax.get_xlim()
    first_line = True

    for x in x_edges:
        ax.plot([x, x], y_lim, **_CONSTRAINT_BOUNDARY, label="Constraint group boundary" if first_line else None)
        first_line = False

    for y in y_edges:
        ax.plot(x_lim, [y, y], **_CONSTRAINT_BOUNDARY, label="Constraint group boundary" if first_line else None)
        first_line = False


def annotate_constraint_text_hor(ax: Axes, x_left: float, x_right: float, y: float, text: str):
    ax_x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax_y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

    # arrow
    ax.annotate(
        "",
        xy=(x_left + (0.01 * ax_x_range), y),
        xytext=(x_right - (0.01 * ax_x_range), y),
        arrowprops=dict(
            arrowstyle="<->",
            color=_CONSTRAINT_BOUNDARY["color"],
            linewidth=_CONSTRAINT_BOUNDARY["linewidth"],
        ),
    )

    # text
    x_center = 0.5 * (x_left + x_right)
    ax.text(
        x_center,
        y + (0.035 * ax_y_range),
        text,
        **_CONSTRAINT_TEXT,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", boxstyle="round,pad=0.1"),
    )


def annotate_constraint_text_vert(ax: Axes, y_bot: float, y_top: float, x: float, text: str):
    ax_x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax_y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

    # arrow
    ax.annotate(
        "",
        xy=(x, y_bot + (0.01 * ax_y_range)),
        xytext=(x, y_top - (0.01 * ax_y_range)),
        arrowprops=dict(
            arrowstyle="<->",
            color=_CONSTRAINT_BOUNDARY["color"],
            linewidth=_CONSTRAINT_BOUNDARY["linewidth"],
        ),
    )

    # text
    y_center = 0.5 * (y_bot + y_top)
    ax.text(
        x + (0.035 * ax_x_range),
        y_center,
        text,
        **_CONSTRAINT_TEXT,
        rotation=-90,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", boxstyle="round,pad=0.1"),
    )


# =================================================================================================
#  Main Entrypoint
# =================================================================================================
if __name__ == "__main__":
    """
    Syntax: python fig_bm_problems_vectors_and_cons.py <target_folder> [--show-plots=true|false]
    """
    if len(sys.argv) < 2:
        print("Syntax: python fig_bm_problems_vectors_and_cons.py <target_folder> [--show-plots=true|false]")
    else:
        _target_folder = Path(sys.argv[1]).absolute()
        _show_plots = True
        for arg in sys.argv[2:]:
            if arg.startswith("--show-plots="):
                _show_plots = arg.split("=")[1].lower() in ("true", "yes", "1")
        create_figures(_target_folder, show_plots=_show_plots)
