"""Generate the figures of the guide pages under `docs/guides/`.

The figures of `geomean_separation.md` plot given selections controlled by a parameter alpha: three
cases as dot rows, and the separation metrics against alpha below them; no solver is involved.

The figures of `uniform_sampling.md` are seven solved selections of one random population, one per
experiment, plus a longer solve of the last experiment; only those run the solver. Each selection is
emitted as an interactive figure (an HTML fragment over a raster of the population) plus its
separations table, and cached as JSON so that `--reuse-solution` re-renders the figures without the
solves, solving only a run that has no cache yet; a closing table compares them all. The longer solve
is emitted as a replay figure that steps through every change of its selection.

Run with: ``uv run --group benchmarks ./scripts/generate_guide_images.py [--reuse-solution]``.
"""

import argparse
import json
import pickle
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from uniform_sampling_explorer import POPULATION_COLOR, explorer_fragment, nearest_neighbors
from uniform_sampling_replay import ReplayFrame, replay_fragment

from benchmarks.figures.style import REPO_ROOT, save_webp, use_docs_style
from max_div.metrics import DistanceMetric, DiversityMetric, HybridDiversityMetric
from max_div.problem import Constraint, MaxDivProblem
from max_div.solver import ParallelMaxDivSolution, ParallelMaxDivSolverBuilder, seconds

GENERATED_DIR = REPO_ROOT / "generated"
IMAGES_DIR = REPO_ROOT / "docs" / "guides" / "images"

# One color per metric, shared by every figure so the reader learns them once.
METRIC_COLORS = {
    "min separation": "#4C72B0",
    "mean separation": "#DD8452",
    "geometric-mean separation": "#55A868",
    "harmonic-mean separation": "#8172B3",
}
CASE_LABELS = ("A", "B", "C")


# ==================================================================================================
#  Metrics of a selection
# ==================================================================================================
def separations(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return each item's distance to its nearest other item, for sorted one-dimensional positions."""
    gaps = np.diff(positions)
    left = np.concatenate(([np.inf], gaps))
    right = np.concatenate((gaps, [np.inf]))
    return np.minimum(left, right)


def harmonic_mean(values: NDArray[np.float64]) -> float:
    """Return the harmonic mean of a positive array, or 0.0 if any value is not positive."""
    return float(values.size / np.sum(1.0 / values)) if np.all(values > 0) else 0.0


def metrics(positions: NDArray[np.float64]) -> dict[str, float]:
    """Return the separation-based diversity metrics of a one-dimensional selection."""
    sep = separations(np.sort(positions))
    is_positive = bool(np.all(sep > 0))
    return {
        "min separation": float(sep.min()),
        "mean separation": float(sep.mean()),
        "geometric-mean separation": float(np.exp(np.mean(np.log(sep)))) if is_positive else 0.0,
        "harmonic-mean separation": harmonic_mean(sep),
    }


# ==================================================================================================
#  Rendering
# ==================================================================================================
def render_example(
    name: str,
    layout: Callable[[float], NDArray[np.float64]],
    case_alphas: tuple[float, float, float],
    alpha_range: tuple[float, float],
    axis_range: tuple[float, float],
    legend: dict | None = None,
    diversity_max: float | None = None,
    highlight: tuple[int, ...] = (),
    dot_size: float = 36,
    position_marks: tuple[float, ...] | None = None,
    use_log_scale: bool = False,
) -> None:
    """Render one figure: the three cases as dot rows, and the metrics against alpha below them.

    Args:
        name: Image file stem under `IMAGES_DIR`.
        layout: Maps alpha to the item positions of the selection.
        alpha_range: The alpha interval the metric curves cover.
        axis_range: The position interval every dot row shows, shared so the cases compare.
        legend: Keyword arguments placing the metric legend where the curves leave room.
        diversity_max: Top of the diversity axis; autoscaled when omitted.
        highlight: Indices of the items alpha acts on, drawn as hollow rings.
        dot_size: Marker area of the items; smaller for selections with many items.
        position_marks: Positions marked by a labeled vertical across the rows; the whole numbers in range when omitted.
        use_log_scale: Draw the positions on a logarithmic axis, and each metric relative to its value at alpha = 0 on a
            logarithmic axis too, so that metrics of different magnitudes share one panel.
    """
    use_docs_style()
    fig, (ax_cases, ax_metrics) = plt.subplots(2, 1, figsize=(8.0, 5.6), height_ratios=(1.3, 2.0))
    fig.subplots_adjust(hspace=0.45, left=0.09, right=0.91, top=0.95, bottom=0.09)

    # --- the cases as dot rows ------------------
    for row, (label, alpha) in enumerate(zip(CASE_LABELS, case_alphas)):
        y = len(CASE_LABELS) - 1 - row
        positions = layout(alpha)
        ax_cases.hlines(y, *axis_range, color="#D8D8D8", linewidth=1.0, zorder=1)
        is_highlighted = np.isin(np.arange(positions.size), highlight)
        ax_cases.scatter(
            positions[~is_highlighted], np.full((~is_highlighted).sum(), y), s=dot_size, c="#222222", zorder=2
        )
        ax_cases.scatter(
            positions[is_highlighted],
            np.full(is_highlighted.sum(), y),
            s=60,
            facecolors="white",
            edgecolors="#222222",
            linewidths=1.8,
            zorder=3,
        )
        left, right = _row_label_positions(axis_range, use_log_scale)
        ax_cases.text(left, y, f"Case {label}", ha="right", va="center")
        ax_cases.text(right, y, f"\u03b1 = {alpha:g}", ha="left", va="center")  # alpha
    # Faint labeled verticals at position_marks let a position be read across the rows.
    y_top = len(CASE_LABELS) - 0.4
    if use_log_scale:
        ax_cases.set_xscale("log")
    if position_marks is None:
        position_marks = tuple(np.arange(np.ceil(axis_range[0]), axis_range[1] + 1e-9) + 0.0)  # + 0.0 turns a -0 into 0
    for tick in position_marks:
        ax_cases.vlines(tick, -0.6, y_top, color="#D8D8D8", linewidth=0.8, linestyle="--", zorder=0)
        label = _position_mark_label(tick, use_log_scale)
        ax_cases.text(tick, y_top, label, ha="center", va="bottom", color="#888888", fontsize="small")
    ax_cases.set_xlim(*axis_range)
    ax_cases.set_ylim(-0.6, y_top)
    ax_cases.set_yticks([])
    ax_cases.set_xticks([])  # the labeled verticals are the position scale
    for side in ("left", "right", "top", "bottom"):
        ax_cases.spines[side].set_visible(False)

    # --- the metrics against alpha --------------
    alphas = np.linspace(*alpha_range, 400)
    curves = {metric: np.array([metrics(layout(a))[metric] for a in alphas]) for metric in METRIC_COLORS}
    if use_log_scale:
        curves = {metric: curve / metrics(layout(0.0))[metric] for metric, curve in curves.items()}
        ax_metrics.set_yscale("log")
    for metric, color in METRIC_COLORS.items():
        ax_metrics.plot(alphas, curves[metric], color=color, linewidth=1.8, label=metric)
    if diversity_max is not None:
        ax_metrics.set_ylim(top=diversity_max)
    for label, alpha in zip(CASE_LABELS, case_alphas):
        ax_metrics.axvline(alpha, color="#555555", linestyle=":", linewidth=1.2)
        ax_metrics.text(alpha, ax_metrics.get_ylim()[1], f" {label}", ha="left", va="top", color="#555555")
    ax_metrics.set_xlim(*alpha_range)
    ax_metrics.set_xlabel("\u03b1")  # alpha
    ax_metrics.set_ylabel("diversity relative to \u03b1 = 0" if use_log_scale else "diversity")
    ax_metrics.legend(**(legend or {"loc": "best"}))

    save_webp(fig, IMAGES_DIR / f"{name}.webp")


def _row_label_positions(axis_range: tuple[float, float], use_log_scale: bool) -> tuple[float, float]:
    """Return the x positions of a dot row's left and right labels, a small pad outside the axis range."""
    if use_log_scale:
        pad = axis_range[1] ** 0.06
        return axis_range[0] / pad, axis_range[1] * pad
    else:
        pad = 0.02 * (axis_range[1] - axis_range[0])
        return axis_range[0] - pad, axis_range[1] + pad


def _position_mark_label(tick: float, use_log_scale: bool) -> str:
    """Return a position mark's label: a power of ten on a logarithmic axis, else the number itself."""
    if use_log_scale:
        return f"$10^{{{int(np.log10(tick))}}}$"
    elif float(tick).is_integer():
        return f"{tick:.1f}"
    else:
        return f"{tick:g}"


# ==================================================================================================
#  Examples
# ==================================================================================================
def layout_closest_pair_fixed(alpha: float) -> NDArray[np.float64]:
    """Return items with the first two at 0.0 and 0.1 and every further item alpha after the previous."""
    return np.concatenate(([0.0, 0.1], 0.1 + alpha * np.arange(1, 10)))


def layout_near_duplicate(alpha: float) -> NDArray[np.float64]:
    """Return items uniform over [0, 1], except the second, which sits at alpha."""
    positions = np.linspace(0.0, 1.0, 11)
    positions[1] = alpha
    return positions


def layout_power_spacing(alpha: float) -> NDArray[np.float64]:
    """Return items power-spaced over [0, 1]: uniform at alpha = 1, crowding toward 0 below it and toward 1 above it."""
    return (np.arange(51) / 50.0) ** ((2.0 - alpha) / alpha)


def layout_decades(alpha: float) -> NDArray[np.float64]:
    """Return 11 items each ten times the previous one.

    alpha > 0 grows the largest gap 10**alpha-fold; alpha < 0 shrinks the smallest gap by the same factor.
    """
    positions = 10.0 ** np.arange(11, dtype=np.float64)
    if alpha > 0:
        positions[-1] *= 10.0**alpha
    else:
        positions[0] = positions[1] - 9.0 * 10.0**alpha
    return positions


def layout_constrained_group(alpha: float) -> NDArray[np.float64]:
    """Return 26 items pinned uniformly over [-0.25, 0] plus free items over (0, 1] crowding toward 0 for alpha > 0."""
    constrained = -0.25 + np.arange(26) / 100.0
    t = np.arange(26, 51) / 25.0 - 1.0
    free = (1.0 - alpha) * t + alpha * t**2
    return np.concatenate((constrained, free))


# ==================================================================================================
#  Uniform sampling: seven solved experiments
# ==================================================================================================
# The keys name the distances of `uniform_sampling_explorer.DISTANCES`.
DISTANCE_METRICS = {
    "l2": DistanceMetric.l2_euclidean(),
    "x": DistanceMetric.along_axis(0),
    "y": DistanceMetric.along_axis(1),
    "linf": DistanceMetric.l_minus_inf(),
    "geomean": DistanceMetric.geometric_mean(),
}
REFERENCE_LABELS = {"l2": "L2", "x": "$x$", "y": "$y$"}
# One row label per experiment, shared by the summary and the convergence tables; the section
# numbers are those of `uniform_sampling.md`, so a row maps to the section that shows the experiment.
EXPERIMENT_LABELS = {
    "l2": "**III.A** L2 distance",
    "x": "**III.B** $x$ distance",
    "y": "**III.C** $y$ distance",
    "linf": "**IV.A** L\u2212\u221e distance",
    "geomean": "**IV.B** geometric-mean distance",
    "hybrid": "**V.A** hybrid: L2, $x$ and $y$ terms",
    "hybrid_banded": "**V.B** hybrid, 20 items per band",
    "hybrid_banded_long": "**V.C** hybrid, 20 items per band, 900 s",
}
# The replay figure of section V.C re-solves the banded hybrid experiment with this budget.
LONG_BUDGET_SEC = 900.0
# A summary cell is colored by its achieved / reference fraction: red below LOW, green above HIGH.
# The classes are styled in docs/stylesheets/extra.css; `uniform_sampling.md` states the rule.
SUMMARY_LOW, SUMMARY_HIGH = 0.5, 0.7


@dataclass(frozen=True)
class ExperimentSettings:
    """The settings shared by every experiment, stored with each cached solution.

    `--reuse-solution` checks that a cache was solved with the settings it is asked to render.
    """

    n: int
    k: int
    budget_sec: float
    n_workers: int
    seed: int


@dataclass(frozen=True)
class Experiment:
    """An experiment maximizes the min separation over one distance, or over several as a hybrid.

    `distance_keys` holds one key for a simple objective and one per term for a hybrid, which is the
    geometric mean of the per-distance terms. `n_bands` cuts each axis into that many equal bands and
    requires every band to hold the same number of selected items, k / n_bands; None leaves the
    selection unconstrained.
    """

    name: str
    distance_keys: tuple[str, ...]
    n_bands: int | None = None

    def band_edges(self) -> tuple[float, ...]:
        """Return the interior band edges along one axis, empty for an unconstrained experiment."""
        if self.n_bands is None:
            return ()
        else:
            return tuple(i / self.n_bands for i in range(1, self.n_bands))

    def constraints(self, vectors: NDArray[np.float32], k: int) -> list[Constraint]:
        """Return one exact-count constraint per band along each axis, built from the population's coordinates.

        Raises:
            ValueError: If `k` is not divisible by `n_bands`, so the bands cannot hold equal counts.
        """
        if self.n_bands is None:
            return []
        elif k % self.n_bands != 0:
            raise ValueError(f"k = {k} does not split evenly over {self.n_bands} bands")
        else:
            per_band_count = k // self.n_bands
            band_indices = np.minimum((vectors * self.n_bands).astype(np.intp), self.n_bands - 1)
            return [
                Constraint(
                    int_set=set(np.flatnonzero(band_indices[:, axis] == band).tolist()),
                    min_count=per_band_count,
                    max_count=per_band_count,
                )
                for axis in (0, 1)
                for band in range(self.n_bands)
            ]

    def diversity_metric(self) -> DiversityMetric | HybridDiversityMetric:
        """Return the objective the solver maximizes."""
        if len(self.distance_keys) == 1:
            return DiversityMetric.MIN_SEPARATION
        else:
            return HybridDiversityMetric.geomean_of(
                *(DiversityMetric.MIN_SEPARATION.over(DISTANCE_METRICS[key]) for key in self.distance_keys)
            )

    def distance_metric(self) -> DistanceMetric:
        """Return the problem's distance metric; a hybrid carries its distances in its terms."""
        return DISTANCE_METRICS[self.distance_keys[0]]


EXPERIMENTS = (
    Experiment("l2", ("l2",)),
    Experiment("x", ("x",)),
    Experiment("y", ("y",)),
    Experiment("linf", ("linf",)),
    Experiment("geomean", ("geomean",)),
    Experiment("hybrid", ("l2", "x", "y")),
    Experiment("hybrid_banded", ("l2", "x", "y"), n_bands=5),
)


def build_uniform_sampling_population(n: int, seed: int) -> NDArray[np.float32]:
    """Return n points drawn uniformly at random from the unit square, in the solver's float32.

    A coordinate value that a draw repeats is redrawn: 10,000 float32 values in [0, 1] repeat a value a
    few times by chance, and a repeated coordinate is a pair at distance zero under the x, y, L-inf and
    geometric-mean distances, which would turn the experiments into a study of the tie-breakers.
    """
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2)).astype(np.float32)
    for axis in (0, 1):
        while True:
            _, first_index = np.unique(points[:, axis], return_index=True)
            repeated = np.setdiff1d(np.arange(n), first_index)
            if repeated.size == 0:
                break
            points[repeated, axis] = rng.random(repeated.size).astype(np.float32)
    return points


def build_experiment_problem(vectors: NDArray[np.float32], experiment: Experiment, k: int) -> MaxDivProblem:
    """Return the experiment's problem over the population."""
    return MaxDivProblem.new(
        vectors=vectors,
        k=k,
        distance_metric=experiment.distance_metric(),
        diversity_metric=experiment.diversity_metric(),
        constraints=experiment.constraints(vectors, k),
    )


def solve_experiment(
    vectors: NDArray[np.float32],
    experiment: Experiment,
    settings: ExperimentSettings,
    should_record_intermediate_selections: bool = False,
) -> ParallelMaxDivSolution:
    """Return the experiment's solution, solved within an end-to-end budget.

    With `should_record_intermediate_selections` every checkpoint also carries its selection, which the
    replay figure steps through.
    """
    problem = build_experiment_problem(vectors, experiment, settings.k)
    solver = (
        ParallelMaxDivSolverBuilder(problem)
        .with_seed(settings.seed)
        .with_workers(seconds(settings.budget_sec), settings.n_workers)
        .with_end_to_end_budget()
        .with_intermediate_selections(should_record_intermediate_selections)
        .build()
    )
    return solver.solve(verbosity=0)


@dataclass(frozen=True)
class ExperimentRun:
    """An `ExperimentRun` holds one experiment's cached outcome.

    The outcome is the selected indices and the best-known convergence trace across workers.
    `checkpoints` lists `(elapsed seconds, iterations, primary diversity)` in the order the solver
    recorded them; the last entry is the run's total. `frames`, kept only for the run the replay
    figure shows, lists the checkpoints at which the best-known selection changed, each as
    `(elapsed seconds, primary diversity, selected indices)`.
    """

    i_selected: NDArray[np.intp]
    checkpoints: list[tuple[float, int, float]]
    frames: list[tuple[float, float, list[int]]] | None = None

    @property
    def n_iterations(self) -> int:
        """Return the iterations run by the worker holding the last checkpoint."""
        return self.checkpoints[-1][1]

    def diversity_at(self, t_sec: float) -> float:
        """Return the primary diversity at the last checkpoint recorded at or before `t_sec`, 0.0 before the first."""
        reached = [diversity for elapsed, _, diversity in self.checkpoints if elapsed <= t_sec]
        return reached[-1] if reached else 0.0


def selection_frames(solution: ParallelMaxDivSolution) -> list[tuple[float, float, list[int]]]:
    """Return the checkpoints at which the best-known selection changed, as `(elapsed seconds, diversity, indices)`.

    A checkpoint whose selection equals the previous kept frame's is dropped, so a reader stepping
    through the frames never looks for a change that is not there.
    """
    frames: list[tuple[float, float, list[int]]] = []
    for checkpoint in solution.score_checkpoints:
        indices = [int(i) for i in checkpoint.i_selected]
        if not frames or indices != frames[-1][2]:
            frames.append((round(checkpoint.elapsed.t_elapsed_sec, 3), checkpoint.score.diversity, indices))
    return frames


def load_or_solve_experiment(
    vectors: NDArray[np.float32],
    experiment: Experiment,
    settings: ExperimentSettings,
    should_reuse_solution: bool,
    cache_name: str | None = None,
    should_record_frames: bool = False,
    should_pickle_solution: bool = False,
) -> ExperimentRun:
    """Return the experiment's run, from its JSON cache when asked and present, else from a fresh solve.

    A fresh solve rewrites the cache; an experiment without a cache is solved even when reuse is asked,
    so a new experiment can be added without re-solving the others. The cache holds the selected
    indices and the convergence trace, plus the selection frames when `should_record_frames` asks for them:
    the population is rebuilt from n and the seed, so the dots and the raster always come from the same
    coordinates. `cache_name` names the cache when one experiment is solved under several settings.
    `should_pickle_solution` also stores the whole solution object beside the cache, gitignored.
    """
    path = GENERATED_DIR / f"uniform_sampling_{cache_name or experiment.name}_solution.json"
    if should_reuse_solution and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if {key: cached[key] for key in asdict(settings)} != asdict(settings):
            raise ValueError(f"{path} was solved with other settings than {settings}")
        frames = (
            [(t_sec, diversity, indices) for t_sec, diversity, indices in cached["frames"]]
            if should_record_frames
            else None
        )
        return ExperimentRun(
            np.asarray(cached["i_selected"], dtype=np.intp), [tuple(row) for row in cached["checkpoints"]], frames
        )
    solution = solve_experiment(
        vectors, experiment, settings, should_record_intermediate_selections=should_record_frames
    )
    if should_pickle_solution:
        # Pickle the whole solution, so figures that need more than the selection (the solve timeline)
        # can be redrawn without re-solving; unlike the JSON cache, this file is gitignored.
        pickle_path = _solution_pickle_path(cache_name or experiment.name)
        pickle_path.write_bytes(pickle.dumps(solution))
        print(f"wrote {pickle_path.relative_to(REPO_ROOT)}")
    run = ExperimentRun(
        np.asarray(solution.i_selected, dtype=np.intp),
        [
            (round(checkpoint.elapsed.t_elapsed_sec, 3), checkpoint.elapsed.n_iterations, checkpoint.score.diversity)
            for checkpoint in solution.score_checkpoints
        ],
        selection_frames(solution) if should_record_frames else None,
    )
    record = {**asdict(settings), "i_selected": [int(i) for i in run.i_selected], "checkpoints": run.checkpoints}
    if run.frames is not None:
        record["frames"] = run.frames
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return run


def _solution_pickle_path(name: str) -> Path:
    """Return where an experiment's whole solution is pickled, beside its JSON cache."""
    return GENERATED_DIR / f"uniform_sampling_{name}_solution.pkl"


def render_uniform_sampling_population(name: str, vectors: NDArray[np.float32], pixels: int = 1000) -> None:
    """Render the population alone as a square lossless raster covering exactly the unit square.

    The interactive figures lay their SVG marks over this image, so the image carries no axes or margins:
    its edges are the square's edges.
    """
    use_docs_style()
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        dpi = plt.rcParams["savefig.dpi"]
        fig = plt.figure(figsize=(pixels / dpi, pixels / dpi))
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")
        ax.scatter(vectors[:, 0], vectors[:, 1], s=0.7, color=POPULATION_COLOR, linewidths=0)
        save_webp(fig, IMAGES_DIR / f"{name}.webp", lossless=True)


def min_separations(selection: NDArray[np.float64]) -> dict[str, float]:
    """Return the selection's min separation under the L2, x and y distances."""
    neighbors = nearest_neighbors(selection[:, 0], selection[:, 1], tuple(REFERENCE_LABELS))
    return {key: float(np.min(distance)) for key, (_, distance) in neighbors.items()}


# The densest known packing of 100 equal circles in a unit square (E. Specht, Packomania,
# https://www.packomania.com/csq/pdf/d9.pdf) has this radius; the circle centers, which are the points,
# then lie in the inner square of side 1 - 2r, so the point spacing in the unit square is 2r / (1 - 2r).
PACKING_RADIUS_100 = 0.051401071774
PACKING_SPACING_100 = 2.0 * PACKING_RADIUS_100 / (1.0 - 2.0 * PACKING_RADIUS_100)


def reference_separations(k: int) -> dict[str, float]:
    """Return the free-placement reference separation per reference distance, as `uniform_sampling.md` derives them.

    Along one axis, k evenly spaced values over [0, 1] are 1 / (k - 1) apart; in the square, the
    reference is the densest known packing of k points, tabulated for k = 100 only.
    """
    if k != 100:
        raise ValueError(f"the L2 reference is tabulated for k = 100 only, not k = {k}")
    return {"l2": PACKING_SPACING_100, "x": 1.0 / (k - 1), "y": 1.0 / (k - 1)}


def write_experiment_separations(run_name: str, selection: NDArray[np.float64], k: int) -> None:
    """Write a run's achieved separations beside their references as a table fragment named after the run.

    `docs/guides/uniform_sampling.md` includes the fragment below the run's figure, so the numbers
    come from the same solve as the figure.
    """
    achieved, references = min_separations(selection), reference_separations(k)
    lines = ["| min separation under … | achieved | reference | achieved / reference |", "|---|---|---|---|"]
    for key, label in REFERENCE_LABELS.items():
        lines.append(
            f"| {label} | {achieved[key]:.4f} | {references[key]:.4f} | {achieved[key] / references[key]:.0%} |"
        )
    path = GENERATED_DIR / f"uniform_sampling_{run_name}_separations.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def _summary_cell(value: float, fraction: float) -> str:
    """Return a summary cell: the value and its fraction, wrapped in a colored span when the fraction is low or high."""
    text = f"{value:.4f} ({fraction:.0%})"
    if fraction < SUMMARY_LOW:
        return f'<span class="usx-low">{text}</span>'
    elif fraction > SUMMARY_HIGH:
        return f'<span class="usx-high">{text}</span>'
    else:
        return text


def write_summary(selections: dict[str, NDArray[np.float64]], k: int) -> None:
    """Write the closing table: every experiment's achieved separations as a fraction of the references."""
    references = reference_separations(k)
    achieved = {name: min_separations(selection) for name, selection in selections.items()}
    fractions = {name: {key: achieved[name][key] / references[key] for key in REFERENCE_LABELS} for name in selections}
    header = " | ".join(f"{label}, achieved / reference" for label in REFERENCE_LABELS.values())
    lines = [f"| experiment | {header} |", "|---|---|---|---|"]
    for name in selections:
        cells = [_summary_cell(achieved[name][key], fractions[name][key]) for key in REFERENCE_LABELS]
        lines.append(f"| {EXPERIMENT_LABELS[name]} | {' | '.join(cells)} |")
    path = GENERATED_DIR / "uniform_sampling_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


CONVERGENCE_MARKS_SEC = (5.0, 15.0, 30.0)


def write_convergence(runs: dict[str, ExperimentRun], budget_sec: float) -> None:
    """Write the table of iterations run and of the objective's progress, per experiment.

    The progress columns give the primary diversity at a few elapsed marks as a fraction of its final
    value, for the best selection any worker held; a column near 100% early on means the run had converged by then.
    """
    marks = " | ".join(f"at {mark:g} s" for mark in CONVERGENCE_MARKS_SEC)
    lines = [
        f"| experiment | iterations in {budget_sec:g} s | {marks} |",
        "|---|---|" + "---|" * len(CONVERGENCE_MARKS_SEC),
    ]
    for name, run in runs.items():
        final = run.checkpoints[-1][2]
        cells = " | ".join(f"{run.diversity_at(mark) / final:.1%}" for mark in CONVERGENCE_MARKS_SEC)
        lines.append(f"| {EXPERIMENT_LABELS[name]} | {run.n_iterations:,} | {cells} |")
    path = GENERATED_DIR / "uniform_sampling_convergence.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def render_uniform_sampling_experiments(settings: ExperimentSettings, should_reuse_solution: bool) -> None:
    """Produce the case study: the population raster, and per experiment its figure fragment and separations table.

    Args:
        settings: Its seed seeds both the population and every solver run.
        should_reuse_solution: Read the cached selections where they exist and skip those solves.
    """
    vectors = build_uniform_sampling_population(settings.n, settings.seed)
    render_uniform_sampling_population("uniform_sampling_population", vectors)
    selections, runs = {}, {}
    for experiment in EXPERIMENTS:
        run = load_or_solve_experiment(vectors, experiment, settings, should_reuse_solution)
        runs[experiment.name] = run
        selection = vectors[run.i_selected].astype(np.float64)
        selections[experiment.name] = selection
        write_experiment_separations(experiment.name, selection, settings.k)
        fragment = explorer_fragment(
            selection[:, 0],
            selection[:, 1],
            settings.n,
            settings.k,
            objective_keys=experiment.distance_keys,
            band_edges=experiment.band_edges(),
            population_image_url="../images/uniform_sampling_population.webp",
            description=(
                "Ten thousand gray points in the unit square with the hundred selected ones in red, and the "
                "selection's x and y values as rug marks along the bottom and left edges"
            ),
        )
        path = GENERATED_DIR / f"uniform_sampling_{experiment.name}_figure.html"
        path.write_text(fragment, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    selections["hybrid_banded_long"] = render_uniform_sampling_replay(vectors, settings, should_reuse_solution)
    write_summary(selections, settings.k)
    write_convergence(runs, settings.budget_sec)


def render_uniform_sampling_replay(
    vectors: NDArray[np.float32], settings: ExperimentSettings, should_reuse_solution: bool
) -> NDArray[np.float64]:
    """Produce the replay figure of the banded hybrid experiment solved for `LONG_BUDGET_SEC`, and return its selection.

    The run is the last experiment's problem under a longer budget, with the selection recorded at every
    checkpoint; the figure steps through the checkpoints at which the selection changed.
    """
    experiment = EXPERIMENTS[-1]
    run = load_or_solve_experiment(
        vectors,
        experiment,
        replace(settings, budget_sec=LONG_BUDGET_SEC),
        should_reuse_solution,
        cache_name="hybrid_banded_long",
        should_record_frames=True,
        should_pickle_solution=True,
    )
    assert run.frames is not None
    render_uniform_sampling_long_run_timeline()
    selection = vectors[run.i_selected].astype(np.float64)
    write_experiment_separations("hybrid_banded_long", selection, settings.k)
    fragment = replay_fragment(
        vectors,
        [ReplayFrame(t_sec, diversity, indices) for t_sec, diversity, indices in run.frames],
        settings.n,
        settings.k,
        objective_keys=experiment.distance_keys,
        band_edges=experiment.band_edges(),
        population_image_url="../images/uniform_sampling_population.webp",
        description=(
            "Ten thousand gray points in the unit square with the hundred selected ones in red, stepping "
            "through every change of the selection during the solve"
        ),
    )
    path = GENERATED_DIR / "uniform_sampling_hybrid_banded_long_replay.html"
    path.write_text(fragment, encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return selection


def render_uniform_sampling_long_run_timeline() -> None:
    """Draw the 900 s solve as a timeline, tie-breaker panels included, from the pickled solution.

    The image is linked from the guide, not shown inline: it is a tall figure that documents the solve
    behind the replay, not a result of the case study.
    """
    pickle_path = _solution_pickle_path("hybrid_banded_long")
    if not pickle_path.exists():
        print(f"no {pickle_path.relative_to(REPO_ROOT)}: the 900 s solve predates the pickling; re-solve it")
        return
    solution: ParallelMaxDivSolution = pickle.loads(pickle_path.read_bytes())
    use_docs_style()
    save_webp(
        solution.plot_timeline(include_tie_breakers=True),
        IMAGES_DIR / "uniform_sampling_hybrid_banded_long_timeline.webp",
    )


def main() -> None:
    """Render every guide figure."""
    parser = argparse.ArgumentParser(description="Regenerate the guide figures.")
    parser.add_argument(
        "--reuse-solution",
        action="store_true",
        help="read the cached experiment selections; solve only experiments without a cache",
    )
    args = parser.parse_args()
    render_example(
        "geomean_separation_I1",
        layout_closest_pair_fixed,
        case_alphas=(0.1, 0.2, 0.3),
        alpha_range=(0.05, 0.35),
        axis_range=(-0.1, 3.1),
        legend={"loc": "upper left", "bbox_to_anchor": (0.19, 1.0)},
        highlight=(0, 1),
    )
    render_example(
        "geomean_separation_I2",
        layout_near_duplicate,
        case_alphas=(0.001, 0.01, 0.1),
        alpha_range=(0.0, 0.2),
        axis_range=(-0.1, 1.1),
        legend={"loc": "lower center"},
        diversity_max=0.12,
        highlight=(1,),
    )
    render_example(
        "geomean_separation_I3",
        layout_power_spacing,
        case_alphas=(0.5, 1.0, 1.75),
        alpha_range=(0.01, 1.99),
        axis_range=(-0.1, 1.1),
        legend={"loc": "upper left", "bbox_to_anchor": (0.01, 0.92)},
        diversity_max=0.022,
        dot_size=14,
    )
    render_example(
        "geomean_separation_I4",
        layout_decades,
        case_alphas=(-0.5, 0.0, 1.0),
        alpha_range=(-1.0, 1.0),
        axis_range=(0.3, 1e12),
        legend={"loc": "lower right"},
        highlight=(0, 10),
        position_marks=tuple(10.0**n for n in range(12)),
        use_log_scale=True,
    )
    render_example(
        "geomean_separation_II",
        layout_constrained_group,
        case_alphas=(-0.5, 0.0, 0.75),
        alpha_range=(-1.0, 1.0),
        axis_range=(-0.35, 1.1),
        dot_size=14,
        position_marks=(-0.25, 0.0, 1.0),
    )
    render_uniform_sampling_experiments(
        ExperimentSettings(n=10_000, k=100, budget_sec=60.0, n_workers=16, seed=42),
        should_reuse_solution=args.reuse_solution,
    )


if __name__ == "__main__":
    main()
