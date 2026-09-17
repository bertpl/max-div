import numpy as np

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._parallel import ParallelMaxDivSolution, WorkerConfig, WorkerSummary
from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._score import Score
from max_div._core.solver._score_checkpoint import ScoreCheckpoint
from max_div._core.solver._step_identity import SolverStepIdentity
from tests._extras import skip_module_unless_extra

skip_module_unless_extra("plot")

from matplotlib.figure import Figure  # noqa: E402

from max_div._core.plotting.timeline import ParallelSolutionTimelinePlot  # noqa: E402


def _plot(first_constraints: float = 1.0) -> ParallelSolutionTimelinePlot:
    """Return a plot whose group 1 dissolves into group 0 at t=5, ending at t=10."""
    plot = ParallelSolutionTimelinePlot(initial_groups=[0, 0, 1], start_offsets=[0.0, 0.0, 0.0])
    plot.set_worker_preset(0, "SMART")
    plot.record_best(0.0, 0)
    plot.record_score(0.0, 0.1, first_constraints)
    plot.dissolve_group(5.0, 1, {2: 0})
    plot.record_best(5.0, 2)
    plot.record_score(5.0, 0.5, 1.0)
    plot.finish(10.0)
    return plot


def test_render_returns_a_two_panel_figure_when_constraints_hold():
    """With the constraint score at one throughout, the figure is the band panel and the diversity panel."""
    # --- arrange / act ----------------
    fig = _plot().render()

    # --- assert -----------------------
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_a_third_panel_appears_when_the_constraint_score_drops_below_one():
    """A constraints trajectory that dips below one gets its own panel below the diversity panel."""
    # --- arrange / act ----------------
    fig = _plot(first_constraints=0.8).render()

    # --- assert -----------------------
    assert len(fig.axes) == 3
    assert fig.axes[2].get_ylabel() == "constraints"


def test_the_title_label_and_color_legend_are_set():
    """The figure carries the title, the elapsed-seconds axis label, and the three-color legend."""
    # --- arrange / act ----------------
    fig = _plot().render()

    # --- assert -----------------------
    assert fig.axes[0].get_title(loc="left") == "Parallel solve timeline"
    assert fig.axes[-1].get_xlabel() == "elapsed seconds"
    legend_labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert legend_labels == ["active worker", "leading group", "leading worker"]


def test_each_dissolution_draws_one_marker_line_on_every_panel():
    """A group dissolution before the end draws one vertical marker line on each panel."""
    # --- arrange / act ----------------
    fig = _plot(first_constraints=0.8).render()

    # --- assert -----------------------
    # the band panel holds only the dissolution lines; one dissolution here, so one line per panel
    assert len(fig.axes[0].lines) == 1
    assert all(line.get_xdata()[0] == 5.0 for line in fig.axes[0].lines)


def test_plot_timeline_on_a_solution_returns_a_figure():
    """`ParallelMaxDivSolution.plot_timeline` builds the plot from the solution and draws it."""
    # --- arrange ----------------------
    checkpoints = [
        ScoreCheckpoint(
            SolverStepIdentity(1, "step"), Elapsed(t, int(10 * t)), Score(1.0, 1.0, (d,)), worker_index=0, group_index=0
        )
        for t, d in [(0.0, 0.1), (5.0, 0.5)]
    ]
    worker = WorkerSummary(
        worker_index=0,
        config=WorkerConfig(preset=SolverPreset.SMART),
        seed=0,
        score=Score(1.0, 1.0, (0.5,)),
        elapsed=Elapsed(5.0, 50),
        has_best_score=True,
        t_start_offset_sec=0.0,
    )
    solution = ParallelMaxDivSolution(
        i_selected=np.array([0], dtype=np.int32),
        score_checkpoints=checkpoints,
        step_durations=[],
        workers=[worker],
        winning_worker=0,
        initial_worker_groups=[0],
        worker_group_changes=[],
    )

    # --- act --------------------------
    fig = solution.plot_timeline()

    # --- assert -----------------------
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title(loc="left") == "Parallel solve timeline"
