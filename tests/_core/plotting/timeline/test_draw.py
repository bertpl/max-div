import numpy as np
import pytest

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
from max_div._core.plotting.timeline.draw import (  # noqa: E402
    _BAND_MAX_INCH,
    _BAND_MIN_INCH,
    _BAND_SHRINK_THRESHOLD,
    _format_elapsed,
    _time_axis_step,
    _top_panel_inch,
)


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
    """The figure carries the title, the elapsed-time axis label, and the three-color legend."""
    # --- arrange / act ----------------
    fig = _plot().render()

    # --- assert -----------------------
    assert fig.axes[0].get_title(loc="left") == "Parallel solve timeline"
    assert fig.axes[-1].get_xlabel() == "Elapsed time"
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


def test_a_reassigned_worker_is_labeled_white_in_each_band_it_occupies():
    """A worker reassigned to another group is labeled again in its new band, and every label is white."""
    # --- arrange / act ----------------
    fig = _plot().render()  # worker 2 starts in group 1 and moves to group 0 at t=5, so it occupies two bands
    texts = fig.axes[0].texts

    # --- assert -----------------------
    w2_labels = [text for text in texts if text.get_text().split()[0] == "2"]
    assert len(w2_labels) == 2
    assert all(text.get_color() == "white" for text in texts)


def test_each_group_block_is_named_beside_its_top_band():
    """The top panel's y tick labels name each group block, one per block."""
    # --- arrange / act ----------------
    fig = _plot().render()  # groups 0 and 1, so two blocks

    # --- assert -----------------------
    labels = sorted(text.get_text() for text in fig.axes[0].get_yticklabels())
    assert labels == ["group 0", "group 1"]


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


@pytest.mark.parametrize(
    "total_extent, expected_band_inch",
    [
        (4.0, _BAND_MAX_INCH),  # few bands: bands are at full height
        (16.0, _BAND_MAX_INCH * (_BAND_SHRINK_THRESHOLD / 16.0) ** 0.5),  # past the threshold: shrunk, above the floor
        (200.0, _BAND_MIN_INCH),  # many bands: band height pinned to the floor, panel grows instead
    ],
)
def test_top_panel_height_clamps_the_band_height_between_its_bounds(total_extent: float, expected_band_inch: float):
    """The panel height is the per-band height, clamped to `[_BAND_MIN_INCH, _BAND_MAX_INCH]`, times the rows."""
    # --- act --------------------------
    top_inch = _top_panel_inch(total_extent)

    # --- assert -----------------------
    assert top_inch == pytest.approx(expected_band_inch * total_extent)
    assert _BAND_MIN_INCH <= top_inch / total_extent <= _BAND_MAX_INCH


@pytest.mark.parametrize(
    "t_end, expected_step",
    [
        (3600, 600),  # one hour -> 10-minute ticks
        (300, 60),  # five minutes -> 1-minute ticks
        (45, 5),  # under a minute -> 5-second ticks
        (7200, 900),  # two hours -> 15-minute ticks
    ],
)
def test_time_axis_step_picks_a_natural_spacing_within_the_tick_budget(t_end: float, expected_step: float):
    """The elapsed-time axis steps on a natural unit, keeping the interval count within budget."""
    # --- act --------------------------
    step = _time_axis_step(t_end)

    # --- assert -----------------------
    assert step == expected_step
    assert t_end / step <= 9


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "0"),
        (10, "10s"),
        (1800, "30m"),
        (3600, "1h"),
        (5400, "1h30m"),
        (1830, "30m30s"),
    ],
)
def test_format_elapsed_uses_whole_time_units(seconds: float, expected: str):
    """An elapsed-second tick formats as its whole hours, minutes and seconds."""
    # --- act / assert -----------------
    assert _format_elapsed(seconds) == expected
