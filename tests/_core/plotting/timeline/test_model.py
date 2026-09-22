import numpy as np
import pytest

from max_div._core.solver._duration import Elapsed
from max_div._core.solver._parallel import ParallelMaxDivSolution, WorkerConfig, WorkerGroupChange, WorkerSummary
from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._score import Score
from max_div._core.solver._score_checkpoint import ScoreCheckpoint
from max_div._core.solver._step_identity import SolverStepIdentity
from tests._extras import skip_module_unless_extra

skip_module_unless_extra("plot")

from max_div._core.plotting.timeline import ParallelSolutionTimelinePlot  # noqa: E402
from max_div._core.plotting.timeline.model import BandSegment, BestInterval, GroupBlock, ScorePoint  # noqa: E402


def _built_plot() -> ParallelSolutionTimelinePlot:
    """Return a plot of three workers, replayed by hand: group 1 dissolves into group 0 at t=5."""
    plot = ParallelSolutionTimelinePlot(initial_groups=[0, 0, 1], start_offsets=[0.0, 0.0, 2.0])
    plot.set_worker_preset(0, "SMART")
    plot.record_best(0.0, 0)
    plot.record_score(0.0, 0.1, 1.0)
    plot.dissolve_group(5.0, 1, {2: 0})
    plot.record_best(5.0, 2)
    plot.record_score(5.0, 0.5, 1.0)
    plot.finish(10.0)
    return plot


# ==================================================================================================
#  Building by event replay
# ==================================================================================================
def test_a_worker_moves_to_the_first_empty_band_of_its_new_group():
    """On reassignment a worker vacates its band and takes the first empty band of its target group."""
    # --- arrange / act ----------------
    segments = _built_plot().segments

    # --- assert -----------------------
    # worker 2 sits in group 1 band 0 until t=5, then group 0's first empty band (0 and 1 are taken)
    assert BandSegment(worker=2, group=1, row=0, t_from=2.0, t_to=5.0) in segments
    assert BandSegment(worker=2, group=0, row=2, t_from=5.0, t_to=10.0) in segments


def test_a_group_block_spans_its_life_and_is_as_tall_as_its_peak_size():
    """A block runs from its earliest member's start to its dissolution (or the end), peak-size bands tall."""
    # --- arrange / act ----------------
    blocks = _built_plot().group_blocks

    # --- assert -----------------------
    # group 0 grows to three workers and survives to the end; group 1 holds one worker until it dissolves
    assert blocks == [
        GroupBlock(group=0, t_start=0.0, t_end=10.0, height=3),
        GroupBlock(group=1, t_start=2.0, t_end=5.0, height=1),
    ]


def test_a_zero_time_interim_assignment_is_dropped_and_does_not_inflate_its_group():
    """A worker routed through a group for no time by a same-instant multi-way merge is filtered out."""
    # --- arrange ----------------------
    plot = ParallelSolutionTimelinePlot(initial_groups=[0, 1, 2], start_offsets=[0.0, 0.0, 0.0])
    plot.dissolve_group(10.0, 1, {1: 2})  # worker 1 lands in group 2 ...
    plot.dissolve_group(10.0, 2, {2: 0, 1: 0})  # ... which itself dissolves at the same instant

    # --- act --------------------------
    plot.finish(20.0)
    segments = plot.segments
    group_2 = next(block for block in plot.group_blocks if block.group == 2)

    # --- assert -----------------------
    assert BandSegment(worker=1, group=2, row=1, t_from=10.0, t_to=10.0) not in segments
    assert all(segment.t_to > segment.t_from for segment in segments)
    assert group_2.height == 1  # only worker 2 ever really sat in group 2


def test_the_best_intervals_name_the_holder_and_its_group_over_each_span():
    """Each best interval carries the worker that held the best score and the group it was in at the time."""
    # --- arrange / act ----------------
    plot = _built_plot()

    # --- assert -----------------------
    assert plot.best_intervals == [
        BestInterval(t_from=0.0, t_to=5.0, worker=0, group=0),
        BestInterval(t_from=5.0, t_to=10.0, worker=2, group=0),
    ]
    assert plot.score_points == [
        ScorePoint(t=0.0, diversity=0.1, constraints=1.0),
        ScorePoint(t=5.0, diversity=0.5, constraints=1.0),
    ]


def test_the_worker_facts_are_read_back():
    """The plot reports its worker count, preset labels and start offsets."""
    # --- arrange / act ----------------
    plot = _built_plot()

    # --- assert -----------------------
    assert plot.n_workers == 3
    assert plot.worker_presets == {0: "SMART"}
    assert plot.start_offsets == [0.0, 0.0, 2.0]
    assert plot.t_end == 10.0


def test_an_event_going_back_in_time_is_rejected():
    """A timed modifier called out of order raises, because a real solve's events never go backwards."""
    # --- arrange ----------------------
    plot = ParallelSolutionTimelinePlot(initial_groups=[0, 0], start_offsets=[0.0, 0.0])
    plot.record_score(5.0, 0.1, 1.0)

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="back in time"):
        plot.record_score(4.0, 0.2, 1.0)


def test_reading_the_end_before_finishing_is_an_error():
    """`t_end` is only known once `finish` has closed the axis."""
    # --- arrange / act / assert -------
    plot = ParallelSolutionTimelinePlot(initial_groups=[0], start_offsets=[0.0])
    with pytest.raises(ValueError, match="not finished"):
        _ = plot.t_end


# ==================================================================================================
#  Building from a solved solution
# ==================================================================================================
def _summary(worker_index: int, offset: float) -> WorkerSummary:
    """Return a worker summary that ran the default preset, starting at `offset` on the shared axis."""
    return WorkerSummary(
        worker_index=worker_index,
        config=WorkerConfig(preset=SolverPreset.DEFAULT),
        seed=worker_index,
        score=Score(size=1.0, constraints=1.0, diversities=(0.5,)),
        elapsed=Elapsed(t_elapsed_sec=5.0, n_iterations=50),
        has_best_score=(worker_index == 2),
        t_start_offset_sec=offset,
    )


def _checkpoint(t: float, worker: int, diversity: float) -> ScoreCheckpoint:
    """Return a checkpoint held by `worker` at shared-axis time `t`."""
    return ScoreCheckpoint(
        SolverStepIdentity(1, "step"),
        Elapsed(t_elapsed_sec=t, n_iterations=int(10 * t)),
        Score(size=1.0, constraints=1.0, diversities=(diversity,)),
        worker_index=worker,
        group_index=0,
    )


def _solution() -> ParallelMaxDivSolution:
    """Return a fabricated three-worker solution: group 1 dissolves into group 0 at t=5."""
    change = WorkerGroupChange(
        executed_by=0,
        elapsed=Elapsed(t_elapsed_sec=5.0, n_iterations=50),
        progress_fraction=0.5,
        dissolved_group=1,
        n_alive_groups_after=1,
        slot_scores={},
        reassignments={2: 0},
    )
    # the solve runs on past the t=5 merge to t=10, so the merged worker holds its new band for real time
    return ParallelMaxDivSolution(
        i_selected=np.array([0], dtype=np.int32),
        score_checkpoints=[_checkpoint(0.0, 0, 0.1), _checkpoint(5.0, 2, 0.5), _checkpoint(10.0, 2, 0.6)],
        step_durations=[],
        workers=[_summary(0, 0.0), _summary(1, 0.0), _summary(2, 0.0)],
        winning_worker=2,
        initial_worker_groups=[0, 0, 1],
        worker_group_changes=[change],
    )


def test_from_solution_replays_the_grouping_and_resolves_the_preset_alias():
    """`from_solution` reads the grouping, offsets and the DEFAULT-to-SMART preset off the solution."""
    # --- arrange / act ----------------
    plot = ParallelSolutionTimelinePlot.from_solution(_solution())

    # --- assert -----------------------
    assert plot.n_workers == 3
    assert plot.worker_presets == {0: "SMART", 1: "SMART", 2: "SMART"}
    assert plot.group_blocks == [
        GroupBlock(group=0, t_start=0.0, t_end=10.0, height=3),
        GroupBlock(group=1, t_start=0.0, t_end=5.0, height=1),
    ]
    assert plot.score_points == [
        ScorePoint(t=0.0, diversity=0.1, constraints=1.0),
        ScorePoint(t=5.0, diversity=0.5, constraints=1.0),
        ScorePoint(t=10.0, diversity=0.6, constraints=1.0),
    ]


def test_from_solution_credits_the_holder_from_the_replayed_grouping():
    """A checkpoint's holder is placed in the group the replayed changes put it in, not the checkpoint's own tag."""
    # --- arrange / act ----------------
    plot = ParallelSolutionTimelinePlot.from_solution(_solution())

    # --- assert -----------------------
    # the change at t=5 runs before the t=5 checkpoint, so worker 2 is already in group 0 when it leads
    assert BestInterval(t_from=0.0, t_to=5.0, worker=0, group=0) in plot.best_intervals


def test_from_solution_records_the_tie_breakers_and_their_labels():
    """Every point carries the checkpoint's tie-breaker scores, and the labels are the solution's, primary excluded."""
    # --- arrange ----------------------
    solution = _solution()
    solution.diversity_objective_labels = ["min_separation", "approx_geomean_separation", "non_zero_fraction"]
    solution.score_checkpoints = [
        ScoreCheckpoint(
            checkpoint.step_identity,
            checkpoint.elapsed,
            Score(size=1.0, constraints=1.0, diversities=(checkpoint.score.diversity, 0.2, 0.9)),
            worker_index=checkpoint.worker_index,
            group_index=checkpoint.group_index,
        )
        for checkpoint in solution.score_checkpoints
    ]

    # --- act --------------------------
    plot = ParallelSolutionTimelinePlot.from_solution(solution)

    # --- assert -----------------------
    assert [point.tie_breakers for point in plot.score_points] == [(0.2, 0.9)] * 3
    assert plot.tie_breaker_labels == ["approx_geomean_separation", "non_zero_fraction"]


def test_a_solution_without_labels_leaves_the_tie_breakers_unnamed():
    """A fabricated solution records no labels, so the plot's tie-breaker labels are empty."""
    assert ParallelSolutionTimelinePlot.from_solution(_solution()).tie_breaker_labels == []
