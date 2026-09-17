"""The geometry of a parallel solve's timeline, built by replaying the solve's events onto one axis.

`ParallelSolutionTimelinePlot` holds everything a timeline figure needs and nothing visual: the group
blocks with their height and span, per worker the bands it occupied and when, which worker held the
best score over each interval, and the score trajectory. `draw` turns this geometry into a figure.

The plot is populated by event replay, not through its constructor: the constructor fixes the initial
grouping and where each worker's band starts, then `dissolve_group`, `record_best` and `record_score`
advance the state in time order. `from_solution` drives that replay from a solved
`ParallelMaxDivSolution`; the modifiers take plain values, so this module imports nothing from the
solver package and a test can build a plot from hand-written events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from max_div._core.solver import ParallelMaxDivSolution


# ==================================================================================================
#  Geometry
# ==================================================================================================
@dataclass(frozen=True)
class BandSegment:
    """One stretch a worker spent in one band of one group, from `t_from` to `t_to` on the shared axis."""

    worker: int
    group: int
    row: int
    t_from: float
    t_to: float


@dataclass(frozen=True)
class GroupBlock:
    """A worker group's block: it is drawn `height` bands tall, over `t_start` to `t_end` on the shared axis."""

    group: int
    t_start: float
    t_end: float
    height: int


@dataclass(frozen=True)
class BestInterval:
    """Over `t_from` to `t_to`, `worker` held the best score, and it was then in group `group`."""

    t_from: float
    t_to: float
    worker: int
    group: int


@dataclass(frozen=True)
class ScorePoint:
    """The diversity and constraints scores of the best-known selection at time `t` on the shared axis."""

    t: float
    diversity: float
    constraints: float


# ==================================================================================================
#  ParallelSolutionTimelinePlot
# ==================================================================================================
@dataclass
class ParallelSolutionTimelinePlot:
    """The drawable geometry of a parallel solve's timeline, populated by replaying the solve's events.

    Build it with `from_solution`, or directly for a test: the constructor takes each worker's initial
    group and where its band starts, then the modifiers replay the solve. Every timed modifier must be
    called in non-decreasing time order; a call that goes back in time raises, because a real solve's
    events never do.
    """

    # --------------------------------------------------------------------------
    #  Construction
    # --------------------------------------------------------------------------
    def __init__(self, initial_groups: list[int], start_offsets: list[float]) -> None:
        """Fix the initial grouping and each worker's band start.

        Args:
            initial_groups: each worker's group at the start, indexed by worker.
            start_offsets: each worker's start on the shared axis (seconds since the earliest worker),
                indexed by worker; the worker's band is drawn from here.
        """
        if len(initial_groups) != len(start_offsets):
            raise ValueError("initial_groups and start_offsets must have one entry per worker")
        self._n_workers = len(initial_groups)
        self._start_offsets = list(start_offsets)
        self._presets: dict[int, str] = {}

        # live replay state
        self._group_of: dict[int, int] = dict(enumerate(initial_groups))
        self._row_of: dict[int, int] = {}
        self._occupied_rows: dict[int, set[int]] = {}
        self._open_from: dict[int, float] = {}
        self._group_start: dict[int, float] = {}
        self._group_end: dict[int, float] = {}
        self._peak_height: dict[int, int] = {}
        self._last_t = 0.0
        self._best_open: tuple[float, int, int] | None = None

        # accumulated geometry
        self._segments: list[BandSegment] = []
        self._best_intervals: list[BestInterval] = []
        self._score_points: list[ScorePoint] = []
        self._t_end: float | None = None

        for worker in range(self._n_workers):
            group = initial_groups[worker]
            self._place_in_group(worker, group, start_offsets[worker])
            self._group_start[group] = min(self._group_start.get(group, start_offsets[worker]), start_offsets[worker])

    # --------------------------------------------------------------------------
    #  Modifiers
    # --------------------------------------------------------------------------
    def set_worker_preset(self, worker: int, preset: str) -> None:
        """Label `worker`'s band with the preset it ran; shown once, at the band's start."""
        self._presets[worker] = preset

    def dissolve_group(self, t: float, group: int, reassignments: dict[int, int]) -> None:
        """Dissolve `group` at time `t`, moving each of its workers to its target group in `reassignments`.

        Args:
            reassignments: target group per worker that was in the dissolved group.
        """
        self._check_time(t)
        for worker in [w for w, g in self._group_of.items() if g == group]:
            self._close_segment(worker, t)
            self._occupied_rows[group].discard(self._row_of[worker])
            self._place_in_group(worker, reassignments[worker], t)
        self._group_end[group] = t

    def record_best(self, t: float, worker: int) -> None:
        """Record that from `t` on, `worker` held the best score; its group is read from the current grouping."""
        self._check_time(t)
        self._close_best(t)
        self._best_open = (t, worker, self._group_of[worker])

    def record_score(self, t: float, diversity: float, constraints: float) -> None:
        """Append the best-known selection's scores at time `t` to the trajectory."""
        self._check_time(t)
        self._score_points.append(ScorePoint(t=t, diversity=diversity, constraints=constraints))

    def finish(self, t_end: float) -> None:
        """Close every open band, group block and best interval at `t_end`, the end of the solve."""
        self._check_time(t_end)
        for worker in range(self._n_workers):
            self._close_segment(worker, t_end)
        for group in self._group_of.values():
            self._group_end.setdefault(group, t_end)
        self._close_best(t_end)
        self._t_end = t_end

    # --------------------------------------------------------------------------
    #  Geometry read by the drawing code and the tests
    # --------------------------------------------------------------------------
    @property
    def n_workers(self) -> int:
        """Return how many workers the solve ran."""
        return self._n_workers

    @property
    def worker_presets(self) -> dict[int, str]:
        """Return the preset label per worker, for those set."""
        return dict(self._presets)

    @property
    def start_offsets(self) -> list[float]:
        """Return each worker's start on the shared axis, indexed by worker."""
        return list(self._start_offsets)

    @property
    def segments(self) -> list[BandSegment]:
        """Return every band segment, one per stretch a worker spent in one band."""
        return list(self._segments)

    @property
    def group_blocks(self) -> list[GroupBlock]:
        """Return the group blocks in group-id order: the units the top panel stacks top-down."""
        return [
            GroupBlock(group=group, t_start=self._group_start[group], t_end=self._group_end[group], height=height)
            for group, height in sorted(self._peak_height.items())
        ]

    @property
    def best_intervals(self) -> list[BestInterval]:
        """Return the intervals over which each worker held the best score, in time order."""
        return list(self._best_intervals)

    @property
    def score_points(self) -> list[ScorePoint]:
        """Return the score trajectory, in time order."""
        return list(self._score_points)

    @property
    def t_end(self) -> float:
        """Return the end of the solve on the shared axis.

        Raises:
            ValueError: If `finish` has not been called yet.
        """
        if self._t_end is None:
            raise ValueError("the plot is not finished; call finish(t_end) before reading t_end")
        return self._t_end

    # --------------------------------------------------------------------------
    #  Rendering
    # --------------------------------------------------------------------------
    def render(self) -> Figure:
        """Draw the timeline and return the Matplotlib figure."""
        from .draw import draw_timeline

        return draw_timeline(self)

    # --------------------------------------------------------------------------
    #  From a solved solution
    # --------------------------------------------------------------------------
    @classmethod
    def from_solution(cls, solution: ParallelMaxDivSolution) -> ParallelSolutionTimelinePlot:
        """Build the timeline geometry from a solved parallel solution.

        Merges the solution's worker group changes and score checkpoints into one time-ordered replay:
        a change dissolves a group, a checkpoint records the best holder and its scores. Ties on time
        put a change before a checkpoint, so a checkpoint reads the grouping the change just produced,
        and equal-time changes keep their happened order (the falling alive-group count).
        """
        start_offsets = [worker.t_start_offset_sec for worker in solution.workers]
        plot = cls(initial_groups=list(solution.initial_worker_groups), start_offsets=start_offsets)
        for worker in solution.workers:
            plot.set_worker_preset(worker.worker_index, str(worker.config.preset.resolve_alias()).upper())

        changes = [(c.elapsed.t_elapsed_sec, 0, -c.n_alive_groups_after, c, None) for c in solution.worker_group_changes]
        checkpoints = [(cp.elapsed.t_elapsed_sec, 1, cp.worker_index, None, cp) for cp in solution.score_checkpoints]
        for t, _kind, _tiebreak, change, checkpoint in sorted(changes + checkpoints, key=lambda e: (e[0], e[1], e[2])):
            if change is not None:
                plot.dissolve_group(t, change.dissolved_group, change.reassignments)
            elif checkpoint is not None and checkpoint.worker_index is not None:
                plot.record_best(t, checkpoint.worker_index)
                plot.record_score(t, checkpoint.score.diversity, checkpoint.score.constraints)

        plot.finish(solution.score_checkpoints[-1].elapsed.t_elapsed_sec)
        return plot

    # --------------------------------------------------------------------------
    #  Internal replay
    # --------------------------------------------------------------------------
    def _place_in_group(self, worker: int, group: int, t_from: float) -> None:
        """Put `worker` in `group`'s first empty band from `t_from`, growing the block's height if needed."""
        occupied = self._occupied_rows.setdefault(group, set())
        row = next(r for r in range(len(occupied) + 1) if r not in occupied)
        occupied.add(row)
        self._group_of[worker] = group
        self._row_of[worker] = row
        self._open_from[worker] = t_from
        self._peak_height[group] = max(self._peak_height.get(group, 0), max(occupied) + 1)

    def _close_segment(self, worker: int, t: float) -> None:
        """Emit `worker`'s open band segment, ending it at `t`."""
        self._segments.append(
            BandSegment(
                worker=worker,
                group=self._group_of[worker],
                row=self._row_of[worker],
                t_from=self._open_from[worker],
                t_to=t,
            )
        )

    def _close_best(self, t: float) -> None:
        """Close the open best-score interval at `t`, dropping it when it spans no time."""
        if self._best_open is not None:
            t_from, worker, group = self._best_open
            if t > t_from:
                self._best_intervals.append(BestInterval(t_from=t_from, t_to=t, worker=worker, group=group))
            self._best_open = None

    def _check_time(self, t: float) -> None:
        """Advance the replay clock, rejecting an event that goes back in time."""
        if t < self._last_t:
            raise ValueError(f"timeline events must not go back in time: {t} follows {self._last_t}")
        self._last_t = t


if TYPE_CHECKING:
    from matplotlib.figure import Figure  # noqa: TID251 — annotation only; the render body imports it
