"""The geometry of a parallel solve's timeline is built by replaying the solve's events onto one axis.

`ParallelSolutionTimelinePlot` holds everything a timeline figure needs and nothing visual:

- the group blocks with their height and span;
- per worker, the bands it occupied and when;
- which worker held the best score over each interval;
- the score trajectory.

`draw` turns this geometry into a figure.

The plot is populated by event replay, not through its constructor: the constructor fixes the initial
grouping and where each worker's band starts, then `dissolve_group`, `record_best` and `record_score`
advance the state in time order.

`from_solution` drives that replay from a solved `ParallelMaxDivSolution`; the modifiers take plain
values, so this module imports nothing from the solver package and a test can build a plot from
hand-written events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from max_div._core.solver import ParallelMaxDivSolution

# a band a worker held for less than this fraction of the whole solve is a merge artifact (see `segments`)
_MIN_SEGMENT_FRACTION = 1 / 1000


# ==================================================================================================
#  Geometry
# ==================================================================================================
@dataclass(frozen=True)
class BandSegment:
    """A band segment is a stretch a worker spent in one band of one group, from `t_from` to `t_to`."""

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
    """A score point is the best-known selection's diversity, tie-breaker and constraints scores at time `t`."""

    t: float
    diversity: float
    constraints: float
    tie_breakers: tuple[float, ...] = ()


# ==================================================================================================
#  ParallelSolutionTimelinePlot
# ==================================================================================================
@dataclass
class ParallelSolutionTimelinePlot:
    """The drawable geometry of a parallel solve's timeline is populated by replaying the solve's events.

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
        self._tie_breaker_labels: list[str] = []
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
        """Dissolve `group` at time `t`, sending each of its workers to its target group in `reassignments`."""
        self._advance_time(t)
        for worker in [w for w, g in self._group_of.items() if g == group]:
            self._close_segment(worker, t)
            self._occupied_rows[group].discard(self._row_of[worker])
            self._place_in_group(worker, reassignments[worker], t)
        self._group_end[group] = t

    def record_best(self, t: float, worker: int) -> None:
        """Record that from `t` on, `worker` held the best score; its group is read from the current grouping."""
        self._advance_time(t)
        self._close_best(t)
        self._best_open = (t, worker, self._group_of[worker])

    def record_score(
        self, t: float, diversity: float, constraints: float, tie_breakers: tuple[float, ...] = ()
    ) -> None:
        """Append the best-known selection's scores at time `t` to the trajectory."""
        self._advance_time(t)
        self._score_points.append(
            ScorePoint(t=t, diversity=diversity, constraints=constraints, tie_breakers=tie_breakers)
        )

    def set_tie_breaker_labels(self, labels: list[str]) -> None:
        """Name the tie-breakers, in the order they break ties; the panels read these."""
        self._tie_breaker_labels = list(labels)

    def finish(self, t_end: float) -> None:
        """Close every open band, group block and best interval at `t_end`, the end of the solve."""
        self._advance_time(t_end)
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
        """Return every band segment a worker held for a meaningful span.

        A merge that cuts the group count by more than one is recorded as a chain of dissolutions that
        all share one timestamp, which routes a worker through an intermediate group for no time.
        Segments shorter than `_MIN_SEGMENT_FRACTION` of the whole solve are those artifacts, dropped
        here so they neither inflate a group's height nor draw a stray worker label.
        """
        min_sec = self.t_end * _MIN_SEGMENT_FRACTION
        return [segment for segment in self._segments if segment.t_to - segment.t_from >= min_sec]

    @property
    def group_blocks(self) -> list[GroupBlock]:
        """Return the group blocks in group-id order, each as tall as its lasting occupancy.

        The height is the top band any lasting member reaches, read from `segments` (which excludes the
        transient bands), so a merge artifact never makes a block taller than it truly was.
        """
        heights: dict[int, int] = {}
        for segment in self.segments:
            heights[segment.group] = max(heights.get(segment.group, 0), segment.row + 1)
        return [
            GroupBlock(
                group=group,
                t_start=self._group_start[group],
                t_end=self._group_end[group],
                height=heights.get(group, self._peak_height[group]),
            )
            for group in sorted(self._peak_height)
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
    @property
    def tie_breaker_labels(self) -> list[str]:
        """Return the tie-breaker labels, one per tie-breaker score of the points (empty when unnamed)."""
        return list(self._tie_breaker_labels)

    def render(self, include_tie_breakers: bool = False) -> Figure:
        """Draw the timeline and return the Matplotlib figure; `include_tie_breakers` adds one panel per tie-breaker."""
        from .draw import draw_timeline

        return draw_timeline(self, include_tie_breakers=include_tie_breakers)

    # --------------------------------------------------------------------------
    #  From a solved solution
    # --------------------------------------------------------------------------
    @classmethod
    def from_solution(cls, solution: ParallelMaxDivSolution) -> ParallelSolutionTimelinePlot:
        """Build the timeline geometry from a solved parallel solution.

        Merge the solution's worker group changes and score checkpoints into one time-ordered replay:
        a change dissolves a group, a checkpoint records the best holder and its scores. Ties on time
        put a change before a checkpoint, so a checkpoint reads the grouping the change just produced,
        and equal-time changes keep the order they happened in, because sorting on the descending count
        of groups still alive after each change reproduces that order.
        """
        start_offsets = [worker.t_start_offset_sec for worker in solution.workers]
        plot = cls(initial_groups=list(solution.initial_worker_groups), start_offsets=start_offsets)
        for worker in solution.workers:
            plot.set_worker_preset(worker.worker_index, str(worker.config.preset.resolve_alias()).upper())

        changes = [
            (c.elapsed.t_elapsed_sec, 0, -c.n_alive_groups_after, c, None) for c in solution.worker_group_changes
        ]
        checkpoints = [(cp.elapsed.t_elapsed_sec, 1, cp.worker_index, None, cp) for cp in solution.score_checkpoints]
        for t, _kind, _tiebreak, change, checkpoint in sorted(changes + checkpoints, key=lambda e: (e[0], e[1], e[2])):
            if change is not None:
                plot.dissolve_group(t, change.dissolved_group, change.reassignments)
            elif checkpoint is not None and checkpoint.worker_index is not None:
                plot.record_best(t, checkpoint.worker_index)
                plot.record_score(
                    t, checkpoint.score.diversity, checkpoint.score.constraints, checkpoint.score.div_tie_breakers
                )
        plot.set_tie_breaker_labels(solution.diversity_objective_labels[1:])

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

    def _advance_time(self, t: float) -> None:
        """Advance the replay clock, rejecting an event that goes back in time."""
        if t < self._last_t:
            raise ValueError(f"timeline events must not go back in time: {t} follows {self._last_t}")
        self._last_t = t


if TYPE_CHECKING:
    from matplotlib.figure import Figure
