from __future__ import annotations

import math
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

from max_div._core._utils import format_long_time_duration, np_int32_array_var_length_hash
from max_div._core._utils._progress_table import ProgressTable

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy._typing import NDArray

    from max_div._core.solver._duration import Progress
    from max_div._core.solver._score import Score
    from max_div._core.solver._solver_state import SolverState


# =================================================================================================
#  Verbosity
# =================================================================================================
class Verbosity(IntEnum):
    """Verbosity names the levels for `solve`; members are plain ints, so plain integers are accepted too.

    The `TABULAR` through `TABULAR_FASTEST` levels differ only in how quickly the update cadence
    slows down over a run: `TABULAR` spaces rows out the fastest (fewest rows), `TABULAR_FASTEST`
    keeps them coming. `TABULAR_DEBUG` is `TABULAR_FASTEST` plus a column of solver-internal
    statistics.
    """

    SILENT = 0
    PROGRESS_BAR = 10
    TABULAR = 20
    TABULAR_FAST = 21
    TABULAR_FASTER = 22
    TABULAR_FASTEST = 23
    TABULAR_DEBUG = 25


# How quickly each tabular level's update cadence slows down; `from_verbosity` reads it, and the
# worker-side forwarding cadence is derived from it, so these numbers have exactly one home.
TABULAR_C_SLOWDOWNS: dict[Verbosity, float] = {
    Verbosity.TABULAR: 1.10,
    Verbosity.TABULAR_FAST: 1.05,
    Verbosity.TABULAR_FASTER: 1.02,
    Verbosity.TABULAR_FASTEST: 1.01,
}


# =================================================================================================
#  ProgressSnapshot
# =================================================================================================
@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """A snapshot records what a reporter can show about one moment of a solve, detached from its live state.

    `ProgressReporter` builds one per reporting call and hands it to the rendering methods, so
    renderers never read `SolverState` — a snapshot is self-contained and stays meaningful outside
    the solve that produced it. The selection is held by reference (never copied), keeping
    construction cheap on the many updates that are throttled away without being shown.
    """

    step_name: str  # name of the solver step this snapshot was taken in
    progress: Progress | None  # step progress; None when the step reports none (solver-state init)
    t_elapsed_solver: float  # seconds since the first step started
    t_elapsed_step: float  # seconds since the current step started
    score: Score
    n_selected: int | np.integer
    k: int | np.integer
    m: int | np.integer  # number of constraints
    selection: NDArray[np.int32] | None  # currently selected indices, by reference into the live state
    ignore_infeasible_diversity: bool  # render diversity as not-yet-meaningful while infeasible

    # --- materialized / multi-worker fields ---
    # A snapshot that crosses a process boundary cannot carry the selection by reference or resolve a
    # debug callable, so the sender materializes these; `selection_hash` replaces `selection` and
    # `debug_info` replaces the `get_debug_info` callable. The worker fields exist only in the
    # combined view a parallel solve renders, where one row draws on several workers.
    selection_hash: str | None = None  # precomputed hash, standing in for `selection`
    debug_info: str | None = None  # pre-resolved debug string, standing in for the callable
    worker_index: int | None = None  # worker this snapshot's result fields came from
    n_active: int | None = None  # number of workers still solving
    worker_finished: bool = False  # whether the result-fields worker has finished solving


# =================================================================================================
#  ReportThrottle
# =================================================================================================
class ReportThrottle:
    """A throttle decides which progress updates get shown, thinning them out as a run progresses.

    An update passes only when both an iteration threshold and an elapsed-time threshold are met;
    each pass raises both, the time spacing growing toward a fixed ceiling and the iteration spacing
    by a factor `c_slowdown` — so the closer `c_slowdown` is to 1.0, the more updates keep coming.
    """

    def __init__(self, c_slowdown: float) -> None:
        """Create a throttle whose pass spacing grows by a factor `c_slowdown` per shown update."""
        self._c_slowdown = c_slowdown
        self._next_iter: int = 0
        self._next_t: float = 0.0
        self._n_passed: int = 0

    def reset(self) -> None:
        """Reset the thresholds, so the next update passes; called at each step start."""
        self._next_iter = 0
        self._next_t = 0.0
        self._n_passed = 0

    def passes(self, iter_now: int, t_elapsed: float) -> bool:
        """Return whether this update should be shown, advancing both thresholds when it is."""
        if (iter_now < self._next_iter) or (t_elapsed < self._next_t):
            return False
        self._n_passed += 1
        self._next_iter = max(iter_now + 1, int(iter_now * self._c_slowdown))
        t_increment = min(1.0, 0.1 * (self._c_slowdown**self._n_passed))
        self._next_t += t_increment * math.ceil((t_elapsed - self._next_t) / t_increment)
        return True


# =================================================================================================
#  SnapshotRequirements
# =================================================================================================
@dataclass(frozen=True, slots=True)
class SnapshotRequirements:
    """The requirements record what a reporter needs materialized in snapshots built in another process.

    In-process, a reporter lazily takes what it renders (the selection by reference, the debug
    callable), so nothing needs declaring. When snapshots are produced in another process, the sender
    must materialize up front exactly what the receiving reporter will render — this record, declared
    by the reporter class about itself, tells the sender what that is.
    """

    debug_info: bool  # resolve the debug callable into `ProgressSnapshot.debug_info`
    selection_hash: bool  # hash the selection into `ProgressSnapshot.selection_hash`


# =================================================================================================
#  Base class
# =================================================================================================
class ProgressReporter(ABC):
    """A progress reporter shows the progress of a running solve; subclasses render `ProgressSnapshot`s.

    The solver and its steps call the `solver_step_*`/`update` methods with live state; this base
    class owns the clocks, builds the snapshot, and delegates to the `show_*` methods. Renderers
    therefore work from snapshots alone and can be driven without a solver attached.
    """

    def __init__(self) -> None:
        self._t_start_solver = -1.0
        self._t_start_step = 0.0
        self._step_name = ""

    @property
    def snapshot_requirements(self) -> SnapshotRequirements | None:
        """Return what to materialize in snapshots built in another process for this reporter.

        `None` means the reporter renders nothing at all, so no snapshots need to reach it. The
        default declares a renderer that shows progress but neither the selection hash nor debug
        info; subclasses override where they render more (tabular) or nothing (silent).
        """
        return SnapshotRequirements(debug_info=False, selection_hash=False)

    # -------------------------------------------------------------------------
    #  Main API (called by the solver and its steps)
    # -------------------------------------------------------------------------
    def solver_step_started(self, step_name: str) -> None:
        """Record that a new solver step with the provided name has started, and notify the renderer."""
        self._step_name = step_name
        self._t_start_step = time.perf_counter()
        if self._t_start_solver < 0:
            self._t_start_solver = self._t_start_step
        self.show_step_started(step_name)

    def update(
        self,
        progress: Progress,
        state: SolverState,
        get_debug_info: Callable[[], str] | None = None,
        *,
        ignore_infeasible_diversity: bool = False,
    ) -> None:
        """Report current progress and state to the renderer."""
        self.show_update(self._build_snapshot(progress, state, ignore_infeasible_diversity), get_debug_info)

    def solver_step_finished(
        self,
        progress: Progress | None,
        state: SolverState,
        get_debug_info: Callable[[], str] | None = None,
        *,
        ignore_infeasible_diversity: bool = False,
    ) -> None:
        """Report that the current solver step has finished."""
        self.show_step_finished(self._build_snapshot(progress, state, ignore_infeasible_diversity), get_debug_info)

    # -------------------------------------------------------------------------
    #  Rendering interface (implemented by subclasses, consuming snapshots only)
    # -------------------------------------------------------------------------
    @abstractmethod
    def show_step_started(self, step_name: str) -> None:
        """Render the start of a new solver step."""

    @abstractmethod
    def show_update(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        """Render a progress update, or skip it (e.g. when updates come too frequently)."""

    @abstractmethod
    def show_step_finished(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        """Render the end of the current solver step."""

    def show_milestone(  # noqa: B027 — deliberately concrete: rendering nothing is the right default
        self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None
    ) -> None:
        """Render a snapshot that must not be throttled away, set off from the regular stream.

        A parallel solve emits one per finishing worker, so that worker's final state is visible in
        scrollback no matter what later rows show. The default renders nothing; only renderers with a
        way to set a row apart (the table) override this.
        """

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _build_snapshot(
        self, progress: Progress | None, state: SolverState, ignore_infeasible_diversity: bool
    ) -> ProgressSnapshot:
        """Build a snapshot of the current progress and state, stamping the elapsed times."""
        t_now = time.perf_counter()
        return ProgressSnapshot(
            step_name=self._step_name,
            progress=progress,
            t_elapsed_solver=t_now - self._t_start_solver,
            t_elapsed_step=t_now - self._t_start_step,
            score=state.score,
            n_selected=state.n_selected,
            k=state.k,
            m=state.m,
            selection=state.selected_index_array,
            ignore_infeasible_diversity=ignore_infeasible_diversity,
        )

    # -------------------------------------------------------------------------
    #  Factory methods
    # -------------------------------------------------------------------------
    @classmethod
    def silent(cls) -> SilentProgressReporter:
        """Create a silent progress reporter that doesn't output anything."""
        return SilentProgressReporter()

    @classmethod
    def tqdm(cls) -> TqdmProgressReporter:
        """Create a tqdm-based progress bar reporter."""
        return TqdmProgressReporter()

    @classmethod
    def tabular(cls, c_slowdown: float = 1.05, debug_info: bool = False) -> TabularProgressReporter:
        """Create a tabular progress reporter."""
        return TabularProgressReporter(c_slowdown=c_slowdown, debug_info=debug_info)

    @classmethod
    def from_verbosity(cls, verbosity: int | Verbosity, worker_columns: bool = False) -> ProgressReporter:
        """Create the reporter for a verbosity level; the integer levels are decoded only here.

        `verbosity` may be a `Verbosity` member or its plain integer value.

        Args:
            verbosity: the level to build the reporter for.
            worker_columns: If `True`, a tabular reporter is laid out for a multi-worker solve;
                the other reporters render identically either way.

        Raises:
            ValueError: If `verbosity` is not one of the `Verbosity` levels.
        """
        match verbosity:
            case 0:
                return cls.silent()
            case 10:
                return cls.tqdm()
            case 20 | 21 | 22 | 23:
                return TabularProgressReporter(
                    c_slowdown=TABULAR_C_SLOWDOWNS[Verbosity(verbosity)],
                    debug_info=False,
                    worker_columns=worker_columns,
                )
            case 25:
                return TabularProgressReporter(
                    c_slowdown=TABULAR_C_SLOWDOWNS[Verbosity.TABULAR_FASTEST],
                    debug_info=True,
                    worker_columns=worker_columns,
                )
            case _:
                raise ValueError(f"Invalid verbosity level: {verbosity}")


# =================================================================================================
#  Silent
# =================================================================================================
class SilentProgressReporter(ProgressReporter):
    """A progress reporter that is fully silent and doesn't output anything."""

    @property
    def snapshot_requirements(self) -> SnapshotRequirements | None:
        """Return None: nothing is rendered, so no snapshots need to reach this reporter."""
        return None

    def show_step_started(self, step_name: str) -> None: ...  # no-op
    def show_update(
        self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None
    ) -> None: ...  # no-op
    def show_step_finished(
        self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None
    ) -> None: ...  # no-op


# =================================================================================================
#  TQDM
# =================================================================================================
class TqdmProgressReporter(ProgressReporter):
    """A progress reporter that shows one tqdm progress bar per solver step."""

    def __init__(self) -> None:
        super().__init__()
        self._current_step_name: str = ""
        self._current_pbar: tqdm | None = None

    # -------------------------------------------------------------------------
    #  Rendering interface
    # -------------------------------------------------------------------------
    def show_step_started(self, step_name: str) -> None:
        if (step_name != self._current_step_name) or (not self._current_pbar):
            self._close_current_pbar()  # close previous pbar, if present
            self._current_pbar = tqdm(desc=f"{step_name} ", total=1, file=sys.stdout)  # initialize new pbar
            self._current_step_name = step_name

    def show_update(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        if (self._current_pbar is not None) and (snapshot.progress is not None):
            # ignore updates coming in before starting a new step or after finishing the current step
            n = snapshot.progress.tqdm_n_current
            if n > self._current_pbar.n:
                self._current_pbar.n = n
                self._current_pbar.total = snapshot.progress.tqdm_n_total
                self._current_pbar.refresh()

    def show_step_finished(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        self._close_current_pbar()

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _close_current_pbar(self) -> None:
        if self._current_pbar is not None:
            # make sure pbar shows 100%
            self._current_pbar.total = max(1, self._current_pbar.total or 0)  # tqdm annotates total as int | None
            self._current_pbar.n = self._current_pbar.total
            self._current_pbar.refresh()

            # cleanup
            self._current_pbar.close()
            self._current_pbar = None  # avoid updates after closing


# =================================================================================================
#  Tabular
# =================================================================================================
class TabularProgressReporter(ProgressReporter):
    """A progress reporter that prints one table row per (throttled) update."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, c_slowdown: float = 1.05, debug_info: bool = False, worker_columns: bool = False) -> None:
        """Initializes a TabularProgressReporter.

        Args:
            c_slowdown: Update-thinning factor; see `ReportThrottle`.
            debug_info: If `True`, includes additional column with solver step debug info.
            worker_columns: If `True`, lay the table out for a multi-worker solve: a worker column
                and an active-worker count replace the per-step columns, which have no
                meaning when the rendered rows draw on several workers at once.
        """
        super().__init__()

        # settings
        self._c_slowdown = c_slowdown
        self._debug_info = debug_info
        self._worker_columns = worker_columns

        self._progress_table: ProgressTable | None = None
        self._throttle = ReportThrottle(c_slowdown=c_slowdown)

    @property
    def snapshot_requirements(self) -> SnapshotRequirements | None:
        """Return the tabular needs: always the selection hash, plus debug info in debug mode."""
        return SnapshotRequirements(debug_info=self._debug_info, selection_hash=True)

    # -------------------------------------------------------------------------
    #  Rendering interface
    # -------------------------------------------------------------------------
    def show_step_started(self, step_name: str) -> None:
        # make sure table is initialized
        if not self._progress_table:
            self._initialize_table(step_name_width=len(step_name))

        # reset progress reporting thresholds
        self._throttle.reset()

    def show_update(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        iter_now = snapshot.progress.iter_count if (snapshot.progress is not None) else 0
        if self._throttle.passes(iter_now, snapshot.t_elapsed_step):
            debug_info = self._resolve_debug_info(snapshot, get_debug_info)
            self._show_table_row(snapshot, debug_info)

    def show_step_finished(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        # show final metrics + horizontal table line
        debug_info = self._resolve_debug_info(snapshot, get_debug_info)
        self._show_table_row(snapshot, debug_info)
        self._show_table_line()

    def show_milestone(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        """Render the snapshot as an unthrottled row, set apart by a horizontal line."""
        self.show_step_finished(snapshot, get_debug_info)

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _resolve_debug_info(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None) -> str:
        """Return the debug column text: pre-materialized when present, else pulled from the callable."""
        if not self._debug_info:
            return ""
        if snapshot.debug_info is not None:
            return snapshot.debug_info
        return get_debug_info() if (get_debug_info is not None) else ""

    def _initialize_table(self, step_name_width: int) -> None:
        """Initialize self._progress_table."""
        if self._worker_columns:
            leading_headers = [
                "Solver t.".ljust(10),
                "Worker".ljust(6),
                "Active".ljust(6),
                "Progress".ljust(10),
                "Iter.".ljust(10),
            ]
        else:
            leading_headers = [
                "Solver t.".ljust(10),
                "Solver step".ljust(step_name_width),
                "Step %".ljust(10),
                "Step it.".ljust(10),
                "Step t.".ljust(10),
            ]
        self._progress_table = ProgressTable(
            headers=leading_headers
            + [
                "Selected".ljust(13),
                "Constraints".ljust(11),
                "Diversity".ljust(14),
                "Selection hash".ljust(32),
            ]
            + (["Debug info".ljust(90)] if self._debug_info else []),
        )
        self._progress_table.show_header()

    def _show_table_row(self, snapshot: ProgressSnapshot, debug_info: str = "") -> None:
        progress = snapshot.progress
        score = snapshot.score

        if snapshot.ignore_infeasible_diversity and (score.constraints < 1.0):
            diversity_str = f"({score.diversity:.4e})"  # between brackets if we're ignoring it
        else:
            diversity_str = f"{score.diversity:.6e}"

        if self._worker_columns:
            worker_str = "" if (snapshot.worker_index is None) else str(snapshot.worker_index)
            leading_values = [
                format_long_time_duration(snapshot.t_elapsed_solver, n_chars=8),
                f"{worker_str}✓" if snapshot.worker_finished else worker_str,
                str(snapshot.n_active) if (snapshot.n_active is not None) else "",
                f"{progress.fraction * 100:.2f}%" if progress else "",
                f"{progress.iter_count:_}".rjust(10) if progress else "",
            ]
        else:
            leading_values = [
                format_long_time_duration(snapshot.t_elapsed_solver, n_chars=8),
                snapshot.step_name,
                f"{progress.fraction * 100:.2f}%" if progress else "",
                f"{progress.iter_count:_}".rjust(10) if progress else "",
                format_long_time_duration(snapshot.t_elapsed_step, n_chars=8),
            ]

        self._progress_table.show_progress(  # ty: ignore[unresolved-attribute]  # table is initialized in show_step_started before any row is shown
            values=leading_values
            + [
                f"{snapshot.n_selected:>6}/{snapshot.k:>6}",
                f"{score.constraints:.6f}" if (snapshot.m > 0) else "/",
                diversity_str,
                self._selection_hash_str(snapshot).ljust(32),
            ]
            + ([debug_info] if self._debug_info else [])
        )

    def _show_table_line(self) -> None:
        if self._progress_table:
            self._progress_table.print_line()

    @staticmethod
    def _selection_hash_str(snapshot: ProgressSnapshot) -> str:
        """Return the hash column text: pre-materialized when present, else hashed from the selection."""
        if snapshot.selection_hash is not None:
            return snapshot.selection_hash
        return selection_hash_str(snapshot)

    @staticmethod
    def _get_selection_hash(selection: NDArray[np.int32], n: int) -> str:
        """Get a hex hash string representing the current selection in the solver state."""
        return _selection_hash_hex(selection, n)


# =================================================================================================
#  Helpers
# =================================================================================================
def selection_hash_str(snapshot: ProgressSnapshot) -> str:
    """Return the hash string of a snapshot's selection, its length proportional to selection size."""
    if snapshot.selection is None:
        return ""
    return _selection_hash_hex(
        selection=snapshot.selection,
        n=math.ceil((32 * snapshot.n_selected) / snapshot.k),
    )


def _selection_hash_hex(selection: NDArray[np.int32], n: int) -> str:
    """Return `n` hex characters hashed from the selected indices; empty when nothing is selected."""
    # --- shortcut ---
    if n == 0:
        return ""

    # --- generate hash ---
    hash_array = np_int32_array_var_length_hash(selection, n)
    return "".join(f"{val & 0xF:x}" for val in hash_array)
