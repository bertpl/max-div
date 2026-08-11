from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
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
    selection: NDArray[np.int32]  # currently selected indices, by reference into the live state
    ignore_infeasible_diversity: bool  # render diversity as not-yet-meaningful while infeasible


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

    # -------------------------------------------------------------------------
    #  Main API (called by the solver and its steps)
    # -------------------------------------------------------------------------
    def solver_step_started(self, step_name: str) -> None:
        """Record that a new solver step with the provided name has started, and notify the renderer."""
        self._step_name = step_name
        self._t_start_step = perf_counter()
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

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _build_snapshot(
        self, progress: Progress | None, state: SolverState, ignore_infeasible_diversity: bool
    ) -> ProgressSnapshot:
        """Build a snapshot of the current progress and state, stamping the elapsed times."""
        t_now = perf_counter()
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
    def from_verbosity(cls, verbosity: int) -> ProgressReporter:
        """Create the reporter a verbosity level names; nothing else decodes the integer levels.

        Levels: 0 = silent, 10 = tqdm progress bar, 20-23 = progress table from slowest to fastest
        update cadence, 25 = fastest cadence plus a debug-info column.

        :raises ValueError: If `verbosity` is not one of the levels above.
        """
        match verbosity:
            case 0:
                return cls.silent()
            case 10:
                return cls.tqdm()
            case 20 | 21 | 22 | 23:
                return cls.tabular(
                    c_slowdown=[1.10, 1.05, 1.02, 1.01][verbosity - 20],
                    debug_info=False,
                )
            case 25:
                # fastest cadence, plus the debug-info column
                return cls.tabular(c_slowdown=1.01, debug_info=True)
            case _:
                raise ValueError(f"Invalid verbosity level: {verbosity}")


# =================================================================================================
#  Silent
# =================================================================================================
class SilentProgressReporter(ProgressReporter):
    """A progress reporter that is fully silent and doesn't output anything."""

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
    def __init__(self, c_slowdown: float = 1.05, debug_info: bool = False) -> None:
        """Initializes a TabularProgressReporter.

        :param c_slowdown: Factor by which to slow down reporting frequency:

            Updates are shown only when both
               a) time elapsed since last report exceeds a threshold
                    0.1sec initially, increasing to 1.0sec eventually
               b) number of iterations since start has exceeded a threshold
                    increasing with factor c_slowdown each report

            c_slowdown influences how quickly both increase.  The closer to 1.0, the more frequents updates keep coming.

        :param debug_info: If `True`, includes additional column with solver step debug info.
        """
        super().__init__()

        # settings
        self._c_slowdown = c_slowdown
        self._debug_info = debug_info

        self._progress_table: ProgressTable | None = None

        # don't show next table line before passing both thresholds below:
        self._next_report_t: float = 0.0
        self._next_report_iter: int = 0

        # other stats
        self._n_progress_reports_this_step = 0

    # -------------------------------------------------------------------------
    #  Rendering interface
    # -------------------------------------------------------------------------
    def show_step_started(self, step_name: str) -> None:
        # make sure table is initialized
        if not self._progress_table:
            self._initialize_table(step_name_width=len(step_name))

        # reset progress reporting thresholds
        self._n_progress_reports_this_step = 0
        self._next_report_t = 0.0
        self._next_report_iter = 0

    def show_update(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        iter_now = snapshot.progress.iter_count if (snapshot.progress is not None) else 0
        t_elapsed_step = snapshot.t_elapsed_step

        if (iter_now >= self._next_report_iter) and (t_elapsed_step >= self._next_report_t):
            # show table row
            debug_info = get_debug_info() if (self._debug_info and (get_debug_info is not None)) else ""
            self._show_table_row(snapshot, debug_info)
            self._n_progress_reports_this_step += 1

            # update next report thresholds
            self._next_report_iter = max(iter_now + 1, int(iter_now * self._c_slowdown))
            t_increment = min(1.0, 0.1 * (self._c_slowdown**self._n_progress_reports_this_step))
            self._next_report_t += t_increment * math.ceil((t_elapsed_step - self._next_report_t) / t_increment)

    def show_step_finished(self, snapshot: ProgressSnapshot, get_debug_info: Callable[[], str] | None = None) -> None:
        # show final metrics + horizontal table line
        debug_info = get_debug_info() if (self._debug_info and (get_debug_info is not None)) else ""
        self._show_table_row(snapshot, debug_info)
        self._show_table_line()

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _initialize_table(self, step_name_width: int) -> None:
        """Initialize self._progress_table."""
        self._progress_table = ProgressTable(
            headers=[
                "Solver t.".ljust(10),
                "Solver step".ljust(step_name_width),
                "Step %".ljust(10),
                "Step it.".ljust(10),
                "Step t.".ljust(10),
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

        self._progress_table.show_progress(  # ty: ignore[unresolved-attribute]  # table is initialized in show_step_started before any row is shown
            values=[
                format_long_time_duration(snapshot.t_elapsed_solver, n_chars=8),
                snapshot.step_name,
                f"{progress.fraction * 100:.2f}%" if progress else "",
                f"{progress.iter_count:_}".rjust(10) if progress else "",
                format_long_time_duration(snapshot.t_elapsed_step, n_chars=8),
                f"{snapshot.n_selected:>6}/{snapshot.k:>6}",
                f"{score.constraints:.6f}" if (snapshot.m > 0) else "/",
                diversity_str,
                self._get_selection_hash(
                    selection=snapshot.selection,  # create hash from currently selected indices...
                    n=math.ceil((32 * snapshot.n_selected) / snapshot.k),  # ...of length proportional to selection size
                ).ljust(32),
            ]
            + ([debug_info] if self._debug_info else [])
        )

    def _show_table_line(self) -> None:
        if self._progress_table:
            self._progress_table.print_line()

    @staticmethod
    def _get_selection_hash(selection: NDArray[np.int32], n: int) -> str:
        """Get a hex hash string representing the current selection in the solver state."""
        # --- shortcut ---
        if n == 0:
            return ""

        # --- generate hash ---
        hash_array = np_int32_array_var_length_hash(selection, n)
        return "".join(f"{val & 0xF:x}" for val in hash_array)
