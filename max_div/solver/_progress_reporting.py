from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Callable

import numpy as np
from numpy._typing import NDArray
from tqdm.auto import tqdm

from max_div.internal.formatting import format_long_time_duration
from max_div.internal.utils._hash import np_int32_array_var_length_hash
from max_div.internal.utils._progress_table import ProgressTable
from max_div.solver._duration import Progress
from max_div.solver._score import Score
from max_div.solver._solver_state import SolverState


# =================================================================================================
#  Base class
# =================================================================================================
class ProgressReporter(ABC):
    @abstractmethod
    def solver_step_started(self, step_name: str):
        """Notify that a new solver step with the provided name has started."""
        pass

    @abstractmethod
    def update(self, progress: Progress, state: SolverState, get_debug_info: Callable[[], str] | None = None, **kwargs):
        """
        Update progress reporter with current progress and state.
        Reporters can choose to not report certain updates they receive, if they come too frequently.
        """
        pass

    @abstractmethod
    def solver_step_finished(
        self, progress: Progress | None, state: SolverState, get_debug_info: Callable[[], str] | None = None, **kwargs
    ):
        """Notify that the current solver step has finished."""
        pass

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


# =================================================================================================
#  Silent
# =================================================================================================
class SilentProgressReporter(ProgressReporter):
    """A progress reporter that is fully silent and doesn't output anything."""

    def solver_step_started(self, step_name: str): ...  # no-op
    def update(
        self, progress: Progress, score: Score, get_debug_info: Callable[[], str] | None = None, **kwargs
    ): ...  # no-op
    def solver_step_finished(
        self, progress: Progress | None, score: Score, get_debug_info: Callable[[], str] | None = None, **kwargs
    ): ...  # no-op


# =================================================================================================
#  TQDM
# =================================================================================================
class TqdmProgressReporter(ProgressReporter):
    def __init__(self):
        super().__init__()
        self._current_step_name: str = ""
        self._current_pbar: tqdm | None = None

    # -------------------------------------------------------------------------
    #  main API
    # -------------------------------------------------------------------------
    def solver_step_started(self, step_name: str):
        if (step_name != self._current_step_name) or (not self._current_pbar):
            self._close_current_pbar()  # close previous pbar, if present
            self._current_pbar = tqdm(desc=f"{step_name} ", total=1, file=sys.stdout)  # initialize new pbar
            self._current_step_name = step_name

    def update(self, progress: Progress, state: SolverState, get_debug_info: Callable[[], str] | None = None, **kwargs):
        if self._current_pbar is not None:
            # ignore updates coming in before starting a new step or after finishing the current step
            n = progress.tqdm_n_current
            if n > self._current_pbar.n:
                self._current_pbar.n = n
                self._current_pbar.total = progress.tqdm_n_total
                self._current_pbar.refresh()

    def solver_step_finished(
        self, progress: Progress | None, state: SolverState, get_debug_info: Callable[[], str] | None = None, **kwargs
    ):
        self._close_current_pbar()

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _close_current_pbar(self):
        if self._current_pbar is not None:
            # make sure pbar shows 100%
            self._current_pbar.total = max(1, self._current_pbar.total)
            self._current_pbar.n = self._current_pbar.total
            self._current_pbar.refresh()

            # cleanup
            self._current_pbar.close()
            self._current_pbar = None  # avoid updates after closing


# =================================================================================================
#  Tabular
# =================================================================================================
class TabularProgressReporter(ProgressReporter):
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, c_slowdown: float = 1.05, debug_info: bool = False):
        """
        :param c_slowdown: Factor by which to slow down reporting frequency:

            Updates are shown only when both
               a) time elapsed since last report exceeds a threshold
                    0.1sec initially, increasing to 1.0sec eventually
               b) number of iterations since start has exceeded a threshold
                    increasing with factor c_slowdown each report

            c_slowdown influences how quickly both increase.  The closer to 1.0, the more frequents updates keep coming.

        :param debug_info: If `True`, includes additional column with solver step debug info.
        """

        # settings
        self._c_slowdown = c_slowdown
        self._debug_info = debug_info

        self._progress_table: ProgressTable | None = None
        self._step_name = ""

        # don't show next table line before passing both thresholds below:
        self._next_report_t_elapsed: float = 0.0
        self._next_report_iter: int = 0

        # start times
        self._t_start_solver = -1.0
        self._t_start_step = 0.0

        # other stats
        self._n_progress_reports_this_step = 0

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def solver_step_started(self, step_name: str):
        # make sure table is initialized
        self._step_name = step_name
        if not self._progress_table:
            self._initialize_table(step_name_width=len(step_name))

        # reset progress reporting thresholds
        self._n_progress_reports_this_step = 0
        self._next_report_t: float = 0.0
        self._next_report_iter: int = 0

        # record step start time
        self._t_start_step = perf_counter()
        if self._t_start_solver < 0:
            self._t_start_solver = self._t_start_step

    def update(self, progress: Progress, state: SolverState, get_debug_info: Callable[[], str] | None = None, **kwargs):
        iter_now = progress.iter_count
        t_now = perf_counter()
        t_elapsed_step = t_now - self._t_start_step

        if (iter_now >= self._next_report_iter) and (t_elapsed_step >= self._next_report_t):
            # show table row
            debug_info = get_debug_info() if (self._debug_info and (get_debug_info is not None)) else ""
            ignore_infeasible_diversity = kwargs.get("ignore_infeasible_diversity", False)
            self._show_table_row(progress, state, debug_info, ignore_infeasible_diversity)
            self._n_progress_reports_this_step += 1

            # update next report thresholds
            self._next_report_iter = max(iter_now + 1, int(iter_now * self._c_slowdown))
            t_increment = min(1.0, 0.1 * (self._c_slowdown**self._n_progress_reports_this_step))
            self._next_report_t += t_increment * math.ceil((t_elapsed_step - self._next_report_t) / t_increment)

    def solver_step_finished(
        self, progress: Progress | None, state: SolverState, get_debug_info: Callable[[], str] | None = None, **kwargs
    ):
        # show final metrics + horizontal table line
        debug_info = get_debug_info() if (self._debug_info and (get_debug_info is not None)) else ""
        ignore_infeasible_diversity = kwargs.get("ignore_infeasible_diversity", False)
        self._show_table_row(progress, state, debug_info, ignore_infeasible_diversity)
        self._show_table_line()

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _initialize_table(self, step_name_width: int):
        """Initialize self._progress_table"""
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

    def _show_table_row(
        self,
        progress: Progress | None,
        state: SolverState,
        debug_info: str = "",
        ignore_infeasible_diversity: bool = False,
    ):
        t_now = perf_counter()
        t_elapsed_solver = t_now - self._t_start_solver
        t_elapsed_step = t_now - self._t_start_step
        score = state.score

        if ignore_infeasible_diversity and (score.constraints < 1.0):
            diversity_str = f"({score.diversity:.4e})"  # between brackets if we're ignoring it
        else:
            diversity_str = f"{score.diversity:.6e}"

        self._progress_table.show_progress(
            values=[
                format_long_time_duration(t_elapsed_solver, n_chars=8),
                self._step_name,
                f"{progress.fraction * 100:.2f}%" if progress else "",
                f"{progress.iter_count:_}".rjust(10) if progress else "",
                format_long_time_duration(t_elapsed_step, n_chars=8),
                f"{state.n_selected:>6}/{state.k:>6}",
                f"{score.constraints:.6f}" if (state.m > 0) else "/",
                diversity_str,
                self._get_selection_hash(
                    selection=state.selected_index_array,  # create hash from currently selected indices...
                    n=math.ceil((32 * state.n_selected) / state.k),  # ...of length proportional to selection size
                ).ljust(32),
            ]
            + ([debug_info] if self._debug_info else [])
        )

    def _show_table_line(self):
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
        hash_str = "".join(f"{val & 0xF:x}" for val in hash_array)
        return hash_str
