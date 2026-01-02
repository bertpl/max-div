import math
from abc import ABC, abstractmethod
from time import perf_counter

from tqdm.auto import tqdm

from max_div.solver._duration import Progress
from max_div.solver._score import Score


# =================================================================================================
#  Base class
# =================================================================================================
class ProgressReporter(ABC):
    @abstractmethod
    def solver_step_started(self, step_name: str):
        """Notify that a new solver step with the provided name has started."""
        pass

    @abstractmethod
    def update(self, progress: Progress, score: Score):
        """
        Update progress reporter with current progress and score.
        Reporters can choose to not report certain updates they receive, if they come too frequently.
        """
        pass

    @abstractmethod
    def solver_step_finished(self, score: Score):
        """Notify that the current solver step has finished."""
        pass


# =================================================================================================
#  Silent
# =================================================================================================
class SilentProgressReporter(ProgressReporter):
    """A progress reporter that is fully silent and doesn't output anything."""

    def solver_step_started(self, step_name: str): ...  # no-op
    def update(self, progress: Progress, score: Score): ...  # no-op
    def solver_step_finished(self, score: Score): ...  # no-op


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
            self._current_pbar = tqdm(desc=f"{step_name} ", total=1)  # initialize new pbar
            self._current_step_name = step_name

    def update(self, progress: Progress, score: Score):
        if self._current_pbar is not None:
            # ignore updates coming in before starting a new step or after finishing the current step
            n = progress.tqdm_n_current
            if n > self._current_pbar.n:
                self._current_pbar.n = n
                self._current_pbar.total = progress.tqdm_n_total
                self._current_pbar.refresh()

    def solver_step_finished(self, score: Score):
        self._close_current_pbar()

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _close_current_pbar(self):
        if self._current_pbar is not None:
            # check pbar.total
            if self._current_pbar.total == 0:
                self._current_pbar.total = 1

            # check pbar.n
            if self._current_pbar.n < self._current_pbar.total:
                self._current_pbar.n = self._current_pbar.total
                self._current_pbar.refresh()

            # cleanup
            self._current_pbar.close()
            self._current_pbar = None  # avoid updates after closing


# =================================================================================================
#  Tabular
# =================================================================================================
class TabularProgressReporter(ProgressReporter):
    pass
