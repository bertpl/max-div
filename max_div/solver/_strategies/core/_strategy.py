from __future__ import annotations

from abc import ABC, abstractmethod

from tqdm.auto import tqdm

from max_div.solver._solver_state import SolverState

from ._progress_tracker import Progress, ProgressTracker
from ._type import StrategyType


# =================================================================================================
#  SolverStrategy
# =================================================================================================
class SolverStrategy(ABC):
    # -------------------------------------------------------------------------
    #  Construction & Configuration
    # -------------------------------------------------------------------------
    def __init__(self, strategy_type: StrategyType, duration: ProgressTracker, name: str | None = None):
        """
        Initialize the solver strategy.
        :param strategy_type: type of the strategy
                               if StrategyType.INITIALIZATION, the class implements an initialization strategy.
                               if StrategyType.OPTIMIZATION,   the class implements an optimization strategy.
        :param duration: duration settings for the strategy, determines how long it will keep iterating until it's done.
        :param name: optional name of the strategy
        """
        self._strategy_type = strategy_type
        self._duration = duration
        self._name = name or self.__class__.__name__

    @property
    def type(self) -> StrategyType:
        return self._strategy_type

    @property
    def duration(self) -> ProgressTracker:
        return self._duration

    @property
    def name(self) -> str:
        return self._name

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def run(self, state: SolverState, tqdm_desc: str = ""):
        # --- init ----------------------------------------
        duration = self._duration
        duration.start()
        pbar = tqdm(desc=tqdm_desc) if tqdm_desc else None

        def _update_pbar(_progress: Progress):
            if pbar is not None:
                pbar.n = _progress.n_current
                pbar.total = _progress.n_total
                pbar.refresh()

        # --- main loop -----------------------------------
        while not (progress := duration.progress()).is_finished:
            # perform work
            self.do_one_iteration(state)

            # update progress
            _update_pbar(progress)
            self._duration.iteration_done()

        # --- wrap up -------------------------------------
        _update_pbar(progress)

    @abstractmethod
    def do_one_iteration(self, state: SolverState):
        """
        Perform one iteration of the strategy, modifying the solver state in-place, to reach a more optimal solution.
        :param state: (SolverState) The current solver state.
        """
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def init_random(cls) -> SolverStrategy:
        from .._init_random import InitRandom

        return InitRandom()

    @classmethod
    def optim_dummy(cls) -> SolverStrategy:
        from .._optim_dummy import OptimDummy

        return OptimDummy()
