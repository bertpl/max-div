import numpy as np
from numpy.typing import NDArray

from max_div.internal.utils import deterministic_hash_int64, int_to_int64
from max_div.random import new_rng_state


class StrategyBase:
    """Base class for OptimizationStrategy & InitializationStrategy, centralizing some overlapping functionality."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, name: str | None = None):
        """
        Initialize the strategy.
        :param name: optional name of the strategy; if omitted class name is used.
        """
        self._name: str = name or self.__class__.__name__
        self._seed: np.int64 = deterministic_hash_int64(self._name)
        self._rng_state: NDArray[np.uint64] = new_rng_state(self._seed)

    # -------------------------------------------------------------------------
    #  Properties
    # -------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def seed(self) -> np.int64:
        """Return _seed without updating it."""
        return self._seed

    def set_seed(self, seed: int | np.int64) -> None:
        """Sets the random seed for the strategy, to be used by child classes."""
        self._seed = int_to_int64(seed)
        self._rng_state = new_rng_state(self._seed)

    # -------------------------------------------------------------------------
    #  Debug info
    # -------------------------------------------------------------------------
    def get_debug_info(self) -> str:
        return "/"
