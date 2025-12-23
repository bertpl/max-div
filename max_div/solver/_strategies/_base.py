import numpy as np

from max_div.internal.utils import deterministic_hash_int64, int_to_int64


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

    @seed.setter
    def seed(self, seed: int | np.int64):
        """Sets the random seed for the strategy, to be used by child classes."""
        self._seed = int_to_int64(seed)

    def next_seed(self) -> np.int64:
        """Get the next seed & update current one."""
        self._seed += 1  # this will silently wrap around in case of overflow
        return self._seed
