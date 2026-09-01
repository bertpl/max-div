from typing import TYPE_CHECKING

import numpy as np

from max_div._core._random import new_rng_state
from max_div._core._utils import deterministic_hash_int64, int_to_int64

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from max_div._core.metrics import DiversityMetric


class StrategyBase:
    """Base class for OptimizationStrategy & InitializationStrategy, centralizing some overlapping functionality."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, name: str | None = None) -> None:
        """Initialize the strategy.

        Args:
            name: optional name of the strategy; if omitted class name is used.
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

    def validate_diversity_metric(self, diversity_metric: "DiversityMetric") -> None:
        """Raise when this strategy does not support the solve's main diversity metric.

        Called when the solver is built, so an unsupported combination fails before any work.
        The default accepts every metric; a strategy whose algorithm is tailored to specific
        metric families overrides `validate_diversity_metric`.
        """

    def set_seed(self, seed: int | np.int64) -> None:
        """Sets the random seed for the strategy, to be used by child classes."""
        self._seed = int_to_int64(int(seed))  # int() since int_to_int64 accepts Python int only; cold path
        self._rng_state = new_rng_state(self._seed)

    # -------------------------------------------------------------------------
    #  Debug info
    # -------------------------------------------------------------------------
    def get_debug_info(self) -> str:
        return "/"
