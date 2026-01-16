"""
Functionality for adaptively sampling from histograms, where through RL-style feedback, we influence future samples,
with the goal of maximizing 'successful' samples.  We provide a base class with core sampling & feedback API,
with specific implementations found in the 'samplers' sub-package.

The following common elements are imposed / assumed:
  - We can get individual samples via .new_sample(), whose value will be remembered for future calls to .feedback()
  - Feedback can be provided via .feedback(success: bool)
      --> success=True
            --> indicates last sample was successful
            --> distribution should be updated to increase future probability of last sample...
            --> ...with time constant tau_learn

      --> success=False
            --> indicates last sample was NOT successful
            --> distribution should be updated to move towards the PRIOR distribution...
            --> ...with time constant tau_forget

"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from max_div.internal.utils import int_to_int64
from max_div.random.rng import new_rng_state

from .base import ParameterValueSource

# shorthand type alias
Floatable = int | float | np.int32 | np.float32 | bool  # all of these can be converted to float32

S = TypeVar("S", bound=Floatable)


# =================================================================================================
#  Core class
# =================================================================================================
class AdaptiveSampler(ParameterValueSource, ABC, Generic[S]):
    # -------------------------------------------------------------------------
    #  Constructor / Configuration
    # -------------------------------------------------------------------------
    def __init__(
        self,
        tau_learn: float,
        tau_forget: float,
        seed: int | np.int64 = 42,
    ):
        # --- dummy initial values --------------
        self._tau_learn, self._tau_forget = -1.0, -1.0
        self._c_learn, self._c_forget = 0.0, 0.0
        self._c_learn_f32, self._c_forget_f32 = np.float32(0.0), np.float32(0.0)
        self._forgetting_enabled = False
        self._rng_state = np.zeros(2, dtype=np.uint64)

        # --- go through update methods ---------
        self.update_tau(tau_learn, tau_forget)
        self.update_seed(seed)

    def update_seed(self, seed: np.int64 | int):
        """Update the seed for the random number generator used by the sampler."""
        self._rng_state = new_rng_state(int_to_int64(seed))  # rng_state matches better with downstream usage than seed

    def update_tau(self, tau_learn: float | None = None, tau_forget: float | None = None):
        """Update the time constants for learning and forgetting."""
        if (tau_learn is not None) and (tau_learn != self._tau_learn):
            self._tau_learn = tau_learn
            self._c_learn = 1.0 - (0.5 ** (1 / tau_learn))  # adjust by 50% in 'tau_learn' # of steps
            self._c_learn_f32 = np.float32(self._c_learn)

        if (tau_forget is not None) and (tau_forget != self._tau_forget):
            self._tau_forget = tau_forget
            if np.isinf(tau_forget):
                self._forgetting_enabled = False
                self._c_forget = 0.0
                self._c_forget_f32 = np.float32(0.0)
            else:
                self._forgetting_enabled = True
                self._c_forget = 1.0 - (0.5 ** (1 / tau_forget))  # adjust by 50% in 'tau_forget' # of steps
                self._c_forget_f32 = np.float32(self._c_forget)

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    @abstractmethod
    def new_sample(self) -> S:
        """Generate new sample from the sampler's distribution."""
        raise NotImplementedError

    @abstractmethod
    def summary_statistic(self) -> float:
        """Return summary statistic of the distribution (expected value, mode, median or a proxy)"""
        raise NotImplementedError

    @abstractmethod
    def feedback(self, success: bool):
        """Provide feedback to the sampler whether the last sample was successful."""
        raise NotImplementedError

    def get_initial_value(self) -> S:
        """Return a valid initial value (any) for the parameter."""
        return self.new_sample()
