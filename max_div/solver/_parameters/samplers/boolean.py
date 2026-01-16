import numpy as np

from max_div.random.rng import rand_float32
from max_div.solver._parameters._adaptive_sampler import AdaptiveSampler


# =================================================================================================
#  Main sampling class
# =================================================================================================
class BooleanAdaptiveSampler(AdaptiveSampler[bool]):
    def __init__(
        self,
        p_true_prior: float,
        tau_learn: float,
        tau_forget: float,
        seed: int | np.int64 = 42,
    ):
        """
        Class that performs adaptive sampling of boolean values (True/False),
        :param p_true_prior: (float in (0,1)) prior expectation of probability of sampling True.
        :param tau_learn: (float) time constant for learning from successful samples.
        :param tau_forget: (float) time constant for forgetting learned probability
                                                     in case of unsuccessful samples.
        :param seed: (int|int64) initial random seed for sampling.
        """
        super().__init__(tau_learn=tau_learn, tau_forget=tau_forget, seed=seed)

        if not 0.0 <= p_true_prior <= 1.0:
            raise ValueError(f"p_true_prior {p_true_prior} not in range [0.0, 1.0].")

        self._p_true_prior: np.float32 = np.float32(p_true_prior)
        self._p_true: np.float32 = self._p_true_prior
        self._last_sample: bool = p_true_prior >= 0.5

    def new_sample(self) -> bool:
        r = rand_float32(self._rng_state)
        sample = bool(r < self._p_true)
        self._last_sample = sample
        return sample

    def summary_statistic(self) -> float:
        """Return probability of sampling True (which can be seen as the expected value if False=0, True=1)."""
        return float(self._p_true)

    def feedback(self, success: bool):
        if success:
            self._p_true += self._c_learn_f32 * (np.float32(self._last_sample) - self._p_true)
        elif self._forgetting_enabled:
            self._p_true += self._c_forget_f32 * (self._p_true_prior - self._p_true)


# =================================================================================================
#  Alias
# =================================================================================================
def sampled_boolean(
    p_true_prior: float = 0.5,
    tau_learn: float = 100.0,
    tau_forget: float | None = None,
    seed: int = 42,
) -> BooleanAdaptiveSampler:
    """Alias for easier access to BooleanAdaptiveSampler."""
    return BooleanAdaptiveSampler(
        p_true_prior=p_true_prior,
        tau_learn=tau_learn,
        tau_forget=tau_forget if tau_forget is not None else (tau_learn * tau_learn),
        seed=seed,
    )
