import numpy as np

from max_div.random.distributions import sample_modified_power_distribution
from max_div.solver._parameters._adaptive_sampler import AdaptiveSampler


# =================================================================================================
#  Main sampling class
# =================================================================================================
class SkewedIntervalAdaptiveSampler(AdaptiveSampler[np.float32]):
    def __init__(
        self,
        min_value: float,
        max_value: float,
        median_prior: float,
        tau_learn: float,
        tau_forget: float,
        seed: int | np.int64 = 42,
    ):
        """
        Class that performs adaptive sampling from a fixed interval, where the median value is adapted using feedback,
        resulting in a skewed distribution over the interval.  Underlying, a modfied power distribution over [0,1]
        is used, which is then scaled to [min_value, max_value].

        For numerical robustness, the median parameter of the underlying distribution, is limited to [0.01, 0.99].

        :param min_value: (int) minimum value to sample.
        :param max_value: (int) maximum value to sample.
        :param median_prior: (float in range [min_value, max_value]) prior expectation of the median of the distribution
        :param tau_learn: (float) time constant for learning from successful samples.
        :param tau_forget: (float) time constant for forgetting learned probability
                                                     in case of unsuccessful samples.
        :param seed: (int|int64) initial random seed for sampling.
        """
        super().__init__(tau_learn=tau_learn, tau_forget=tau_forget, seed=seed)

        if not min_value <= median_prior <= max_value:
            raise ValueError(f"median_prior {median_prior} not in range [{min_value}, {max_value}].")

        self._min_value = np.float32(min_value)
        self._max_value = np.float32(max_value)
        self._median_prior: np.float32 = np.float32(median_prior)
        self._median: np.float32 = self._median_prior
        self._last_sample: np.float32 = self._median

    def new_sample(self) -> np.float32:
        normalized_median = (self._median - self._min_value) / (self._max_value - self._min_value)
        normalized_median = np.clip(normalized_median, np.float32(0.01), np.float32(0.99))
        normalized_sample = sample_modified_power_distribution(normalized_median, self._rng_state)
        sample = self._min_value + normalized_sample * (self._max_value - self._min_value)
        self._last_sample = sample
        return sample

    def summary_statistic(self) -> float:
        """Returns median value of distribution."""
        return float(self._median)

    def feedback(self, success: bool):
        if success:
            self._median += self._c_learn_f32 * (np.float32(self._last_sample) - self._median)
        elif self._forgetting_enabled:
            self._median += self._c_forget_f32 * (self._median_prior - self._median)


# =================================================================================================
#  Alias
# =================================================================================================
def sampled_interval(
    min_value: float,
    max_value: float,
    median_prior: float | None = None,
    tau_learn: float = 100.0,
    tau_forget: float | None = None,
    seed: int = 42,
) -> SkewedIntervalAdaptiveSampler:
    """Alias for easier access to SkewedIntervalAdaptiveSampler."""
    return SkewedIntervalAdaptiveSampler(
        min_value=min_value,
        max_value=max_value,
        median_prior=median_prior if median_prior is not None else (0.5 * (min_value + max_value)),
        tau_learn=tau_learn,
        tau_forget=tau_forget if tau_forget is not None else (tau_learn * tau_learn),
        seed=seed,
    )
