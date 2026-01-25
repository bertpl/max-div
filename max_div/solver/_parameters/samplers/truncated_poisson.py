import numpy as np

from max_div.random.distributions import sample_truncated_poisson, truncated_poisson_expected_value
from max_div.solver._parameters._adaptive_sampler import AdaptiveSampler


# =================================================================================================
#  Main sampling class
# =================================================================================================
class TruncatedPoissonAdaptiveSampler(AdaptiveSampler[int]):
    def __init__(
        self,
        min_value: int,
        max_value: int,
        lambda_prior: float,
        tau_learn: float,
        tau_forget: float,
        large_value_penalty_exponent: float = 0.0,
        seed: int | np.int64 = 42,
    ):
        """
        Class that performs adaptive sampling of a truncated Poisson distribution over interval [min_value, max_value]
        with a prior on the Lambda parameter.  The Lambda parameter is adapted based on feedback.

        :param min_value: (int) minimum value to sample.
        :param max_value: (int) maximum value to sample.
        :param lambda_prior: (float in range [min_value, max_value]) prior expectation of the Poisson Lambda param
        :param tau_learn: (float) time constant for learning from successful samples.
        :param tau_forget: (float) time constant for forgetting learned probability
                                                     in case of unsuccessful samples.
        :param large_value_penalty_exponent: (float) exponent for penalizing larger values when learning, by using
                                             learning factor  c = c_learn * ((lambda/sampled_value) ^ exponent).
        :param seed: (int|int64) initial random seed for sampling.
        """
        super().__init__(tau_learn=tau_learn, tau_forget=tau_forget, seed=seed)

        if not min_value <= lambda_prior <= max_value:
            raise ValueError(f"lambda_prior {lambda_prior} not in range [{min_value}, {max_value}].")

        self._min_value = np.int32(min_value)
        self._max_value = np.int32(max_value)
        self._lambda_prior: np.float32 = np.float32(lambda_prior)
        self._lambda: np.float32 = self._lambda_prior
        self._last_sample: int = round(lambda_prior)
        self._large_value_penalty_exponent_f32 = np.float32(large_value_penalty_exponent)

    def new_sample(self) -> int:
        sample = int(
            sample_truncated_poisson(
                min_value=self._min_value,
                max_value=self._max_value,
                _lambda=self._lambda,
                rng_state=self._rng_state,
            )
        )
        self._last_sample = sample
        return sample

    def summary_statistic(self) -> float:
        """
        Returns value of the lambda parameter of the truncated Poisson distribution, which is a fast proxy
        for the expected value of the distribution. (in the non-truncated case they are equal).
        """
        return float(self._lambda)

    def feedback(self, success: bool):
        if success:
            c_correction = np.pow(self._lambda / np.float32(self._last_sample), self._large_value_penalty_exponent_f32)
            c_learn_f32 = self._c_learn_f32 * c_correction
            self._lambda += c_learn_f32 * (np.float32(self._last_sample) - self._lambda)
        elif self._forgetting_enabled:
            self._lambda += self._c_forget_f32 * (self._lambda_prior - self._lambda)


# =================================================================================================
#  Alias
# =================================================================================================
def sampled_poisson(
    min_value: int,
    max_value: int,
    lambda_prior: float | None = None,
    tau_learn: float = 100.0,
    tau_forget: float | None = None,
    large_value_penalty_exponent: float = 0.0,
    seed: int = 42,
) -> TruncatedPoissonAdaptiveSampler:
    """Alias for easier access to TruncatedPoissonAdaptiveSampler."""
    return TruncatedPoissonAdaptiveSampler(
        min_value=min_value,
        max_value=max_value,
        lambda_prior=lambda_prior if lambda_prior is not None else (0.5 * (min_value + max_value)),
        tau_learn=tau_learn,
        tau_forget=tau_forget if tau_forget is not None else (tau_learn * tau_learn),
        large_value_penalty_exponent=large_value_penalty_exponent,
        seed=seed,
    )
