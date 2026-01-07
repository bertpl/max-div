from abc import ABC, abstractmethod


class ParameterValueSource(ABC):
    """
    Base class for parameter value sources, i.e. classes that can generate values for parameters in different ways:

      - scheduled: deterministic function of progress fraction (e.g., linear, ease_in, ease_out, ease_in_out)
                    --> ParameterSchedule subclass

      - sampled: stochastic sampling from a distribution, adapted based on iteration success
                    --> AdaptiveSampler subclass
    """

    @abstractmethod
    def get_initial_value(self) -> float:
        """Return a valid initial value (any) for the parameter."""
        raise NotImplementedError()
