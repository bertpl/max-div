from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class SpeedParam:
    """A benchmark-scope parameter interpolated between its slow (full-scope) and fast (turbo) values.

    Every benchmark CLI exposes a `speed` parameter: 0.0 runs the full scope and 1.0 the
    turbo scope.  A `SpeedParam` declares one scope parameter's value at both ends; `at()`
    interpolates in between, and `at_int()` is the rounding variant for integer-valued
    parameters such as counts.  Construction is keyword-only.

    Args:
        slow: The value at speed 0.0.
        fast: The value at speed 1.0.
        scale: "log" interpolates geometrically — for values spanning orders of magnitude,
            such as time budgets or problem sizes; "linear" arithmetically — for counts
            with a narrow range.

    Raises:
        ValueError: If `scale` is "log" and `slow` and `fast` are not both strictly
            positive (geometric interpolation is undefined there).
    """

    slow: float
    fast: float
    scale: Literal["log", "linear"] = "log"

    def __post_init__(self) -> None:
        if self.scale == "log" and (self.slow <= 0 or self.fast <= 0):
            raise ValueError("log-scale interpolation requires strictly positive `slow` and `fast` values")

    def at(self, speed: float) -> float:
        """Return the parameter's value at the given speed in [0.0, 1.0]."""
        if self.scale == "log":
            return self.slow * (self.fast / self.slow) ** speed
        return self.slow + speed * (self.fast - self.slow)

    def at_int(self, speed: float) -> int:
        """Return the parameter's value at the given speed, rounded to the nearest integer."""
        return round(self.at(speed))
