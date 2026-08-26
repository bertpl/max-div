from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, cast

T = TypeVar("T", int, float)


@dataclass(frozen=True, kw_only=True)
class SpeedParam(Generic[T]):
    """A benchmark-scope parameter interpolated between its slow (full-scope) and fast (turbo) values.

    Every benchmark CLI exposes a `speed` parameter: 0.0 runs the full scope and 1.0 the
    turbo scope.  A `SpeedParam` declares one scope parameter's value at both ends;
    `at()` interpolates in between.  Construction is keyword-only, and the endpoint type
    decides the result type: two `int` endpoints make `at()` round to an `int`, any
    `float` endpoint makes it return a `float` — so integer-valued parameters (counts,
    sizes) declare integer literals and float-valued ones (time budgets) float literals.

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

    slow: T
    fast: T
    scale: Literal["log", "linear"] = "log"

    def __post_init__(self) -> None:
        if self.scale == "log" and (self.slow <= 0 or self.fast <= 0):
            raise ValueError("log-scale interpolation requires strictly positive `slow` and `fast` values")

    def at(self, speed: float) -> T:
        """Return the parameter's value at the given speed in [0.0, 1.0]."""
        if self.scale == "log":
            value = self.slow * (self.fast / self.slow) ** speed
        else:
            value = self.slow + speed * (self.fast - self.slow)
        if isinstance(self.slow, int) and isinstance(self.fast, int):
            return cast("T", round(value))
        return cast("T", value)
