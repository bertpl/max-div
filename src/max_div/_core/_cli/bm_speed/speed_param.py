from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SpeedParam:
    """A benchmark-scope parameter interpolated between its full-scope and turbo values.

    Every benchmark CLI exposes a `speed` knob: 0.0 runs the full scope (the configuration
    the docs pages are generated with) and 1.0 the turbo scope (the cheap path used by tests
    and smoke runs; `--turbo` is shorthand for it).  A `SpeedParam` declares one scope
    parameter's value at both ends; `at()` interpolates in between, and `at_int()` is the
    rounding variant for integer-valued parameters such as counts.

    Args:
        full: The value at speed 0.0.
        turbo: The value at speed 1.0.
        scale: "log" interpolates geometrically — for values spanning orders of magnitude,
            such as time budgets or problem sizes; "linear" arithmetically — for counts
            with a narrow range.

    Raises:
        ValueError: If `scale` is "log" and `full` and `turbo` are not both strictly
            positive (geometric interpolation is undefined there).
    """

    full: float
    turbo: float
    scale: Literal["log", "linear"] = "log"

    def __post_init__(self) -> None:
        if self.scale == "log" and (self.full <= 0 or self.turbo <= 0):
            raise ValueError("log-scale interpolation requires strictly positive `full` and `turbo` values")

    def at(self, speed: float) -> float:
        """Return the parameter's value at the given speed in [0.0, 1.0]."""
        if self.scale == "log":
            return self.full * (self.turbo / self.full) ** speed
        return self.full + speed * (self.turbo - self.full)

    def at_int(self, speed: float) -> int:
        """Return the parameter's value at the given speed, rounded to the nearest integer."""
        return round(self.at(speed))
