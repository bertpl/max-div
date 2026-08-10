"""A worker configuration says what one worker in a portfolio runs.

Only the search varies per worker; the `SolverBuilderBase` docstring explains why the rest cannot.
"""

from dataclasses import dataclass

from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._strategies import InitializationStrategy


@dataclass(frozen=True)
class WorkerConfig:
    """A worker configuration holds one worker's search: its preset, and where it starts from.

    :param init_strategy: replaces the preset's own initialization; None keeps it.  Two workers on
                          the same preset need this to start from different points.
    """

    preset: SolverPreset = SolverPreset.DEFAULT
    init_strategy: InitializationStrategy | None = None
