"""A worker configuration says what one worker of a parallel solver runs.

Only the search varies per worker; the `SolverBuilderBase` docstring explains why the rest cannot.
"""

from dataclasses import dataclass

from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._strategies import InitializationStrategy


@dataclass(frozen=True)
class WorkerConfig:
    """A worker configuration holds one worker's search: its preset, and where it starts from.

    Args:
        init_strategy: replaces the preset's own initialization; None keeps it.  Different
            seeds already vary a random initialization, so this is for giving workers a
            different kind of start, not merely a different one.
    """

    preset: SolverPreset = SolverPreset.DEFAULT
    init_strategy: InitializationStrategy | None = None
