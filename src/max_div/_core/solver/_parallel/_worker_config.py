"""A worker configuration says what one worker in a portfolio runs.

Only the search varies per worker; `SolverBuilderBase` owns why the rest cannot.
"""

from dataclasses import dataclass

from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._strategies import InitializationStrategy


@dataclass(frozen=True)
class WorkerConfig:
    """A worker configuration holds one worker's search: its preset, and where it starts from.

    :param preset: the preset this worker solves with.
    :param init_strategy: replaces the preset's own initialization, which two workers running the
                          same preset need in order to start from different points; None keeps the
                          preset's.
    """

    preset: SolverPreset = SolverPreset.DEFAULT
    init_strategy: InitializationStrategy | None = None
