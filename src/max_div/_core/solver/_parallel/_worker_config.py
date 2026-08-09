"""What one worker in a portfolio runs.

A worker configuration carries the search and nothing else.  Everything that defines *which
selection is better* — the metric, the tie-breakers, the constraint penalty — is set once on the
portfolio, because comparing what workers found requires one answer to that question.

A preset bundles an initialization strategy with its optimization steps, so `init_strategy` is how
two workers run the same preset from different starting points.
"""

from dataclasses import dataclass

from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._strategies import InitializationStrategy


@dataclass(frozen=True)
class WorkerConfig:
    """One worker's search: which preset it runs, and optionally where it starts from.

    :param preset: the preset this worker solves with.
    :param init_strategy: overrides the preset's own initialization; the preset's is used when None.
    """

    preset: SolverPreset = SolverPreset.DEFAULT
    init_strategy: InitializationStrategy | None = None
