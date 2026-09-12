"""Build the objectives that solver tests hand to `SolverState.new` and `ScoreGenerator`."""

from max_div._core.metrics import (
    DiversityMetric,
    DiversityObjective,
    DiversityObjectiveHybridFlattened,
    DiversityObjectiveSimple,
)


def single_term_objective(diversity_metric: DiversityMetric) -> DiversityObjectiveSimple:
    """Return a simple objective of `diversity_metric` over the problem's own distance (distance `None`)."""
    return DiversityObjectiveSimple(diversity_metric)


def tie_breaker_objectives(tie_breaker_metrics: list[DiversityMetric]) -> list[DiversityObjective]:
    """Return each metric as a tie-breaker objective over the problem's own distance.

    Each is a `DiversityObjectiveHybridFlattened`, as the solver builder constructs a tie-breaker.
    """
    return [DiversityObjectiveHybridFlattened(metric, (None,)) for metric in tie_breaker_metrics]
