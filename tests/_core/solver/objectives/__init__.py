"""Build the objectives that solver tests hand to `SolverState.new` and `ScoreGenerator`."""

from max_div._core.metrics import DiversityMetric, DiversityObjective, DiversityObjectiveSimple


def simple_objective(diversity_metric: DiversityMetric) -> DiversityObjectiveSimple:
    """Return a simple objective of `diversity_metric` over the problem's own distance (distance `None`)."""
    return DiversityObjectiveSimple(diversity_metric)


def tie_breaker_objectives(tie_breaker_metrics: list[DiversityMetric]) -> list[DiversityObjective]:
    """Return each metric as a simple tie-breaker objective over the problem's own distance.

    A single-metric problem's tie-breakers are simple objectives, as the solver builder constructs them.
    """
    return [DiversityObjectiveSimple(metric) for metric in tie_breaker_metrics]
