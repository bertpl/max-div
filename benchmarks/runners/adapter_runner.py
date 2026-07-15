"""Run a single-shot adapter, once per seed, emitting the same record schema as max-div runs."""

from benchmarks.adapters.base import SelectionAdapter
from benchmarks.common.quality import evaluate_selection, n_constraints_satisfied
from benchmarks.common.records import RunRecord
from max_div.problem import MaxDivProblem


def run_adapter(
    adapter: SelectionAdapter,
    problem: MaxDivProblem,
    problem_name: str,
    size: int,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> list[RunRecord]:
    """Run the adapter once per seed and record measured time + quality.

    Returns:
        One record per seed, with the ``single-shot`` budget tag.
    """
    records = []
    for seed in seeds:
        indices, measured_sec = adapter.timed_select(problem, seed)
        records.append(
            RunRecord(
                tool=adapter.name,
                problem=problem_name,
                size=size,
                n=problem.n,
                k=problem.k,
                diversity_metric=problem.diversity_metric.name,
                seed=seed,
                budget="single-shot",
                measured_sec=measured_sec,
                n_iterations=None,
                quality=evaluate_selection(problem, indices),
                n_constraints=problem.m,
                n_constraints_satisfied=n_constraints_satisfied(problem, indices),
            )
        )
    return records
