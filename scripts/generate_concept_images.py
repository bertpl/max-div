"""Generate the figures of the concept pages under `docs/concepts/`.

The figures are the 3 solve timelines on `parallel_solving.md`: one parallel solve per grouping
(12 independent workers, 4 fixed groups of 3, 12 dynamically grouped workers) of the uniform-sampling
case study's banded hybrid experiment, each drawn with `ParallelMaxDivSolution.plot_timeline()`.
The problem and population are the guide script's own, imported from it, so the figures show the
problem the case study reports on.

Every solution is pickled whole under `generated/`, gitignored, so `--reuse-solution` re-renders the
figures without the solves; a pickle is bound to the max-div version that wrote it.

Run from the repo root, which must be on the path for the `benchmarks` import:
``PYTHONPATH=. uv run --group benchmarks ./scripts/generate_concept_images.py [--reuse-solution]``.
"""

import argparse
import pickle
from collections.abc import Callable
from dataclasses import dataclass

from generate_guide_images import EXPERIMENTS, build_experiment_problem, build_uniform_sampling_population

from benchmarks.figures.style import REPO_ROOT, save_webp, use_docs_style
from max_div.problem import MaxDivProblem
from max_div.solver import ParallelMaxDivSolution, ParallelMaxDivSolverBuilder, seconds

GENERATED_DIR = REPO_ROOT / "generated"
IMAGES_DIR = REPO_ROOT / "docs" / "concepts" / "images"

# the case study's banded hybrid experiment, at the case study's size, is the problem of every timeline
TIMELINE_EXPERIMENT = next(experiment for experiment in EXPERIMENTS if experiment.name == "hybrid_banded")
TIMELINE_N = 10_000
TIMELINE_K = 100
TIMELINE_SEED = 42
TIMELINE_BUDGET_SEC = 60.0
TIMELINE_N_WORKERS = 12


@dataclass(frozen=True)
class TimelineExample:
    """A timeline example bundles a grouping of the workers with the builder configured for it."""

    name: str
    configure_workers: Callable[[ParallelMaxDivSolverBuilder], ParallelMaxDivSolverBuilder]

    @property
    def solution_path(self):
        """Return where this example's solution is pickled."""
        return GENERATED_DIR / f"parallel_solving_timeline_{self.name}_solution.pkl"

    @property
    def image_path(self):
        """Return where this example's figure is written."""
        return IMAGES_DIR / f"parallel_solving_timeline_{self.name}.webp"


TIMELINE_EXAMPLES = (
    TimelineExample(
        "independent",
        lambda builder: builder.with_custom_worker_groups(
            seconds(TIMELINE_BUDGET_SEC), TIMELINE_N_WORKERS, n_groups=TIMELINE_N_WORKERS
        ),
    ),
    TimelineExample(
        "fixed_groups",
        lambda builder: builder.with_custom_worker_groups(seconds(TIMELINE_BUDGET_SEC), TIMELINE_N_WORKERS, n_groups=4),
    ),
    TimelineExample(
        "dynamic_groups",
        lambda builder: builder.with_workers(seconds(TIMELINE_BUDGET_SEC), TIMELINE_N_WORKERS),
    ),
)


def solve_timeline_example(problem: MaxDivProblem, example: TimelineExample) -> ParallelMaxDivSolution:
    """Return the example's solution: the shared problem solved under the example's grouping, end to end."""
    builder = ParallelMaxDivSolverBuilder(problem).with_seed(TIMELINE_SEED)
    solver = example.configure_workers(builder).with_end_to_end_budget().build()
    return solver.solve(verbosity=0)


def load_or_solve_timeline_example(
    problem: MaxDivProblem, example: TimelineExample, should_reuse_solution: bool
) -> ParallelMaxDivSolution:
    """Return the example's solution, from its pickle when asked and present, else from a fresh solve.

    A fresh solve rewrites the pickle.
    """
    if should_reuse_solution and example.solution_path.exists():
        with example.solution_path.open("rb") as file:
            return pickle.load(file)
    else:
        solution = solve_timeline_example(problem, example)
        example.solution_path.parent.mkdir(parents=True, exist_ok=True)
        with example.solution_path.open("wb") as file:
            pickle.dump(solution, file)
        return solution


def render_parallel_solving_timelines(should_reuse_solution: bool) -> None:
    """Write the solve timelines of the parallel-solving concept page, one per grouping."""
    vectors = build_uniform_sampling_population(TIMELINE_N, TIMELINE_SEED)
    problem = build_experiment_problem(vectors, TIMELINE_EXPERIMENT, TIMELINE_K)
    use_docs_style()  # the timeline styles itself, but the docs dpi applies at save time
    for example in TIMELINE_EXAMPLES:
        solution = load_or_solve_timeline_example(problem, example, should_reuse_solution)
        print(f"{example.name}: {solution}")
        save_webp(solution.plot_timeline(include_tie_breakers=True), example.image_path)


def main() -> None:
    """Render every concept-page figure."""
    parser = argparse.ArgumentParser(description="Regenerate the concept-page figures.")
    parser.add_argument(
        "--reuse-solution",
        action="store_true",
        help="read the pickled example solutions instead of solving them again",
    )
    args = parser.parse_args()
    render_parallel_solving_timelines(args.reuse_solution)


if __name__ == "__main__":
    main()
