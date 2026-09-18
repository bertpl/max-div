"""Generate the figures of the concept pages under `docs/concepts/`.

The one figure so far is the solve timeline on `parallel_solving.md`: a parallel solve of the
uniform-sampling case study's banded hybrid experiment, drawn with `ParallelMaxDivSolution.plot_timeline()`.
The problem, population and solver settings are the guide script's own, imported from it, so the
figure shows the same solve the case study reports.

The whole solution is pickled under `generated/`, gitignored, so `--reuse-solution` re-renders the
figure without the solve; the pickle is bound to the max-div version that wrote it.

Run from the repo root, which must be on the path for the `benchmarks` import:
``PYTHONPATH=. uv run --group benchmarks ./scripts/generate_concept_images.py [--reuse-solution]``.
"""

import argparse
import pickle

from generate_guide_images import EXPERIMENTS, ExperimentSettings, build_uniform_sampling_population, solve_experiment

from benchmarks.figures.style import REPO_ROOT, save_webp, use_docs_style
from max_div.solver import ParallelMaxDivSolution

GENERATED_DIR = REPO_ROOT / "generated"
IMAGES_DIR = REPO_ROOT / "docs" / "concepts" / "images"

TIMELINE_EXPERIMENT = next(experiment for experiment in EXPERIMENTS if experiment.name == "hybrid_banded")
TIMELINE_SETTINGS = ExperimentSettings(n=10_000, k=100, budget_sec=60.0, n_workers=16, seed=42)
TIMELINE_SOLUTION_PATH = GENERATED_DIR / "parallel_solving_timeline_solution.pkl"


def load_or_solve_timeline_example(should_reuse_solution: bool) -> ParallelMaxDivSolution:
    """Return the timeline example's solution, from its pickle when asked and present, else from a fresh solve.

    A fresh solve rewrites the pickle.
    """
    if should_reuse_solution and TIMELINE_SOLUTION_PATH.exists():
        with TIMELINE_SOLUTION_PATH.open("rb") as file:
            return pickle.load(file)
    else:
        vectors = build_uniform_sampling_population(TIMELINE_SETTINGS.n, TIMELINE_SETTINGS.seed)
        solution = solve_experiment(vectors, TIMELINE_EXPERIMENT, TIMELINE_SETTINGS)
        TIMELINE_SOLUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TIMELINE_SOLUTION_PATH.open("wb") as file:
            pickle.dump(solution, file)
        return solution


def render_parallel_solving_timeline(should_reuse_solution: bool) -> None:
    """Write the solve timeline of the parallel-solving concept page."""
    solution = load_or_solve_timeline_example(should_reuse_solution)
    print(solution)
    use_docs_style()  # the timeline styles itself, but the docs dpi applies at save time
    save_webp(solution.plot_timeline(), IMAGES_DIR / "parallel_solving_timeline.webp")


def main() -> None:
    """Render every concept-page figure."""
    parser = argparse.ArgumentParser(description="Regenerate the concept-page figures.")
    parser.add_argument(
        "--reuse-solution",
        action="store_true",
        help="read the pickled example solution instead of solving it again",
    )
    args = parser.parse_args()
    render_parallel_solving_timeline(args.reuse_solution)


if __name__ == "__main__":
    main()
