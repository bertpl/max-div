# The solve pipeline

## I. Steps { #steps }

When you call `solver.solve()`, the solver executes a pipeline of **steps**:

1. **Initialization step** -- builds the initial [selection](glossary.md#selection) of `k`
   [items](glossary.md#item)
2. **One or more optimization steps** -- iteratively improves the selection

Each step runs for a configured duration (wall-clock time or iteration count) and operates
on a shared solver state that tracks the current selection, separations, and constraint
satisfaction.

```mermaid
flowchart LR
    init["Initialization step"] --> opt1["Optimization step 1"] --> opt2["Optimization step 2"] --> sol(["Solution"])
```

## II. Initialization strategies { #initialization-strategies }

The initialization step selects the initial `k` items. Different strategies trade off
speed vs quality of the starting point:

| Strategy | How it works |
|----------|-------------|
| `random_selection` | Selects all `k` items at random: uniformly when the problem has no constraints or `ignore_constraints=True`, otherwise steering the draw so the selection satisfies the constraints. **Default for a builder without an explicit initialization, and for the RANDOM and GUIDED presets** (which set `ignore_constraints=True`). |
| `farthest_point` | A seeded random start item, then greedily adds an item far from the selection (farthest-point sampling; under `MEAN_PAIRWISE_DISTANCE`, greedily maximizes mean distance to the selection). Each pick samples uniformly among the `top_k` best candidates (default 8; 1 is the exact greedy construction). When every term of the diversity objective is a separation-family metric over one distance, the picks are drawn in rounds of up to `candidate_pool_size` (default 256) per pass over the dataset. Each pick gets the same candidates as when picking one item per pass, and the initialization is several times faster at large n; `candidate_pool_size=None` picks 1 item per pass. Constraint-unaware. **The SMART and THOROUGH presets initialize unconstrained problems this way.** |
| `most_feasible` | Constructs a selection satisfying every constraint where one can be found, so optimization starts feasible instead of searching for feasibility; where the constraints provably cannot all be met, starts from a least-infeasible one; and otherwise from the least-violating one found. **Constrained problems only** — raises on a problem with no constraints. |
| `given_selection` | Starts from a given list of `k` item indices, such as an earlier solution's `i_selected`. Constraint-unaware. Usually set through the builder's `with_initial_selection`; see [Starting from a given selection](#hot-starts). |

## III. Optimization strategies { #optimization-strategies }

Optimization steps iteratively improve the selection through [**swap operations**](glossary.md#swap): in each
iteration, the strategy removes one or more items from the current selection and replaces
them with new ones. The swap is kept only if it improves the score.

| Strategy | How it works |
|----------|-------------|
| `random_swaps` | Randomly selects items to remove and add. Simple baseline. |
| `guided_swaps` | Biased toward removing low-separation items and adding high-separation ones. |
| `smart_swaps` | Adaptively learns which swap sizes and candidate selection strategies work best during the run. |

## IV. Presets vs custom configuration { #presets-vs-custom-configuration }

**Presets** (configured via `with_preset`) select appropriate initialization and optimization
strategies automatically. They are the recommended starting point for most users.

For advanced use cases, you can configure the pipeline manually:

```python
from max_div import (
    MaxDivSolverBuilder, MaxDivProblem,
    InitializationStrategy, OptimizationStrategy,
    seconds, iterations,
)
from max_div._core.solver._solver_step import OptimizationStep

solver = (
    MaxDivSolverBuilder(problem)
    .set_initialization_strategy(InitializationStrategy.farthest_point(top_k=8))
    .add_solver_step(OptimizationStep(OptimizationStrategy.guided_swaps(), seconds(10)))
    .add_solver_step(OptimizationStep(OptimizationStrategy.smart_swaps(
        swap_size_max=4, nc_remove_max=8, nc_add_max=8,
    ), seconds(30)))
    .build()
)
```

This gives you full control over which strategies run, in what order, and for how long.

## V. Starting from a given selection { #hot-starts }

A solve can start from a selection of your own, not one built by an initialization strategy: a **hot start**. Pass exactly `k` distinct item indices to `with_initial_selection`, and the optimization steps start from exactly those items.

```python
from max_div import MaxDivSolverBuilder, seconds

first_solution = MaxDivSolverBuilder(problem).with_preset(seconds(10)).build().solve()

# refine the first result with a longer budget
refined = (
    MaxDivSolverBuilder(problem)
    .with_preset(seconds(60))
    .with_initial_selection(first_solution.i_selected)
    .build()
    .solve()
)
```

You can also build a selection with a process of your own and let the solver improve it.

- **The selection replaces the preset's initialization**, whether `with_preset` is called before or after `with_initial_selection`. The preset's optimization steps still run.
- **The selection may violate the constraints.** The optimization steps then try to satisfy them, as after any constraint-unaware initialization.
- **The indices are checked when you call `with_initial_selection`**: exactly `k` distinct integers, each in `0..n-1`. Anything else raises `ValueError`.
- **A solve has one starting point.** Combining `with_initial_selection` with `set_initialization_strategy` raises `ValueError` when the solver is built, unless a later `with_preset` replaced that strategy.

### V.A. In a parallel solve { #hot-starts-in-a-parallel-solve }

`with_initial_selection` works the same on `ParallelMaxDivSolverBuilder`, and every worker starts from the selection:

```python
from max_div import ParallelMaxDivSolverBuilder, seconds

solution = (
    ParallelMaxDivSolverBuilder(problem)
    .with_initial_selection(first_solution.i_selected)
    .with_workers(seconds(600), 12)
    .build()
    .solve()
)
```

- **Every worker starts from the same selection**, so workers with the same preset differ only in their random seeds, and together they cover less of the search space early on than workers that each build their own starting selection.
- **To give workers different starting selections**, pass `InitializationStrategy.given_selection(indices)` as a worker's `init_strategy` in `with_custom_worker_groups` (see [What varies per worker](parallel_solving.md#what-varies-per-worker)). `given_selection` rejects non-integer, negative or duplicate indices when you create it, and checks the count and upper bound when the solve starts, since only then are `n` and `k` known; each check raises `ValueError`.
- **A worker that sets its own `init_strategy` cannot be combined with `with_initial_selection`**: building the solver raises `ValueError`.
