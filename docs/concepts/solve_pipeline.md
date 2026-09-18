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
| `random_one_shot` | Selects all `k` items in one batch, with probabilities biased by global separation. **Default for the RANDOM and GUIDED presets.** |
| `random_batched` | Selects in batches of `b`, re-evaluating separations between batches. |
| `farthest_point` | A seeded random start item, then greedily adds the item farthest from the selection (farthest-point sampling; under `MEAN_PAIRWISE_DISTANCE`, greedily maximizes mean distance to the selection). An optional `top_k` samples each pick uniformly among the `top_k` best candidates (default 1 keeps the exact greedy construction). Constraint-unaware. |
| `farthest_point_batched` | The farthest-point construction with one pass over the dataset per batch of picks, not per pick. Every draw ranges over the same candidates that `farthest_point` would offer, so quality is equal while large problems initialize several times faster. Separation-family diversity metrics only; constraint-unaware. **The SMART and THOROUGH presets initialize unconstrained problems this way** (`farthest_point` under `MEAN_PAIRWISE_DISTANCE`). |
| `eager` | Evaluates `nc` random candidates per step, picks the best. Slower but higher quality. |
| `most_feasible` | Constructs a selection satisfying every constraint where one can be found, so optimization starts feasible instead of searching for feasibility; where the constraints provably cannot all be met, starts from a least-infeasible one; and otherwise from the least-violating one found. **Constrained problems only** — raises on a problem with no constraints. |
| `fast` | Selects the first `k` items. Trivial deterministic baseline for testing and benchmarking. |

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
    .set_initialization_strategy(InitializationStrategy.eager(nc=50))
    .add_solver_step(OptimizationStep(OptimizationStrategy.guided_swaps(), seconds(10)))
    .add_solver_step(OptimizationStep(OptimizationStrategy.smart_swaps(
        swap_size_max=4, nc_remove_max=8, nc_add_max=8,
    ), seconds(30)))
    .build()
)
```

This gives you full control over which strategies run, in what order, and for how long.
