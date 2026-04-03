# How the Solver Works

## The Solver Pipeline

When you call `solver.solve()`, the solver executes a pipeline of **steps**:

1. **Initialization step** -- builds the initial selection of `k` vectors
2. **One or more optimization steps** -- iteratively improves the selection

Each step runs for a configured duration (wall-clock time or iteration count) and operates
on a shared solver state that tracks the current selection, separations, and constraint
satisfaction.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Initialization │ ──> │ Optimization #1 │ ──> │ Optimization #2 │ ──> Solution
│     Step        │     │     Step        │     │     Step        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Initialization Strategies

The initialization step selects the initial `k` vectors. Different strategies trade off
speed vs quality of the starting point:

| Strategy | How it works |
|----------|-------------|
| `random_one_shot` | Selects all `k` vectors in one batch, with probabilities biased by global separation. **Default for all presets.** |
| `random_batched` | Selects in batches of `b`, re-evaluating separations between batches. |
| `eager` | Evaluates `nc` random candidates per step, picks the best. Slower but higher quality. |
| `fast` | Deterministic greedy: always picks the vector maximizing separation. Fast but seed-independent. |

## Optimization Strategies

Optimization steps iteratively improve the selection through **swap operations**: in each
iteration, the strategy removes one or more vectors from the current selection and replaces
them with new ones. The swap is kept only if it improves the score.

| Strategy | How it works |
|----------|-------------|
| `random_swaps` | Randomly selects vectors to remove and add. Simple baseline. |
| `guided_swaps` | Biased towards removing low-separation vectors and adding high-separation ones. |
| `smart_swaps` | Adaptively learns which swap sizes and candidate selection strategies work best during the run. |

## Presets vs Custom Configuration

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
