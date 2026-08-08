# How the Solver Works

## The Solver Pipeline

When you call `solver.solve()`, the solver executes a pipeline of **steps**:

1. **Initialization step** -- builds the initial [selection](glossary.md#selection) of `k`
   [items](glossary.md#item)
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

The initialization step selects the initial `k` items. Different strategies trade off
speed vs quality of the starting point:

| Strategy | How it works |
|----------|-------------|
| `random_one_shot` | Selects all `k` items in one batch, with probabilities biased by global separation. **Default for all presets.** |
| `random_batched` | Selects in batches of `b`, re-evaluating separations between batches. |
| `farthest_point` | A seeded random start item, then greedily adds the item farthest from the selection (farthest-point sampling; under `MEAN_PAIRWISE_DISTANCE`, greedily maximizes mean distance to the selection). Constraint-unaware. |
| `eager` | Evaluates `nc` random candidates per step, picks the best. Slower but higher quality. |
| `fast` | Selects the first `k` items. Trivial deterministic baseline for testing and benchmarking. |

## Optimization Strategies

Optimization steps iteratively improve the selection through [**swap operations**](glossary.md#swap): in each
iteration, the strategy removes one or more items from the current selection and replaces
them with new ones. The swap is kept only if it improves the score.

| Strategy | How it works |
|----------|-------------|
| `random_swaps` | Randomly selects items to remove and add. Simple baseline. |
| `guided_swaps` | Biased towards removing low-separation items and adding high-separation ones. |
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

## Distance Storage

During search the solver reads pairwise distances constantly, and how they are stored is
selectable on the builder:

```python
from max_div.solver import DistanceStorage

solver = (
    MaxDivSolverBuilder(problem)
    .with_preset(seconds(5))
    .with_distance_storage(DistanceStorage.FULL_MATRIX)  # optional; AUTO is the default
    .build()
)
```

- **`CONDENSED`** — one entry per pair (`n·(n-1)/2` values); the most memory-lean stored layout.
- **`FULL_MATRIX`** — a full `n x n` matrix; twice the memory of condensed, but distance reads
  become contiguous row scans, which speeds up solving roughly 2x at large problem sizes.
- **`LAZY`** — no stored distances at all: each distance is computed on demand from the vectors.
  Slower per read, but removes the O(n²) memory requirement entirely, so much larger problems
  become feasible. Available only when the problem is built from vectors.
- **`AUTO`** (default) — for vector problems, picks the fastest layout that fits comfortably in
  memory (full matrix, else condensed, else lazy); for problems built via `from_distances`, keeps
  the format the distances were provided in. The resolved backend is reported in the solution
  summary, e.g. `storage=full_matrix (auto)` — pin a backend explicitly to override.

### Reproducibility

**On one machine, with the same installed versions — max-div's and numba's — and the same
backend, a seeded solve is exactly reproducible**: run it again and you get the same selection,
bit for bit.

**Change any of those three and you may get a different — equally diverse — selection.**
Distances are accumulated sums, and the compiler is allowed to reorder such a sum to vectorize it;
how it does so depends on the processor and on the numba version that compiled the kernels, which
is why a numba upgrade counts here as much as a max-div one. The resulting differences are in the
last bits, but the search is a chaotic process, so one differing comparison can send it down a
different path to an equally good answer. The difference is not a slightly different selection —
it is a different one of comparable quality.

Two practical consequences:

- **`AUTO` picks a backend from available memory**, so the same problem can resolve differently on
  a machine with more or less RAM. Pin the backend explicitly if you want that variable removed —
  though on its own that does not make results portable across different machines.
- **Comparing runs meaningfully** means comparing achieved diversity, not selected indices.

(With a time budget rather than an iteration budget, a faster backend also completes more
iterations — the machine-dependence any time budget carries.)
