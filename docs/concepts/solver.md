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
| `random_one_shot` | Selects all `k` items in one batch, with probabilities biased by global separation. **Default for the RANDOM and GUIDED presets.** |
| `random_batched` | Selects in batches of `b`, re-evaluating separations between batches. |
| `farthest_point` | A seeded random start item, then greedily adds the item farthest from the selection (farthest-point sampling; under `MEAN_PAIRWISE_DISTANCE`, greedily maximizes mean distance to the selection). An optional `top_k` samples each pick uniformly among the `top_k` best candidates (default 1 keeps the exact greedy construction). Constraint-unaware. **The SMART and THOROUGH presets initialize this way.** |
| `eager` | Evaluates `nc` random candidates per step, picks the best. Slower but higher quality. |
| `most_feasible` | Constructs a selection satisfying every constraint where one can be found, so optimization starts feasible instead of searching for feasibility; where the constraints provably cannot all be met, starts from a least-infeasible one; and otherwise from the least-violating one found. **Constrained problems only** — raises on a problem with no constraints. |
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

## Solving in Parallel

`ParallelMaxDivSolverBuilder` runs several workers on one problem at once — an **algorithm
portfolio** — and keeps the best result any of them reached. The workers share one copy of the
distances, which are usually the most memory-intensive structure in a solve, so N workers cost N
processes but not N copies of that data.

The workers form **[worker groups](glossary.md#worker-group)** — the parallel-metaheuristics
literature calls them *islands*: within a group, every worker adopts the best selection any
member has found so far, exchanged many times per second while solving; groups never communicate
with each other. Groups of one worker are fully independent — a fully
independent set of workers is the special case where every group has one member.

```python
from max_div.solver import ParallelMaxDivSolverBuilder, WorkerConfig, seconds

solution = (
    ParallelMaxDivSolverBuilder(problem)
    .with_seed(42)
    .with_workers(seconds(60), 8, n_groups=4)   # 8 workers in 4 groups of 2
    .build()
    .solve()
)
```

### Why Run Several

The two counts buy different things:

- **More groups**: variance reduction. A run's quality depends on its seed, and keeping the best
  over several independent groups insures against drawing a bad one.
- **Larger groups**: shared search capacity. A group's members pool their effort on promising
  selections — a member stuck with a poor selection picks up a sibling's better one and continues
  from there — at the cost of searching less independently.

How much the variance reduction buys depends on the budget. The [published preset quantiles](../benchmarks/solver/presets_u1.md)
show the seed spread narrowing sharply as budgets grow — roughly tenfold over the first stretch —
and then flattening rather than vanishing.

Even at that floor the bands of neighboring budgets overlap, so an unlucky seed with more budget can
still finish below a lucky one with less.

### Workers and Groups

`with_workers` accepts the worker set in three forms:

- **an integer** — that many default workers, grouped into `n_groups` groups (both counts have
  defaults, below); the only form `n_groups` combines with;
- **a flat sequence of `WorkerConfig`** — one configuration per worker, grouped by the default
  rule;
- **a nested sequence** — one inner sequence per group, fixing the grouping and every
  configuration at once; groups may differ in size and mix presets freely.

When the counts are not given:

- the worker total defaults to **3/4 of the logical cores**;
- the group count defaults to **groups of about four workers** (the count nearest a quarter of
  the worker total; five workers or fewer form a single group), spreading the risk of a bad seed
  over independent groups without losing quality;
- a worker total that does not divide evenly over an explicit `n_groups` hands the extra workers
  to the first groups.

### What Varies per Worker

Each worker is configured by a `WorkerConfig`: the preset it runs, and optionally the
initialization strategy it starts from. `init_strategy` lets two workers run the same preset from
different starting points.

Everything that decides **which selection is better** is fixed for all workers, whether that
setting comes from the problem (the diversity metric, the constraints) or from the builder (the
tie-breakers, the constraint penalty). Comparing what workers found requires a single answer to that
question.

Distance storage is fixed for a different reason: the workers read one shared buffer.

### Seeds and Reproducibility

The parallel solver takes one seed and derives a seed per worker from that seed, so the workers search
differently while the whole configuration derives from a single number.

**Reproducibility follows the grouping.** A fully independent set of workers (`n_groups` equal
to the worker count) repeated from one seed returns the same selection. With cooperating groups
it does not: which selections get adopted depends on how far each worker happens to have
come when it reaches an exchange, and that inter-worker timing varies from run to run.

Each worker's `WorkerSummary` carries its derived seed next to the configuration it ran. For an
independent worker that is enough to replay it on its own with `MaxDivSolverBuilder`; a
cooperative worker's trajectory also depends on what its group mates published, so the replay
contract is independent-only. The limits in the [Reproducibility](#reproducibility) section apply
on top.

### Reading the Result

`solve()` returns a `ParallelMaxDivSolution`: the winning worker's solution, with a `WorkerSummary`
per worker attached. The number worth looking at is `n_workers_with_best_score`:

- **Well below the worker count**: seeds mattered on this problem, and the parallel solve earned its
  cost.
- **Equal to the worker count**: every worker tied. With a fully independent set of workers that means
  the run found nothing a single worker would not have — lower the worker count or solve once.
  With cooperating groups, ties *within* a group are partly structural (members adopt each
  other's best), so read the count against the number of groups rather than of workers.

A `ParallelSolvingWarning` is raised for configurations that cannot help — a single worker, or more
workers than the machine has cores.

### Watching Progress

`solve(verbosity=...)` takes the same levels as a single solve (see `Verbosity`), rendered as **one
combined live view** rather than N interleaved streams; the default is the progress table, the level
suited to longer runs. A row combines two halves with different sources:

- **Progress** (the fraction, iteration count and elapsed time) follows the *slowest still-running
  worker* — the fraction tracks when `solve()` will return, and reaches 100% exactly when it does.
- **The result columns** show the *best score found so far* by any worker, running or finished,
  with the `Worker` column naming the worker it came from (marked `✓` once that worker finished).

So a frozen best while progress keeps advancing simply means the leading worker is done and nobody
has beaten it yet — the `Active` column shows how many workers are still trying. Each worker prints
one set-off row with its final state the moment it finishes.

### On the Word "Portfolio"

Running several configurations of one solver concurrently and keeping the best is known as an
algorithm portfolio, an idea introduced by Huberman, Lukose and Hogg (1997) and developed by Gomes
and Selman (2001).

The word also names a different technique, **algorithm selection**: reading a problem's features to
predict, and then run, the single algorithm best suited to it. That is not max-div's sense — max-div
runs several at once and keeps the best.

Portfolio workers may run independently or share what they learn as they go — ManySAT (Hamadi,
Jabbour and Sais, 2009) shares. max-div sits in between: group members share their best selection,
while groups never exchange information.

**References**

- Huberman, B. A., Lukose, R. M., & Hogg, T. (1997). An economics approach to hard computational
  problems. *Science*, 275(5296), 51–54.
- Gomes, C. P., & Selman, B. (2001). Algorithm portfolios. *Artificial Intelligence*, 126(1–2),
  43–62.
- Hamadi, Y., Jabbour, S., & Sais, L. (2009). ManySAT: a parallel SAT solver. *Journal on
  Satisfiability, Boolean Modeling and Computation*, 6(4), 245–262.
