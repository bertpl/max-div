# Getting Started

## Installation

```bash
pip install max-div
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add max-div
```

## Basic Usage

### 1. Define a problem

A Maximum Diversity Problem consists of a set of `n` vectors in `d` dimensions, from which you want
to select `k` that are as diverse as possible.

```python
import numpy as np
from max_div import MaxDivProblem, MaxDivSolverBuilder, seconds

# 200 random points in 5 dimensions
rng = np.random.default_rng(42)
vectors = rng.random((200, 5))

# Select 20 that maximize diversity
problem = MaxDivProblem.new(vectors, k=20)
```

Vectors are passed as a 2D numpy array.

### 2. Solve

Use the builder to configure and run the solver. The simplest approach is to just specify a
**time budget**. Here we don't specify a preset, so the default (`SMART`) is used:

```python
solver = MaxDivSolverBuilder(problem).with_preset(seconds(5)).build()
solution = solver.solve()
```

The solver will print progress as it runs. Pass `verbosity=0` for silent operation, or
`verbosity=20` for a detailed progress table.

### 3. Interpret the results

```python
# One-line summary
print(solution)
# MaxDivSolution: 20 vectors selected | diversity=0.7792 | 5.17s (39_008 iterations)

# Indices of the selected vectors
print(solution.i_selected)       # e.g. array([ 3,  7, 14, ...], dtype=int32)

# The selected vectors themselves
selected_vectors = vectors[solution.i_selected]

# How long the solver ran
print(solution.duration)          # 5.17s (39_008 iterations)

# Multi-component score of the solution
print(solution.score)
# size=1.0000 | constraints=1.0000 | diversity=0.7792
```

The `score` has three main components, shown in strict priority order.
In a final solution, `size` is always 1.0. For an unconstrained problem, `constraints` is also
always 1.0:

- **`size`** -- 1.0 when exactly `k` vectors are selected
- **`constraints`** -- 1.0 when all fairness constraints are satisfied
- **`diversity`** -- the main diversity score (higher is better)

This priority ordering means that solutions with better constraint satisfaction will always be
preferred over solutions with higher diversity.


## Adding Fairness Constraints

Constraints let you enforce that a minimum and/or maximum number of selected vectors come from
specific subsets. This is useful for ensuring fair representation across groups.

```python
from max_div import Constraint

# Vectors 0-99 are "group A", vectors 100-199 are "group B"
# Require at least 8 and at most 12 from each group
constraints = [
    Constraint(int_set=set(range(0, 100)),   min_count=8, max_count=12),
    Constraint(int_set=set(range(100, 200)), min_count=8, max_count=12),
]

problem = MaxDivProblem.new(vectors, k=20, constraints=constraints)
solver = MaxDivSolverBuilder(problem).with_preset(seconds(5)).build()
solution = solver.solve(verbosity=0)

print(solution)
# MaxDivSolution: 20 vectors selected | diversity=0.7702 | constraints: 2/2 satisfied | ...
```

Note that the diversity is slightly lower than the unconstrained solution (0.7702 vs 0.7792) --
this is expected, since satisfying constraints restricts the search space.

Constraints can overlap -- a single vector can belong to multiple constraint groups.

If constraints are infeasible (or jointly infeasible), the solver will not fail. Instead, it
gracefully finds the *least infeasible* solution -- the one that violates constraints as little as
possible. This follows naturally from the score's priority ordering: a higher constraint score always
outweighs a higher diversity score.


## Choosing Metrics

### Distance metrics

The distance metric determines how the distance between two vectors is measured.

```python
from max_div import DistanceMetric

problem = MaxDivProblem.new(
    vectors, k=20,
    distance_metric=DistanceMetric.L2_EUCLIDEAN,       # default
    # distance_metric=DistanceMetric.L1_MANHATTAN,
    # distance_metric=DistanceMetric.L2S_EUCLIDEAN_SQUARED,
)
```

| Metric | Description |
|--------|-------------|
| `L2_EUCLIDEAN` | Standard Euclidean distance (default) |
| `L1_MANHATTAN` | Sum of absolute differences |
| `L2S_EUCLIDEAN_SQUARED` | Squared Euclidean -- avoids the square root, produces identical solutions when used with `GEOMEAN_SEPARATION` |

### Diversity metrics

The diversity metric defines what "diverse" means for the selected subset. It operates on the
*separation* of each selected vector -- its distance to its nearest neighbor in the selection.
A well-spread selection will have high separations across all vectors; a clustered selection will
have low separations for the vectors that are close together.

```python
from max_div import DiversityMetric

problem = MaxDivProblem.new(
    vectors, k=20,
    diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,   # default
    # diversity_metric=DiversityMetric.MIN_SEPARATION,
    # diversity_metric=DiversityMetric.MEAN_SEPARATION,
    # diversity_metric=DiversityMetric.APPROX_GEOMEAN_SEPARATION,
)
```

| Metric | Description | Best for |
|--------|-------------|----------|
| `GEOMEAN_SEPARATION` | Geometric mean of all separations (default) | General use -- balances spread across all selected vectors |
| `MIN_SEPARATION` | Minimum separation (= p-dispersion) | When the closest pair matters most |
| `MEAN_SEPARATION` | Arithmetic mean of all separations | When total spread is the objective |
| `APPROX_GEOMEAN_SEPARATION` | Fast approximation of `GEOMEAN_SEPARATION` | Large-scale problems where speed matters |


## Solver Presets

Presets configure the initialization and optimization strategies. All presets accept a
**time budget** (via `seconds()`, `minutes()`, `hours()`) or an **iteration count**
(via `iterations()`).

```python
from max_div import SolverPreset, minutes, iterations

# Using a named preset
solver = (
    MaxDivSolverBuilder(problem)
    .with_preset(minutes(2), preset=SolverPreset.SMART)
    .build()
)

# Or using iteration count instead of wall-clock time
solver = (
    MaxDivSolverBuilder(problem)
    .with_preset(iterations(10_000))
    .build()
)
```

| Preset | Description |
|--------|-------------|
| `RANDOM` | Baseline -- random initialization + random swap optimization. Fast but lower quality. |
| `GUIDED` | Distance-guided swaps -- biased towards removing low-separation vectors and adding high-separation ones. |
| `SMART` | **Default.** Adaptive swap-based optimization that learns which swap sizes and candidate selection strategies work best. |
| `THOROUGH` | Like `SMART` but with wider parameter ranges, exploring more of the search space at the cost of slower convergence. Best for long runs. |

As a rule of thumb, `SMART` (the default) is a good starting point. Use `THOROUGH` for long runs
(minutes to hours) where you want the best possible solution quality. Use `RANDOM` or `GUIDED` as
baselines for comparison.

The time budget should be chosen relative to the problem size. A good starting point is to give the
solver at least `10 * k` to `100 * k` iterations.


## Reproducibility

The solver is deterministic for a given seed. Change the seed to get different solutions:

```python
solver = (
    MaxDivSolverBuilder(problem)
    .with_preset(seconds(5))
    .with_seed(123)
    .build()
)
```

Running the same solver multiple times with different seeds and keeping the best result is a simple
way to improve solution quality.


## Next Steps

- **[API Reference](reference/solver/MaxDivSolverBuilder.md)** -- full details on all configuration options
- **[Benchmark Results](benchmarks/_index.md)** -- see how different strategies and presets perform across various test problems
- **[CLI](cli.md)** -- run benchmarks and solve problems from the command line
