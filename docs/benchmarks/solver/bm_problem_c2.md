# Benchmark Results - Problem C2

## I. Problem Description

### A. Overall Approach

Identical to test problem `C1` — same vectors, same group partition — but with the exact per-group quotas relaxed to lower bounds: from each group between $\min(4, \lfloor k/m \rfloor)$ and $k$ vectors need to be selected.  The cap on the lower bound keeps the $m$ bounds summing to at most $k$ at every $n$.

### B. Visualization

This image shows problem C2 with $n=200$ (thus $d=2$, $k=20$, $m=4$):

![Problem C2](./images/problem_C2.webp){ .center }

The image below shows an example solution, obtained by using the `DEFAULT` solver preset over 10.000 iterations
using the L2 distance metric and the `geomean_separation` diversity metric:

![Problem C2 with Solution](./images/problem_C2_with_solution.webp){ .center }

### C. Separation statistics

The image below shows distribution of vector separations (distances to nearest neighbor for all vectors in the population),
for different problem sizes:

![Problem C2 - Vector Separations](./images/problem_C2_separations.webp){ .center75 }

### D. Feasibility

--8<-- "docs/benchmarks/solver/results/note_measured_feasibility.md"

--8<-- "docs/benchmarks/solver/results/note_feasibility_methodology.md"

See [Proving Feasibility](../../concepts/feasibility.md) for the machinery behind these verdicts, and
[Scoring](../../concepts/scoring.md) for the constraints-score scale.

--8<-- "docs/benchmarks/solver/results/feasibility_verdicts_C2.md"

## II. Benchmark results

- [Initialization Strategies](bm_problem_c2_init.md)
- [Optimization Strategies](bm_problem_c2_optim.md)
- [Solver Presets](bm_problem_c2_presets.md)
