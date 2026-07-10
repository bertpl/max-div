# Benchmark Results - Problem C1

## I. Problem Description

### A. Overall Approach

Vectors are 2-dimensional and have...

- first component drawn from a uniform distribution over $[0, 1]$,
- second component drawn from a standard normal distribution $\mathcal{N}(0, 1)$.

All vectors are split in $m=2s$ non-overlapping groups, by splitting the range $[0, 1]$ of the first component into $m$ equal segments.
From each group between $4$ and $k$ vectors need to be selected.  This is always feasible, since $k=5m$ and $n=50m$.

### B. Visualization

This image shows problem C1 with size parameter $s=2$ (thus $d=2$, $n=200$, $k=20$, $m=4$):

![Problem C1](./images/problem_C1.webp){ .center }

The image below shows an example solution, obtained by using the `DEFAULT` solver preset over 10.000 iterations
using the L2 distance metric and the `geomean_separation` diversity metric:

![Problem C1 with Solution](./images/problem_C1_with_solution.webp){ .center }

### C. Separation statistics

The image below shows distribution of vector separations (distances to nearest neighbor for all vectors in the population),
for different problem sizes:

![Problem C1 - Vector Separations](./images/problem_C1_separations.webp){ .center75 }

## II. Benchmark results

- [Initialization Strategies](bm_problem_c1_init.md)
- [Optimization Strategies](bm_problem_c1_optim.md)
- [Solver Presets](bm_problem_c1_presets.md)
