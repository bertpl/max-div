# Benchmark Results - Problem U3

## I. Problem Description

### A. Overall Approach

Vector components are drawn from a standard normal distribution $\mathcal{N}(0, 1)$, with $d = \lceil n/100 \rceil$.

### B. Visualization

This image shows problem U3 with $n=200$ (thus $d=2$, $k=20$, $m=0$):

![Problem U3](./images/problem_U3.webp){ .center }

The image below shows an example solution, obtained by using the `DEFAULT` solver preset over 10.000 iterations
using the L2 distance metric and the `geomean_separation` diversity metric:

![Problem U3 with Solution](./images/problem_U3_with_solution.webp){ .center }

### C. Separation statistics

The image below shows distribution of vector separations (distances to nearest neighbor for all vectors in the population),
for different problem sizes:

![Problem U3 - Vector Separations](./images/problem_U3_separations.webp){ .center75 }

## II. Benchmark results

- [Initialization Strategies](bm_problem_u3_init.md)
- [Optimization Strategies](bm_problem_u3_optim.md)
- [Solver Presets](bm_problem_u3_presets.md)
