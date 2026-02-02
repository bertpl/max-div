# Benchmark Results - Problem U2

## I. Problem Description

### A. Overall Approach

Vector components are drawn from a standard normal distribution $\mathcal{N}(0, 1)$.

### B. Visualization

This image shows problem U2 with size parameter $s=2$ (thus $d=2$, $n=200$, $k=20$, $m=0$):

![Problem U2](./images/problem_U2.png){ .center }

The image below shows an example solution, obtained by using the `DEFAULT` solver preset over 10.000 iterations 
using the L2 distance metric and the `geomean_separation` diversity metric:

![Problem U2 with Solution](./images/problem_U2_with_solution.png){ .center }

### C. Separation statistics

The image below shows distribution of vector separations (distances to nearest neighbor for all vectors in the population),
for different problem sizes:

![Problem U2 - Vector Separations](./images/problem_U2_separations.png){ .center75 }

## II. Benchmark results

- [Initialization Strategies](bm_problem_u2_init.md)
- [Optimization Strategies](bm_problem_u2_optim.md)
- Solver Presets (TODO)
