# Solver Benchmarking Problems

The package comes with 5 built-in benchmarking problems (each parametrized by size), which can be triggered [via the CLI](../../cli.md) and allow testing of initialization
and optimization strategies & presets under controlled conditions.

### High-level overview

| Problem Name | $d$ | $n$    | $k$    | $m$  | Description                                                     | Results                  |
|--------------|-----|--------|--------|------|-----------------------------------------------------------------|--------------------------|
| `A1`         | $s$ | $100s$ | $10s$  | $0$  | Unconstrained, uniform vector density                           | [link](bm_problem_a1.md) |
| `A2`         | $s$ | $100s$ | $10s$  | $0$  | Unconstrained, non-uniform vector density                       | [link](bm_problem_a2.md) |
| `A3`         | $2$ | $100s$ | $10s$  | $2s$ | semi-uniform vector density, simple non-overlapping constraints | [link](bm_problem_a3.md) |
| `A4`         | $s$ | $150s$ | $10s$  | $2s$ | non-uniform vector density, overlapping constraints             | [link](bm_problem_a4.md) |
| `A5`         | $s$ | $150s$ | $10s$  | $3s$ | non-uniform vector density, strongly coupled constraints        | [link](bm_problem_a5.md) |

**Legend:**

- $s$: size parameter ($1, 2, \ldots$)
- $d$: dimensionality of vectors
- $n$: number of vectors to choose from
- $k$: number of vectors to select
- $m$: number of constraint groups

## Test Problem `A1`

Vectors are drawn from a uniform distribution over $[0, 1]^d$.

## Test Problem `A2`

Vector components are drawn from a standard normal distribution $\mathcal{N}(0, 1)$.

## Test Problem `A3`

Vectors are 2-dimensional and have...

- first component drawn from a uniform distribution over $[0, 1]$,
- second component drawn from a standard normal distribution $\mathcal{N}(0, 1)$.

All vectors are split in $m=2s$ non-overlapping groups, by splitting the range $[0, 1]$ of the first component into $m$ equal segments.
From each group between $4$ and $k$ vectors need to be selected.  This is always feasible, since $k=5m$ and $n=50m$.

## Test Problem `A4`

Vector components are drawn from a standard normal distribution $\mathcal{N}(0.5, 1)$, i.e. centered around $0.5$ and therefore have
a probability of approximately $\sim 69\%$ to be positive and $\sim 31\%$ to be negative.

In each of the $d=s$ dimensions, we split the vectors in $2$ groups, based on the sign of that dimension's vector component:

- between $\frac{4}{10}k$ and $k$ vectors with positive component in that dimension need to be selected
- between $\frac{4}{10}k$ and $k$ vectors with negative component in that dimension need to be selected

For example, for $s=2$ ($d=2, n=300$) we expect group sizes of approximately $207$ and $93$ vectors, from each of which $8$ out of $k=20$ vectors need to be sampled.

Note that in a single dimension, groups of that dimension do not overlap, but across dimensions, there is an intricate overlap
between groups, creating $2^d$ possible combinations of group membership, most of which are expected to be empty for large $d$.

## Test Problem `A5`

Vector components are drawn from a standard normal distribution $\mathcal{N}(0.5, 1)$, i.e. centered around $0.5$ and therefore...

- $\sim 69\%$ of values are expected to be positive
- $\sim 31\%$ of values are expected to be negative
- $\sim 62\%$ of values are expected to be in the range $[-1, +1]$

In each of the $d=s$ dimensions, we define $3$ groups, based on the value of that dimension's vector component:

- between $\frac{4}{10}k$ and $k$ vectors with positive component in that dimension need to be selected
- between $\frac{4}{10}k$ and $k$ vectors with negative component in that dimension need to be selected
- between $\frac{7}{10}k$ and $k$ vectors with component in the range $[-1, +1]$ in that dimension need to be selected

Note that in this example, we also have overlapping groups within a single dimension, as well as across dimensions,
creating $4^d$ possible combinations of group membership.