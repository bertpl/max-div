# Solver Benchmarking Problems

The package comes with 5 built-in benchmarking problems (each parametrized by size), which can be triggered [via the CLI](../../cli.md) and allow testing of initialization
and optimization strategies & presets under controlled conditions.

### High-level overview

We have 4 unconstrained (`U1-4`) and 4 constrained (`C1-4`) benchmark problems:

| Problem Name | $d$ | $n$    | $k$    | $m$  | Description                                                          | Results                  |
|--------------|-----|--------|--------|------|----------------------------------------------------------------------|--------------------------|
| `U1`         | $s$ | $100s$ | $10s$  | $0$  | Unconstrained, uniform vector density                                | [link](bm_problem_u1.md) |
| `U2`         | $s$ | $100s$ | $10s$  | $0$  | Unconstrained, non-uniform vector density (Gaussian)                 | [link](bm_problem_u2.md) |
| `U3`         | $s$ | $100s$ | $10s$  | $0$  | Unconstrained, non-uniform vector density (Exponential)              | [link](bm_problem_u3.md) |
| `U4`         | $s$ | $100s$ | $10s$  | $0$  | Unconstrained, non-uniform vector density (Conic)                    | [link](bm_problem_u4.md) |
| `C1`         | $2$ | $100s$ | $10s$  | $2s$ | semi-uniform vector density, simple non-overlapping constraints      | [link](bm_problem_c1.md) |
| `C2`         | $2$ | $100s$ | $10s$  | $2s$ | semi-uniform vector density, medium-hard non-overlapping constraints | [link](bm_problem_c2.md) |
| `C3`         | $s$ | $150s$ | $10s$  | $2s$ | non-uniform vector density, overlapping constraints                  | [link](bm_problem_c3.md) |
| `C4`         | $s$ | $150s$ | $10s$  | $3s$ | non-uniform vector density, strongly coupled constraints             | [link](bm_problem_c4.md) |

**Legend:**

- $s$: size parameter ($1, 2, \ldots$)
- $d$: dimensionality of vectors
- $n$: number of vectors to choose from
- $k$: number of vectors to select
- $m$: number of constraint groups

## Unconstrained test problems

### Test Problem `U1`

Vectors are drawn from a uniform distribution over $[0, 1]^d$.

### Test Problem `U2`

Vector components are drawn from a standard normal distribution $\mathcal{N}(0, 1)$.

### Test Problem `U3`

Vectors are drawn from a uniform distribution over $[0, 1]^d$ and then exponentially scaled to range $[0.1, 10]^d$

### Test Problem `U4`

Samples are drawn in a cone-shaped region around the line spanning $(0,0,\ldots,0) \in \mathbb{R}^d$ to $(1,1,\ldots,1) \in \mathbb{R}^d$, where each sample is constructed as follows:

- choose random center point $c$ as $(a,a,\ldots,a)$ with $a$ drawn from a uniform distribution over $[0, 1]$
- determine radius $r = \frac{1}{10}\sqrt{d}$   (ensuring a cone with constant angular width as $d$ increases)
- choose point randomly at a distance $r$ from $c$ in a random direction


## Constrained test problems

### Test Problem `C1`

Vectors are 2-dimensional and have...

- first component drawn from a uniform distribution over $[0, 1]$,
- second component drawn from a standard normal distribution $\mathcal{N}(0, 1)$.

All vectors are split in $m=2s$ non-overlapping groups, by splitting the range $[0, 1]$ of the first component into $m$ equal segments.
From each group between $4$ and $k$ vectors need to be selected.  This is always feasible, since $k=5m$ and $n=50m$.

### Test Problem `C2`

Identical to test problem `C1`, but with group constraint boundaries changed from $[4, k]$ to $[5,5]$.

### Test Problem `C3`

Vector components are drawn from a standard normal distribution $\mathcal{N}(0.5, 1)$, i.e. centered around $0.5$ and therefore have
a probability of approximately $\sim 69\%$ to be positive and $\sim 31\%$ to be negative.

In each of the $d=s$ dimensions, we split the vectors in $2$ groups, based on the sign of that dimension's vector component:

- between $\frac{4}{10}k$ and $k$ vectors with positive component in that dimension need to be selected
- between $\frac{4}{10}k$ and $k$ vectors with negative component in that dimension need to be selected

For example, for $s=2$ ($d=2, n=300$) we expect group sizes of approximately $207$ and $93$ vectors, from each of which $8$ out of $k=20$ vectors need to be sampled.

Note that in a single dimension, groups of that dimension do not overlap, but across dimensions, there is an intricate overlap
between groups, creating $2^d$ possible combinations of group membership, most of which are expected to be empty for large $d$.

### Test Problem `C4`

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