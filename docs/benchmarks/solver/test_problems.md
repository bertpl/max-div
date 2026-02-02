# Solver Benchmarking Problems

The package comes with 5 built-in benchmarking problems (each parametrized by size), which can be triggered [via the CLI](../../cli.md) and allow testing of initialization
and optimization strategies & presets under controlled conditions.

## High-level overview

We have 4 unconstrained (`U1-U4`) and 4 constrained (`C1-C4`) benchmark problems.  Detailed descriptions of each problem
can be found by following the provided links.

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
