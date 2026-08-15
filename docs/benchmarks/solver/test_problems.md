# Solver Benchmarking Problems

The package comes with 8 built-in benchmarking problems, each parametrized directly by the problem size $n$ (all other dimensions derive from it), which can be triggered [via the CLI](../../cli.md) and allow testing of initialization
and optimization strategies & presets under controlled conditions.

## High-level overview

We have 4 unconstrained (`U1-U4`) and 4 constrained (`C1-C4`) benchmark problems:

- **`U1` and `C1` are the reference problems for cross-tool comparisons**: fixed at $d=2$ — and, for `C1`, using the most restrictive constraint form (exact stratified quotas over a partition) — so that every third-party subset-selection tool can run them.
- **The remaining problems carry each series' native difficulty axis**: increasingly non-uniform density for `U2-U4` (dimensionality scaling with $n$), increasingly free-form constraints for `C2-C4`.

Detailed descriptions of each problem can be found by following the provided links.

| Problem Name | $d$                    | $k$                   | $m$                  | Description                                                          | Results                  |
|--------------|------------------------|-----------------------|----------------------|----------------------------------------------------------------------|--------------------------|
| `U1`         | $2$                    | $\lceil n/10 \rceil$  | $0$                  | Clustered 2D density + background + outlier halo (cross-tool ref.)   | [link](bm_problem_u1.md) |
| `U2`         | $\lceil n/100 \rceil$  | $\lceil n/10 \rceil$  | $0$                  | Uniform vector density                                               | [link](bm_problem_u2.md) |
| `U3`         | $\lceil n/100 \rceil$  | $\lceil n/10 \rceil$  | $0$                  | Non-uniform vector density (Gaussian)                                | [link](bm_problem_u3.md) |
| `U4`         | $\lceil n/100 \rceil$  | $\lceil n/10 \rceil$  | $0$                  | Non-uniform vector density (Conic)                                   | [link](bm_problem_u4.md) |
| `C1`         | $2$                    | $\lceil n/10 \rceil$  | $\lceil k/5 \rceil$  | Exact per-band quotas; non-overlapping partition (cross-tool ref.)   | [link](bm_problem_c1.md) |
| `C2`         | $2$                    | $\lceil n/10 \rceil$  | $\lceil k/5 \rceil$  | Per-band lower bounds; non-overlapping partition                     | [link](bm_problem_c2.md) |
| `C3`         | $\lceil n/150 \rceil$  | $\lceil n/15 \rceil$  | $2d$                 | Non-uniform vector density, overlapping constraints                  | [link](bm_problem_c3.md) |
| `C4`         | $\lceil n/150 \rceil$  | $\lceil n/15 \rceil$  | $3d$                 | Non-uniform vector density, strongly coupled constraints             | [link](bm_problem_c4.md) |

**Legend:**

- $n$: number of vectors to choose from — the problem's single size parameter (any integer $\geq 20$, the enforced minimum)
- $d$: dimensionality of vectors
- $k$: number of vectors to select
- $m$: number of constraint groups
