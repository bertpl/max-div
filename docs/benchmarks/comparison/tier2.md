# Comparison Benchmarks — vs. Python Heuristics

How does `max-div` compare to the subset-selection tools a Python user would otherwise
reach for? This page benchmarks it against the surveyed single-shot heuristics (see the
[Comparison](../../comparison.md) page for the tool landscape) on the built-in
[benchmark problems](../solver/test_problems.md).

## Protocol

- **max-div** runs a wall-clock budget ladder (2× steps from 1 ms; the last rung is the
  first ≥ 10 s), one independent solve per budget × seed (3 seeds), `DEFAULT` preset (an
  alias of `SMART` — figures label it `max-div[DEFAULT]`). Figures
  plot *measured* solve time, never the nominal budget. One ladder per diversity metric —
  max-div optimizes the metric it is scored under.
- **Competitors** are single-shot: one run per seed, plotted as a dot at (measured time,
  quality).
- **Every** tool's selection is scored under max-div's own diversity metrics, computed
  identically for all tools.
- Labels matter: `apricot[facility-location]` optimizes coverage and `kmedoids[FasterPAM]`
  representativeness — they are included as different-objective references, not as
  dispersion competitors. `qc-selector` (GPL) is included when installed.
- Hardware: 16" MacBook Pro with M3-class CPU, single sequential run.
- Reproduce with `uv run --group benchmarks python -m benchmarks.tier2.full` (records),
  then `... -m benchmarks.tier2.report` (figures/tables).

## Key findings

**There is a crossover, and it is driven by k (the selection size).** max-div wins clearly
on the separation objectives up to n ≈ 1000, wins narrowly at n = 5000 given ~10 s, and
loses at n = 20000 (k = 2000) — there, farthest-point-sampling pickers reach better
separation values in a few seconds than the local search reaches at any tried budget. At
k = 2000, a 1 s budget fits only a few hundred iterations: not even one improvement pass
over the selected set.

**On `MEAN_PAIRWISE_DISTANCE` (classical max-sum), the greedy construction is already at
par.** max-div matches `greedy[max-sum]` at small-to-mid sizes and trails slightly at the
largest; it does not beat it. If pure max-sum at scale is the goal, the greedy baseline is
the pragmatic choice.

**Where max-div is differentiated:** the separation-family objectives below n ≈ 5000, the
anytime property (near-plateau quality within ~1 s at most sizes), and constrained
selection (below) — where the heuristic field thins out to a single competitor that stops
at n ≈ 2000.

## Unconstrained results (U1, uniform density)

Anytime curves on U1 for the max-min and geomean objectives; the margin tables further
down cover all problems and metrics.

### MIN_SEPARATION

![U1 size 2](./images/tier2_U1_2_min_separation.svg)
![U1 size 10](./images/tier2_U1_10_min_separation.svg)
![U1 size 50](./images/tier2_U1_50_min_separation.svg)
![U1 size 200](./images/tier2_U1_200_min_separation.svg)

### GEOMEAN_SEPARATION

![U1 size 2](./images/tier2_U1_2_geomean_separation.svg)
![U1 size 10](./images/tier2_U1_10_geomean_separation.svg)
![U1 size 50](./images/tier2_U1_50_geomean_separation.svg)
![U1 size 200](./images/tier2_U1_200_geomean_separation.svg)

## Margin tables (all problems)

Each cell: max-div's margin vs. the *best* competitor under that metric, at ~1 s / ~16 s
of budget. Positive = max-div ahead.

### MIN_SEPARATION

--8<-- "docs/benchmarks/comparison/results/tier2_margins_min_separation.md"

### MEAN_SEPARATION

--8<-- "docs/benchmarks/comparison/results/tier2_margins_mean_separation.md"

### GEOMEAN_SEPARATION

--8<-- "docs/benchmarks/comparison/results/tier2_margins_geomean_separation.md"

### MEAN_PAIRWISE_DISTANCE

--8<-- "docs/benchmarks/comparison/results/tier2_margins_mean_pairwise_distance.md"

## Constrained results

With fairness constraints, the only surveyed heuristic competitor is
[code-FDM](https://github.com/yhwang1990/code-FDM) (FairFlow), and it bounds the comparison
twice over: it scales roughly cubically (~150 s at n = 2000, impractical beyond), and its
disjoint-color fairness model cannot even *express* overlapping constraint groups — so the
harder constrained problems (C3/C4) have no competitor at all. The comparison therefore
covers C1/C2 up to n = 2000; beyond either limit, no surveyed Python heuristic produces
constraint-satisfying selections (exact solvers can, at small n — see the
[exact-reference benchmarks](tier1.md)).

![C1 size 2](./images/tier2_C1_2_min_separation.svg)
![C1 size 10](./images/tier2_C1_10_min_separation.svg)
![C1 size 20](./images/tier2_C1_20_min_separation.svg)

--8<-- "docs/benchmarks/comparison/results/tier2_margins_constrained.md"
