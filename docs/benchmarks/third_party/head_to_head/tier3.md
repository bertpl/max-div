# Comparison Benchmarks — vs. MDPLIB Best-Known Values

How does `max-div` fare on the *literature's* shared benchmark? The MMDP instance sets
(Glover / Geo / Ran) and their published best-known max-min values come from Resende,
Martí, Gallego & Duarte (2010), *"GRASP and path relinking for the max–min diversity
problem"* (Computers & Operations Research 37(3)), hosted with the
[MDPLIB library](https://www.uv.es/rmarti/paper/mdp.html). Instances are fetched at run
time and never redistributed; the reference values are vendored in the repo so the
comparison is pinned.

## Protocol

- Objective: `MIN_SEPARATION` (the MMDP objective). All 195 published (instance, k)
  pairings: 75 Glover (n ≤ 30), 60 Geo and 60 Ran (n ∈ {100, 250, 500}).
- max-div runs the wall-clock budget ladder (2× steps from 1 ms, `SMART` preset, 3 seeds)
  to ~2 s per pairing, extended to ~16 s on the n = 500 instances — calibration showed
  quality plateaus well inside 1 s on the smaller ones.
- Hardware: 16" MacBook Pro with M3-class CPU, single sequential run.
- Reproduce with `uv run --group benchmarks python -m benchmarks.tier3.full` (records),
  then `... -m benchmarks.tier3.report` (tables).
- max-div figures measured against **v0.10.1**; the reference values are the published
  2010 ones, vendored.

## Reading the reference values honestly

The published values are the best over the six algorithms of the 2010 paper. They are a
**historical reference, not today's state of the art**: later work (e.g. Porumbel, Hao &
Glover 2011) improved some bounds, and — as the Ran results below show — max-div itself
exceeds the published value on parts of that set. The claim on this page is therefore
*"gap at budget t against the published 2010 values"*, never "matches modern SOTA".
Two of the three sets are also reported as counts rather than percentage gaps: the Ran
distances are integers in [1, 200], so at small reference values one integer step is a
double-digit "percentage", and the Glover instances are small enough that every published
algorithm agrees on every value.

## Geo — percentage gap to the published values

The discriminating set: Euclidean instances where a residual gap exists and budget moves
it. Gap of the best-of-3-seeds solution (positive = below the published value):

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier3_geo.md"

## Ran — matched / exceeded / below counts

Ran's distances are integers in [1, 200], so results are reported as counts (best over
seeds and budgets), not percentage gaps. The picture splits by size:

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier3_ran.md"

On the larger instances (n ≥ 250) max-div matches the reference on most — all 10 of 10 at
n = 500, k = 150 — and **exceeds** it on a couple: where "exceeded" is non-zero it found a
strictly better max-min value than the published one, direct evidence the 2010 values are
not tight there. On the small, dense n = 100 instances it trails on a fair share: those
solutions are fully converged well within budget (they plateau by ~16 ms), so the
shortfall is a genuine quality gap, not a matter of more time. This is the honest shape of
the result — max-div is strong on the harder, larger instances and mixed on the small
dense ones.

## Glover — matched / exceeded counts

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier3_glover.md"

These instances (n ≤ 30) are trivial for modern methods; they are included for
completeness of the published benchmark, not as a discriminating test.
