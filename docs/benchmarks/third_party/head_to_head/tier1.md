# Comparison Benchmarks — vs. Exact Solvers

How close does `max-div` get to *provably optimal* solutions, and what does proving
optimality cost? This page compares the heuristic quality numbers with exact references
(CP-SAT and SCIP), on the built-in [benchmark problems](../../solver/test_problems.md).

## Protocol

- **max-div** runs the same series of wall-clock budgets as the
  [heuristics comparison](tier2.md) (2× steps from 1 ms, `SMART` preset, 3 seeds).
- **Exact max-min reference:** CP-SAT via threshold binary search over the distinct
  pairwise distances (the max-min optimum is always one of them), one worker for
  determinism, 120 s cap. Gaps are only reported where optimality was *certified*.
- **Exact mean/geomean reference:** the nearest-neighbor assignment model (Lei–Church
  style), on both SCIP and CP-SAT backends. As shown below, no backend certifies it at the
  generators' sizes, so no mean/geomean gap-to-optimum is published — an
  incumbent-at-budget comparison is shown instead, clearly labeled as uncertified.
- Hardware: 16" MacBook Pro with M3-class CPU, single sequential run.
- Reproduce with `uv run --group benchmarks python -m benchmarks.tier1.full` (results),
  then `... -m benchmarks.tier1.report` (tables).
- max-div figures measured against **v0.10.1**. The exact-solver references — optima,
  proof times, incumbents — are properties of the problems, kept as tracked reference
  records.

## Max-min: gap to the proven optimum

CP-SAT certifies the max-min optimum up to n = 300 within the cap; it stops certifying
around n = 400. Constraints *help*: they prune the conflict graph. The table shows max-div's
gap to the certified optimum, mean over 3 seeds:

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_maxmin_gap.md"

Reading the table:

- max-div lands within roughly 10 % of the proven optimum in ~100 ms and ~2–9 % at the
  largest budget, occasionally hitting the optimum exactly.
- The exact solver needs up to seconds-to-minutes to *certify*, and stops certifying
  entirely a factor ~2 above these sizes.
- Max-min is the harshest gap measure: the objective is the single worst pair, so one
  suboptimal swap shows in full.

## Mean/geomean: why there is no gap-to-optimum

The canonical exact formulation for mean/geomean-of-NN separation is the NN-assignment
model. Its proofs stop far below the generators' smallest problem (n = 100), whichever
backend runs it. The table shows the time to certified optimality on a d=4 random family
(k = n/10, geomean; each backend stops at its first timeout):

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_scaling.md"

CP-SAT with 8 parallel workers is the strongest backend and still times out at n = 100
(cap 1 h; the trend suggests ~2 h would be needed, and n = 110+ is out of reach). The
model's closest-assignment constraints relax weakly, so cost grows ~4–5× per +10 items —
a property of the formulation, not of any solver. Hence: **no certified mean/geomean
optimum exists at any published problem size.**

## Geomean: incumbent-at-budget comparison (uncertified)

What a practitioner can still ask: *"what if I just gave a MIP/CP solver a big time budget
and took its best solution?"* Below, CP-SAT (8 workers) runs the assignment model at a
generous cap on two shipped problems. Its **incumbent** — best solution found, no
optimality certificate; the bound gap column shows how far the proof remained — is compared
against max-div's 1-second budget:

--8<-- "docs/benchmarks/third_party/head_to_head/results/tier1_incumbent_geomean.md"

On the unconstrained problem (U3), max-div reaches within ~1 % of the exact solver's 3-hour
incumbent in one second. On the constrained problem (C4), max-div's 1-second solution
*matches* the one the exact solver reached in 15 minutes (neither certified optimal).

Both results are point comparisons on two problems, not certified gaps. They answer the
budget-parity question the certified experiment cannot reach: given comparable or far
greater time, the exact solver does not pull ahead on these instances.
