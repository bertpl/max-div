---
solver:
  name: OR-Tools CP-SAT
  source: https://developers.google.com/optimization/cp/cp_solver
  verified: 2026-07-27
  scale:
    max_n_memory: 20000
    max_n_time: 20000
    max_n_quality_50: 1000
    max_n_quality_90: 1000
  metadata:
    guarantee: proven optimum
    license: Apache-2.0
    last_release: {version: "9.15.6755", date: "2026-01-14"}
    determinism: deterministic with a fixed worker count
    input: a hand-built model
  capabilities:
    distance.l1:
      mark: partial
      note: &integer_scaled
        text: >-
          Reachable, but CP-SAT is an integer solver: distances must be scaled to integers before they enter the model, so the achievable precision is a modeling choice rather than a property of the metric.
    distance.l2:
      mark: partial
      note: *integer_scaled
    distance.linf:
      mark: partial
      note: *integer_scaled
    distance.cosine:
      mark: partial
      note: *integer_scaled
    distance.minkowski:
      mark: partial
      note: *integer_scaled
    distance.custom:
      mark: full
      note:
        text: >-
          Any distance you can compute and round to integers is usable, which in practice means any metric at all.
    objective.max_min:
      mark: partial
      note:
        text: >-
          Reachable through a threshold feasibility search: ask whether a selection exists with every pair at least t apart, then binary-search t. It is a natural fit for CP-SAT and the reason this solver appears here at all.
    objective.mean_nn:
      mark: partial
      note: &no_cp_encoding
        text: >-
          No natural constraint-programming encoding. Expressing a nearest-neighbor mean needs the same auxiliary assignment structure a MILP would use, at which point a MILP solver is the better tool.
    objective.geomean_nn:
      mark: partial
      note: *no_cp_encoding
    objective.max_sum:
      mark: partial
      note: *no_cp_encoding
    objective.optimality_proofs: {mark: full}
    constraints.disjoint_groups:
      mark: partial
      note: &linear_cons
        text: >-
          Reachable as linear constraints over the selection variables, which you write yourself. Any counting constraint expressible that way is available.
    constraints.overlapping_groups:
      mark: partial
      note: *linear_cons
    constraints.ranged_counts:
      mark: partial
      note: *linear_cons
    constraints.feasibility_proofs:
      mark: partial
      note:
        text: >-
          A feasibility-only solve of the hand-built constraint model returns a proven
          feasible-or-infeasible verdict: the solver itself produces the proof; only the model
          is yours to write.
    budget.iterations: {mark: full}
    budget.wall_clock: {mark: full}
    budget.improves:
      mark: partial
      note:
        text: >-
          The incumbent improves as the branch-and-bound search proceeds, but that is a proof search rather than an anytime budget: progress is uneven, and time spent may go entirely into tightening the bound rather than improving the solution.
    parallelism.independent:
      mark: full
      note:
        text: >-
          CP-SAT runs a portfolio of differently configured search workers on the same instance by
          default, set by the num_workers parameter.
    parallelism.cooperative:
      mark: full
      note:
        text: >-
          The portfolio workers share learned clauses, solutions and objective bounds while
          solving.
---

# OR-Tools CP-SAT

CP-SAT is Google's constraint-programming solver, and like every exact solver here it is not a
diversity tool: you write a model, it proves an answer. What makes it worth a profile of its own is
that one diversity objective suits it unusually well.

Max-min is a *threshold* question in disguise. Rather than maximizing a minimum, you ask whether a
selection exists in which every pair is at least $t$ apart — pure feasibility, which is what
constraint propagation is built for — and then binary-search $t$. That plays to CP-SAT's strengths
in a way the mean-based objectives do not.

The catch is integrality. CP-SAT reasons over integers, so distances must be scaled and rounded
before the model is built, and the scaling factor becomes a precision decision you own.

## Problem targeted

Given $n$ items, a distance $d(i,j)$, and a target size $k$, the max-min formulation is a sequence
of feasibility problems parameterized by a threshold $t$:

$$
\exists\, S \subseteq \{1,\dots,n\},\; |S| = k
\quad\text{such that}\quad
d(i,j) \ge t \;\; \forall\, i,j \in S,\; i \ne j.
$$

The largest $t$ for which this is satisfiable is the max-min optimum. In practice the search is
driven by a binary search over $t$ with each feasibility question handed to the solver.

**Guarantee: proven optimum.** Each feasibility answer is exact, so the resulting threshold is the
true optimum rather than a bound on it.

## Reference

--8<-- "generated/features/ortools-cpsat.md"
