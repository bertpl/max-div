---
solver:
  name: HiGHS
  source: https://ergo-code.github.io/HiGHS/
  verified: 2026-07-27
  scale:
    max_n_memory: 2000
    max_n_time: 2000
    max_n_quality_50: 200
    max_n_quality_90: 200
  metadata:
    guarantee: proven optimum
    license: MIT
    last_release: {version: "1.15.1", date: "2026-07-02"}
    determinism: deterministic
    input: a hand-built model
  capabilities:
    distance.l1:
      mark: partial
      note: &coeffs
        text: >-
          Reachable, but you build the model: distances enter as precomputed objective coefficients, so the metric is whatever you computed before the solver ever sees it.
    distance.l2:
      mark: partial
      note: *coeffs
    distance.linf:
      mark: partial
      note: *coeffs
    distance.minkowski:
      mark: partial
      note: *coeffs
    distance.cosine:
      mark: partial
      note: *coeffs
    distance.custom:
      mark: full
      note:
        text: >-
          Distances are coefficients you supply, so an unusual metric costs no more than a common one.
    objective.max_min:
      mark: partial
      note: &linearize
        text: >-
          Reachable, but the objective must be linearized by hand — for max-min, a threshold variable bounded below every selected pair via big-M constraints.
    objective.mean_nn:
      mark: partial
      note: &assignment
        text: >-
          Reachable only through an assignment MILP that pairs each selected item with its nearest selected neighbor; the formulation is considerably larger than the max-min one and is what drives the practical size limit down.
    objective.geomean_nn:
      mark: partial
      note: *assignment
    objective.max_sum:
      mark: partial
      note: *linearize
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
    parallelism.independent: {mark: none}
    parallelism.cooperative:
      mark: full
      note:
        text: >-
          The dual simplex, the interior-point factorization and the MIP branch-and-bound each run
          in parallel, controlled by the parallel and threads options — threads jointly advancing
          one solve rather than racing independent ones.
---

# HiGHS

HiGHS is an open-source linear and mixed-integer programming solver. Its profile here is close to
SCIP's, for the same reason: both are general MILP engines, so the diversity problem has to be
written as a model before either is involved, and the same linearizations apply to both.

Where it differs is licensing and footprint. HiGHS is MIT-licensed and comparatively light to
install, which makes it the easier of the two to embed in a project that only occasionally needs an
exact answer.

It is included as a second optimality reference rather than as a distinct approach. If the two
disagree on a small instance, one of them has a bug — which is exactly why having two is useful.

## Problem targeted

Identical in form to any MILP-based selection: given $n$ items, a distance $d(i,j)$ and a size $k$,

$$
\max_{S \subseteq \{1,\dots,n\},\; |S| = k} f(S),
$$

with $f$ supplied in linearized form. For max-min that is the big-M threshold formulation:

$$
\max\; t
\qquad
t \le d(i,j) + M\,(2 - x_i - x_j)
\quad \forall\, i < j,
\qquad
\sum_i x_i = k,
\qquad
x_i \in \{0,1\}.
$$

**Guarantee: proven optimum.** On termination the reported selection is optimal for the model as
written — which makes the modeling, not the solving, the place errors live.

## Reference

--8<-- "generated/features/highs.md"
