---
solver:
  name: SCIP (PySCIPOpt)
  source: https://www.scipopt.org/
  verified: 2026-07-27
  scale:
    max_practical_n: "3"
    rationale: >-
      The limit differs sharply by objective. A max-min model with big-M linearization is
      routinely solved to optimality around n ≈ 10³; the mean-of-NN objective needs an assignment
      MILP whose size grows far faster and which stops being tractable near n ≈ 60 — below the
      floor of the benchmark generators used here. The single value reported is the max-min one,
      because that is the formulation this comparison actually exercises.
  metadata:
    guarantee: proven optimum
    license: Apache-2.0 (SCIP)
    last_release: {version: "6.2.1", date: "2026-05-16"}
    determinism: deterministic
    input: a hand-built model
    notes:
      license:
        text: >-
          The two halves carry different licenses: the SCIP solver itself is Apache-2.0, while the
          PySCIPOpt binding installed from PyPI is MIT.
        url: https://pypi.org/project/PySCIPOpt/
  capabilities:
    distance.l1:
      mark: partial
      note: &coefficients
        text: >-
          Reachable, but you build the model: distances enter as precomputed objective
          coefficients, so the metric is whatever you computed before the solver ever sees it.
    distance.l2:
      mark: partial
      note: *coefficients
    distance.linf:
      mark: partial
      note: *coefficients
    distance.cosine:
      mark: partial
      note: *coefficients
    distance.custom:
      mark: full
      note:
        text: >-
          Since every distance is a coefficient you supply, an arbitrary metric costs nothing
          extra — this is the one distance axis where a modelling solver is at no disadvantage.
    objective.max_min:
      mark: partial
      note: &linearize
        text: >-
          Reachable, but the objective must be linearized by hand — for max-min, a threshold
          variable bounded below every selected pair via big-M constraints.
    objective.mean_nn:
      mark: partial
      note: &assignment
        text: >-
          Reachable only through an assignment MILP that pairs each selected item with its
          nearest selected neighbor; the formulation is considerably larger than the max-min one
          and is what drives the practical size limit down.
    objective.geomean_nn:
      mark: partial
      note: *assignment
    objective.max_sum:
      mark: partial
      note: *linearize
    constraints.disjoint_groups:
      mark: partial
      note: &linear_constraints
        text: >-
          Reachable as linear constraints over the selection variables, which you write yourself.
          Any counting constraint expressible that way is available.
    constraints.overlapping_groups:
      mark: partial
      note: *linear_constraints
    constraints.ranged_counts:
      mark: partial
      note: *linear_constraints
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
          The incumbent improves as the branch-and-bound search proceeds, but that is a proof
          search rather than an anytime budget: progress is uneven, and time spent may go entirely
          into tightening the bound rather than improving the solution.
    parallelism.independent: {mark: none}
    parallelism.cooperative:
      mark: partial
      note:
        text: >-
          Reachable through SCIP's concurrent solving and the FiberSCIP / UG frameworks, which
          share bounds and solutions between the racing solvers but need a TPI-enabled build; the
          default PySCIPOpt wheel does not expose it.
---

# SCIP (PySCIPOpt)

SCIP is a general-purpose mixed-integer programming solver, used here through its Python
interface PySCIPOpt. It is not a diversity tool: it solves whatever model you hand it, which
means the diversity problem has to be written as a MILP before SCIP is involved at all.

That framing explains most of its profile below. Distance metrics, objectives and constraints
are all *reachable* rather than *provided* — reachable by writing the right model, at whatever
size that model turns out to be tractable. What SCIP offers in exchange is the one thing no
heuristic can: when it finishes, the answer is provably optimal.

In this comparison it serves as the optimality reference for the small instances, not as a
competitor. Below roughly a thousand items it establishes what the best achievable objective
value actually is; above that, it stops being able to answer in reasonable time.

## Problem targeted

Given $n$ items, a distance $d(i,j)$ between them, and a target size $k$, SCIP solves any
selection problem whose objective and constraints you can linearize:

$$
\max_{S \subseteq \{1,\dots,n\},\; |S| = k} f(S)
\quad\text{subject to your linear constraints,}
$$

where $f$ must be supplied in linearized form. For the max-min objective this is the standard
threshold formulation, introducing $t \le d(i,j)$ for every selected pair:

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

**Guarantee: proven optimum.** SCIP terminates with a certificate that no better selection
exists, which is what distinguishes it from every heuristic in this comparison.

## Reference

--8<-- "generated/features/scip.md"
