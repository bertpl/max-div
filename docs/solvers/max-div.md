---
solver:
  name: max-div
  source: https://max-div.readthedocs.io/
  verified: 2026-08-11
  scale:
    max_n_memory: 50000000
    max_n_time: 1000000
    max_n_quality: pending
  metadata:
    guarantee: heuristic
    license: Apache-2.0
    last_release: {version: "0.11.0", date: "2026-08-11"}
    determinism: seeded, fully reproducible
    input: vectors (L1 / L2 / L2² / Linf / cosine) or a precomputed distance matrix
  capabilities:
    distance.l1: {mark: full}
    distance.l2: {mark: full}
    distance.linf: {mark: full}
    distance.cosine: {mark: full}
    distance.custom:
      mark: full
      note:
        text: >-
          A problem can be built from a precomputed distance matrix, square or condensed, so any
          metric the caller can compute is usable without max-div implementing it.
    objective.max_min: {mark: full}
    objective.mean_nn: {mark: full}
    objective.geomean_nn: {mark: full}
    objective.max_sum: {mark: full}
    constraints.disjoint_groups: {mark: full}
    constraints.overlapping_groups: {mark: full}
    constraints.ranged_counts:
      mark: full
      note:
        text: >-
          Counts are per-group minimum and maximum bounds, and carry per-constraint weights and a
          choice of linear or quadratic penalty shaping when a problem is over-constrained.
    constraints.feasibility_proofs:
      mark: full
      note:
        text: >-
          `check_feasibility` returns a three-valued verdict whose definite answers are proofs.
          Feasible comes with a satisfying selection; infeasible with an independently
          re-checkable certificate. Unknown claims nothing.
    budget.iterations: {mark: full}
    budget.wall_clock: {mark: full}
    budget.improves: {mark: full}
    parallelism.independent:
      mark: full
      note:
        text: >-
          Several workers solve the same problem at once and the best result wins.
    parallelism.cooperative:
      mark: full
      note:
        text: >-
          Workers form groups whose members adopt the best selection any member has found so
          far; groups stay independent of each other, and the best result over all workers wins.
---

# max-div

This record exists so that max-div has a row in the capability grid and the comparison table.
It is deliberately excluded from the third-party solver reference — see the note on `reference`
in `data/solver_registry.yaml` — and is not published as a page.
