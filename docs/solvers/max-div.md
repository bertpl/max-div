---
solver:
  name: max-div
  source: https://max-div.readthedocs.io/
  verified: 2026-07-27
  scale:
    max_practical_n: "4-5"
    rationale: >-
      Bounded by the pairwise distance matrix, which is materialized once at O(n²)/2 float32
      entries; beyond n ≈ 10⁵ that no longer fits comfortably in memory. The solver itself is
      anytime, so the practical ceiling is memory rather than time.
  metadata:
    guarantee: heuristic
    license: Apache-2.0
    last_release: {version: "0.8.3", date: "2026-07-27"}
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
    budget.iterations: {mark: full}
    budget.wall_clock: {mark: full}
    budget.improves: {mark: full}
---

# max-div

This record exists so that max-div has a row in the capability grid and the comparison table.
It is deliberately excluded from the third-party solver reference — see the note on `reference`
in `data/solver_registry.yaml` — and is not published as a page.
