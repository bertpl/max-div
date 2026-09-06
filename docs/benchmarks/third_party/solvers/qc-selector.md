---
solver:
  name: qc-selector
  source: https://selector.qcdevs.org/
  verified: 2026-07-27
  scale:
    max_n_memory: 20000
    max_n_time: 20000
    max_n_quality_50: 20000
    max_n_quality_90: none
  metadata:
    guarantee: heuristic
    license: GPL-3.0
    last_release: {version: "0.1.4", date: "2026-02-03"}
    determinism: seeded
    input: vectors or a distance matrix
  capabilities:
    distance.l1: {mark: full}
    distance.l2: {mark: full}
    distance.linf:
      mark: full
      note:
        text: >-
          Available through the same Minkowski-exponent parameter that provides L1: its radius-based
          methods forward `p` to scipy, which treats p=∞ as the Chebyshev norm; the greedy pickers
          accept any metric as a precomputed distance matrix or callable.
    distance.minkowski:
      mark: full
      note:
        text: >-
          The same Minkowski-exponent parameter that provides L1 and Chebyshev takes any p.
    distance.cosine:
      mark: partial
      note:
        text: >-
          Reachable by L2-normalizing the vectors first: on the unit sphere, cosine distance is a monotone function of Euclidean distance, so a Euclidean picker returns the same ordering.
    distance.geomean:
      mark: partial
      note:
        text: >-
          The Minkowski exponent cannot express the geometric mean (the p → 0 limit of the power
          mean, which divides the sum by d before the root — not a Minkowski form), so the
          geometric mean is reachable only as a precomputed distance matrix.
    distance.custom:
      mark: full
      note:
        text: >-
          A precomputed distance matrix is accepted directly, so any metric you can compute is usable.
    objective.max_min: {mark: full}
    objective.mean_nn:
      mark: none
      note: &no_nn_family
        text: >-
          Its diversity measures are computed over the whole selection rather than over nearest-neighbor pairs, so the nearest-neighbor family is absent.
    objective.geomean_nn:
      mark: none
      note: *no_nn_family
    objective.max_sum: {mark: full}
    objective.optimality_proofs: {mark: none}
    constraints.disjoint_groups:
      mark: full
      note:
        text: >-
          Label-stratified selection: given class labels, it picks proportionally across them. That covers disjoint groups with proportional targets, but not arbitrary minimum and maximum counts, and not groups an item can belong to more than once.
    constraints.overlapping_groups: {mark: none}
    constraints.ranged_counts: {mark: none}
    constraints.feasibility_proofs: {mark: none}
    budget.iterations:
      mark: partial
      note:
        text: >-
          Several of its methods take a parameter that controls how much work they do — a sphere-exclusion radius, an OptiSim subsample size — but these change the character of the search rather than lengthening it. Raising one does not mean a better answer.
    budget.wall_clock: {mark: none}
    budget.improves:
      mark: none
      note:
        text: >-
          A single construction pass, so there is no budget to spend: the answer is whatever one greedy sweep produces, and waiting longer does not change it.
    parallelism.independent:
      mark: none
      note:
        text: >-
          MaxMin and MaxSum selection is sequential; OptiSim and DISE parallelize only their KDTree
          neighbor queries, not the search over selections.
    parallelism.cooperative: {mark: none}
---

# qc-selector

qc-selector, from the QC-Devs group, is a collection of subset-selection methods aimed at chemical
datasets: max-min and max-sum pickers, OptiSim, sphere exclusion, and several diversity measures
for scoring a selection after the fact.

It is the broadest of the one-shot tools — the only one here offering both a max-min and a max-sum
picker, plus label-stratified selection when your items carry classes. That last feature is the
closest anything in this comparison comes to max-div's fairness constraints, and the gap is worth
being precise about: proportional picks across labels are not the same as minimum and maximum
counts per group, and its groups cannot overlap.

Two practical notes. It is **GPL-3 licensed** — as is kmedoids; every other tool surveyed is more permissive —
which may decide the question for you. And it is young — first released in 2025 — so
its API is less settled than the others'.

## Problem targeted

Its max-min picker targets the same dispersion objective as the other pickers,

$$
\max_{|S| = k} \; \min_{i \ne j \in S} d(i,j),
$$

while its max-sum picker targets

$$
\max_{|S| = k} \; \sum_{i \ne j \in S} d(i,j),
$$

both by greedy construction. Sphere exclusion and OptiSim instead select by a radius or a sampled
candidate pool, which are procedures rather than objectives — they define what gets picked without
defining a quantity being maximized.

**Guarantee: heuristic.** The greedy max-min picker inherits the usual 2-approximation, but the
library as a whole makes no optimality claims, and its procedural methods have no objective for a
bound to be stated against.

## Reference

--8<-- "generated/features/qc-selector.md"
