---
solver:
  name: RDKit MaxMinPicker
  source: https://www.rdkit.org/docs/source/rdkit.SimDivFilters.rdSimDivPickers.html
  verified: 2026-07-27
  scale:
    max_practical_n: "6"
    rationale: >-
      Never materializes a distance matrix: it calls your distance function lazily and keeps one
      running nearest-selected distance per candidate, so memory is O(n) and the ceiling is set by
      how fast your callback is rather than by n² storage.
  metadata:
    guarantee: 2-approximation (farthest-point traversal)
    license: BSD-3-Clause
    last_release: {version: "2026.3.4", date: "2026-07-16"}
    determinism: seeded
    input: a caller-supplied distance function
  capabilities:
    distance.l1:
      mark: partial
      note: &callback
        text: >-
          Reachable through the distance callback you supply — RDKit itself has no opinion about the metric, it only calls your function.
    distance.l2:
      mark: partial
      note: *callback
    distance.linf:
      mark: partial
      note: *callback
    distance.cosine:
      mark: partial
      note: *callback
    distance.custom:
      mark: full
      note:
        text: >-
          The callback is the native interface rather than an escape hatch: it never needs all n² distances to exist at once, which is what lets this picker scale into the millions.
    objective.max_min: {mark: full}
    objective.mean_nn: {mark: none}
    objective.geomean_nn: {mark: none}
    objective.max_sum: {mark: none}
    constraints.disjoint_groups:
      mark: none
      note:
        text: >-
          No per-group counting of any kind. It does accept a set of items that must appear in the result, which is a different guarantee entirely: membership for named items, not proportions across groups.
    constraints.overlapping_groups: {mark: none}
    constraints.ranged_counts: {mark: none}
    budget.iterations: {mark: none}
    budget.wall_clock: {mark: none}
    budget.improves:
      mark: none
      note:
        text: >-
          A single construction pass, so there is no budget to spend: the answer is whatever one greedy sweep produces, and waiting longer does not change it.
    parallelism.independent: {mark: none}
    parallelism.cooperative: {mark: none}
---

# RDKit MaxMinPicker

RDKit is a cheminformatics toolkit, and its MaxMinPicker is farthest-point traversal: start
somewhere, then repeatedly take whichever remaining item is furthest from everything picked so far.

Its distinguishing feature here is not the algorithm — several tools implement the same traversal —
but the interface. It takes a *distance function*, not a matrix, and evaluates it lazily. That
single decision is what keeps it practical into the millions: nothing ever needs all $n^2$
distances to exist simultaneously.

The trade is that it does one thing. One objective, no constraints, no budget to spend, and no way
to improve the answer by waiting.

## Problem targeted

Given $n$ items, a distance $d(i,j)$ and a target size $k$, it greedily constructs $S$ by

$$
S \leftarrow S \cup \Bigl\{ \arg\max_{i \notin S} \; \min_{j \in S} d(i,j) \Bigr\},
$$

repeated until $|S| = k$. This is the classical greedy heuristic for

$$
\max_{|S| = k} \; \min_{i \ne j \in S} d(i,j).
$$

**Guarantee: 2-approximation.** Farthest-point traversal is guaranteed to reach at least half the
optimal minimum separation, and no better ratio is achievable in polynomial time unless P = NP. In
practice it usually lands far closer to optimal than the bound suggests.

## Reference

--8<-- "generated/features/rdkit.md"
