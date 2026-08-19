---
solver:
  name: apricot-select
  source: https://apricot-select.readthedocs.io/
  verified: 2026-07-27
  scale:
    memory_ceiling: pending
    time_ceiling: pending
    quality_ceiling: pending
  metadata:
    guarantee: 1−1/e (lazy greedy on a monotone submodular function)
    license: MIT
    last_release: {version: "0.6.1", date: "2021-02-18"}
    determinism: deterministic
    input: vectors or a precomputed similarity matrix
    notes:
      last_release:
        text: >-
          The PyPI release is from 2021, but the project is not abandoned: its repository was last
          updated in November 2025. Installing from PyPI gets you considerably older code than the
          repository holds.
        url: https://github.com/jmschrei/apricot
  capabilities:
    distance.l1: {mark: full}
    distance.l2: {mark: full}
    distance.linf: {mark: full}
    distance.cosine: {mark: full}
    distance.custom:
      mark: full
      note:
        text: >-
          A precomputed similarity matrix is a first-class input, so any similarity you can compute is usable. Note the sign convention: apricot maximizes similarity coverage, so a distance has to be turned into a similarity first.
    objective.max_min:
      mark: none
      note: &coverage_not_dispersion
        text: >-
          Submodular coverage objectives, not dispersion ones. Nothing here maximizes a minimum separation, and no reformulation makes facility location do so.
    objective.mean_nn:
      mark: none
      note: *coverage_not_dispersion
    objective.geomean_nn:
      mark: none
      note: *coverage_not_dispersion
    objective.max_sum:
      mark: full
      note:
        text: >-
          Facility location rewards how well the selection covers the remaining items, which on a distance-derived kernel is the standard submodular surrogate for a max-sum style objective — related to, but not identical with, maximizing the mean pairwise distance.
    constraints.disjoint_groups: {mark: none}
    constraints.overlapping_groups: {mark: none}
    constraints.ranged_counts: {mark: none}
    constraints.feasibility_proofs: {mark: none}
    budget.iterations: {mark: none}
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
          The optimizers accept an n_jobs argument, but it is unused: the greedy selection runs
          single-threaded.
    parallelism.cooperative: {mark: none}
---

# apricot-select

apricot maximizes submodular functions, most usefully *facility location*: pick the subset that
best represents everything you did not pick. That is a coverage objective, and it is worth being
precise that coverage and dispersion are not the same goal.

A dispersion objective asks that selected items be far from *each other*. A coverage objective asks
that unselected items be close to *something selected*. The two often agree — a well-spread
selection covers well — but they can diverge, and apricot optimizes the second.

What it brings in exchange is a real approximation guarantee. Greedy maximization of a monotone
submodular function is within $1 - 1/e$ of optimal, which is a stronger statement than any
heuristic here can make, on an objective that is genuinely different.

## Problem targeted

Given $n$ items and a similarity $s(i,j)$, facility location selects

$$
\max_{S \subseteq \{1,\dots,n\},\; |S| = k}
\;\sum_{i=1}^{n} \max_{j \in S} \, s(i,j).
$$

Each unselected item contributes its similarity to whichever selected item represents it best, so
the objective rewards a selection that leaves nothing far from everything.

**Guarantee: $1 - 1/e$.** The function is monotone and submodular, so greedy maximization is within
a factor $1 - 1/e \approx 0.63$ of optimal — a bound that holds for this objective, not for the
dispersion objectives the other tools target.

## Reference

--8<-- "generated/features/apricot-select.md"
