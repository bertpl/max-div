---
solver:
  name: skmatter
  source: https://scikit-matter.readthedocs.io/
  verified: 2026-07-27
  scale:
    max_practical_n: "5"
    rationale: >-
      A NumPy implementation following scikit-learn conventions: no distance matrix is stored, but
      each pick scans all remaining candidates, so cost grows as n·k. Comfortable into the
      hundreds of thousands, an order below the compiled implementations.
  metadata:
    guarantee: 2-approximation (farthest-point traversal)
    license: BSD-3-Clause
    last_release: {version: "0.3.3", date: "2026-01-06"}
    determinism: deterministic
    input: vectors
  capabilities:
    distance.l1: {mark: none}
    distance.l2: {mark: full}
    distance.cosine:
      mark: partial
      note:
        text: >-
          Reachable by L2-normalizing the vectors first: on the unit sphere, cosine distance is a monotone function of Euclidean distance, so a Euclidean picker returns the same ordering.
    distance.custom: {mark: none}
    objective.max_min: {mark: full}
    objective.mean_nn: {mark: none}
    objective.geomean_nn: {mark: none}
    objective.max_sum: {mark: none}
    constraints.disjoint_groups: {mark: none}
    constraints.overlapping_groups: {mark: none}
    constraints.ranged_counts: {mark: none}
    budget.iterations: {mark: none}
    budget.wall_clock: {mark: none}
    budget.improves:
      mark: none
      note:
        text: >-
          A single construction pass, so there is no budget to spend: the answer is whatever one greedy sweep produces, and waiting longer does not change it.
---

# skmatter

skmatter — scikit-matter — is a scikit-learn-compatible library of feature and sample selection
methods aimed at atomistic machine learning. Farthest-point sampling is one entry in a larger
catalogue that also includes CUR decomposition and several information-theoretic selectors.

For this comparison it is the *idiomatic* choice rather than the fastest one: a selector with
`fit`/`transform` semantics that drops into an existing scikit-learn pipeline without new
vocabulary. If a project already lives in that ecosystem, that convenience is worth more than a
constant factor in speed.

Its selection behavior is the same greedy traversal as the other pickers, with the same guarantee.

## Problem targeted

Given $n$ vectors and a target size $k$, farthest-point sampling approximates

$$
\max_{|S| = k} \; \min_{i \ne j \in S} \lVert x_i - x_j \rVert_2 ,
$$

selecting at each step the candidate whose nearest already-selected neighbor is furthest away.

**Guarantee: 2-approximation.** The traversal's standard bound. skmatter also exposes
non-diversity selectors (CUR, PCovR-based) whose objectives are different problems entirely and
are out of scope here.

## Capabilities

--8<-- "generated/features/skmatter.md"
