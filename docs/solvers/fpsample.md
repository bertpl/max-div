---
solver:
  name: fpsample
  source: https://github.com/leonardodalinky/fpsample
  verified: 2026-07-27
  scale:
    max_practical_n: "6"
    rationale: >-
      A Rust implementation with KD-tree accelerated variants, so the traversal avoids the O(nk)
      distance evaluations a naive sweep needs. The tree variants degrade in high dimension and are
      documented as practical below roughly d ≈ 9; the plain variant has no such limit.
  metadata:
    guarantee: 2-approximation (farthest-point traversal)
    license: MIT
    last_release: {version: "1.0.2", date: "2025-12-20"}
    determinism: deterministic
    input: vectors
  capabilities:
    distance.l1: {mark: none}
    distance.l2: {mark: full}
    distance.cosine:
      mark: partial
      note: &cosine_norm
        text: >-
          Reachable by L2-normalizing the vectors first: on the unit sphere, cosine distance is a monotone function of Euclidean distance, so a Euclidean picker returns the same ordering.
    distance.custom:
      mark: none
      note:
        text: >-
          Euclidean only. The KD-tree variants depend on it structurally, so this is not a gap waiting to be filled.
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

# fpsample

fpsample is a small Rust library exposing farthest-point sampling to Python. It solves exactly one
problem — Euclidean max-min selection — and solves it very fast.

Its interest here is as the speed reference. Where RDKit buys scale through laziness, fpsample buys
it through a compiled implementation and spatial indexing: its KD-tree variants avoid scanning
every candidate on every pick. That indexing is also its constraint, since tree structures lose
their advantage as dimension grows, and the documented practical range stops around $d \approx 9$.

If your problem is Euclidean, unconstrained, and you want an answer in milliseconds, this is a
better tool than max-div.

## Problem targeted

Given $n$ vectors in $\mathbb{R}^d$ and a target size $k$, it approximates

$$
\max_{|S| = k} \; \min_{i \ne j \in S} \lVert x_i - x_j \rVert_2
$$

by farthest-point traversal, with the same greedy step RDKit uses but accelerated by a spatial
index rather than evaluated lazily.

**Guarantee: 2-approximation.** Identical to any farthest-point traversal — the implementation
changes the speed, not the bound.

## Reference

--8<-- "generated/features/fpsample.md"
