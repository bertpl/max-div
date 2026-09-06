---
solver:
  name: kmedoids
  source: https://github.com/kno10/python-kmedoids
  verified: 2026-09-05
  scale:
    max_n_memory: 50000
    max_n_time: 50000
    max_n_quality_50: none
    max_n_quality_90: none
  metadata:
    guarantee: heuristic (local optimum of the k-medoids objective)
    license: GPL-3.0-or-later
    last_release: {version: "0.5.5", date: "2026-05-23"}
    determinism: seeded
    input: a precomputed dissimilarity matrix, or vectors through the scikit-learn-style estimator
  capabilities:
    distance.l1:
      mark: full
      note: &via_sklearn_estimator
        text: >-
          The core functions take a dissimilarity matrix only; the scikit-learn-style estimator
          accepts a metric name and forwards it to scikit-learn's pairwise distances, which needs
          scikit-learn installed.
    distance.l2:
      mark: full
      note: *via_sklearn_estimator
    distance.linf:
      mark: full
      note: *via_sklearn_estimator
    distance.minkowski:
      mark: full
      note: *via_sklearn_estimator
    distance.cosine:
      mark: full
      note: *via_sklearn_estimator
    distance.geomean:
      mark: partial
      note:
        text: >-
          Not among the metric names scikit-learn understands, so it is reachable only as a
          precomputed dissimilarity matrix.
    distance.custom:
      mark: full
      note:
        text: >-
          A precomputed dissimilarity matrix is the primary input and the one the documentation
          recommends, so any dissimilarity you can compute is usable as is.
    objective.max_min:
      mark: none
      note: &medoids_not_dispersion
        text: >-
          k-medoids minimizes the total dissimilarity of every item to its nearest medoid, a
          representativeness objective that places the medoids in dense regions; none of the
          dispersion objectives is optimized.
    objective.mean_nn:
      mark: none
      note: *medoids_not_dispersion
    objective.geomean_nn:
      mark: none
      note: *medoids_not_dispersion
    objective.max_sum:
      mark: none
      note: *medoids_not_dispersion
    objective.optimality_proofs: {mark: none}
    constraints.disjoint_groups: {mark: none}
    constraints.overlapping_groups: {mark: none}
    constraints.ranged_counts: {mark: none}
    constraints.feasibility_proofs: {mark: none}
    budget.iterations:
      mark: partial
      note: &converges_then_stops
        text: >-
          The iteration cap bounds the swap phase, but the cap is a convergence limit, not a budget:
          the search stops at the first local optimum it reaches, usually after a handful of
          passes, whatever cap is set.
    budget.wall_clock: {mark: none}
    budget.improves:
      mark: none
      note: *converges_then_stops
    parallelism.independent: {mark: none}
    parallelism.cooperative:
      mark: none
      note:
        text: >-
          The swap search can evaluate candidates on several threads, which speeds up one search;
          it is not several workers searching the problem.
---

# kmedoids

kmedoids is a Rust implementation of k-medoids clustering with Python bindings, whose FasterPAM
algorithm makes the classical PAM swap descent fast enough for tens of thousands of items. The
medoids are actual data items, so a run yields a size-$k$ selection.

kmedoids is included here as a representativeness reference, not as a dispersion competitor. A
set of cluster centers is usually less spread out than a selection chosen for separation, and
measuring one under the dispersion metrics shows how a representativeness selection scores when
spread is what you are after.

## Problem targeted

Given $n$ items with pairwise dissimilarities $d$ and a target size $k$, kmedoids approximately solves

$$
\min_{|S| = k} \; \sum_{i=1}^{n} \min_{m \in S} d(i, m)
$$

by swap descent from a random initial medoid set: any swap of a medoid for a non-medoid that
lowers the total is applied eagerly, until no swap improves it.

**Guarantee: heuristic.** The descent stops at a local optimum of its own objective; no bound is
claimed relative to the global optimum.

## Reference

--8<-- "generated/features/kmedoids.md"
