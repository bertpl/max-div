---
solver:
  name: code-FDM
  source: https://github.com/yhwang1990/code-FDM
  verified: 2026-07-27
  scale:
    max_practical_n: "4"
    rationale: >-
      Research code built around a full distance matrix and repeated threshold scans over it. No
      effort has gone into constant factors, so the practical ceiling sits below the maintained
      libraries at comparable memory.
  metadata:
    guarantee: heuristic with approximation bounds stated in the paper
    license: none declared
    last_release: {version: "none", date: "none"}
    determinism: not documented
    input: vectors
    notes:
      last_release:
        text: >-
          Never packaged: this is a research repository, not a release. Its last commit was in July
          2022 and it carries no license file, which means no permission to use it is granted —
          a blocker for anything beyond reading the code, regardless of how well it performs.
        url: https://github.com/yhwang1990/code-FDM
  capabilities:
    distance.l1:
      mark: partial
      note: &edit_source
        text: >-
          The implementation is written around Euclidean distance; another metric means editing the source rather than passing an argument.
    distance.l2: {mark: full}
    distance.linf:
      mark: partial
      note: *edit_source
    distance.cosine:
      mark: partial
      note: *edit_source
    distance.custom:
      mark: full
      note:
        text: >-
          Since you are editing the source in any case, an arbitrary metric is no harder than a common one — a consequence of it being research code rather than a designed feature.
    objective.max_min: {mark: full}
    objective.mean_nn: {mark: none}
    objective.geomean_nn: {mark: none}
    objective.max_sum: {mark: none}
    constraints.disjoint_groups:
      mark: full
      note:
        text: >-
          This is the whole point of the tool: it selects a maximally dispersed subset subject to a required count from each group, which is the fair-diversity problem max-div also targets.
    constraints.overlapping_groups:
      mark: none
      note: &one_group_each
        text: >-
          Each item belongs to exactly one group. Overlapping membership — an item that is both in-region and in-category — is outside the model the algorithms are built on.
    constraints.ranged_counts:
      mark: none
      note:
        text: >-
          Counts are exact requirements per group, not a minimum and a maximum, so a range cannot be expressed.
    budget.iterations: {mark: none}
    budget.wall_clock: {mark: none}
    budget.improves:
      mark: none
      note:
        text: >-
          A single construction pass, so there is no budget to spend: the answer is whatever one greedy sweep produces, and waiting longer does not change it.
    parallelism.parallel: {mark: none}
---

# code-FDM

code-FDM is the reference implementation accompanying published work on *fair diversity
maximization* — max-min selection subject to a required number of items from each group. It is the
only surveyed tool besides max-div that treats group constraints as part of the problem rather than
as post-processing.

That makes it the closest comparison point for max-div's fairness constraints, and the differences
are instructive. Its groups partition the items, so each item belongs to exactly one; max-div's may
overlap. Its counts are exact requirements; max-div's are minimum and maximum bounds. And it is a
construction algorithm, so it produces one answer rather than improving one.

**Before considering it for anything beyond reading: it carries no license.** No permission to use,
copy or modify it has been granted, which is a legal question rather than a technical one. Its last
commit was July 2022.

## Problem targeted

Given $n$ items partitioned into groups $G_1,\dots,G_m$, a distance $d(i,j)$, and a required count
$k_g$ for each group, it targets

$$
\max_{S} \; \min_{i \ne j \in S} d(i,j)
\qquad\text{subject to}\qquad
|S \cap G_g| = k_g \;\; \forall\, g .
$$

The algorithms proceed by fixing a candidate threshold, building the graph of pairs closer than it,
and searching for a group-feasible independent set — then adjusting the threshold.

**Guarantee: heuristic, with bounds stated in the accompanying paper** for particular variants. The
implementation is a research artifact, so treat published ratios as claims about the algorithms
rather than about this code.

## Reference

--8<-- "generated/features/code-fdm.md"
