---
solver:
  name: DPPy
  source: https://dppy.readthedocs.io/
  verified: 2026-07-27
  scale:
    max_practical_n: "4"
    rationale: >-
      An n × n kernel matrix must be materialized, and exact k-DPP sampling eigendecomposes it —
      an O(n³) step that dominates well before memory does. Around n ≈ 10⁴ that is still minutes;
      beyond it, approximate samplers are the only route.
  metadata:
    guarantee: sampler, not optimizer
    license: MIT
    last_release: {version: "0.3.3", date: "2024-08-14"}
    determinism: seeded sampling
    input: a similarity kernel matrix
  capabilities:
    distance.l1:
      mark: partial
      note: &via_kernel
        text: >-
          Reachable only through the kernel you supply. DPPy never sees vectors or a distance
          function: you choose how similarity is defined when you build the kernel, and any
          metric you can express that way is available.
    distance.l2:
      mark: partial
      note: *via_kernel
    distance.linf:
      mark: partial
      note: *via_kernel
    distance.cosine:
      mark: partial
      note: *via_kernel
    distance.custom:
      mark: full
      note:
        text: >-
          The kernel is entirely caller-supplied, so a custom similarity is the native case rather
          than a workaround. Note the sign convention differs from a distance: the kernel encodes
          similarity, and diversity comes from its determinant being large.
    objective.max_min:
      mark: none
      note: &samples_not_optimizes
        text: >-
          DPPy draws samples from a determinantal point process rather than maximizing anything.
          Diversity is a property of the distribution it samples from — subsets whose kernel
          submatrix has a large determinant are more likely — so no objective is optimized and
          no selection is claimed to be best.
    objective.mean_nn:
      mark: none
      note: *samples_not_optimizes
    objective.geomean_nn:
      mark: none
      note: *samples_not_optimizes
    objective.max_sum:
      mark: none
      note: *samples_not_optimizes
    constraints.disjoint_groups: {mark: none}
    constraints.overlapping_groups: {mark: none}
    constraints.ranged_counts: {mark: none}
    budget.iterations:
      mark: partial
      note:
        text: >-
          You can ask for more samples, but a sample count is a number of independent draws, not
          an improvement budget — the tenth draw is no better than the first, merely different.
    budget.wall_clock: {mark: none}
    budget.improves: {mark: none}
---

# DPPy

DPPy samples from determinantal point processes. Given a kernel that says how similar any two
items are, a k-DPP places more probability on subsets whose members are mutually dissimilar —
so drawing from it tends to produce a spread-out selection.

It is included here because that makes it a genuine alternative for "give me a diverse subset",
and excluded from any ranking because it is not solving the same problem. Every other tool in
this comparison returns an answer it claims is good by some measure. DPPy returns a *draw*. Ask
twice and you get two different subsets, neither of which is put forward as better.

That distinction is why its objective row is empty rather than partially filled. The temptation
is to read the empty row as a weakness; it is more accurately a different contract. Where you
want a diverse sample — for downstream ensembling, or to explore rather than commit — being able
to draw repeatedly is the feature, and a maximizer is the wrong tool.

## Problem targeted

Given $n$ items and a positive semi-definite similarity kernel $L \in \mathbb{R}^{n \times n}$,
a k-DPP defines a probability distribution over subsets of fixed size $k$:

$$
\Pr(S) \;\propto\; \det(L_S),
\qquad S \subseteq \{1,\dots,n\},\; |S| = k,
$$

where $L_S$ is the submatrix of $L$ indexed by $S$. The determinant is large when the selected
items are close to mutually orthogonal in the kernel's feature space, which is the sense in
which the distribution favors diversity. DPPy draws from this distribution; it does not search
for the $S$ that maximizes $\det(L_S)$ — that problem is NP-hard.

**Guarantee: sampler, not optimizer.** The guarantee is distributional. Exact k-DPP sampling
draws from precisely the distribution above, with no claim whatsoever about any individual
draw's diversity relative to the best possible subset.

## Reference

--8<-- "generated/features/dppy.md"
