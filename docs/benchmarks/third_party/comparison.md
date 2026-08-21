# Comparison with Other Tools

How does max-div relate to other freely available tools that select diverse subsets? The honest
answer starts with recognizing that the alternatives fall into **three different categories**, and
a flat ranking across them would be misleading:

- **Exact solvers** (SCIP, OR-Tools CP-SAT, HiGHS) are general MIP/CP engines. Given a hand-built
  model they *prove* optimality — but only at small problem sizes, at minutes-to-hours cost. They
  are best read as the optimality reference, not as competitors.
- **One-shot pickers & samplers** (RDKit, skmatter, fpsample, apricot, qc-selector, DPPy) run a
  single-shot construction: one pass, one answer, typically in milliseconds. Very fast, but the
  answer does not improve if you can afford more time, and none of them supports general
  per-group selection constraints.
- **Anytime heuristic optimizers** — max-div's category: start from a construction, then keep
  improving the selection for as long as the time budget allows, under fairness constraints if
  given.

The tables below make those categories explicit. Terminology for the objective columns is defined
in [Objectives & the Diversity-Problem Landscape](../../concepts/objectives.md), and each tool's name
links to its [profile](solvers/index.md), where the same capabilities are laid out one tool at a
time with the reasoning behind every mark.

--8<-- "generated/comparison.md"

## What the table means in practice

- **If you need a proven optimum** (and your problem is small): build a MIP/CP model — CP-SAT for
  max-min, SCIP for the mean-of-NN objectives. max-div cannot prove optimality.
- **If you need one diverse subset as fast as possible, unconstrained**: greedy FPS (fpsample,
  skmatter, RDKit) is excellent — within a factor 2 of the max-min optimum, in milliseconds.
- **If you can spend more than milliseconds, want NN-separation or max-sum objectives beyond
  max-min, or have per-group selection constraints**: this is max-div's territory. General
  min/max per-group counts are supported by none of the pickers (code-FDM comes closest, for
  max-min only), and the mean/geomean-of-NN objectives are — as far as we know — implemented
  nowhere else.

Notable omission: **submodlib** (a submodular-optimization zoo including disparity-min, i.e.
max-min) currently cannot be installed on Python 3.13 (no wheels or sdist for it on PyPI as of
the verification dates above) and is therefore listed here only as a mention.
