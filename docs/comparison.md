# Comparison with Other Tools

How does max-div relate to other freely available tools that select diverse subsets? The honest
answer starts with recognizing that the alternatives fall into **three different categories**, and
a flat ranking across them would be misleading:

- **Exact solvers** (SCIP, OR-Tools CP-SAT, HiGHS) are general MIP/CP engines. Given a hand-built
  model they *prove* optimality — but only at small problem sizes, at minutes-to-hours cost. They
  are best read as the optimality reference, not as competitors.
- **Greedy pickers & samplers** (RDKit, skmatter, fpsample, apricot, qc-selector, DPPy) run a
  single-shot construction: one pass, one answer, typically in milliseconds. Very fast, but the
  answer does not improve if you can afford more time, and none of them supports general
  per-group selection constraints.
- **Anytime heuristic optimizers** — max-div's category: start from a construction, then keep
  improving the selection for as long as the time budget allows, under fairness constraints if
  given.

The table below makes those categories explicit. Terminology for the objective column is defined
in [Objectives & the Diversity-Problem Landscape](concepts/objectives.md).

<div class="comparison" markdown>

| Tool | Category | Diversity objectives | Input | Constraints beyond k | Guarantee | Practical scale | Determinism | License · activity |
|---|---|---|---|---|---|---|---|---|
| **max-div** | **anytime heuristic** | **max-min, mean-of-NN, geomean-of-NN, mean-pairwise (max-sum)** | **vectors (L1 / L2 / L2² / cosine) or precomputed distances** | **overlapping per-group min/max counts, per-constraint weights, penalty shaping** | **heuristic** | **n ≈ 10⁴–10⁵ (memory-bound: O(n²)/2 float32 distances)** | **seeded, fully reproducible** | **Apache-2.0 · active** |
| SCIP (PySCIPOpt) | exact MIP | any linearizable objective (max-min; mean-of-NN via assignment MILP) | hand-built model | any linear constraints | proven optimum | max-min n ≲ 10³; mean-of-NN n ≲ 10²–10²·⁵ | deterministic | Apache-2.0 · very active |
| OR-Tools CP-SAT | exact CP | max-min via threshold feasibility search (integer-scaled distances) | hand-built model | any linear constraints | proven optimum | n ≲ 10³ | deterministic (fixed workers) | Apache-2.0 · very active |
| HiGHS | exact MIP | any linearizable objective (big-M formulations) | hand-built model | any linear constraints | proven optimum | n ≲ 10³ | deterministic | MIT · very active |
| RDKit MaxMinPicker | greedy picker | max-min | custom distance function (lazy evaluation) | must-include seeds | 2-approx (FPS) | very large (lazy distances) | seeded | BSD-3 · very active |
| skmatter | greedy picker | max-min (FPS), CUR | vectors | seed point only | 2-approx (FPS) | large | deterministic | BSD-3 · active |
| fpsample | greedy picker | max-min (FPS) | vectors (KD-tree variants need d ≲ 9) | seed index only | 2-approx (FPS) | very large, very fast (Rust) | deterministic | MIT · active |
| apricot-select | greedy submodular | facility location, feature-based, other submodular | vectors or precomputed similarities | none | 1−1/e (submodular) | large | deterministic | MIT · repo active, PyPI release stale (2021) |
| qc-selector | greedy picker | max-min, max-sum, OptiSim, sphere exclusion | vectors or distance matrix | label-stratified proportional picks | heuristic | medium | seeded | GPL-3 · young (2025) |
| DPPy | probabilistic sampler | k-DPP (log-det kernel) — samples diverse sets, does not maximize | kernel matrix | none | sampler, not optimizer | medium (O(n²) kernel) | seeded sampling | MIT · low activity (last release 2024) |
| code-FDM | fair heuristic | max-min under per-group counts | vectors | per-group counts | heuristic | medium | — | no license (research code, dormant since 2022) |

</div>

Facts (versions, licenses, activity) verified against PyPI / GitHub on 2026-07-14.

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
the verification date) and is therefore listed here only as a mention.
