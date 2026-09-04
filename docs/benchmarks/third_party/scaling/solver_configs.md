# Solver Scaling — Solver Configurations

Each solver enters the [scaling measurements](protocol.md) as one or more of the configurations below. The **Version** column names the solver version every scaling measurement ran with (for the SCIP and HiGHS rows: the Python package's version — `PySCIPOpt` and `highspy`).

For every configuration, the time budget is handed to the solver where it accepts one, and a run counts as successful when a valid size-`k` selection is available when the run ends — for the exact solvers, a proven-optimal solution and an incumbent held at the budget kill are both valid outcomes.

| Solver | Version | Config | Description |
|---|---|---|---|
| max-div | 0.17.2 | `lean` | uniform random one-shot initialization only, no optimization step, lazy distance storage, 1 worker |
| max-div | 0.17.2 | `optimal-eager` | SMART preset, full end-to-end time budget, full-matrix distance storage forced, 12 workers with dynamic grouping |
| max-div | 0.17.2 | `optimal-lazy` | SMART preset, full end-to-end time budget, lazy distance storage forced, 12 workers with dynamic grouping |
| OR-Tools CP-SAT | 9.15.6755 | `feasible` | max-min CP-SAT model, stop at the first feasible solution, 1 worker |
| OR-Tools CP-SAT | 9.15.6755 | `optimal` | max-min CP-SAT model, full time budget, 12 portfolio workers |
| SCIP | 6.2.1 | `feasible` | big-M max-min MIP, stop at the first feasible solution |
| SCIP | 6.2.1 | `optimal` | big-M max-min MIP, full time budget |
| HiGHS | 1.15.1 | `feasible` | big-M max-min MIP, stop at the first improving solution |
| HiGHS | 1.15.1 | `optimal` | big-M max-min MIP, full time budget, parallel branch-and-bound |
| RDKit MaxMinPicker | 2026.3.5 | `default` | MaxMinPicker with a Euclidean distance callable (its only mode) |
| fpsample | 1.0.2 | `vanilla` | plain farthest-point sampling |
| fpsample | 1.0.2 | `kdline` | bucket KD-line farthest-point sampling — the tree-accelerated variant, well suited to `d=2` |
| skmatter | 0.3.3 | `default` | FPS selector (its only mode) |
| apricot-select | 0.6.1 | `default` | facility-location selection, lazy greedy, RBF similarity matrix |
| qc-selector | 0.1.4 | `maxmin` | max-min selection on a precomputed distance matrix |
| DPPy | 0.3.3 | `default` | one exact k-DPP sample over an RBF likelihood kernel, bandwidth set by the median-pairwise-distance heuristic |
| code-FDM | commit `d18758a` | `default` | FairFlow with a single color spanning all items (its unconstrained reduction) |

Notes:

- **Worker counts differ per configuration on purpose**: parallel workers improve quality within a wall-clock budget, but they multiply working memory and add startup overhead — so they appear only in the quality-oriented configurations, never in the lean ones.  The 12 workers match the reference machine's performance-core count.
- **SCIP runs single-threaded throughout**: the PySCIPOpt package on PyPI ships without SCIP's concurrent-solve support, and building a custom parallel SCIP is out of scope — the protocol tests solvers as installable.
- **One-shot solvers get a single configuration** where they genuinely have a single mode: no budget knob, no worker knob, and no memory/speed/quality trade-off a user could configure.
