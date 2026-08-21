# Solver Scaling — Solver Configurations

The solver configurations entered into the [scaling measurements](protocol.md).  Each configuration is tested independently; per axis and problem size, a solver's best result across its configurations is reported.

For every configuration, the time budget is handed to the solver where it accepts one, and a run counts as successful when a valid size-`k` selection is available when the run ends — for the exact solvers, a proven-optimal solution and an incumbent held at the budget kill are both valid outcomes.

| Solver | Config | Description |
|---|---|---|
| max-div | `lean` | random one-shot initialization only, no optimization step, lazy distance storage, 1 worker |
| max-div | `optimal-eager` | SMART preset, full end-to-end time budget, full-matrix distance storage forced, 12 cooperative workers |
| max-div | `optimal-lazy` | SMART preset, full end-to-end time budget, lazy distance storage forced, 12 cooperative workers |
| OR-Tools CP-SAT | `feasible` | max-min CP-SAT model, stop at the first feasible solution, 1 worker |
| OR-Tools CP-SAT | `optimal` | max-min CP-SAT model, full time budget, 12 portfolio workers |
| SCIP | `feasible` | big-M max-min MIP, stop at the first feasible solution |
| SCIP | `optimal` | big-M max-min MIP, full time budget |
| HiGHS | `feasible` | big-M max-min MIP, stop at the first improving solution |
| HiGHS | `optimal` | big-M max-min MIP, full time budget, parallel branch-and-bound |
| RDKit MaxMinPicker | `default` | MaxMinPicker with a Euclidean distance callable (its only mode) |
| fpsample | `vanilla` | plain farthest-point sampling |
| fpsample | `kdline` | bucket KD-line farthest-point sampling — the tree-accelerated variant, well suited to `d=2` |
| skmatter | `default` | FPS selector (its only mode) |
| apricot-select | `default` | facility-location selection, lazy greedy, RBF similarity matrix |
| qc-selector | `maxmin` | max-min selection on a precomputed distance matrix |
| DPPy | `default` | one exact k-DPP sample over an RBF likelihood kernel, bandwidth set by the median-pairwise-distance heuristic |
| code-FDM | `default` | FairFlow with a single color spanning all items (its unconstrained reduction) |

Notes:

- **Worker counts differ per configuration on purpose**: parallel workers improve quality within a wall-clock budget, but they multiply working memory and add startup overhead — so they appear only in the quality-oriented configurations, never in the lean ones.  The 12 workers match the reference machine's performance-core count.
- **SCIP runs single-threaded throughout**: the PySCIPOpt package on PyPI ships without SCIP's concurrent-solve support, and building a custom parallel SCIP is out of scope — the protocol tests solvers as installable.
- **One-shot solvers get a single configuration** where they genuinely have a single mode: no budget knob, no worker knob, and no memory/speed/quality trade-off a user could configure.
