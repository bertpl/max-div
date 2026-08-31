| Solver | Config | Largest n closing 50% of the gap | Largest n closing 90% of the gap |
|---|---|---|---|
| max-div | `lean` | — | — |
| max-div | `optimal-eager` | **50,000** | **50,000** |
| max-div | `optimal-lazy` | **200,000** | **200,000** |
| OR-Tools CP-SAT | `feasible` | — | — |
| OR-Tools CP-SAT | `optimal` | **1,000** | **1,000** |
| SCIP (PySCIPOpt) | `feasible` | **20** | **20** |
| SCIP (PySCIPOpt) | `optimal` | **500** | **200** |
| HiGHS | `feasible` | **20** | — |
| HiGHS | `optimal` | **200** | **200** |
| RDKit MaxMinPicker | `default` | **20,000** | **50** |
| fpsample | `vanilla` | **500,000** | — |
| fpsample | `kdline` | **5,000,000** | — |
| skmatter | `default` | **500,000** | — |
| apricot-select | `default` | — | — |
| qc-selector | `maxmin` | **20,000** | — |
| DPPy | `default` | — | — |
| code-FDM | `default` | **10,000** | **50** |
