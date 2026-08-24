| Solver | Config | Largest n within memory | Determination |
|---|---|---|---|
| max-div | `lean` | **1,000,000,000** | linear fit over 19 sizes |
| max-div | `optimal-eager` | — | not measured: spawns worker processes |
| max-div | `optimal-lazy` | — | not measured: spawns worker processes |
| OR-Tools CP-SAT | `feasible` | **20,000** | quadratic fit over 8 sizes |
| OR-Tools CP-SAT | `optimal` | **2,000** | quadratic fit over 5 sizes |
| SCIP (PySCIPOpt) | `feasible` | **2,000** | quadratic fit over 5 sizes |
| SCIP (PySCIPOpt) | `optimal` | **2,000** | quadratic fit over 6 sizes |
| HiGHS | `feasible` | **2,000** | quadratic fit over 5 sizes |
| HiGHS | `optimal` | **2,000** | quadratic fit over 5 sizes |
| RDKit MaxMinPicker | `default` | **1,000,000,000** | linear fit over 19 sizes |
| fpsample | `vanilla` | **1,000,000,000** | linear fit over 19 sizes |
| fpsample | `kdline` | **500,000,000** | linear fit over 18 sizes |
| skmatter | `default` | **500,000,000** | linear fit over 20 sizes |
| apricot-select | `default` | **50,000** | quadratic fit over 9 sizes |
| qc-selector | `maxmin` | **20,000** | quadratic fit over 8 sizes |
| DPPy | `default` | **500** | measurement series truncated: the solver fails at the next size (`ValueError: size k=100 > rank=98`) |
| code-FDM | `default` | **100,000,000** | linear fit over 15 sizes |
