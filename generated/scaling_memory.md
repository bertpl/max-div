| Solver | Config | Largest n within memory | Determination |
|---|---|---|---|
| max-div | `lean` | **1,000,000,000** | linear fit over 21 sizes |
| max-div | `optimal-eager` | — | not measured: spawns worker processes |
| max-div | `optimal-lazy` | — | not measured: spawns worker processes |
| OR-Tools CP-SAT | `feasible` | **20,000** | quadratic fit over 9 sizes |
| OR-Tools CP-SAT | `optimal` | **2,000** | quadratic fit over 5 sizes |
| SCIP (PySCIPOpt) | `feasible` | **2,000** | quadratic fit over 6 sizes |
| SCIP (PySCIPOpt) | `optimal` | **2,000** | quadratic fit over 6 sizes |
| HiGHS | `feasible` | **5,000** | quadratic fit over 7 sizes |
| HiGHS | `optimal` | **5,000** | quadratic fit over 6 sizes |
| RDKit MaxMinPicker | `default` | **1,000,000,000** | linear fit over 21 sizes |
| fpsample | `vanilla` | **1,000,000,000** | linear fit over 21 sizes |
| fpsample | `kdline` | **500,000,000** | linear fit over 21 sizes |
| skmatter | `default` | **500,000,000** | linear fit over 21 sizes |
| apricot-select | `default` | **20,000** | quadratic fit over 10 sizes |
| qc-selector | `maxmin` | **20,000** | quadratic fit over 9 sizes |
| DPPy | `default` | **500** | measurement series truncated: the solver fails at the next size (`ValueError: size k=100 > rank=98`) |
| code-FDM | `default` | **100,000,000** | linear fit over 17 sizes |
| kmedoids | `default` | **50,000** | quadratic fit over 10 sizes |
