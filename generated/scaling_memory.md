| Solver | Largest n within memory | Determination |
|---|---|---|
| max-div `lean` | **1,000,000,000** | linear fit over 18 sizes |
| max-div `optimal-eager` | — | not measured: spawns worker processes |
| max-div `optimal-lazy` | — | not measured: spawns worker processes |
| OR-Tools CP-SAT `feasible` | **20,000** | quadratic fit over 8 sizes |
| OR-Tools CP-SAT `optimal` | **2,000** | quadratic fit over 4 sizes |
| SCIP (PySCIPOpt) `feasible` | **2,000** | quadratic fit over 5 sizes |
| SCIP (PySCIPOpt) `optimal` | **1,000** | quadratic fit over 3 sizes |
| HiGHS `feasible` | **2,000** | quadratic fit over 5 sizes |
| HiGHS `optimal` | **1,000** | quadratic fit over 4 sizes |
| RDKit MaxMinPicker | **1,000,000,000** | linear fit over 18 sizes |
| fpsample `vanilla` | **1,000,000,000** | linear fit over 18 sizes |
| fpsample `kdline` | **500,000,000** | linear fit over 17 sizes |
| skmatter | **200,000,000** | linear fit over 16 sizes |
| apricot-select | **50,000** | quadratic fit over 9 sizes |
| qc-selector | **20,000** | quadratic fit over 8 sizes |
| DPPy | **500** | measurement series truncated: the solver fails at the next size (`ValueError: size k=100 > rank=98`) |
| code-FDM | — | measurement series truncated: the solver fails at the next size (`RuntimeError: code-FDM (FairFlow) produced no selection`) |
