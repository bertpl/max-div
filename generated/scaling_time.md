| Solver | Config | Largest n within the time budget | Sweep ended by |
|---|---|---|---|
| max-div | `lean` | **500,000** | time budget exceeded at n=1,000,000 (≥75 s) |
| max-div | `optimal-eager` | **50,000** | memory cap exceeded at n=100,000 |
| max-div | `optimal-lazy` | **200,000** | time budget exceeded at n=500,000 (≥75 s) |
| OR-Tools CP-SAT | `feasible` | **20,000** | memory cap exceeded at n=50,000 |
| OR-Tools CP-SAT | `optimal` | **500** | time budget exceeded at n=1,000 (61 s) |
| SCIP (PySCIPOpt) | `feasible` | **2,000** | time budget exceeded at n=5,000 (≥75 s) |
| SCIP (PySCIPOpt) | `optimal` | **500** | time budget exceeded at n=1,000 (62 s) |
| HiGHS | `feasible` | **2,000** | time budget exceeded at n=5,000 (≥75 s) |
| HiGHS | `optimal` | **500** | time budget exceeded at n=1,000 (61 s) |
| RDKit MaxMinPicker | `default` | **20,000** | time budget exceeded at n=50,000 (≥75 s) |
| fpsample | `vanilla` | **500,000** | time budget exceeded at n=1,000,000 (≥75 s) |
| fpsample | `kdline` | **5,000,000** | time budget exceeded at n=10,000,000 (≥75 s) |
| skmatter | `default` | **500,000** | time budget exceeded at n=1,000,000 (≥75 s) |
| apricot-select | `default` | **50,000** | memory cap exceeded at n=100,000 |
| qc-selector | `maxmin` | **20,000** | memory cap exceeded at n=50,000 |
| DPPy | `default` | **500** | failure at n=1,000: `ValueError: size k=100 > rank=98` |
| code-FDM | `default` | **10,000** | time budget exceeded at n=20,000 (≥75 s) |
