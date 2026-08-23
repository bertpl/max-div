| Solver | Largest n within the time budget | Sweep ended by |
|---|---|---|
| max-div `lean` | **500,000** | completed past the time budget at n=1,000,000 (137 s) |
| max-div `optimal-eager` | **50,000** | completed past the time budget at n=100,000 (60 s) |
| max-div `optimal-lazy` | **200,000** | time budget exceeded at n=500,000 |
| OR-Tools CP-SAT `feasible` | **20,000** | memory cap exceeded at n=50,000 |
| OR-Tools CP-SAT `optimal` | **500** | completed past the time budget at n=1,000 (61 s) |
| SCIP (PySCIPOpt) `feasible` | **2,000** | time budget exceeded at n=5,000 |
| SCIP (PySCIPOpt) `optimal` | **500** | completed past the time budget at n=1,000 (62 s) |
| HiGHS `feasible` | **2,000** | time budget exceeded at n=5,000 |
| HiGHS `optimal` | **500** | completed past the time budget at n=1,000 (60 s) |
| RDKit MaxMinPicker | **20,000** | completed past the time budget at n=50,000 (92 s) |
| fpsample `vanilla` | **500,000** | time budget exceeded at n=1,000,000 |
| fpsample `kdline` | **5,000,000** | time budget exceeded at n=10,000,000 |
| skmatter | **500,000** | time budget exceeded at n=1,000,000 |
| apricot-select | **50,000** | memory cap exceeded at n=100,000 |
| qc-selector | **20,000** | memory cap exceeded at n=50,000 |
| DPPy | **500** | failure at n=1,000: `ValueError: size k=100 > rank=98` |
| code-FDM | — | failure at n=20: `RuntimeError: code-FDM (FairFlow) produced no selection` |
