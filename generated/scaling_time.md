| Solver | Largest n within the time budget | Sweep ended by |
|---|---|---|
| max-div `lean` | **500,000** | completed past the time budget at n=1,000,000 (137 s) |
| max-div `optimal-eager` | **50,000** | completed past the time budget at n=100,000 (60 s) |
| max-div `optimal-lazy` | **200,000** | time budget exceeded at n=500,000 |
| RDKit MaxMinPicker | **20,000** | completed past the time budget at n=50,000 (92 s) |
| DPPy | **500** | failure at n=1,000: `ValueError: size k=100 > rank=98` |
