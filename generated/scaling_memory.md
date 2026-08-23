| Solver | Largest n within memory | Determination |
|---|---|---|
| max-div `lean` | **1,000,000,000** | linear fit over 18 sizes |
| max-div `optimal-eager` | — | not measured: spawns worker processes |
| max-div `optimal-lazy` | — | not measured: spawns worker processes |
| RDKit MaxMinPicker | **1,000,000,000** | linear fit over 18 sizes |
| DPPy | **500** | bracketed: fails at the next size (`ValueError: size k=100 > rank=98`) |
