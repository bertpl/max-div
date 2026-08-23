| Solver | Largest n within memory | Determination |
|---|---|---|
| max-div `lean` | **500,000,000** | linear fit over 5 sizes |
| max-div `optimal-eager` | **50,000** | bracketed: the memory cap was reached at the next size |
| max-div `optimal-lazy` | **1,000,000,000** | linear fit over 5 sizes |
| RDKit MaxMinPicker | **200,000,000** | linear fit over 5 sizes |
| DPPy | **20,000** | quadratic fit over 5 sizes |
