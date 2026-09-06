| n | best one-shot tool | its quality | its time | 1 worker @1 s | 1 worker @60 s | 12 workers @1 s | 12 workers @60 s | overtake budget, 1 worker | overtake budget, 12 workers |
|---|---|---|---|---|---|---|---|---|---|
| 100 | RDKit[MaxMinPicker] | 0.2727 | 0.00987 s | 0.3283 | 0.3407 | 0.3407 | 0.3407 | 0.001 s | 1 s |
| 1,000 | RDKit[MaxMinPicker] | 0.0863 | 0.0355 s | 0.0909 | 0.0941 | 0.0941 | 0.0966 | 0.05 s | 1 s |
| 10,000 | RDKit[MaxMinPicker] | 0.0250 | 3.67 s | 0.0258 | 0.0269 | 0.0252 | 0.0274 | 0.5 s | 1 s |
| 100,000 | fpsample[FPS] | 0.0079 | 1.76 s | 0.0079 | 0.0079 | 0.0079 | 0.0079 | 0.001 s | 1 s |
