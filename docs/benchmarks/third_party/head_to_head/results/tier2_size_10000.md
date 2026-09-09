| tool | quality (min separation) | time |
|---|---|---|
| max-div[DEFAULT, 12 workers] @ 60 s | 0.0273 | 60.3 s |
| max-div[DEFAULT] @ 60 s | 0.0269 | 60 s |
| max-div[DEFAULT] @ 1 s | 0.0254 | 1.02 s |
| max-div[DEFAULT, 12 workers] @ 1 s | 0.0252 | 1.36 s |
| fpsample[KDLine] | 0.0250 | 0.00235 s |
| fpsample[FPS] | 0.0250 | 0.018 s |
| skmatter[FPS] | 0.0250 | 0.0367 s |
| RDKit[MaxMinPicker] | 0.0250 | 3.67 s |
| qc-selector[MaxMin] | 0.0250 | 5.52 s |
| code-FDM[single-color] | 0.0250 | 38.8 s |
| apricot[facility-location] | 0.0054 | 1.77 s |
| kmedoids[FasterPAM] | 0.0034 | 5.08 s |
| qc-selector[MaxSum] | 0.0007 | 9.98 s |

`max-div` reaches the best one-shot result (RDKit[MaxMinPicker]) at a budget of 1 s with one worker and at a budget of 1 s with 12 workers.
