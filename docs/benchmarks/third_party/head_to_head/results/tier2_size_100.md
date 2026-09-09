| tool | quality (min separation) | time |
|---|---|---|
| max-div[DEFAULT, 12 workers] @ 1 s | 0.3407 | 1.35 s |
| max-div[DEFAULT] @ 60 s | 0.3407 | 60 s |
| max-div[DEFAULT, 12 workers] @ 60 s | 0.3407 | 60.4 s |
| max-div[DEFAULT] @ 1 s | 0.3283 | 1 s |
| RDKit MaxMinPicker[default] | 0.2727 | 0.00987 s |
| fpsample[kdline] | 0.2622 | 4.82e-05 s |
| fpsample[vanilla] | 0.2622 | 0.0489 s |
| qc-selector[maxmin] | 0.2622 | 0.0543 s |
| skmatter[default] | 0.2622 | 3.66 s |
| code-FDM[default] | 0.2514 | 0.11 s |
| DPPy[default] | 0.1200 | 0.00935 s |
| apricot-select[default] | 0.1016 | 0.662 s |
| qc-selector[maxsum] | 0.0833 | 9.85e-05 s |
| kmedoids[default] | 0.0696 | 0.0921 s |

`max-div` reaches the best one-shot result (RDKit MaxMinPicker[default]) at a budget of 0.001 s with one worker and at a budget of 1 s with 12 workers.
