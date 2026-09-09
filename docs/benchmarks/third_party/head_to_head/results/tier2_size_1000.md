| tool | quality (min separation) | time |
|---|---|---|
| max-div[DEFAULT, 12 workers] @ 60 s | 0.0962 | 60.4 s |
| max-div[DEFAULT] @ 60 s | 0.0941 | 60 s |
| max-div[DEFAULT, 12 workers] @ 1 s | 0.0938 | 1.35 s |
| max-div[DEFAULT] @ 1 s | 0.0909 | 1 s |
| RDKit MaxMinPicker[default] | 0.0863 | 0.0355 s |
| fpsample[kdline] | 0.0859 | 0.000231 s |
| fpsample[vanilla] | 0.0837 | 0.000192 s |
| skmatter[default] | 0.0837 | 0.00183 s |
| qc-selector[maxmin] | 0.0837 | 0.00575 s |
| code-FDM[default] | 0.0816 | 0.428 s |
| apricot-select[default] | 0.0208 | 0.47 s |
| kmedoids[default] | 0.0116 | 0.197 s |
| qc-selector[maxsum] | 0.0069 | 0.0115 s |

`max-div` reaches the best one-shot result (RDKit MaxMinPicker[default]) at a budget of 0.05 s with one worker and at a budget of 1 s with 12 workers.
