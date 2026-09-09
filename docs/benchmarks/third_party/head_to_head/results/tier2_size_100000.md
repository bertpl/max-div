| tool | quality (min separation) | time |
|---|---|---|
| fpsample[kdline] | 0.0079 | 0.0464 s |
| fpsample[vanilla] | 0.0079 | 1.76 s |
| skmatter[default] | 0.0079 | 2.26 s |
| max-div[DEFAULT] @ 1 s | 0.0079 | 2.95 s |
| max-div[DEFAULT, 12 workers] @ 1 s | 0.0079 | 3.32 s |
| max-div[DEFAULT, 12 workers] @ 60 s | 0.0079 | 60.4 s |
| max-div[DEFAULT] @ 60 s | 0.0079 | 62 s |

`max-div` reaches the best one-shot result (fpsample[vanilla]) at a budget of 0.001 s with one worker and at a budget of 1 s with 12 workers.
