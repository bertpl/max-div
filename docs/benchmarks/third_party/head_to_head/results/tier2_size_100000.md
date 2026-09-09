| tool | quality (min separation) | time |
|---|---|---|
| max-div[DEFAULT, 12 workers] @ 60 s | 0.0079 | 60.4 s |
| max-div[DEFAULT, 12 workers] @ 1 s | 0.0079 | 3.32 s |
| max-div[DEFAULT] @ 1 s | 0.0079 | 2.95 s |
| max-div[DEFAULT] @ 60 s | 0.0079 | 62 s |
| fpsample[FPS] | 0.0079 | 1.76 s |
| fpsample[KDLine] | 0.0079 | 0.0464 s |
| skmatter[FPS] | 0.0079 | 2.26 s |

`max-div` reaches the best one-shot result (fpsample[FPS]) at a budget of 0.001 s with one worker and at a budget of 1 s with 12 workers.
