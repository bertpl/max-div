| Problem size n | Best-known quality (min. separation) | Solver | Config | Measured time |
|---|---|---|---|---|
| 20 | 1.1746 | max-div | `optimal-eager` | 895 s |
| 50 | 0.6236 | max-div | `optimal-eager` | 895 s |
| 100 | 0.3407 | max-div | `optimal-eager` | 895 s |
| 200 | 0.2561 | max-div | `optimal-eager` | 895 s |
| 500 | 0.1529 | OR-Tools CP-SAT | `optimal` | 18 s |
| 1,000 | 0.1015 | OR-Tools CP-SAT | `optimal` | 131 s |
| 2,000 | 0.0657 | OR-Tools CP-SAT | `optimal` | 899 s |
| 5,000 | 0.0400 | max-div | `optimal-eager` | 895 s |
| 10,000 | 0.0276 | max-div | `optimal-lazy` | 895 s |
| 20,000 | 0.0192 | max-div | `optimal-lazy` | 895 s |
| 50,000 | 0.0120 | max-div | `optimal-eager` | 895 s |
| 100,000 | 0.0083 | max-div | `optimal-lazy` | 895 s |
| 200,000 | 0.0056 | max-div | `optimal-lazy` | 895 s |
| 500,000 | 0.0035 | max-div | `optimal-lazy` | 895 s |
| 1,000,000 | 0.0025 | fpsample | `kdline` | 2 s |
| 2,000,000 | 0.0017 | fpsample | `vanilla` | 703 s |
| 5,000,000 | 0.0011 | fpsample | `kdline` | 48 s |
