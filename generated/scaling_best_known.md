| Problem size n | Q_random | 50% threshold | 90% threshold | Best-known quality (min. separation) | Solver | Config | Measured time |
|---|---|---|---|---|---|---|---|
| 20 | 0.3962 | 0.7854 | 1.0967 | 1.1746 | max-div | `optimal-eager` | 895 s |
| 50 | 0.0427 | 0.3332 | 0.5655 | 0.6236 | max-div | `optimal-eager` | 895 s |
| 100 | 0.0114 | 0.1761 | 0.3078 | 0.3407 | max-div | `optimal-eager` | 895 s |
| 200 | 0.0054 | 0.1307 | 0.2310 | 0.2561 | max-div | `optimal-eager` | 895 s |
| 500 | 0.0029 | 0.0779 | 0.1379 | 0.1529 | OR-Tools CP-SAT | `optimal` | 18 s |
| 1,000 | 0.0012 | 0.0513 | 0.0915 | 0.1015 | OR-Tools CP-SAT | `optimal` | 131 s |
| 2,000 | 0.0006 | 0.0331 | 0.0592 | 0.0657 | OR-Tools CP-SAT | `optimal` | 899 s |
| 5,000 | 0.0002 | 0.0201 | 0.0360 | 0.0400 | max-div | `optimal-eager` | 895 s |
| 10,000 | 0.0001 | 0.0139 | 0.0249 | 0.0276 | max-div | `optimal-lazy` | 895 s |
| 20,000 | 0.0001 | 0.0096 | 0.0173 | 0.0192 | max-div | `optimal-lazy` | 895 s |
| 50,000 | 0.0000 | 0.0060 | 0.0108 | 0.0120 | max-div | `optimal-eager` | 895 s |
| 100,000 | 0.0000 | 0.0041 | 0.0074 | 0.0083 | max-div | `optimal-lazy` | 895 s |
| 200,000 | 0.0000 | 0.0028 | 0.0050 | 0.0056 | max-div | `optimal-lazy` | 895 s |
| 500,000 | 0.0000 | 0.0018 | 0.0032 | 0.0035 | max-div | `optimal-lazy` | 895 s |
| 1,000,000 | 0.0000 | 0.0012 | 0.0022 | 0.0025 | fpsample | `kdline` | 2 s |
| 2,000,000 | 0.0000 | 0.0009 | 0.0016 | 0.0017 | fpsample | `kdline` | 8 s |
| 5,000,000 | 0.0000 | 0.0006 | 0.0010 | 0.0011 | fpsample | `kdline` | 48 s |
