# `randint_constrained`

([API reference][max_div.sampling.con.randint_constrained])

Command:
```bash
uv tool install max-div
max-div benchmark --markdown randint_constrained
```
or 
```bash
uv run max-div benchmark --markdown randint_constrained
```

We compare the following situations:
- use `randint_numba`, i.e. we simply sample without taking constraints into account
- use `randint_constrained`, i.e. we sample taking constraints into account
- use `randint_constrained_robust`, i.e. we sample taking constraints into account with multiple trials to improve success rate

We test both speed (absolute time of a single call) and accuracy (% of cases the generated samples satisfied the constraints).


## Scenario A

Varying n & k with 10 non-overlapping constraints spanning equal portions of the n-range

### Uniform sampling.

#### Timing Results                                                                                                                                                                           

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**1.135 μsec ± 0.3%**</span> | 2.388 μsec ± 0.1%                      | 2.404 μsec ± 0.1%                     | 1.803 μsec ± 0.4%                            |
| 4   | 10   | 10           | <span style="color:#00aa00">**1.224 μsec ± 0.2%**</span> | 3.372 μsec ± 0.4%                      | 3.383 μsec ± 0.3%                     | 1.926 μsec ± 0.2%                            |
| 8   | 10   | 10           | <span style="color:#00aa00">**1.230 μsec ± 0.1%**</span> | 5.251 μsec ± 0.6%                      | 5.212 μsec ± 0.7%                     | 1.988 μsec ± 0.4%                            |
| 2   | 100  | 10           | <span style="color:#00aa00">**1.157 μsec ± 0.1%**</span> | 3.066 μsec ± 0.1%                      | 3.062 μsec ± 0.1%                     | 2.028 μsec ± 0.1%                            |
| 4   | 100  | 10           | <span style="color:#00aa00">**1.157 μsec ± 0.2%**</span> | 4.288 μsec ± 0.2%                      | 4.287 μsec ± 0.1%                     | 3.253 μsec ± 0.3%                            |
| 8   | 100  | 10           | <span style="color:#00aa00">**1.188 μsec ± 0.1%**</span> | 7.199 μsec ± 0.1%                      | 7.183 μsec ± 0.1%                     | 7.286 μsec ± 0.2%                            |
| 16  | 100  | 10           | <span style="color:#00aa00">**1.226 μsec ± 0.4%**</span> | 13.06 μsec ± 0.0%                      | 13.04 μsec ± 0.0%                     | 12.74 μsec ± 0.1%                            |
| 32  | 100  | 10           | <span style="color:#00aa00">**1.268 μsec ± 0.6%**</span> | 24.61 μsec ± 0.1%                      | 24.60 μsec ± 0.0%                     | 23.35 μsec ± 0.2%                            |
| 64  | 100  | 10           | <span style="color:#00aa00">**1.362 μsec ± 0.3%**</span> | 48.64 μsec ± 0.1%                      | 48.64 μsec ± 0.0%                     | 39.38 μsec ± 1.1%                            |
| 2   | 1000 | 10           | <span style="color:#00aa00">**1.247 μsec ± 0.9%**</span> | 7.783 μsec ± 0.1%                      | 7.788 μsec ± 0.2%                     | 2.698 μsec ± 0.5%                            |
| 4   | 1000 | 10           | <span style="color:#00aa00">**1.266 μsec ± 0.8%**</span> | 13.96 μsec ± 0.1%                      | 13.93 μsec ± 0.0%                     | 8.318 μsec ± 0.7%                            |
| 8   | 1000 | 10           | <span style="color:#00aa00">**1.296 μsec ± 0.5%**</span> | 27.13 μsec ± 0.1%                      | 27.19 μsec ± 0.1%                     | 27.04 μsec ± 0.2%                            |
| 16  | 1000 | 10           | <span style="color:#00aa00">**1.300 μsec ± 0.9%**</span> | 53.55 μsec ± 0.1%                      | 53.55 μsec ± 0.0%                     | 53.42 μsec ± 0.1%                            |
| 32  | 1000 | 10           | <span style="color:#00aa00">**1.334 μsec ± 0.2%**</span> | 105.6 μsec ± 0.0%                      | 105.6 μsec ± 0.1%                     | 105.4 μsec ± 0.3%                            |
| 64  | 1000 | 10           | <span style="color:#00aa00">**1.424 μsec ± 0.4%**</span> | 215.2 μsec ± 0.1%                      | 216.4 μsec ± 0.1%                     | 209.8 μsec ± 0.4%                            |
| 128 | 1000 | 10           | <span style="color:#00aa00">**1.603 μsec ± 0.3%**</span> | 440.8 μsec ± 0.1%                      | 438.0 μsec ± 0.1%                     | 419.4 μsec ± 0.4%                            |
| 256 | 1000 | 10           | <span style="color:#00aa00">**1.850 μsec ± 0.4%**</span> | 894.4 μsec ± 0.1%                      | 894.7 μsec ± 0.1%                     | 840.7 μsec ± 0.8%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**1.300 μsec ± 0.4%**</span> | 22.38 μsec ± 0.1%                      | 22.37 μsec ± 0.1%                     | 17.17 μsec ± 0.4%                            |

#### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                               | `randint_constrained`<br>(eager=False)         | `randint_constrained`<br>(eager=True)          | `randint_constrained_robust`<br>(n_trials=5)   |
| --- | ---- | --------- | --------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 90.6%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 51.2%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 2.4%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 1.7%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 4.1%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 18.2%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 90.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 46.3%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 1.4%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.8%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 1.7%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 1.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.4%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.7%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 35.91%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |

### Non-uniform sampling (custom p).

#### Timing Results                                                                                                                                                                           

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**1.394 μsec ± 0.0%**</span> | 2.632 μsec ± 0.1%                      | 2.632 μsec ± 0.1%                     | 2.069 μsec ± 0.1%                            |
| 4   | 10   | 10           | <span style="color:#00aa00">**1.370 μsec ± 0.2%**</span> | 3.424 μsec ± 0.1%                      | 3.412 μsec ± 0.1%                     | 2.082 μsec ± 0.1%                            |
| 8   | 10   | 10           | <span style="color:#00aa00">**1.357 μsec ± 0.1%**</span> | 5.214 μsec ± 0.1%                      | 5.204 μsec ± 0.3%                     | 2.131 μsec ± 0.3%                            |
| 2   | 100  | 10           | <span style="color:#00aa00">**1.783 μsec ± 0.0%**</span> | 3.058 μsec ± 0.2%                      | 3.071 μsec ± 0.1%                     | 2.722 μsec ± 0.1%                            |
| 4   | 100  | 10           | <span style="color:#00aa00">**2.031 μsec ± 0.1%**</span> | 4.365 μsec ± 0.3%                      | 4.367 μsec ± 0.1%                     | 4.561 μsec ± 0.2%                            |
| 8   | 100  | 10           | <span style="color:#00aa00">**2.463 μsec ± 0.0%**</span> | 7.230 μsec ± 0.1%                      | 7.225 μsec ± 0.0%                     | 9.026 μsec ± 0.1%                            |
| 16  | 100  | 10           | <span style="color:#00aa00">**3.271 μsec ± 0.1%**</span> | 13.13 μsec ± 0.1%                      | 13.14 μsec ± 0.1%                     | 15.22 μsec ± 0.1%                            |
| 32  | 100  | 10           | <span style="color:#00aa00">**3.388 μsec ± 0.1%**</span> | 24.81 μsec ± 0.1%                      | 24.80 μsec ± 0.0%                     | 26.47 μsec ± 0.1%                            |
| 64  | 100  | 10           | <span style="color:#00aa00">**3.363 μsec ± 0.2%**</span> | 48.88 μsec ± 0.0%                      | 48.80 μsec ± 0.0%                     | 48.96 μsec ± 0.0%                            |
| 2   | 1000 | 10           | <span style="color:#00aa00">**5.267 μsec ± 0.2%**</span> | 8.256 μsec ± 0.1%                      | 8.275 μsec ± 0.1%                     | 7.424 μsec ± 0.2%                            |
| 4   | 1000 | 10           | <span style="color:#00aa00">**5.741 μsec ± 0.2%**</span> | 14.92 μsec ± 0.1%                      | 14.92 μsec ± 0.0%                     | 16.28 μsec ± 0.4%                            |
| 8   | 1000 | 10           | <span style="color:#00aa00">**6.836 μsec ± 0.2%**</span> | 29.03 μsec ± 0.0%                      | 29.00 μsec ± 0.0%                     | 35.20 μsec ± 0.0%                            |
| 16  | 1000 | 10           | <span style="color:#00aa00">**9.081 μsec ± 0.1%**</span> | 57.35 μsec ± 0.0%                      | 57.33 μsec ± 0.1%                     | 65.06 μsec ± 0.0%                            |
| 32  | 1000 | 10           | <span style="color:#00aa00">**12.90 μsec ± 0.2%**</span> | 113.2 μsec ± 0.0%                      | 113.1 μsec ± 0.1%                     | 124.5 μsec ± 0.0%                            |
| 64  | 1000 | 10           | <span style="color:#00aa00">**14.53 μsec ± 0.1%**</span> | 228.5 μsec ± 0.1%                      | 228.5 μsec ± 0.1%                     | 237.8 μsec ± 0.1%                            |
| 128 | 1000 | 10           | <span style="color:#00aa00">**15.26 μsec ± 0.2%**</span> | 462.7 μsec ± 0.0%                      | 462.7 μsec ± 0.1%                     | 466.2 μsec ± 0.1%                            |
| 256 | 1000 | 10           | <span style="color:#00aa00">**16.32 μsec ± 0.2%**</span> | 937.4 μsec ± 0.1%                      | 936.6 μsec ± 0.1%                     | 926.5 μsec ± 0.1%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**4.364 μsec ± 0.1%**</span> | 23.22 μsec ± 0.1%                      | 23.22 μsec ± 0.1%                     | 22.11 μsec ± 0.1%                            |

#### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                               | `randint_constrained`<br>(eager=False)         | `randint_constrained`<br>(eager=True)          | `randint_constrained_robust`<br>(n_trials=5)   |
| --- | ---- | --------- | --------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 86.7%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 40.2%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 0.2%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 79.2%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 29.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 0.2%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 31.53%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |

## Scenario B

Fixed n=1000 & k=100 with varying number of constraints spanning random 1% portions of the n-range

### Uniform sampling.

#### Timing Results                                                                                                                                                                           

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**1.547 μsec ± 1.0%**</span> | 262.9 μsec ± 0.0%                      | 262.6 μsec ± 0.0%                     | 259.8 μsec ± 0.0%                            |
| 100 | 1000 | 4            | <span style="color:#00aa00">**1.536 μsec ± 1.1%**</span> | 266.2 μsec ± 0.0%                      | 266.2 μsec ± 0.0%                     | 264.0 μsec ± 0.0%                            |
| 100 | 1000 | 8            | <span style="color:#00aa00">**1.540 μsec ± 0.9%**</span> | 273.9 μsec ± 0.0%                      | 273.2 μsec ± 0.1%                     | 274.2 μsec ± 0.0%                            |
| 100 | 1000 | 16           | <span style="color:#00aa00">**1.594 μsec ± 1.0%**</span> | 289.2 μsec ± 0.0%                      | 284.5 μsec ± 0.1%                     | 291.4 μsec ± 0.0%                            |
| 100 | 1000 | 32           | <span style="color:#00aa00">**1.564 μsec ± 1.1%**</span> | 317.4 μsec ± 0.1%                      | 304.0 μsec ± 0.1%                     | 331.0 μsec ± 0.0%                            |
| 100 | 1000 | 64           | <span style="color:#00aa00">**1.561 μsec ± 1.2%**</span> | 379.0 μsec ± 0.1%                      | 342.4 μsec ± 0.0%                     | 409.3 μsec ± 0.0%                            |
| 100 | 1000 | 128          | <span style="color:#00aa00">**1.575 μsec ± 0.9%**</span> | 502.0 μsec ± 0.0%                      | 433.6 μsec ± 0.0%                     | 565.2 μsec ± 0.1%                            |
| 100 | 1000 | 256          | <span style="color:#00aa00">**1.554 μsec ± 1.0%**</span> | 700.4 μsec ± 0.0%                      | 618.8 μsec ± 0.0%                     | 885.1 μsec ± 1.7%                            |
| 100 | 1000 | 384          | <span style="color:#00aa00">**1.534 μsec ± 1.1%**</span> | 917.6 μsec ± 0.1%                      | 811.9 μsec ± 0.1%                     | 1.947 msec ± 0.2%                            |
| 100 | 1000 | 512          | <span style="color:#00aa00">**1.526 μsec ± 0.8%**</span> | 1.170 msec ± 0.1%                      | 1.040 msec ± 0.1%                     | 3.340 msec ± 3.4%                            |
| 100 | 1000 | 768          | <span style="color:#00aa00">**1.546 μsec ± 1.0%**</span> | 1.683 msec ± 0.2%                      | 1.568 msec ± 0.1%                     | 6.396 msec ± 0.1%                            |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**1.579 μsec ± 0.9%**</span> | 2.399 msec ± 0.1%                      | 2.316 msec ± 0.2%                     | 9.357 msec ± 0.3%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**1.555 μsec ± 1.0%**</span> | 563.1 μsec ± 0.1%                      | 527.6 μsec ± 0.1%                     | 851.2 μsec ± 0.5%                            |

#### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                             | `randint_constrained`<br>(eager=False)        | `randint_constrained`<br>(eager=True)         | `randint_constrained_robust`<br>(n_trials=5)  |
| --- | ---- | --------- | ------------------------------------------- | --------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| 100 | 1000 | 2         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 4         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 8         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 16        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 32        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 64        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 128       | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 256       | 0.0%                                        | 90.6%                                         | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 384       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | 46.6%                                         | <span style="color:#00aa00">**76.6%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 65.88%                                        | 78.88%                                        | <span style="color:#00aa00">**81.38%**</span> |

### Non-uniform sampling (custom p).

#### Timing Results                                                                                                                                                                           

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**15.15 μsec ± 0.2%**</span> | 287.8 μsec ± 0.0%                      | 287.8 μsec ± 0.0%                     | 299.3 μsec ± 0.1%                            |
| 100 | 1000 | 4            | <span style="color:#00aa00">**15.15 μsec ± 0.1%**</span> | 291.0 μsec ± 0.0%                      | 291.0 μsec ± 0.0%                     | 303.1 μsec ± 0.1%                            |
| 100 | 1000 | 8            | <span style="color:#00aa00">**15.11 μsec ± 0.2%**</span> | 299.0 μsec ± 0.0%                      | 298.4 μsec ± 0.0%                     | 313.1 μsec ± 0.0%                            |
| 100 | 1000 | 16           | <span style="color:#00aa00">**15.15 μsec ± 0.2%**</span> | 313.9 μsec ± 0.0%                      | 309.4 μsec ± 0.0%                     | 332.8 μsec ± 0.1%                            |
| 100 | 1000 | 32           | <span style="color:#00aa00">**15.19 μsec ± 0.1%**</span> | 342.5 μsec ± 0.0%                      | 329.1 μsec ± 0.1%                     | 369.9 μsec ± 0.1%                            |
| 100 | 1000 | 64           | <span style="color:#00aa00">**15.08 μsec ± 0.3%**</span> | 406.2 μsec ± 0.1%                      | 369.1 μsec ± 0.1%                     | 449.3 μsec ± 0.1%                            |
| 100 | 1000 | 128          | <span style="color:#00aa00">**15.10 μsec ± 0.2%**</span> | 527.1 μsec ± 0.1%                      | 458.8 μsec ± 0.0%                     | 604.4 μsec ± 0.1%                            |
| 100 | 1000 | 256          | <span style="color:#00aa00">**15.15 μsec ± 0.2%**</span> | 725.2 μsec ± 0.0%                      | 643.6 μsec ± 0.0%                     | 930.9 μsec ± 0.7%                            |
| 100 | 1000 | 384          | <span style="color:#00aa00">**15.10 μsec ± 0.1%**</span> | 991.8 μsec ± 0.1%                      | 867.9 μsec ± 0.1%                     | 2.075 msec ± 0.4%                            |
| 100 | 1000 | 512          | <span style="color:#00aa00">**15.07 μsec ± 0.1%**</span> | 1.190 msec ± 0.1%                      | 1.059 msec ± 0.1%                     | 3.345 msec ± 5.2%                            |
| 100 | 1000 | 768          | <span style="color:#00aa00">**15.19 μsec ± 0.3%**</span> | 1.708 msec ± 0.1%                      | 1.594 msec ± 0.1%                     | 6.536 msec ± 0.2%                            |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**15.15 μsec ± 0.2%**</span> | 2.428 msec ± 0.2%                      | 2.345 msec ± 0.2%                     | 9.490 msec ± 0.3%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**15.13 μsec ± 0.2%**</span> | 596.9 μsec ± 0.1%                      | 560.1 μsec ± 0.1%                     | 922.6 μsec ± 0.6%                            |

#### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                             | `randint_constrained`<br>(eager=False)        | `randint_constrained`<br>(eager=True)         | `randint_constrained_robust`<br>(n_trials=5)  |
| --- | ---- | --------- | ------------------------------------------- | --------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| 100 | 1000 | 2         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 4         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 8         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 16        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 32        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 64        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 128       | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 256       | 0.0%                                        | 89.5%                                         | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 384       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | 48.1%                                         | <span style="color:#00aa00">**74.3%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 65.79%                                        | 79.01%                                        | <span style="color:#00aa00">**81.19%**</span> |
