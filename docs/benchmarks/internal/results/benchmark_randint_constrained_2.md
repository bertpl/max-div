### Non-uniform sampling (custom p).

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**655.5 nsec ± 0.4%**</span> | 1.442 μsec ± 1.0%                      | 1.457 μsec ± 0.3%                     | 1.026 μsec ± 0.6%                            |
| 4   | 10   | 10           | <span style="color:#00aa00">**673.4 nsec ± 0.2%**</span> | 1.957 μsec ± 0.5%                      | 1.950 μsec ± 0.3%                     | 1.063 μsec ± 0.3%                            |
| 8   | 10   | 10           | <span style="color:#00aa00">**665.9 nsec ± 0.4%**</span> | 3.159 μsec ± 1.5%                      | 3.041 μsec ± 0.4%                     | 1.081 μsec ± 0.3%                            |
| 2   | 100  | 10           | <span style="color:#00aa00">**1.083 μsec ± 1.0%**</span> | 2.018 μsec ± 1.1%                      | 1.978 μsec ± 0.3%                     | 1.642 μsec ± 0.3%                            |
| 4   | 100  | 10           | <span style="color:#00aa00">**1.239 μsec ± 0.2%**</span> | 3.027 μsec ± 0.4%                      | 3.027 μsec ± 0.2%                     | 2.949 μsec ± 1.0%                            |
| 8   | 100  | 10           | <span style="color:#00aa00">**1.525 μsec ± 0.4%**</span> | 5.227 μsec ± 1.1%                      | 5.024 μsec ± 0.3%                     | 6.514 μsec ± 1.8%                            |
| 16  | 100  | 10           | <span style="color:#00aa00">**2.309 μsec ± 1.0%**</span> | 9.804 μsec ± 0.7%                      | 9.582 μsec ± 2.1%                     | 11.78 μsec ± 2.4%                            |
| 32  | 100  | 10           | <span style="color:#00aa00">**2.358 μsec ± 1.1%**</span> | 19.20 μsec ± 3.8%                      | 18.90 μsec ± 2.0%                     | 19.59 μsec ± 0.5%                            |
| 64  | 100  | 10           | <span style="color:#00aa00">**2.934 μsec ± 9.1%**</span> | 47.64 μsec ± 15.7%                     | 39.44 μsec ± 4.6%                     | 37.65 μsec ± 1.5%                            |
| 2   | 1000 | 10           | <span style="color:#00aa00">**4.133 μsec ± 4.4%**</span> | 5.856 μsec ± 1.4%                      | 5.648 μsec ± 1.4%                     | 5.289 μsec ± 3.3%                            |
| 4   | 1000 | 10           | <span style="color:#00aa00">**4.359 μsec ± 1.1%**</span> | 9.717 μsec ± 1.9%                      | 9.986 μsec ± 2.9%                     | 9.949 μsec ± 1.8%                            |
| 8   | 1000 | 10           | <span style="color:#00aa00">**5.084 μsec ± 2.8%**</span> | 19.27 μsec ± 2.0%                      | 19.55 μsec ± 0.2%                     | 23.70 μsec ± 1.6%                            |
| 16  | 1000 | 10           | <span style="color:#00aa00">**6.514 μsec ± 1.0%**</span> | 36.70 μsec ± 0.8%                      | 37.03 μsec ± 1.8%                     | 39.98 μsec ± 0.3%                            |
| 32  | 1000 | 10           | <span style="color:#00aa00">**8.972 μsec ± 0.6%**</span> | 72.84 μsec ± 1.4%                      | 71.81 μsec ± 0.2%                     | 78.66 μsec ± 1.3%                            |
| 64  | 1000 | 10           | <span style="color:#00aa00">**10.99 μsec ± 1.9%**</span> | 146.2 μsec ± 2.7%                      | 157.5 μsec ± 7.4%                     | 168.8 μsec ± 0.6%                            |
| 128 | 1000 | 10           | <span style="color:#00aa00">**13.18 μsec ± 0.7%**</span> | 344.0 μsec ± 0.6%                      | 345.1 μsec ± 0.7%                     | 327.0 μsec ± 2.1%                            |
| 256 | 1000 | 10           | <span style="color:#00aa00">**14.05 μsec ± 0.9%**</span> | 703.4 μsec ± 1.5%                      | 609.8 μsec ± 0.8%                     | 551.0 μsec ± 0.4%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**2.963 μsec ± 1.6%**</span> | 15.90 μsec ± 2.3%                      | 15.54 μsec ± 1.5%                     | 14.19 μsec ± 1.2%                            |

#### Accuracy Results

| `k` | `n`  | `m`       | `randint_numba`                               | `randint_constrained`<br>(eager=False)         | `randint_constrained`<br>(eager=True)          | `randint_constrained_robust`<br>(n_trials=5)   |
| --- | ---- | --------- | --------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 87.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 42.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 85.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 43.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 32.88%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |

