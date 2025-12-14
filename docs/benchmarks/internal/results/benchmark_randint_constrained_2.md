### Non-uniform sampling (custom p).

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**654.6 nsec ± 1.4%**</span> | 1.332 μsec ± 0.9%                      | 1.321 μsec ± 1.0%                     | 1.012 μsec ± 0.5%                            |
| 4   | 10   | 10           | <span style="color:#00aa00">**675.2 nsec ± 1.4%**</span> | 1.828 μsec ± 1.6%                      | 1.820 μsec ± 0.7%                     | 1.062 μsec ± 1.2%                            |
| 8   | 10   | 10           | <span style="color:#00aa00">**656.0 nsec ± 1.3%**</span> | 2.871 μsec ± 0.5%                      | 2.870 μsec ± 0.4%                     | 1.057 μsec ± 0.6%                            |
| 2   | 100  | 10           | <span style="color:#00aa00">**1.099 μsec ± 0.7%**</span> | 1.934 μsec ± 1.3%                      | 1.884 μsec ± 1.7%                     | 1.646 μsec ± 0.5%                            |
| 4   | 100  | 10           | <span style="color:#00aa00">**1.307 μsec ± 1.2%**</span> | 2.951 μsec ± 1.6%                      | 3.011 μsec ± 1.9%                     | 3.025 μsec ± 2.4%                            |
| 8   | 100  | 10           | <span style="color:#00aa00">**1.559 μsec ± 1.0%**</span> | 5.227 μsec ± 1.1%                      | 4.841 μsec ± 3.1%                     | 6.246 μsec ± 1.3%                            |
| 16  | 100  | 10           | <span style="color:#00aa00">**2.281 μsec ± 0.8%**</span> | 9.365 μsec ± 1.7%                      | 9.439 μsec ± 0.5%                     | 11.27 μsec ± 1.0%                            |
| 32  | 100  | 10           | <span style="color:#00aa00">**2.313 μsec ± 0.7%**</span> | 18.13 μsec ± 0.5%                      | 18.22 μsec ± 2.2%                     | 20.21 μsec ± 1.0%                            |
| 64  | 100  | 10           | <span style="color:#00aa00">**2.356 μsec ± 1.1%**</span> | 36.42 μsec ± 2.6%                      | 36.20 μsec ± 1.7%                     | 38.38 μsec ± 0.8%                            |
| 2   | 1000 | 10           | <span style="color:#00aa00">**3.956 μsec ± 0.6%**</span> | 5.510 μsec ± 0.5%                      | 5.525 μsec ± 1.0%                     | 5.108 μsec ± 0.8%                            |
| 4   | 1000 | 10           | <span style="color:#00aa00">**4.299 μsec ± 1.0%**</span> | 9.704 μsec ± 0.5%                      | 9.595 μsec ± 0.3%                     | 9.829 μsec ± 0.5%                            |
| 8   | 1000 | 10           | <span style="color:#00aa00">**4.951 μsec ± 0.6%**</span> | 18.29 μsec ± 0.3%                      | 18.33 μsec ± 0.4%                     | 23.48 μsec ± 0.5%                            |
| 16  | 1000 | 10           | <span style="color:#00aa00">**6.318 μsec ± 0.4%**</span> | 35.93 μsec ± 0.9%                      | 36.02 μsec ± 1.1%                     | 42.23 μsec ± 0.2%                            |
| 32  | 1000 | 10           | <span style="color:#00aa00">**8.909 μsec ± 0.2%**</span> | 70.91 μsec ± 0.6%                      | 70.78 μsec ± 0.8%                     | 80.04 μsec ± 0.4%                            |
| 64  | 1000 | 10           | <span style="color:#00aa00">**10.73 μsec ± 0.3%**</span> | 143.2 μsec ± 0.1%                      | 142.0 μsec ± 0.3%                     | 153.0 μsec ± 0.2%                            |
| 128 | 1000 | 10           | <span style="color:#00aa00">**11.14 μsec ± 0.2%**</span> | 290.3 μsec ± 0.2%                      | 289.6 μsec ± 0.2%                     | 294.9 μsec ± 0.2%                            |
| 256 | 1000 | 10           | <span style="color:#00aa00">**11.88 μsec ± 0.3%**</span> | 589.7 μsec ± 0.2%                      | 589.4 μsec ± 0.1%                     | 583.1 μsec ± 0.2%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**2.851 μsec ± 0.8%**</span> | 14.80 μsec ± 0.9%                      | 14.71 μsec ± 1.0%                     | 14.05 μsec ± 0.7%                            |

#### Accuracy Results

| `k` | `n`  | `m`       | `randint_numba`                               | `randint_constrained`<br>(eager=False)         | `randint_constrained`<br>(eager=True)          | `randint_constrained_robust`<br>(n_trials=5)   |
| --- | ---- | --------- | --------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 87.8%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 43.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 0.7%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 0.2%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 88.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 45.6%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.1%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 0.1%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 33.32%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |

