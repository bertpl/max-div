### Non-uniform sampling (custom p).

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**640.3 nsec ± 0.5%**</span> | 1.294 μsec ± 0.4%                      | 1.305 μsec ± 0.8%                     | 998.4 nsec ± 0.5%                            |
| 4   | 10   | 10           | <span style="color:#00aa00">**678.4 nsec ± 0.8%**</span> | 1.801 μsec ± 0.8%                      | 1.784 μsec ± 0.5%                     | 1.050 μsec ± 0.8%                            |
| 8   | 10   | 10           | <span style="color:#00aa00">**654.3 nsec ± 1.6%**</span> | 2.878 μsec ± 0.3%                      | 2.879 μsec ± 0.4%                     | 1.066 μsec ± 0.8%                            |
| 2   | 100  | 10           | <span style="color:#00aa00">**1.070 μsec ± 0.6%**</span> | 1.857 μsec ± 1.8%                      | 1.845 μsec ± 1.1%                     | 1.643 μsec ± 0.3%                            |
| 4   | 100  | 10           | <span style="color:#00aa00">**1.240 μsec ± 0.7%**</span> | 2.842 μsec ± 0.5%                      | 2.767 μsec ± 0.2%                     | 2.943 μsec ± 0.4%                            |
| 8   | 100  | 10           | <span style="color:#00aa00">**1.529 μsec ± 0.1%**</span> | 5.187 μsec ± 1.0%                      | 4.998 μsec ± 0.4%                     | 6.270 μsec ± 1.0%                            |
| 16  | 100  | 10           | <span style="color:#00aa00">**2.258 μsec ± 0.8%**</span> | 9.686 μsec ± 1.3%                      | 9.744 μsec ± 2.4%                     | 11.62 μsec ± 1.3%                            |
| 32  | 100  | 10           | <span style="color:#00aa00">**2.373 μsec ± 1.9%**</span> | 18.17 μsec ± 1.3%                      | 18.00 μsec ± 0.8%                     | 19.48 μsec ± 0.4%                            |
| 64  | 100  | 10           | <span style="color:#00aa00">**2.295 μsec ± 0.4%**</span> | 36.52 μsec ± 2.6%                      | 35.74 μsec ± 0.2%                     | 36.03 μsec ± 0.2%                            |
| 2   | 1000 | 10           | <span style="color:#00aa00">**3.917 μsec ± 0.2%**</span> | 5.473 μsec ± 0.3%                      | 5.498 μsec ± 0.2%                     | 5.015 μsec ± 0.2%                            |
| 4   | 1000 | 10           | <span style="color:#00aa00">**4.259 μsec ± 0.6%**</span> | 9.708 μsec ± 0.9%                      | 9.708 μsec ± 1.5%                     | 9.821 μsec ± 1.5%                            |
| 8   | 1000 | 10           | <span style="color:#00aa00">**4.928 μsec ± 0.3%**</span> | 18.18 μsec ± 0.3%                      | 18.20 μsec ± 0.2%                     | 23.46 μsec ± 0.3%                            |
| 16  | 1000 | 10           | <span style="color:#00aa00">**6.324 μsec ± 0.4%**</span> | 35.71 μsec ± 0.3%                      | 35.76 μsec ± 0.2%                     | 42.61 μsec ± 1.1%                            |
| 32  | 1000 | 10           | <span style="color:#00aa00">**8.906 μsec ± 0.1%**</span> | 70.23 μsec ± 0.2%                      | 70.07 μsec ± 0.2%                     | 79.61 μsec ± 0.2%                            |
| 64  | 1000 | 10           | <span style="color:#00aa00">**10.68 μsec ± 0.3%**</span> | 142.0 μsec ± 0.1%                      | 142.2 μsec ± 0.2%                     | 151.7 μsec ± 0.2%                            |
| 128 | 1000 | 10           | <span style="color:#00aa00">**11.20 μsec ± 0.6%**</span> | 289.1 μsec ± 0.2%                      | 289.1 μsec ± 0.1%                     | 295.3 μsec ± 0.2%                            |
| 256 | 1000 | 10           | <span style="color:#00aa00">**11.89 μsec ± 0.2%**</span> | 589.5 μsec ± 0.2%                      | 588.9 μsec ± 0.1%                     | 582.1 μsec ± 0.1%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**2.826 μsec ± 0.6%**</span> | 14.69 μsec ± 0.7%                      | 14.61 μsec ± 0.6%                     | 13.94 μsec ± 0.6%                            |

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
| 4   | 1000 | 10        | 45.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.1%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 0.1%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 33.32%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |

