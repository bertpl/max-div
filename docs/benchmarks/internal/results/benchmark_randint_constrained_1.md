## Scenario A

Varying n & k with 10 non-overlapping constraints spanning equal portions of the n-range

### Uniform sampling.

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**552.8 nsec ± 1.4%**</span> | 1.432 μsec ± 0.7%                      | 1.429 μsec ± 0.5%                     | 921.6 nsec ± 0.3%                            |
| 4   | 10   | 10           | <span style="color:#00aa00">**538.1 nsec ± 0.3%**</span> | 1.976 μsec ± 0.5%                      | 1.944 μsec ± 0.8%                     | 953.1 nsec ± 0.3%                            |
| 8   | 10   | 10           | <span style="color:#00aa00">**570.7 nsec ± 1.1%**</span> | 3.089 μsec ± 0.2%                      | 3.076 μsec ± 0.4%                     | 1.027 μsec ± 1.2%                            |
| 2   | 100  | 10           | <span style="color:#00aa00">**584.9 nsec ± 0.5%**</span> | 2.001 μsec ± 2.3%                      | 1.941 μsec ± 0.3%                     | 1.135 μsec ± 0.4%                            |
| 4   | 100  | 10           | <span style="color:#00aa00">**593.2 nsec ± 0.7%**</span> | 2.916 μsec ± 1.0%                      | 2.889 μsec ± 0.3%                     | 2.066 μsec ± 0.6%                            |
| 8   | 100  | 10           | <span style="color:#00aa00">**602.6 nsec ± 0.6%**</span> | 5.439 μsec ± 1.6%                      | 5.311 μsec ± 2.7%                     | 5.499 μsec ± 1.7%                            |
| 16  | 100  | 10           | <span style="color:#00aa00">**669.9 nsec ± 2.0%**</span> | 10.16 μsec ± 1.4%                      | 10.41 μsec ± 1.6%                     | 9.860 μsec ± 1.7%                            |
| 32  | 100  | 10           | <span style="color:#00aa00">**681.2 nsec ± 1.2%**</span> | 19.02 μsec ± 2.8%                      | 17.97 μsec ± 0.5%                     | 17.94 μsec ± 1.3%                            |
| 64  | 100  | 10           | <span style="color:#00aa00">**775.6 nsec ± 1.5%**</span> | 35.68 μsec ± 0.3%                      | 36.34 μsec ± 0.2%                     | 29.37 μsec ± 2.3%                            |
| 2   | 1000 | 10           | <span style="color:#00aa00">**608.9 nsec ± 1.0%**</span> | 5.073 μsec ± 1.0%                      | 5.293 μsec ± 3.4%                     | 1.594 μsec ± 0.8%                            |
| 4   | 1000 | 10           | <span style="color:#00aa00">**624.1 nsec ± 1.1%**</span> | 9.109 μsec ± 0.1%                      | 9.495 μsec ± 1.5%                     | 5.306 μsec ± 2.5%                            |
| 8   | 1000 | 10           | <span style="color:#00aa00">**658.8 nsec ± 1.3%**</span> | 18.11 μsec ± 0.9%                      | 18.22 μsec ± 2.4%                     | 17.63 μsec ± 1.9%                            |
| 16  | 1000 | 10           | <span style="color:#00aa00">**647.0 nsec ± 1.2%**</span> | 38.71 μsec ± 7.4%                      | 41.63 μsec ± 8.5%                     | 35.77 μsec ± 2.1%                            |
| 32  | 1000 | 10           | <span style="color:#00aa00">**732.3 nsec ± 1.2%**</span> | 71.93 μsec ± 0.8%                      | 71.34 μsec ± 0.2%                     | 66.62 μsec ± 0.4%                            |
| 64  | 1000 | 10           | <span style="color:#00aa00">**793.8 nsec ± 0.5%**</span> | 145.8 μsec ± 0.2%                      | 145.1 μsec ± 0.4%                     | 137.8 μsec ± 2.0%                            |
| 128 | 1000 | 10           | <span style="color:#00aa00">**952.3 nsec ± 0.9%**</span> | 298.7 μsec ± 0.2%                      | 298.4 μsec ± 0.2%                     | 267.9 μsec ± 0.3%                            |
| 256 | 1000 | 10           | <span style="color:#00aa00">**1.315 μsec ± 0.3%**</span> | 609.3 μsec ± 0.1%                      | 609.7 μsec ± 0.1%                     | 544.5 μsec ± 0.2%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**681.6 nsec ± 1.0%**</span> | 15.16 μsec ± 1.3%                      | 15.21 μsec ± 1.4%                     | 10.90 μsec ± 1.2%                            |

#### Accuracy Results

| `k` | `n`  | `m`       | `randint_numba`                               | `randint_constrained`<br>(eager=False)         | `randint_constrained`<br>(eager=True)          | `randint_constrained_robust`<br>(n_trials=5)   |
| --- | ---- | --------- | --------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 90.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 53.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 2.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 1.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 4.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 14.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 90.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 45.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 1.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 1.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 1.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 35.62%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |

