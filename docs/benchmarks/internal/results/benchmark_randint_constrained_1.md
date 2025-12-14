## Scenario A

Varying n & k with 10 non-overlapping constraints spanning equal portions of the n-range

### Uniform sampling.

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**532.8 nsec ± 0.6%**</span> | 1.298 μsec ± 0.8%                      | 1.281 μsec ± 0.6%                     | 917.8 nsec ± 1.0%                            |
| 4   | 10   | 10           | <span style="color:#00aa00">**552.2 nsec ± 0.9%**</span> | 1.795 μsec ± 0.4%                      | 1.790 μsec ± 0.6%                     | 942.8 nsec ± 1.1%                            |
| 8   | 10   | 10           | <span style="color:#00aa00">**576.8 nsec ± 1.0%**</span> | 2.958 μsec ± 0.5%                      | 2.956 μsec ± 0.6%                     | 1.028 μsec ± 1.8%                            |
| 2   | 100  | 10           | <span style="color:#00aa00">**605.2 nsec ± 1.0%**</span> | 1.814 μsec ± 0.5%                      | 1.824 μsec ± 0.3%                     | 1.137 μsec ± 0.7%                            |
| 4   | 100  | 10           | <span style="color:#00aa00">**609.6 nsec ± 1.8%**</span> | 2.806 μsec ± 1.3%                      | 2.931 μsec ± 2.3%                     | 2.109 μsec ± 2.0%                            |
| 8   | 100  | 10           | <span style="color:#00aa00">**619.0 nsec ± 0.7%**</span> | 5.075 μsec ± 1.2%                      | 5.020 μsec ± 1.3%                     | 5.139 μsec ± 3.0%                            |
| 16  | 100  | 10           | <span style="color:#00aa00">**694.0 nsec ± 2.0%**</span> | 9.450 μsec ± 0.2%                      | 9.074 μsec ± 0.1%                     | 9.615 μsec ± 0.3%                            |
| 32  | 100  | 10           | <span style="color:#00aa00">**708.3 nsec ± 0.8%**</span> | 18.40 μsec ± 1.4%                      | 18.24 μsec ± 0.2%                     | 17.84 μsec ± 3.3%                            |
| 64  | 100  | 10           | <span style="color:#00aa00">**785.4 nsec ± 0.7%**</span> | 37.05 μsec ± 1.0%                      | 36.35 μsec ± 0.4%                     | 29.60 μsec ± 1.1%                            |
| 2   | 1000 | 10           | <span style="color:#00aa00">**620.2 nsec ± 1.0%**</span> | 4.821 μsec ± 0.7%                      | 4.812 μsec ± 0.7%                     | 1.625 μsec ± 1.3%                            |
| 4   | 1000 | 10           | <span style="color:#00aa00">**626.3 nsec ± 1.2%**</span> | 8.890 μsec ± 0.5%                      | 8.933 μsec ± 1.0%                     | 5.406 μsec ± 0.8%                            |
| 8   | 1000 | 10           | <span style="color:#00aa00">**630.6 nsec ± 1.1%**</span> | 17.89 μsec ± 1.3%                      | 17.67 μsec ± 0.5%                     | 18.15 μsec ± 0.5%                            |
| 16  | 1000 | 10           | <span style="color:#00aa00">**662.6 nsec ± 2.4%**</span> | 35.21 μsec ± 0.7%                      | 35.28 μsec ± 1.1%                     | 36.02 μsec ± 0.3%                            |
| 32  | 1000 | 10           | <span style="color:#00aa00">**691.7 nsec ± 0.9%**</span> | 69.94 μsec ± 0.1%                      | 70.35 μsec ± 0.1%                     | 71.24 μsec ± 0.6%                            |
| 64  | 1000 | 10           | <span style="color:#00aa00">**792.7 nsec ± 0.9%**</span> | 145.0 μsec ± 1.1%                      | 144.5 μsec ± 0.4%                     | 141.9 μsec ± 0.3%                            |
| 128 | 1000 | 10           | <span style="color:#00aa00">**959.4 nsec ± 0.4%**</span> | 294.6 μsec ± 0.1%                      | 294.3 μsec ± 0.1%                     | 285.0 μsec ± 0.3%                            |
| 256 | 1000 | 10           | <span style="color:#00aa00">**1.314 μsec ± 0.4%**</span> | 605.3 μsec ± 0.3%                      | 610.2 μsec ± 1.4%                     | 577.5 μsec ± 0.9%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**686.7 nsec ± 1.1%**</span> | 14.51 μsec ± 0.7%                      | 14.47 μsec ± 0.7%                     | 11.03 μsec ± 1.1%                            |

#### Accuracy Results

| `k` | `n`  | `m`       | `randint_numba`                               | `randint_constrained`<br>(eager=False)         | `randint_constrained`<br>(eager=True)          | `randint_constrained_robust`<br>(n_trials=5)   |
| --- | ---- | --------- | --------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 90.6%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 51.2%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 2.4%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 1.7%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 4.1%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 18.3%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 90.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 46.2%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 1.4%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.8%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 1.7%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 1.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.4%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.7%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 35.91%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |

