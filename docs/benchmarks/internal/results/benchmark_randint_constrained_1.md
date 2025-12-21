## Scenario A

Varying n & k with 10 non-overlapping constraints spanning equal portions of the n-range

### Uniform sampling.

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**545.9 nsec ± 1.5%**</span> | 1.276 μsec ± 0.2%                      | 1.272 μsec ± 0.7%                     | 905.4 nsec ± 0.9%                            |
| 4   | 10   | 10           | <span style="color:#00aa00">**549.1 nsec ± 0.3%**</span> | 1.784 μsec ± 0.3%                      | 1.783 μsec ± 0.3%                     | 936.5 nsec ± 0.8%                            |
| 8   | 10   | 10           | <span style="color:#00aa00">**559.4 nsec ± 0.7%**</span> | 2.932 μsec ± 0.3%                      | 2.934 μsec ± 1.0%                     | 1.096 μsec ± 5.4%                            |
| 2   | 100  | 10           | <span style="color:#00aa00">**597.3 nsec ± 0.8%**</span> | 1.817 μsec ± 0.4%                      | 1.807 μsec ± 0.9%                     | 1.132 μsec ± 0.9%                            |
| 4   | 100  | 10           | <span style="color:#00aa00">**671.4 nsec ± 1.4%**</span> | 3.278 μsec ± 1.8%                      | 3.214 μsec ± 2.3%                     | 2.098 μsec ± 0.5%                            |
| 8   | 100  | 10           | <span style="color:#00aa00">**616.8 nsec ± 1.3%**</span> | 4.988 μsec ± 1.4%                      | 5.094 μsec ± 1.6%                     | 5.264 μsec ± 0.4%                            |
| 16  | 100  | 10           | <span style="color:#00aa00">**609.8 nsec ± 0.2%**</span> | 9.531 μsec ± 0.2%                      | 9.144 μsec ± 0.1%                     | 9.270 μsec ± 0.1%                            |
| 32  | 100  | 10           | <span style="color:#00aa00">**657.9 nsec ± 0.5%**</span> | 18.08 μsec ± 0.1%                      | 17.72 μsec ± 2.2%                     | 17.70 μsec ± 1.1%                            |
| 64  | 100  | 10           | <span style="color:#00aa00">**765.8 nsec ± 0.4%**</span> | 35.32 μsec ± 0.2%                      | 36.32 μsec ± 0.3%                     | 29.63 μsec ± 0.8%                            |
| 2   | 1000 | 10           | <span style="color:#00aa00">**610.3 nsec ± 0.4%**</span> | 4.788 μsec ± 0.3%                      | 4.799 μsec ± 0.2%                     | 1.590 μsec ± 0.6%                            |
| 4   | 1000 | 10           | <span style="color:#00aa00">**609.4 nsec ± 0.7%**</span> | 8.842 μsec ± 0.1%                      | 8.895 μsec ± 0.2%                     | 5.391 μsec ± 0.6%                            |
| 8   | 1000 | 10           | <span style="color:#00aa00">**632.0 nsec ± 0.7%**</span> | 17.56 μsec ± 0.3%                      | 17.69 μsec ± 0.1%                     | 18.12 μsec ± 0.2%                            |
| 16  | 1000 | 10           | <span style="color:#00aa00">**653.6 nsec ± 1.2%**</span> | 35.03 μsec ± 0.1%                      | 35.07 μsec ± 0.6%                     | 35.78 μsec ± 0.3%                            |
| 32  | 1000 | 10           | <span style="color:#00aa00">**696.2 nsec ± 1.0%**</span> | 69.84 μsec ± 0.1%                      | 69.94 μsec ± 0.1%                     | 70.48 μsec ± 0.3%                            |
| 64  | 1000 | 10           | <span style="color:#00aa00">**780.7 nsec ± 1.0%**</span> | 143.1 μsec ± 0.3%                      | 143.4 μsec ± 0.1%                     | 141.1 μsec ± 0.4%                            |
| 128 | 1000 | 10           | <span style="color:#00aa00">**956.0 nsec ± 0.6%**</span> | 293.4 μsec ± 0.1%                      | 293.7 μsec ± 0.1%                     | 284.1 μsec ± 0.6%                            |
| 256 | 1000 | 10           | <span style="color:#00aa00">**1.316 μsec ± 0.4%**</span> | 602.4 μsec ± 0.1%                      | 602.7 μsec ± 0.1%                     | 570.3 μsec ± 0.5%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**677.6 nsec ± 0.8%**</span> | 14.50 μsec ± 0.4%                      | 14.49 μsec ± 0.6%                     | 11.00 μsec ± 0.9%                            |

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
| 64  | 100  | 10        | 18.2%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 90.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 46.2%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 1.4%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.8%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 1.7%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 1.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.4%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.7%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 35.91%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |

