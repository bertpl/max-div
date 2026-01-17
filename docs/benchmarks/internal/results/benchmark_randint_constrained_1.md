## Scenario A

Varying n & k with 10 non-overlapping constraints spanning equal portions of the n-range

### Uniform sampling.

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**560.4 nsec ± 1.0%**</span> | 1.403 μsec ± 0.2%                      | 1.376 μsec ± 0.2%                     |
| 4   | 10   | 10           | <span style="color:#00aa00">**549.6 nsec ± 2.5%**</span> | 1.960 μsec ± 0.5%                      | 1.939 μsec ± 0.4%                     |
| 8   | 10   | 10           | <span style="color:#00aa00">**558.6 nsec ± 0.5%**</span> | 3.055 μsec ± 0.3%                      | 3.044 μsec ± 0.1%                     |
| 2   | 100  | 10           | <span style="color:#00aa00">**585.2 nsec ± 0.3%**</span> | 1.934 μsec ± 1.7%                      | 1.868 μsec ± 0.1%                     |
| 4   | 100  | 10           | <span style="color:#00aa00">**585.0 nsec ± 0.4%**</span> | 2.875 μsec ± 0.7%                      | 2.840 μsec ± 0.1%                     |
| 8   | 100  | 10           | <span style="color:#00aa00">**588.7 nsec ± 0.4%**</span> | 5.006 μsec ± 0.2%                      | 5.052 μsec ± 0.1%                     |
| 16  | 100  | 10           | <span style="color:#00aa00">**623.3 nsec ± 0.2%**</span> | 9.366 μsec ± 0.4%                      | 9.303 μsec ± 0.5%                     |
| 32  | 100  | 10           | <span style="color:#00aa00">**652.6 nsec ± 0.3%**</span> | 18.38 μsec ± 0.4%                      | 17.75 μsec ± 0.1%                     |
| 64  | 100  | 10           | <span style="color:#00aa00">**760.1 nsec ± 0.4%**</span> | 36.33 μsec ± 0.2%                      | 36.60 μsec ± 0.3%                     |
| 2   | 1000 | 10           | <span style="color:#00aa00">**622.4 nsec ± 0.6%**</span> | 4.978 μsec ± 0.2%                      | 4.985 μsec ± 0.2%                     |
| 4   | 1000 | 10           | <span style="color:#00aa00">**619.6 nsec ± 0.6%**</span> | 9.176 μsec ± 0.2%                      | 9.114 μsec ± 0.2%                     |
| 8   | 1000 | 10           | <span style="color:#00aa00">**626.0 nsec ± 0.9%**</span> | 18.07 μsec ± 0.2%                      | 17.94 μsec ± 0.2%                     |
| 16  | 1000 | 10           | <span style="color:#00aa00">**648.9 nsec ± 0.5%**</span> | 35.82 μsec ± 0.2%                      | 35.76 μsec ± 0.1%                     |
| 32  | 1000 | 10           | <span style="color:#00aa00">**695.2 nsec ± 0.7%**</span> | 71.24 μsec ± 0.2%                      | 71.17 μsec ± 0.2%                     |
| 64  | 1000 | 10           | <span style="color:#00aa00">**771.6 nsec ± 0.5%**</span> | 145.8 μsec ± 0.2%                      | 145.8 μsec ± 0.1%                     |
| 128 | 1000 | 10           | <span style="color:#00aa00">**960.3 nsec ± 0.3%**</span> | 298.4 μsec ± 0.2%                      | 299.1 μsec ± 0.1%                     |
| 256 | 1000 | 10           | <span style="color:#00aa00">**1.318 μsec ± 0.4%**</span> | 608.8 μsec ± 0.3%                      | 609.2 μsec ± 0.2%                     |
|     |      | **Geomean:** | <span style="color:#00aa00">**671.2 nsec ± 0.6%**</span> | 14.83 μsec ± 0.4%                      | 14.73 μsec ± 0.2%                     |

#### Accuracy Results

| `k` | `n`  | `m`       | `randint_numba`                               | `randint_constrained`<br>(eager=False)         | `randint_constrained`<br>(eager=True)          |
| --- | ---- | --------- | --------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 90.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 54.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 2.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 1.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 4.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 13.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 90.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 45.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 1.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 1.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 1.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 35.62%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |

