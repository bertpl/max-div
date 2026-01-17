### Non-uniform sampling (custom p).

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**644.0 nsec ± 0.3%**</span> | 1.411 μsec ± 0.3%                      | 1.386 μsec ± 0.2%                     |
| 4   | 10   | 10           | <span style="color:#00aa00">**674.7 nsec ± 0.5%**</span> | 1.933 μsec ± 0.3%                      | 1.940 μsec ± 0.2%                     |
| 8   | 10   | 10           | <span style="color:#00aa00">**652.3 nsec ± 0.4%**</span> | 3.017 μsec ± 0.2%                      | 3.019 μsec ± 0.5%                     |
| 2   | 100  | 10           | <span style="color:#00aa00">**1.102 μsec ± 0.5%**</span> | 1.973 μsec ± 0.4%                      | 1.957 μsec ± 0.1%                     |
| 4   | 100  | 10           | <span style="color:#00aa00">**1.230 μsec ± 0.2%**</span> | 2.892 μsec ± 0.2%                      | 2.967 μsec ± 0.1%                     |
| 8   | 100  | 10           | <span style="color:#00aa00">**1.516 μsec ± 0.2%**</span> | 5.104 μsec ± 0.2%                      | 5.094 μsec ± 0.3%                     |
| 16  | 100  | 10           | <span style="color:#00aa00">**2.190 μsec ± 0.4%**</span> | 9.549 μsec ± 0.5%                      | 9.220 μsec ± 0.2%                     |
| 32  | 100  | 10           | <span style="color:#00aa00">**2.266 μsec ± 0.1%**</span> | 18.22 μsec ± 0.5%                      | 17.57 μsec ± 0.1%                     |
| 64  | 100  | 10           | <span style="color:#00aa00">**2.320 μsec ± 0.7%**</span> | 36.20 μsec ± 0.3%                      | 35.79 μsec ± 0.6%                     |
| 2   | 1000 | 10           | <span style="color:#00aa00">**3.890 μsec ± 0.2%**</span> | 5.594 μsec ± 0.2%                      | 5.608 μsec ± 0.2%                     |
| 4   | 1000 | 10           | <span style="color:#00aa00">**4.214 μsec ± 0.2%**</span> | 9.722 μsec ± 0.2%                      | 9.706 μsec ± 0.1%                     |
| 8   | 1000 | 10           | <span style="color:#00aa00">**4.907 μsec ± 0.2%**</span> | 18.54 μsec ± 0.4%                      | 18.47 μsec ± 0.1%                     |
| 16  | 1000 | 10           | <span style="color:#00aa00">**6.305 μsec ± 0.1%**</span> | 36.30 μsec ± 0.3%                      | 36.40 μsec ± 0.1%                     |
| 32  | 1000 | 10           | <span style="color:#00aa00">**8.935 μsec ± 0.1%**</span> | 71.61 μsec ± 0.4%                      | 71.56 μsec ± 0.2%                     |
| 64  | 1000 | 10           | <span style="color:#00aa00">**10.62 μsec ± 0.4%**</span> | 145.0 μsec ± 0.3%                      | 144.6 μsec ± 0.2%                     |
| 128 | 1000 | 10           | <span style="color:#00aa00">**11.11 μsec ± 0.4%**</span> | 292.8 μsec ± 0.2%                      | 293.9 μsec ± 0.2%                     |
| 256 | 1000 | 10           | <span style="color:#00aa00">**11.91 μsec ± 0.5%**</span> | 595.7 μsec ± 0.2%                      | 594.6 μsec ± 0.2%                     |
|     |      | **Geomean:** | <span style="color:#00aa00">**2.811 μsec ± 0.3%**</span> | 15.01 μsec ± 0.3%                      | 14.94 μsec ± 0.2%                     |

#### Accuracy Results

| `k` | `n`  | `m`       | `randint_numba`                               | `randint_constrained`<br>(eager=False)         | `randint_constrained`<br>(eager=True)          |
| --- | ---- | --------- | --------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 87.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 42.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 85.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 43.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 32.91%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |

