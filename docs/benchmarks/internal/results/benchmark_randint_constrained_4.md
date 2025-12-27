### Non-uniform sampling (custom p).

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**11.02 μsec ± 0.6%**</span> | 166.9 μsec ± 0.2%                      | 167.0 μsec ± 0.3%                     | 163.2 μsec ± 0.2%                            |
| 100 | 1000 | 4            | <span style="color:#00aa00">**10.99 μsec ± 0.3%**</span> | 169.8 μsec ± 0.5%                      | 168.3 μsec ± 0.2%                     | 164.3 μsec ± 0.2%                            |
| 100 | 1000 | 8            | <span style="color:#00aa00">**10.97 μsec ± 0.5%**</span> | 172.5 μsec ± 0.3%                      | 170.8 μsec ± 0.2%                     | 172.1 μsec ± 0.2%                            |
| 100 | 1000 | 16           | <span style="color:#00aa00">**10.93 μsec ± 0.5%**</span> | 180.5 μsec ± 0.5%                      | 178.3 μsec ± 1.6%                     | 191.1 μsec ± 1.0%                            |
| 100 | 1000 | 32           | <span style="color:#00aa00">**11.04 μsec ± 1.2%**</span> | 208.1 μsec ± 2.4%                      | 191.4 μsec ± 0.8%                     | 215.4 μsec ± 0.4%                            |
| 100 | 1000 | 64           | <span style="color:#00aa00">**11.36 μsec ± 1.3%**</span> | 246.8 μsec ± 0.4%                      | 216.4 μsec ± 0.5%                     | 287.5 μsec ± 2.1%                            |
| 100 | 1000 | 128          | <span style="color:#00aa00">**11.31 μsec ± 2.7%**</span> | 336.6 μsec ± 6.5%                      | 284.6 μsec ± 1.7%                     | 375.1 μsec ± 0.8%                            |
| 100 | 1000 | 256          | <span style="color:#00aa00">**10.98 μsec ± 0.3%**</span> | 432.9 μsec ± 0.4%                      | 387.1 μsec ± 1.3%                     | 568.4 μsec ± 3.7%                            |
| 100 | 1000 | 384          | <span style="color:#00aa00">**11.12 μsec ± 1.0%**</span> | 557.8 μsec ± 0.3%                      | 486.7 μsec ± 0.6%                     | 1.196 msec ± 0.8%                            |
| 100 | 1000 | 512          | <span style="color:#00aa00">**11.59 μsec ± 5.1%**</span> | 700.5 μsec ± 1.4%                      | 625.7 μsec ± 0.5%                     | 2.139 msec ± 10.0%                           |
| 100 | 1000 | 768          | <span style="color:#00aa00">**10.95 μsec ± 0.4%**</span> | 966.8 μsec ± 0.3%                      | 910.6 μsec ± 0.6%                     | 3.705 msec ± 0.3%                            |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**11.15 μsec ± 0.6%**</span> | 1.375 msec ± 0.4%                      | 1.322 msec ± 0.5%                     | 5.320 msec ± 1.1%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**11.12 μsec ± 1.2%**</span> | 350.4 μsec ± 1.1%                      | 325.8 μsec ± 0.7%                     | 537.9 μsec ± 1.7%                            |

#### Accuracy Results

| `k` | `n`  | `m`       | `randint_numba`                             | `randint_constrained`<br>(eager=False)        | `randint_constrained`<br>(eager=True)         | `randint_constrained_robust`<br>(n_trials=5)  |
| --- | ---- | --------- | ------------------------------------------- | --------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| 100 | 1000 | 2         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 4         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 8         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 16        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 32        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 64        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 128       | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 256       | 0.0%                                        | 87.5%                                         | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 384       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | 48.0%                                         | <span style="color:#00aa00">**67.5%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 65.62%                                        | 79.00%                                        | <span style="color:#00aa00">**80.62%**</span> |

