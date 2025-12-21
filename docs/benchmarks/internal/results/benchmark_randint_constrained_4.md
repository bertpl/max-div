### Non-uniform sampling (custom p).

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**10.94 μsec ± 0.1%**</span> | 167.5 μsec ± 0.2%                      | 166.7 μsec ± 0.1%                     | 175.1 μsec ± 0.3%                            |
| 100 | 1000 | 4            | <span style="color:#00aa00">**10.98 μsec ± 0.3%**</span> | 168.4 μsec ± 0.1%                      | 168.7 μsec ± 0.1%                     | 177.3 μsec ± 0.2%                            |
| 100 | 1000 | 8            | <span style="color:#00aa00">**10.95 μsec ± 0.4%**</span> | 176.5 μsec ± 2.4%                      | 171.5 μsec ± 0.2%                     | 182.5 μsec ± 0.1%                            |
| 100 | 1000 | 16           | <span style="color:#00aa00">**10.97 μsec ± 0.2%**</span> | 182.0 μsec ± 0.4%                      | 178.1 μsec ± 0.2%                     | 195.2 μsec ± 0.2%                            |
| 100 | 1000 | 32           | <span style="color:#00aa00">**10.95 μsec ± 0.2%**</span> | 201.6 μsec ± 0.2%                      | 191.0 μsec ± 0.1%                     | 222.3 μsec ± 0.1%                            |
| 100 | 1000 | 64           | <span style="color:#00aa00">**10.94 μsec ± 0.4%**</span> | 245.7 μsec ± 0.1%                      | 218.0 μsec ± 0.1%                     | 278.5 μsec ± 0.1%                            |
| 100 | 1000 | 128          | <span style="color:#00aa00">**10.97 μsec ± 0.2%**</span> | 320.3 μsec ± 0.1%                      | 269.9 μsec ± 0.3%                     | 377.1 μsec ± 0.1%                            |
| 100 | 1000 | 256          | <span style="color:#00aa00">**10.96 μsec ± 0.2%**</span> | 439.1 μsec ± 0.1%                      | 377.6 μsec ± 0.1%                     | 578.6 μsec ± 0.8%                            |
| 100 | 1000 | 384          | <span style="color:#00aa00">**10.94 μsec ± 0.3%**</span> | 565.3 μsec ± 0.2%                      | 492.8 μsec ± 0.1%                     | 1.223 msec ± 0.3%                            |
| 100 | 1000 | 512          | <span style="color:#00aa00">**10.97 μsec ± 0.7%**</span> | 708.3 μsec ± 0.1%                      | 625.2 μsec ± 0.2%                     | 1.985 msec ± 3.4%                            |
| 100 | 1000 | 768          | <span style="color:#00aa00">**10.97 μsec ± 0.2%**</span> | 991.7 μsec ± 0.2%                      | 919.1 μsec ± 0.3%                     | 3.772 msec ± 0.7%                            |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**11.01 μsec ± 0.3%**</span> | 1.401 msec ± 0.2%                      | 1.341 msec ± 0.1%                     | 5.411 msec ± 0.2%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**10.96 μsec ± 0.3%**</span> | 351.1 μsec ± 0.3%                      | 324.9 μsec ± 0.2%                     | 548.4 μsec ± 0.5%                            |

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
| 100 | 1000 | 256       | 0.0%                                        | 89.5%                                         | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 384       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | 48.1%                                         | <span style="color:#00aa00">**74.2%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 65.79%                                        | 79.01%                                        | <span style="color:#00aa00">**81.18%**</span> |

