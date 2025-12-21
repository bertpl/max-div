## Scenario B

Fixed n=1000 & k=100 with varying number of constraints spanning random 1% portions of the n-range

### Uniform sampling.

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**861.2 nsec ± 1.5%**</span> | 166.0 μsec ± 0.4%                      | 165.6 μsec ± 0.3%                     | 162.5 μsec ± 0.5%                            |
| 100 | 1000 | 4            | <span style="color:#00aa00">**877.1 nsec ± 1.4%**</span> | 167.8 μsec ± 0.1%                      | 167.5 μsec ± 0.2%                     | 166.3 μsec ± 0.2%                            |
| 100 | 1000 | 8            | <span style="color:#00aa00">**883.4 nsec ± 1.5%**</span> | 176.3 μsec ± 2.2%                      | 172.2 μsec ± 0.3%                     | 173.4 μsec ± 1.0%                            |
| 100 | 1000 | 16           | <span style="color:#00aa00">**881.7 nsec ± 1.1%**</span> | 180.8 μsec ± 0.2%                      | 177.4 μsec ± 0.1%                     | 184.5 μsec ± 0.4%                            |
| 100 | 1000 | 32           | <span style="color:#00aa00">**887.0 nsec ± 0.8%**</span> | 235.2 μsec ± 1.8%                      | 225.3 μsec ± 2.5%                     | 249.2 μsec ± 2.5%                            |
| 100 | 1000 | 64           | <span style="color:#00aa00">**882.7 nsec ± 1.5%**</span> | 243.9 μsec ± 0.1%                      | 216.9 μsec ± 0.2%                     | 267.6 μsec ± 0.3%                            |
| 100 | 1000 | 128          | <span style="color:#00aa00">**869.3 nsec ± 1.4%**</span> | 319.3 μsec ± 0.2%                      | 268.3 μsec ± 0.1%                     | 365.0 μsec ± 0.1%                            |
| 100 | 1000 | 256          | <span style="color:#00aa00">**866.9 nsec ± 1.5%**</span> | 437.7 μsec ± 0.2%                      | 378.3 μsec ± 0.2%                     | 571.1 μsec ± 0.7%                            |
| 100 | 1000 | 384          | <span style="color:#00aa00">**886.6 nsec ± 1.6%**</span> | 565.5 μsec ± 0.1%                      | 492.6 μsec ± 0.2%                     | 1.212 msec ± 0.1%                            |
| 100 | 1000 | 512          | <span style="color:#00aa00">**856.2 nsec ± 0.5%**</span> | 705.6 μsec ± 0.2%                      | 624.9 μsec ± 0.2%                     | 2.044 msec ± 1.4%                            |
| 100 | 1000 | 768          | <span style="color:#00aa00">**878.6 nsec ± 1.4%**</span> | 990.9 μsec ± 0.2%                      | 920.5 μsec ± 0.2%                     | 3.772 msec ± 0.2%                            |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**863.1 nsec ± 0.4%**</span> | 1.395 msec ± 0.3%                      | 1.341 msec ± 0.2%                     | 5.426 msec ± 1.2%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**874.4 nsec ± 1.2%**</span> | 354.4 μsec ± 0.5%                      | 328.8 μsec ± 0.4%                     | 539.6 μsec ± 0.7%                            |

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
| 100 | 1000 | 256       | 0.0%                                        | 90.6%                                         | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 384       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | 46.6%                                         | <span style="color:#00aa00">**76.7%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 65.88%                                        | 78.88%                                        | <span style="color:#00aa00">**81.39%**</span> |

