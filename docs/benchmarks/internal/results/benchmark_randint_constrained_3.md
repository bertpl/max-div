## Scenario B

Fixed n=1000 & k=100 with varying number of constraints spanning random 1% portions of the n-range

### Uniform sampling.

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**856.3 nsec ± 1.6%**</span> | 164.8 μsec ± 0.1%                      | 164.9 μsec ± 0.4%                     | 162.0 μsec ± 0.6%                            |
| 100 | 1000 | 4            | <span style="color:#00aa00">**888.7 nsec ± 1.7%**</span> | 167.1 μsec ± 0.7%                      | 168.0 μsec ± 0.6%                     | 166.1 μsec ± 0.4%                            |
| 100 | 1000 | 8            | <span style="color:#00aa00">**887.1 nsec ± 1.7%**</span> | 172.4 μsec ± 0.2%                      | 169.8 μsec ± 0.2%                     | 169.7 μsec ± 0.9%                            |
| 100 | 1000 | 16           | <span style="color:#00aa00">**882.7 nsec ± 1.7%**</span> | 178.8 μsec ± 2.2%                      | 193.1 μsec ± 2.1%                     | 185.4 μsec ± 3.3%                            |
| 100 | 1000 | 32           | <span style="color:#00aa00">**890.1 nsec ± 1.1%**</span> | 199.9 μsec ± 0.6%                      | 204.9 μsec ± 2.1%                     | 226.6 μsec ± 1.2%                            |
| 100 | 1000 | 64           | <span style="color:#00aa00">**889.4 nsec ± 1.8%**</span> | 245.1 μsec ± 1.3%                      | 217.8 μsec ± 0.2%                     | 266.9 μsec ± 0.2%                            |
| 100 | 1000 | 128          | <span style="color:#00aa00">**883.1 nsec ± 2.2%**</span> | 319.6 μsec ± 0.3%                      | 269.2 μsec ± 0.3%                     | 366.7 μsec ± 0.5%                            |
| 100 | 1000 | 256          | <span style="color:#00aa00">**876.6 nsec ± 1.9%**</span> | 436.3 μsec ± 0.2%                      | 379.8 μsec ± 0.6%                     | 562.0 μsec ± 0.9%                            |
| 100 | 1000 | 384          | <span style="color:#00aa00">**870.5 nsec ± 2.0%**</span> | 565.1 μsec ± 0.4%                      | 492.2 μsec ± 0.3%                     | 1.207 msec ± 0.9%                            |
| 100 | 1000 | 512          | <span style="color:#00aa00">**853.6 nsec ± 0.3%**</span> | 703.6 μsec ± 0.1%                      | 619.7 μsec ± 0.4%                     | 2.041 msec ± 1.7%                            |
| 100 | 1000 | 768          | <span style="color:#00aa00">**885.1 nsec ± 1.7%**</span> | 990.5 μsec ± 0.8%                      | 931.8 μsec ± 0.9%                     | 3.760 msec ± 0.5%                            |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**864.1 nsec ± 0.3%**</span> | 1.391 msec ± 0.1%                      | 1.339 msec ± 0.2%                     | 5.399 msec ± 0.2%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**877.2 nsec ± 1.5%**</span> | 348.2 μsec ± 0.6%                      | 328.4 μsec ± 0.7%                     | 533.2 μsec ± 0.9%                            |

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
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | 46.7%                                         | <span style="color:#00aa00">**76.6%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 65.88%                                        | 78.89%                                        | <span style="color:#00aa00">**81.38%**</span> |

