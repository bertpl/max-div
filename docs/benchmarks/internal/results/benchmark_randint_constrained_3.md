## Scenario B

Fixed n=1000 & k=100 with varying number of constraints spanning random 1% portions of the n-range

### Uniform sampling.

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**920.0 nsec ± 1.4%**</span> | 169.5 μsec ± 1.6%                      | 166.7 μsec ± 1.0%                     | 150.6 μsec ± 0.4%                            |
| 100 | 1000 | 4            | <span style="color:#00aa00">**1.036 μsec ± 9.4%**</span> | 182.9 μsec ± 6.8%                      | 172.5 μsec ± 0.9%                     | 158.8 μsec ± 1.3%                            |
| 100 | 1000 | 8            | <span style="color:#00aa00">**904.7 nsec ± 1.3%**</span> | 188.3 μsec ± 3.3%                      | 177.3 μsec ± 1.1%                     | 168.8 μsec ± 3.2%                            |
| 100 | 1000 | 16           | <span style="color:#00aa00">**929.3 nsec ± 1.1%**</span> | 183.3 μsec ± 1.3%                      | 178.7 μsec ± 0.8%                     | 179.8 μsec ± 2.4%                            |
| 100 | 1000 | 32           | <span style="color:#00aa00">**928.3 nsec ± 1.7%**</span> | 213.6 μsec ± 19.0%                     | 196.4 μsec ± 1.4%                     | 211.2 μsec ± 1.1%                            |
| 100 | 1000 | 64           | <span style="color:#00aa00">**934.8 nsec ± 1.8%**</span> | 253.4 μsec ± 1.8%                      | 222.4 μsec ± 1.2%                     | 267.8 μsec ± 0.7%                            |
| 100 | 1000 | 128          | <span style="color:#00aa00">**919.2 nsec ± 1.7%**</span> | 326.4 μsec ± 0.7%                      | 274.4 μsec ± 1.0%                     | 370.3 μsec ± 0.7%                            |
| 100 | 1000 | 256          | <span style="color:#00aa00">**909.7 nsec ± 1.3%**</span> | 440.9 μsec ± 0.8%                      | 381.7 μsec ± 0.8%                     | 578.8 μsec ± 3.1%                            |
| 100 | 1000 | 384          | <span style="color:#00aa00">**915.9 nsec ± 1.9%**</span> | 571.9 μsec ± 1.1%                      | 493.4 μsec ± 1.0%                     | 1.202 msec ± 0.8%                            |
| 100 | 1000 | 512          | <span style="color:#00aa00">**924.0 nsec ± 2.0%**</span> | 707.6 μsec ± 0.7%                      | 624.6 μsec ± 1.1%                     | 2.024 msec ± 10.2%                           |
| 100 | 1000 | 768          | <span style="color:#00aa00">**892.6 nsec ± 1.9%**</span> | 975.4 μsec ± 0.3%                      | 910.0 μsec ± 0.7%                     | 3.656 msec ± 0.3%                            |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**881.7 nsec ± 1.6%**</span> | 1.378 msec ± 0.4%                      | 1.316 msec ± 0.4%                     | 5.334 msec ± 0.4%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**924.0 nsec ± 2.3%**</span> | 358.7 μsec ± 3.0%                      | 327.8 μsec ± 1.0%                     | 522.9 μsec ± 2.0%                            |

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
| 100 | 1000 | 256       | 0.0%                                        | 93.5%                                         | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 384       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | 43.5%                                         | <span style="color:#00aa00">**75.0%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 66.12%                                        | 78.62%                                        | <span style="color:#00aa00">**81.25%**</span> |

