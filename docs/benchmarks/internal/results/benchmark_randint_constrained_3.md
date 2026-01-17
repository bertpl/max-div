## Scenario B

Fixed n=1000 & k=100 with varying number of constraints spanning random 1% portions of the n-range

### Uniform sampling.

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**880.5 nsec ± 1.2%**</span> | 165.9 μsec ± 0.9%                      | 165.7 μsec ± 0.1%                     |
| 100 | 1000 | 4            | <span style="color:#00aa00">**880.4 nsec ± 1.2%**</span> | 168.2 μsec ± 0.2%                      | 168.1 μsec ± 0.2%                     |
| 100 | 1000 | 8            | <span style="color:#00aa00">**883.5 nsec ± 1.3%**</span> | 170.8 μsec ± 0.4%                      | 169.7 μsec ± 0.2%                     |
| 100 | 1000 | 16           | <span style="color:#00aa00">**880.3 nsec ± 1.3%**</span> | 180.8 μsec ± 0.4%                      | 175.3 μsec ± 0.1%                     |
| 100 | 1000 | 32           | <span style="color:#00aa00">**870.8 nsec ± 1.4%**</span> | 199.1 μsec ± 0.4%                      | 189.5 μsec ± 0.3%                     |
| 100 | 1000 | 64           | <span style="color:#00aa00">**899.2 nsec ± 1.2%**</span> | 243.8 μsec ± 0.1%                      | 217.3 μsec ± 0.1%                     |
| 100 | 1000 | 128          | <span style="color:#00aa00">**893.1 nsec ± 0.9%**</span> | 316.1 μsec ± 0.2%                      | 265.9 μsec ± 0.2%                     |
| 100 | 1000 | 256          | <span style="color:#00aa00">**875.8 nsec ± 1.6%**</span> | 430.3 μsec ± 0.2%                      | 371.3 μsec ± 0.1%                     |
| 100 | 1000 | 384          | <span style="color:#00aa00">**885.8 nsec ± 0.8%**</span> | 558.1 μsec ± 0.2%                      | 486.2 μsec ± 0.2%                     |
| 100 | 1000 | 512          | <span style="color:#00aa00">**909.9 nsec ± 1.8%**</span> | 702.8 μsec ± 0.5%                      | 612.7 μsec ± 0.3%                     |
| 100 | 1000 | 768          | <span style="color:#00aa00">**875.8 nsec ± 1.5%**</span> | 979.3 μsec ± 0.3%                      | 907.8 μsec ± 0.3%                     |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**882.3 nsec ± 1.1%**</span> | 1.425 msec ± 0.3%                      | 1.368 msec ± 0.4%                     |
|     |      | **Geomean:** | <span style="color:#00aa00">**884.7 nsec ± 1.3%**</span> | 347.6 μsec ± 0.3%                      | 322.1 μsec ± 0.2%                     |

#### Accuracy Results

| `k` | `n`  | `m`       | `randint_numba`                             | `randint_constrained`<br>(eager=False)        | `randint_constrained`<br>(eager=True)         |
| --- | ---- | --------- | ------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| 100 | 1000 | 2         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 4         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 8         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 16        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 32        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 64        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 128       | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 256       | 0.0%                                        | 93.5%                                         | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 384       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**43.5%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 66.12%                                        | <span style="color:#00aa00">**78.62%**</span> |

