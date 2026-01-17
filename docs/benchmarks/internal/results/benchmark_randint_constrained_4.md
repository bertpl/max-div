### Non-uniform sampling (custom p).

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**10.92 μsec ± 0.4%**</span> | 166.3 μsec ± 0.2%                      | 166.2 μsec ± 0.1%                     |
| 100 | 1000 | 4            | <span style="color:#00aa00">**11.03 μsec ± 1.0%**</span> | 168.2 μsec ± 0.3%                      | 168.3 μsec ± 0.2%                     |
| 100 | 1000 | 8            | <span style="color:#00aa00">**10.89 μsec ± 0.4%**</span> | 172.5 μsec ± 0.1%                      | 171.8 μsec ± 0.2%                     |
| 100 | 1000 | 16           | <span style="color:#00aa00">**11.31 μsec ± 2.2%**</span> | 180.8 μsec ± 0.2%                      | 177.2 μsec ± 0.2%                     |
| 100 | 1000 | 32           | <span style="color:#00aa00">**10.90 μsec ± 0.4%**</span> | 200.1 μsec ± 0.2%                      | 189.0 μsec ± 0.2%                     |
| 100 | 1000 | 64           | <span style="color:#00aa00">**10.91 μsec ± 0.5%**</span> | 243.3 μsec ± 0.3%                      | 214.5 μsec ± 0.1%                     |
| 100 | 1000 | 128          | <span style="color:#00aa00">**10.97 μsec ± 0.4%**</span> | 315.5 μsec ± 0.2%                      | 265.9 μsec ± 0.2%                     |
| 100 | 1000 | 256          | <span style="color:#00aa00">**10.96 μsec ± 0.4%**</span> | 430.7 μsec ± 0.1%                      | 372.0 μsec ± 0.2%                     |
| 100 | 1000 | 384          | <span style="color:#00aa00">**10.89 μsec ± 0.4%**</span> | 558.6 μsec ± 0.3%                      | 484.2 μsec ± 0.2%                     |
| 100 | 1000 | 512          | <span style="color:#00aa00">**10.95 μsec ± 0.8%**</span> | 698.9 μsec ± 0.2%                      | 609.8 μsec ± 0.4%                     |
| 100 | 1000 | 768          | <span style="color:#00aa00">**10.97 μsec ± 0.3%**</span> | 981.1 μsec ± 0.3%                      | 909.3 μsec ± 0.2%                     |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**10.96 μsec ± 0.4%**</span> | 1.413 msec ± 0.2%                      | 1.360 msec ± 0.3%                     |
|     |      | **Geomean:** | <span style="color:#00aa00">**10.97 μsec ± 0.6%**</span> | 347.7 μsec ± 0.2%                      | 322.1 μsec ± 0.2%                     |

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
| 100 | 1000 | 256       | 0.0%                                        | 87.5%                                         | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 384       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**48.0%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 65.62%                                        | <span style="color:#00aa00">**79.00%**</span> |

