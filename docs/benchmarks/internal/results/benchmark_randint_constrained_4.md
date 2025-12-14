### Non-uniform sampling (custom p).

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) | `randint_constrained_robust`<br>(n_trials=5) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- | -------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**10.94 μsec ± 0.3%**</span> | 166.9 μsec ± 0.4%                      | 166.2 μsec ± 0.2%                     | 174.8 μsec ± 0.2%                            |
| 100 | 1000 | 4            | <span style="color:#00aa00">**10.96 μsec ± 0.2%**</span> | 168.8 μsec ± 0.3%                      | 168.2 μsec ± 0.2%                     | 176.8 μsec ± 0.2%                            |
| 100 | 1000 | 8            | <span style="color:#00aa00">**10.93 μsec ± 0.3%**</span> | 172.6 μsec ± 0.2%                      | 172.8 μsec ± 0.4%                     | 183.7 μsec ± 0.2%                            |
| 100 | 1000 | 16           | <span style="color:#00aa00">**10.93 μsec ± 0.3%**</span> | 181.7 μsec ± 0.4%                      | 178.7 μsec ± 0.4%                     | 195.2 μsec ± 0.2%                            |
| 100 | 1000 | 32           | <span style="color:#00aa00">**10.93 μsec ± 0.7%**</span> | 201.3 μsec ± 0.4%                      | 190.1 μsec ± 0.4%                     | 222.6 μsec ± 0.5%                            |
| 100 | 1000 | 64           | <span style="color:#00aa00">**10.97 μsec ± 0.3%**</span> | 246.4 μsec ± 0.3%                      | 217.8 μsec ± 0.2%                     | 278.8 μsec ± 0.3%                            |
| 100 | 1000 | 128          | <span style="color:#00aa00">**10.94 μsec ± 0.2%**</span> | 319.9 μsec ± 0.3%                      | 268.5 μsec ± 0.1%                     | 373.6 μsec ± 0.2%                            |
| 100 | 1000 | 256          | <span style="color:#00aa00">**10.92 μsec ± 0.4%**</span> | 436.4 μsec ± 0.1%                      | 376.7 μsec ± 0.1%                     | 575.6 μsec ± 0.8%                            |
| 100 | 1000 | 384          | <span style="color:#00aa00">**10.96 μsec ± 0.8%**</span> | 561.0 μsec ± 0.4%                      | 491.4 μsec ± 0.1%                     | 1.214 msec ± 0.3%                            |
| 100 | 1000 | 512          | <span style="color:#00aa00">**10.94 μsec ± 0.3%**</span> | 704.8 μsec ± 0.2%                      | 615.3 μsec ± 0.2%                     | 1.974 msec ± 3.1%                            |
| 100 | 1000 | 768          | <span style="color:#00aa00">**10.95 μsec ± 0.2%**</span> | 974.5 μsec ± 0.1%                      | 911.9 μsec ± 0.2%                     | 3.737 msec ± 0.2%                            |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**10.95 μsec ± 0.3%**</span> | 1.396 msec ± 0.3%                      | 1.345 msec ± 0.2%                     | 5.387 msec ± 0.2%                            |
|     |      | **Geomean:** | <span style="color:#00aa00">**10.94 μsec ± 0.4%**</span> | 349.2 μsec ± 0.3%                      | 324.1 μsec ± 0.2%                     | 546.7 μsec ± 0.5%                            |

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
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | 48.1%                                         | <span style="color:#00aa00">**74.3%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 65.79%                                        | 79.01%                                        | <span style="color:#00aa00">**81.19%**</span> |

