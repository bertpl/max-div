### Non-uniform sampling (custom p).

#### Timing Results

| `k` | `n`  | `m`          | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**9.460 μsec ± 5.2%**</span> | 161.4 μsec ± 0.2%                      | 161.5 μsec ± 0.2%                     |
| 100 | 1000 | 4            | <span style="color:#00aa00">**9.262 μsec ± 0.2%**</span> | 163.6 μsec ± 0.3%                      | 164.2 μsec ± 0.2%                     |
| 100 | 1000 | 8            | <span style="color:#00aa00">**9.234 μsec ± 0.3%**</span> | 167.6 μsec ± 0.6%                      | 167.2 μsec ± 0.3%                     |
| 100 | 1000 | 16           | <span style="color:#00aa00">**9.251 μsec ± 5.5%**</span> | 175.4 μsec ± 0.3%                      | 172.2 μsec ± 0.1%                     |
| 100 | 1000 | 32           | <span style="color:#00aa00">**9.289 μsec ± 5.2%**</span> | 194.5 μsec ± 0.1%                      | 184.4 μsec ± 0.5%                     |
| 100 | 1000 | 64           | <span style="color:#00aa00">**9.211 μsec ± 0.3%**</span> | 236.7 μsec ± 0.3%                      | 210.7 μsec ± 0.1%                     |
| 100 | 1000 | 128          | <span style="color:#00aa00">**9.188 μsec ± 0.2%**</span> | 310.7 μsec ± 0.2%                      | 260.2 μsec ± 0.3%                     |
| 100 | 1000 | 256          | <span style="color:#00aa00">**10.24 μsec ± 4.9%**</span> | 421.6 μsec ± 0.2%                      | 363.7 μsec ± 0.1%                     |
| 100 | 1000 | 384          | <span style="color:#00aa00">**9.248 μsec ± 0.5%**</span> | 550.0 μsec ± 0.2%                      | 476.9 μsec ± 0.1%                     |
| 100 | 1000 | 512          | <span style="color:#00aa00">**9.217 μsec ± 0.5%**</span> | 690.1 μsec ± 0.2%                      | 605.5 μsec ± 0.2%                     |
| 100 | 1000 | 768          | <span style="color:#00aa00">**9.471 μsec ± 1.3%**</span> | 971.2 μsec ± 0.4%                      | 900.7 μsec ± 0.5%                     |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**9.181 μsec ± 0.6%**</span> | 1.402 msec ± 0.3%                      | 1.350 msec ± 0.4%                     |
|     |      | **Geomean:** | <span style="color:#00aa00">**9.350 μsec ± 2.0%**</span> | 340.4 μsec ± 0.3%                      | 315.9 μsec ± 0.3%                     |

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
| 100 | 1000 | 256       | 0.0%                                        | 98.5%                                         | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 384       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**44.0%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 66.54%                                        | <span style="color:#00aa00">**78.67%**</span> |
