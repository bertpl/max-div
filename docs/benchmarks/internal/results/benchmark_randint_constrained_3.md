## Scenario B

Fixed n=1000 & k=100 with varying number of constraints spanning random 1% portions of the n-range

### Uniform sampling.

#### Timing Results

| `k` | `n`  | `m`          | `randint`                                                | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**734.8 nsec ± 1.6%**</span> | 141.8 μsec ± 0.3%                      | 142.8 μsec ± 0.9%                     |
| 100 | 1000 | 4            | <span style="color:#00aa00">**754.8 nsec ± 2.4%**</span> | 142.0 μsec ± 0.7%                      | 143.3 μsec ± 0.4%                     |
| 100 | 1000 | 8            | <span style="color:#00aa00">**748.4 nsec ± 1.3%**</span> | 145.5 μsec ± 0.8%                      | 143.9 μsec ± 0.3%                     |
| 100 | 1000 | 16           | <span style="color:#00aa00">**758.5 nsec ± 0.9%**</span> | 150.5 μsec ± 1.3%                      | 145.6 μsec ± 0.7%                     |
| 100 | 1000 | 32           | <span style="color:#00aa00">**751.4 nsec ± 1.3%**</span> | 161.4 μsec ± 0.4%                      | 150.0 μsec ± 0.3%                     |
| 100 | 1000 | 64           | <span style="color:#00aa00">**745.4 nsec ± 1.4%**</span> | 186.2 μsec ± 0.9%                      | 158.1 μsec ± 0.9%                     |
| 100 | 1000 | 128          | <span style="color:#00aa00">**744.2 nsec ± 1.9%**</span> | 220.3 μsec ± 1.5%                      | 170.1 μsec ± 0.9%                     |
| 100 | 1000 | 256          | <span style="color:#00aa00">**750.1 nsec ± 1.2%**</span> | 239.8 μsec ± 1.2%                      | 200.2 μsec ± 0.4%                     |
| 100 | 1000 | 384          | <span style="color:#00aa00">**753.1 nsec ± 1.9%**</span> | 263.8 μsec ± 0.5%                      | 223.1 μsec ± 0.6%                     |
| 100 | 1000 | 512          | <span style="color:#00aa00">**742.0 nsec ± 1.4%**</span> | 291.7 μsec ± 0.4%                      | 250.1 μsec ± 0.3%                     |
| 100 | 1000 | 768          | <span style="color:#00aa00">**738.3 nsec ± 1.0%**</span> | 333.5 μsec ± 0.2%                      | 298.5 μsec ± 0.3%                     |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**754.2 nsec ± 1.4%**</span> | 401.9 μsec ± 0.3%                      | 378.4 μsec ± 0.3%                     |
|     |      | **Geomean:** | <span style="color:#00aa00">**747.9 nsec ± 1.5%**</span> | 209.7 μsec ± 0.7%                      | 189.8 μsec ± 0.5%                     |

#### Accuracy Results

| `k` | `n`  | `m`       | `randint`                                   | `randint_constrained`<br>(eager=False)        | `randint_constrained`<br>(eager=True)         |
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
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**39.5%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 66.54%                                        | <span style="color:#00aa00">**78.29%**</span> |
