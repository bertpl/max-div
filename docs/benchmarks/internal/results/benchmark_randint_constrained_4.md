### Non-uniform sampling (custom p).

#### Timing Results

| `k` | `n`  | `m`          | `randint`                                                | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**9.864 μsec ± 0.8%**</span> | 139.4 μsec ± 0.3%                      | 139.5 μsec ± 0.4%                     |
| 100 | 1000 | 4            | <span style="color:#00aa00">**9.887 μsec ± 0.6%**</span> | 140.6 μsec ± 0.3%                      | 140.1 μsec ± 0.2%                     |
| 100 | 1000 | 8            | <span style="color:#00aa00">**9.868 μsec ± 0.9%**</span> | 146.8 μsec ± 1.2%                      | 142.6 μsec ± 0.8%                     |
| 100 | 1000 | 16           | <span style="color:#00aa00">**10.02 μsec ± 0.9%**</span> | 150.4 μsec ± 0.2%                      | 144.2 μsec ± 0.4%                     |
| 100 | 1000 | 32           | <span style="color:#00aa00">**9.934 μsec ± 0.3%**</span> | 161.6 μsec ± 0.5%                      | 148.9 μsec ± 0.7%                     |
| 100 | 1000 | 64           | <span style="color:#00aa00">**9.902 μsec ± 0.4%**</span> | 185.3 μsec ± 0.2%                      | 155.7 μsec ± 0.3%                     |
| 100 | 1000 | 128          | <span style="color:#00aa00">**9.920 μsec ± 0.8%**</span> | 217.9 μsec ± 0.4%                      | 167.7 μsec ± 0.2%                     |
| 100 | 1000 | 256          | <span style="color:#00aa00">**9.892 μsec ± 0.5%**</span> | 241.3 μsec ± 0.5%                      | 193.1 μsec ± 0.3%                     |
| 100 | 1000 | 384          | <span style="color:#00aa00">**9.922 μsec ± 0.3%**</span> | 262.3 μsec ± 0.7%                      | 219.0 μsec ± 0.4%                     |
| 100 | 1000 | 512          | <span style="color:#00aa00">**9.856 μsec ± 0.5%**</span> | 286.4 μsec ± 0.2%                      | 247.0 μsec ± 0.2%                     |
| 100 | 1000 | 768          | <span style="color:#00aa00">**9.897 μsec ± 0.4%**</span> | 334.5 μsec ± 0.3%                      | 301.4 μsec ± 0.3%                     |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**9.836 μsec ± 0.4%**</span> | 400.4 μsec ± 0.3%                      | 374.6 μsec ± 0.3%                     |
|     |      | **Geomean:** | <span style="color:#00aa00">**9.900 μsec ± 0.6%**</span> | 208.8 μsec ± 0.4%                      | 187.1 μsec ± 0.4%                     |

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
| 100 | 1000 | 384       | 0.0%                                        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**43.5%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 66.58%                                        | <span style="color:#00aa00">**78.62%**</span> |
