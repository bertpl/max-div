### Non-uniform sampling (custom p).

#### Timing Results

| `k` | `n`  | `m`          | `randint`                                                | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**718.1 nsec ± 1.0%**</span> | 1.608 μsec ± 2.0%                      | 1.600 μsec ± 1.0%                     |
| 4   | 10   | 10           | <span style="color:#00aa00">**729.5 nsec ± 1.6%**</span> | 1.899 μsec ± 1.7%                      | 1.913 μsec ± 1.5%                     |
| 8   | 10   | 10           | <span style="color:#00aa00">**714.2 nsec ± 1.0%**</span> | 2.554 μsec ± 1.4%                      | 2.502 μsec ± 1.2%                     |
| 2   | 100  | 10           | <span style="color:#00aa00">**1.056 μsec ± 3.8%**</span> | 2.141 μsec ± 1.9%                      | 2.186 μsec ± 0.8%                     |
| 4   | 100  | 10           | <span style="color:#00aa00">**1.118 μsec ± 0.5%**</span> | 2.973 μsec ± 0.4%                      | 2.912 μsec ± 0.2%                     |
| 8   | 100  | 10           | <span style="color:#00aa00">**1.429 μsec ± 0.6%**</span> | 4.602 μsec ± 0.7%                      | 4.607 μsec ± 0.6%                     |
| 16  | 100  | 10           | <span style="color:#00aa00">**2.308 μsec ± 0.5%**</span> | 8.148 μsec ± 0.4%                      | 8.354 μsec ± 0.6%                     |
| 32  | 100  | 10           | <span style="color:#00aa00">**2.388 μsec ± 0.5%**</span> | 13.17 μsec ± 0.9%                      | 13.13 μsec ± 0.4%                     |
| 64  | 100  | 10           | <span style="color:#00aa00">**2.361 μsec ± 1.3%**</span> | 23.27 μsec ± 1.0%                      | 23.20 μsec ± 1.8%                     |
| 2   | 1000 | 10           | <span style="color:#00aa00">**2.431 μsec ± 0.8%**</span> | 7.021 μsec ± 1.0%                      | 7.001 μsec ± 0.8%                     |
| 4   | 1000 | 10           | <span style="color:#00aa00">**2.760 μsec ± 0.4%**</span> | 11.94 μsec ± 0.8%                      | 11.98 μsec ± 1.2%                     |
| 8   | 1000 | 10           | <span style="color:#00aa00">**3.510 μsec ± 9.4%**</span> | 21.85 μsec ± 1.3%                      | 21.77 μsec ± 0.5%                     |
| 16  | 1000 | 10           | <span style="color:#00aa00">**4.907 μsec ± 0.6%**</span> | 45.48 μsec ± 0.5%                      | 45.44 μsec ± 0.5%                     |
| 32  | 1000 | 10           | <span style="color:#00aa00">**7.557 μsec ± 0.4%**</span> | 65.83 μsec ± 1.1%                      | 65.13 μsec ± 1.1%                     |
| 64  | 1000 | 10           | <span style="color:#00aa00">**9.652 μsec ± 0.7%**</span> | 107.3 μsec ± 0.6%                      | 107.0 μsec ± 0.6%                     |
| 128 | 1000 | 10           | <span style="color:#00aa00">**10.20 μsec ± 0.4%**</span> | 197.6 μsec ± 0.7%                      | 199.1 μsec ± 1.1%                     |
| 256 | 1000 | 10           | <span style="color:#00aa00">**10.98 μsec ± 0.9%**</span> | 386.1 μsec ± 0.9%                      | 381.5 μsec ± 0.3%                     |
|     |      | **Geomean:** | <span style="color:#00aa00">**2.543 μsec ± 1.4%**</span> | 13.86 μsec ± 1.0%                      | 13.84 μsec ± 0.8%                     |

#### Accuracy Results

| `k` | `n`  | `m`       | `randint`                                     | `randint_constrained`<br>(eager=False)         | `randint_constrained`<br>(eager=True)          |
| --- | ---- | --------- | --------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 87.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 42.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 84.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 43.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 32.85%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |
