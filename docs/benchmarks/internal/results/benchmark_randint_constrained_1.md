## Scenario A

Varying n & k with 10 non-overlapping constraints spanning equal portions of the n-range

### Uniform sampling.

#### Timing Results

| `k` | `n`  | `m`          | `randint`                                                | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------- | ------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**563.0 nsec ± 0.9%**</span> | 1.581 μsec ± 1.9%                      | 1.599 μsec ± 1.0%                     |
| 4   | 10   | 10           | <span style="color:#00aa00">**564.0 nsec ± 1.8%**</span> | 1.895 μsec ± 1.4%                      | 1.899 μsec ± 0.6%                     |
| 8   | 10   | 10           | <span style="color:#00aa00">**578.1 nsec ± 1.6%**</span> | 2.529 μsec ± 1.1%                      | 2.579 μsec ± 1.3%                     |
| 2   | 100  | 10           | <span style="color:#00aa00">**581.2 nsec ± 0.8%**</span> | 2.109 μsec ± 0.6%                      | 2.094 μsec ± 0.8%                     |
| 4   | 100  | 10           | <span style="color:#00aa00">**582.7 nsec ± 1.0%**</span> | 2.970 μsec ± 0.6%                      | 2.979 μsec ± 1.7%                     |
| 8   | 100  | 10           | <span style="color:#00aa00">**575.5 nsec ± 2.9%**</span> | 4.620 μsec ± 0.4%                      | 4.581 μsec ± 0.3%                     |
| 16  | 100  | 10           | <span style="color:#00aa00">**611.0 nsec ± 1.0%**</span> | 8.376 μsec ± 0.3%                      | 8.118 μsec ± 0.3%                     |
| 32  | 100  | 10           | <span style="color:#00aa00">**618.5 nsec ± 1.1%**</span> | 13.30 μsec ± 0.3%                      | 13.74 μsec ± 1.1%                     |
| 64  | 100  | 10           | <span style="color:#00aa00">**678.3 nsec ± 1.5%**</span> | 23.87 μsec ± 2.9%                      | 24.61 μsec ± 2.0%                     |
| 2   | 1000 | 10           | <span style="color:#00aa00">**797.5 nsec ± 0.8%**</span> | 6.345 μsec ± 1.3%                      | 6.385 μsec ± 1.2%                     |
| 4   | 1000 | 10           | <span style="color:#00aa00">**826.8 nsec ± 2.3%**</span> | 11.31 μsec ± 1.3%                      | 11.25 μsec ± 1.3%                     |
| 8   | 1000 | 10           | <span style="color:#00aa00">**886.5 nsec ± 0.7%**</span> | 20.77 μsec ± 2.0%                      | 22.05 μsec ± 1.0%                     |
| 16  | 1000 | 10           | <span style="color:#00aa00">**614.3 nsec ± 0.9%**</span> | 43.89 μsec ± 0.8%                      | 44.82 μsec ± 1.0%                     |
| 32  | 1000 | 10           | <span style="color:#00aa00">**654.2 nsec ± 0.4%**</span> | 65.66 μsec ± 0.5%                      | 65.27 μsec ± 0.4%                     |
| 64  | 1000 | 10           | <span style="color:#00aa00">**703.8 nsec ± 1.0%**</span> | 106.3 μsec ± 0.4%                      | 108.9 μsec ± 1.3%                     |
| 128 | 1000 | 10           | <span style="color:#00aa00">**792.4 nsec ± 0.8%**</span> | 195.8 μsec ± 1.6%                      | 194.9 μsec ± 1.0%                     |
| 256 | 1000 | 10           | <span style="color:#00aa00">**978.7 nsec ± 0.6%**</span> | 372.2 μsec ± 2.1%                      | 377.6 μsec ± 1.6%                     |
|     |      | **Geomean:** | <span style="color:#00aa00">**672.5 nsec ± 1.2%**</span> | 13.63 μsec ± 1.1%                      | 13.76 μsec ± 1.0%                     |

#### Accuracy Results

| `k` | `n`  | `m`       | `randint`                                     | `randint_constrained`<br>(eager=False)         | `randint_constrained`<br>(eager=True)          |
| --- | ---- | --------- | --------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 90.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 54.0%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 2.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 1.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 4.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 13.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 90.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 49.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 2.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 1.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 1.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 35.94%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |
