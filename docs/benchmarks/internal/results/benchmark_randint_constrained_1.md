## Scenario A

Varying n & k with 10 non-overlapping constraints spanning equal portions of the n-range

### Uniform sampling.

#### Timing Results

| `k` | `n`  | `m`          | `randint`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------- | -------------------------------------- | ------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**485.9 nsec ± 0.7%**</span> | 1.133 μsec ± 0.6%                      | 1.162 μsec ± 1.9%                     |
| 4   | 10   | 10           | <span style="color:#00aa00">**483.7 nsec ± 0.5%**</span> | 1.533 μsec ± 0.3%                      | 1.542 μsec ± 0.3%                     |
| 8   | 10   | 10           | <span style="color:#00aa00">**519.0 nsec ± 1.2%**</span> | 2.495 μsec ± 0.4%                      | 2.508 μsec ± 0.2%                     |
| 2   | 100  | 10           | <span style="color:#00aa00">**522.0 nsec ± 0.3%**</span> | 1.651 μsec ± 0.4%                      | 1.652 μsec ± 0.8%                     |
| 4   | 100  | 10           | <span style="color:#00aa00">**529.6 nsec ± 0.2%**</span> | 2.522 μsec ± 0.6%                      | 2.489 μsec ± 0.3%                     |
| 8   | 100  | 10           | <span style="color:#00aa00">**538.2 nsec ± 0.5%**</span> | 4.414 μsec ± 0.2%                      | 4.384 μsec ± 0.1%                     |
| 16  | 100  | 10           | <span style="color:#00aa00">**550.2 nsec ± 0.6%**</span> | 8.351 μsec ± 0.2%                      | 8.401 μsec ± 0.2%                     |
| 32  | 100  | 10           | <span style="color:#00aa00">**587.7 nsec ± 0.4%**</span> | 16.12 μsec ± 0.4%                      | 16.22 μsec ± 0.3%                     |
| 64  | 100  | 10           | <span style="color:#00aa00">**662.4 nsec ± 0.3%**</span> | 32.06 μsec ± 0.5%                      | 31.91 μsec ± 0.2%                     |
| 2   | 1000 | 10           | <span style="color:#00aa00">**537.8 nsec ± 2.1%**</span> | 4.779 μsec ± 0.6%                      | 4.763 μsec ± 0.2%                     |
| 4   | 1000 | 10           | <span style="color:#00aa00">**558.0 nsec ± 2.9%**</span> | 8.830 μsec ± 0.3%                      | 8.762 μsec ± 0.2%                     |
| 8   | 1000 | 10           | <span style="color:#00aa00">**546.2 nsec ± 2.1%**</span> | 17.50 μsec ± 0.3%                      | 17.51 μsec ± 0.2%                     |
| 16  | 1000 | 10           | <span style="color:#00aa00">**589.6 nsec ± 1.8%**</span> | 34.91 μsec ± 0.1%                      | 35.12 μsec ± 0.3%                     |
| 32  | 1000 | 10           | <span style="color:#00aa00">**608.7 nsec ± 2.2%**</span> | 69.33 μsec ± 0.8%                      | 69.15 μsec ± 0.8%                     |
| 64  | 1000 | 10           | <span style="color:#00aa00">**645.0 nsec ± 2.2%**</span> | 142.9 μsec ± 0.7%                      | 142.4 μsec ± 0.1%                     |
| 128 | 1000 | 10           | <span style="color:#00aa00">**732.8 nsec ± 1.8%**</span> | 288.9 μsec ± 0.8%                      | 291.0 μsec ± 0.3%                     |
| 256 | 1000 | 10           | <span style="color:#00aa00">**911.3 nsec ± 1.5%**</span> | 595.5 μsec ± 0.3%                      | 598.9 μsec ± 0.6%                     |
|     |      | **Geomean:** | <span style="color:#00aa00">**581.2 nsec ± 1.3%**</span> | 13.43 μsec ± 0.4%                      | 13.45 μsec ± 0.4%                     |

#### Accuracy Results

| `k` | `n`  | `m`       | `randint`                               | `randint_constrained`<br>(eager=False)         | `randint_constrained`<br>(eager=True)          |
| --- | ---- | --------- | --------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 90.0%                                   | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 54.0%                                   | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 2.0%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 1.0%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 4.5%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 13.5%                                   | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 90.5%                                   | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 45.5%                                   | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 2.0%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 1.0%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 1.0%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.5%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.5%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.5%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 35.68%                                  | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |
