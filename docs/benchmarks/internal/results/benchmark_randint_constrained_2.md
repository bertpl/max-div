### Non-uniform sampling (custom p).

#### Timing Results

| `k` | `n`  | `m`          | `randint`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------- | -------------------------------------- | ------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**601.0 nsec ± 1.6%**</span> | 1.132 μsec ± 1.2%                      | 1.141 μsec ± 0.9%                     |
| 4   | 10   | 10           | <span style="color:#00aa00">**612.3 nsec ± 0.4%**</span> | 1.545 μsec ± 0.6%                      | 1.531 μsec ± 0.2%                     |
| 8   | 10   | 10           | <span style="color:#00aa00">**588.0 nsec ± 0.5%**</span> | 2.405 μsec ± 0.6%                      | 2.399 μsec ± 0.2%                     |
| 2   | 100  | 10           | <span style="color:#00aa00">**875.7 nsec ± 0.6%**</span> | 1.694 μsec ± 0.3%                      | 1.690 μsec ± 0.1%                     |
| 4   | 100  | 10           | <span style="color:#00aa00">**1.047 μsec ± 4.3%**</span> | 2.545 μsec ± 0.1%                      | 2.562 μsec ± 0.1%                     |
| 8   | 100  | 10           | <span style="color:#00aa00">**1.346 μsec ± 2.0%**</span> | 4.354 μsec ± 0.7%                      | 4.526 μsec ± 0.1%                     |
| 16  | 100  | 10           | <span style="color:#00aa00">**2.082 μsec ± 0.3%**</span> | 8.351 μsec ± 2.1%                      | 8.172 μsec ± 0.1%                     |
| 32  | 100  | 10           | <span style="color:#00aa00">**2.192 μsec ± 0.3%**</span> | 15.82 μsec ± 0.3%                      | 16.06 μsec ± 2.7%                     |
| 64  | 100  | 10           | <span style="color:#00aa00">**2.220 μsec ± 0.5%**</span> | 33.49 μsec ± 1.8%                      | 31.47 μsec ± 0.2%                     |
| 2   | 1000 | 10           | <span style="color:#00aa00">**2.357 μsec ± 0.5%**</span> | 5.458 μsec ± 3.1%                      | 5.454 μsec ± 0.9%                     |
| 4   | 1000 | 10           | <span style="color:#00aa00">**2.776 μsec ± 0.8%**</span> | 9.345 μsec ± 0.4%                      | 9.365 μsec ± 0.2%                     |
| 8   | 1000 | 10           | <span style="color:#00aa00">**3.488 μsec ± 2.3%**</span> | 19.02 μsec ± 1.4%                      | 19.08 μsec ± 0.8%                     |
| 16  | 1000 | 10           | <span style="color:#00aa00">**5.050 μsec ± 0.6%**</span> | 37.32 μsec ± 0.7%                      | 37.51 μsec ± 5.7%                     |
| 32  | 1000 | 10           | <span style="color:#00aa00">**7.399 μsec ± 0.3%**</span> | 70.06 μsec ± 0.2%                      | 69.68 μsec ± 0.2%                     |
| 64  | 1000 | 10           | <span style="color:#00aa00">**9.065 μsec ± 5.7%**</span> | 140.9 μsec ± 0.3%                      | 141.1 μsec ± 0.2%                     |
| 128 | 1000 | 10           | <span style="color:#00aa00">**9.421 μsec ± 2.7%**</span> | 287.3 μsec ± 0.2%                      | 286.4 μsec ± 0.2%                     |
| 256 | 1000 | 10           | <span style="color:#00aa00">**10.20 μsec ± 0.4%**</span> | 580.7 μsec ± 0.2%                      | 581.8 μsec ± 0.2%                     |
|     |      | **Geomean:** | <span style="color:#00aa00">**2.348 μsec ± 1.4%**</span> | 13.69 μsec ± 0.8%                      | 13.67 μsec ± 0.8%                     |

#### Accuracy Results

| `k` | `n`  | `m`       | `randint`                               | `randint_constrained`<br>(eager=False)         | `randint_constrained`<br>(eager=True)          |
| --- | ---- | --------- | --------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 87.5%                                   | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 42.0%                                   | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 0.5%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 0.5%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 0.0%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 0.0%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 84.5%                                   | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 43.0%                                   | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 0.5%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.0%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 0.0%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.0%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.0%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.0%                                    | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 32.85%                                  | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |
