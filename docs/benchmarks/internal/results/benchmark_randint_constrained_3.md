## Scenario B

Fixed n=1000 & k=100 with varying number of constraints spanning random 1% portions of the n-range

### Uniform sampling.

#### Timing Results

| `k` | `n`  | `m`          | `randint`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------- | -------------------------------------- | ------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**717.3 nsec ± 1.6%**</span> | 160.9 μsec ± 0.7%                      | 160.1 μsec ± 0.2%                     |
| 100 | 1000 | 4            | <span style="color:#00aa00">**719.9 nsec ± 2.0%**</span> | 162.6 μsec ± 0.2%                      | 162.7 μsec ± 0.2%                     |
| 100 | 1000 | 8            | <span style="color:#00aa00">**699.3 nsec ± 1.9%**</span> | 166.4 μsec ± 0.4%                      | 166.0 μsec ± 0.2%                     |
| 100 | 1000 | 16           | <span style="color:#00aa00">**706.0 nsec ± 1.7%**</span> | 174.7 μsec ± 0.6%                      | 171.8 μsec ± 0.1%                     |
| 100 | 1000 | 32           | <span style="color:#00aa00">**686.4 nsec ± 2.2%**</span> | 193.9 μsec ± 0.2%                      | 183.8 μsec ± 0.2%                     |
| 100 | 1000 | 64           | <span style="color:#00aa00">**687.1 nsec ± 1.5%**</span> | 236.6 μsec ± 0.3%                      | 207.8 μsec ± 0.2%                     |
| 100 | 1000 | 128          | <span style="color:#00aa00">**696.4 nsec ± 1.8%**</span> | 311.4 μsec ± 0.3%                      | 259.6 μsec ± 0.5%                     |
| 100 | 1000 | 256          | <span style="color:#00aa00">**706.4 nsec ± 1.3%**</span> | 423.3 μsec ± 0.5%                      | 362.5 μsec ± 0.4%                     |
| 100 | 1000 | 384          | <span style="color:#00aa00">**709.6 nsec ± 1.6%**</span> | 548.4 μsec ± 0.4%                      | 477.1 μsec ± 0.5%                     |
| 100 | 1000 | 512          | <span style="color:#00aa00">**692.5 nsec ± 1.9%**</span> | 690.6 μsec ± 0.8%                      | 608.6 μsec ± 0.5%                     |
| 100 | 1000 | 768          | <span style="color:#00aa00">**717.0 nsec ± 4.1%**</span> | 970.6 μsec ± 0.3%                      | 899.2 μsec ± 0.3%                     |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**708.2 nsec ± 2.0%**</span> | 1.394 msec ± 0.4%                      | 1.357 msec ± 0.3%                     |
|     |      | **Geomean:** | <span style="color:#00aa00">**703.8 nsec ± 2.0%**</span> | 339.6 μsec ± 0.4%                      | 314.8 μsec ± 0.3%                     |

#### Accuracy Results

| `k` | `n`  | `m`       | `randint`                             | `randint_constrained`<br>(eager=False)        | `randint_constrained`<br>(eager=True)         |
| --- | ---- | --------- | ------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| 100 | 1000 | 2         | 0.0%                                  | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 4         | 0.0%                                  | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 8         | 0.0%                                  | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 16        | 0.0%                                  | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 32        | 0.0%                                  | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 64        | 0.0%                                  | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 128       | 0.0%                                  | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 256       | 0.0%                                  | 98.0%                                         | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 384       | 0.0%                                  | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                  | 0.0%                                          | <span style="color:#00aa00">**50.0%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                 | 66.50%                                        | <span style="color:#00aa00">**79.17%**</span> |
