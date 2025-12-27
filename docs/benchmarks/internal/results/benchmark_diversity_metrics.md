## DiversityMetric Performance


| `size`       | `min_separation`  | `mean_separation`                                        | `geomean_separation` | `approx_geomean_separation` | `non_zero_separation_frac`                               |
| ------------ | ----------------- | -------------------------------------------------------- | -------------------- | --------------------------- | -------------------------------------------------------- |
| 10           | 110.2 nsec ± 0.5% | 110.5 nsec ± 0.9%                                        | 149.1 nsec ± 1.4%    | 120.6 nsec ± 0.3%           | <span style="color:#00aa00">**108.4 nsec ± 0.5%**</span> |
| 20           | 112.9 nsec ± 0.7% | <span style="color:#00aa00">**110.8 nsec ± 0.6%**</span> | 164.6 nsec ± 0.3%    | 119.9 nsec ± 0.3%           | 113.1 nsec ± 0.3%                                        |
| 50           | 123.7 nsec ± 0.3% | 112.8 nsec ± 0.3%                                        | 222.6 nsec ± 0.2%    | 127.1 nsec ± 0.2%           | <span style="color:#00aa00">**111.7 nsec ± 1.1%**</span> |
| 100          | 150.5 nsec ± 0.4% | <span style="color:#00aa00">**113.9 nsec ± 0.7%**</span> | 318.7 nsec ± 0.4%    | 142.9 nsec ± 0.5%           | 123.2 nsec ± 0.4%                                        |
| 200          | 203.2 nsec ± 0.3% | <span style="color:#00aa00">**120.1 nsec ± 1.7%**</span> | 517.7 nsec ± 2.3%    | 172.5 nsec ± 1.5%           | 136.9 nsec ± 0.4%                                        |
| 500          | 366.4 nsec ± 0.2% | <span style="color:#00aa00">**135.3 nsec ± 0.2%**</span> | 1.080 μsec ± 0.1%    | 261.1 nsec ± 0.5%           | 195.3 nsec ± 1.8%                                        |
| 1000         | 659.1 nsec ± 0.9% | <span style="color:#00aa00">**163.1 nsec ± 0.9%**</span> | 2.028 μsec ± 0.1%    | 390.6 nsec ± 0.3%           | 249.8 nsec ± 0.2%                                        |
| 2000         | 1.166 μsec ± 0.2% | <span style="color:#00aa00">**195.7 nsec ± 0.3%**</span> | 3.921 μsec ± 0.1%    | 653.4 nsec ± 0.2%           | 382.5 nsec ± 0.3%                                        |
| 5000         | 2.760 μsec ± 0.2% | <span style="color:#00aa00">**336.9 nsec ± 0.3%**</span> | 9.613 μsec ± 0.1%    | 1.460 μsec ± 0.1%           | 782.6 nsec ± 0.2%                                        |
| 10000        | 5.426 μsec ± 0.4% | <span style="color:#00aa00">**567.6 nsec ± 1.5%**</span> | 19.07 μsec ± 0.1%    | 2.791 μsec ± 0.1%           | 1.449 μsec ± 0.2%                                        |
| 20000        | 10.75 μsec ± 0.1% | <span style="color:#00aa00">**1.027 μsec ± 0.6%**</span> | 38.05 μsec ± 0.1%    | 5.468 μsec ± 0.1%           | 2.776 μsec ± 0.1%                                        |
| **Geomean:** | 571.7 nsec ± 0.4% | <span style="color:#00aa00">**195.6 nsec ± 0.7%**</span> | 1.437 μsec ± 0.5%    | 417.5 nsec ± 0.4%           | 292.8 nsec ± 0.5%                                        |

