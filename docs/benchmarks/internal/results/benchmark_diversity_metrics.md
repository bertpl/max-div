## DiversityMetric Performance


| `size`       | `min_separation`  | `mean_separation`                                        | `geomean_separation` | `approx_geomean_separation` | `non_zero_separation_frac`                               |
| ------------ | ----------------- | -------------------------------------------------------- | -------------------- | --------------------------- | -------------------------------------------------------- |
| 10           | 107.2 nsec ± 0.8% | 108.6 nsec ± 1.3%                                        | 139.0 nsec ± 0.8%    | 114.7 nsec ± 0.8%           | <span style="color:#00aa00">**106.6 nsec ± 1.6%**</span> |
| 20           | 113.9 nsec ± 0.9% | <span style="color:#00aa00">**112.9 nsec ± 0.8%**</span> | 161.5 nsec ± 1.6%    | 120.1 nsec ± 1.0%           | 114.7 nsec ± 0.9%                                        |
| 50           | 125.8 nsec ± 0.9% | <span style="color:#00aa00">**119.1 nsec ± 3.2%**</span> | 222.9 nsec ± 0.9%    | 128.8 nsec ± 0.4%           | <span style="color:#00aa00">**117.2 nsec ± 0.3%**</span> |
| 100          | 148.5 nsec ± 0.6% | <span style="color:#00aa00">**113.6 nsec ± 1.2%**</span> | 319.5 nsec ± 0.5%    | 142.2 nsec ± 0.6%           | 124.8 nsec ± 1.2%                                        |
| 200          | 193.7 nsec ± 1.2% | <span style="color:#00aa00">**120.5 nsec ± 1.5%**</span> | 509.0 nsec ± 0.4%    | 164.6 nsec ± 0.3%           | 138.6 nsec ± 0.5%                                        |
| 500          | 344.8 nsec ± 0.6% | <span style="color:#00aa00">**138.8 nsec ± 1.7%**</span> | 1.080 μsec ± 0.1%    | 240.4 nsec ± 3.0%           | 178.1 nsec ± 1.0%                                        |
| 1000         | 610.5 nsec ± 0.4% | <span style="color:#00aa00">**159.5 nsec ± 1.0%**</span> | 2.025 μsec ± 0.1%    | 362.1 nsec ± 0.5%           | 239.3 nsec ± 0.8%                                        |
| 2000         | 1.091 μsec ± 0.3% | <span style="color:#00aa00">**198.6 nsec ± 0.8%**</span> | 3.924 μsec ± 0.1%    | 617.0 nsec ± 0.4%           | 357.7 nsec ± 0.2%                                        |
| 5000         | 2.581 μsec ± 0.2% | <span style="color:#00aa00">**319.3 nsec ± 0.6%**</span> | 9.604 μsec ± 0.1%    | 1.367 μsec ± 0.6%           | 734.7 nsec ± 0.2%                                        |
| 10000        | 5.069 μsec ± 0.1% | <span style="color:#00aa00">**529.7 nsec ± 0.1%**</span> | 19.07 μsec ± 0.1%    | 2.680 μsec ± 0.1%           | 1.354 μsec ± 0.2%                                        |
| 20000        | 10.04 μsec ± 0.2% | <span style="color:#00aa00">**967.3 nsec ± 0.2%**</span> | 38.02 μsec ± 0.1%    | 5.353 μsec ± 0.1%           | 2.594 μsec ± 0.1%                                        |
| **Geomean:** | 547.8 nsec ± 0.6% | <span style="color:#00aa00">**193.7 nsec ± 1.1%**</span> | 1.423 μsec ± 0.4%    | 401.5 nsec ± 0.7%           | 284.1 nsec ± 0.6%                                        |

