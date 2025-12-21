## DiversityMetric Performance


| `size`       | `min_separation`  | `mean_separation`                                        | `geomean_separation` | `approx_geomean_separation` | `non_zero_separation_frac`                               |
| ------------ | ----------------- | -------------------------------------------------------- | -------------------- | --------------------------- | -------------------------------------------------------- |
| 10           | 110.8 nsec ± 1.6% | 113.0 nsec ± 1.7%                                        | 139.5 nsec ± 1.2%    | 123.8 nsec ± 1.6%           | <span style="color:#00aa00">**109.0 nsec ± 1.3%**</span> |
| 20           | 112.7 nsec ± 0.5% | <span style="color:#00aa00">**109.1 nsec ± 0.2%**</span> | 157.4 nsec ± 0.2%    | 137.0 nsec ± 0.4%           | 115.2 nsec ± 1.5%                                        |
| 50           | 124.2 nsec ± 0.7% | <span style="color:#00aa00">**111.1 nsec ± 0.4%**</span> | 214.9 nsec ± 0.2%    | 178.2 nsec ± 0.2%           | 113.7 nsec ± 0.9%                                        |
| 100          | 150.0 nsec ± 0.5% | <span style="color:#00aa00">**112.3 nsec ± 0.2%**</span> | 308.4 nsec ± 0.1%    | 248.8 nsec ± 0.1%           | 123.1 nsec ± 0.1%                                        |
| 200          | 203.9 nsec ± 0.2% | <span style="color:#00aa00">**117.1 nsec ± 0.4%**</span> | 494.7 nsec ± 0.1%    | 394.7 nsec ± 0.1%           | 135.8 nsec ± 0.7%                                        |
| 500          | 361.0 nsec ± 0.6% | <span style="color:#00aa00">**134.7 nsec ± 0.2%**</span> | 1.058 μsec ± 0.1%    | 823.9 nsec ± 0.3%           | 182.8 nsec ± 0.2%                                        |
| 1000         | 599.1 nsec ± 0.4% | <span style="color:#00aa00">**159.8 nsec ± 1.2%**</span> | 1.990 μsec ± 0.1%    | 1.520 μsec ± 0.5%           | 251.7 nsec ± 0.7%                                        |
| 2000         | 1.093 μsec ± 0.2% | <span style="color:#00aa00">**196.2 nsec ± 0.2%**</span> | 3.852 μsec ± 0.1%    | 2.912 μsec ± 0.3%           | 379.9 nsec ± 0.1%                                        |
| 5000         | 2.585 μsec ± 0.1% | <span style="color:#00aa00">**337.4 nsec ± 0.6%**</span> | 9.435 μsec ± 0.0%    | 7.062 μsec ± 0.1%           | 772.1 nsec ± 0.2%                                        |
| 10000        | 5.070 μsec ± 0.2% | <span style="color:#00aa00">**561.5 nsec ± 0.1%**</span> | 18.74 μsec ± 0.1%    | 14.05 μsec ± 0.2%           | 1.631 μsec ± 0.5%                                        |
| 20000        | 11.54 μsec ± 8.7% | <span style="color:#00aa00">**1.032 μsec ± 0.3%**</span> | 37.37 μsec ± 0.1%    | 28.25 μsec ± 0.3%           | 2.760 μsec ± 0.4%                                        |
| **Geomean:** | 559.9 nsec ± 1.3% | <span style="color:#00aa00">**194.3 nsec ± 0.5%**</span> | 1.394 μsec ± 0.2%    | 1.106 μsec ± 0.4%           | 294.6 nsec ± 0.6%                                        |

