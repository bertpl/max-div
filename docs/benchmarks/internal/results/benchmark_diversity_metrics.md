## DiversityMetric Performance


| `size`       | `min_separation`  | `mean_separation`                                        | `geomean_separation` | `approx_geomean_separation` | `non_zero_separation_frac`                               |
| ------------ | ----------------- | -------------------------------------------------------- | -------------------- | --------------------------- | -------------------------------------------------------- |
| 10           | 111.4 nsec ± 0.3% | 110.6 nsec ± 0.3%                                        | 146.6 nsec ± 0.2%    | 119.8 nsec ± 0.4%           | <span style="color:#00aa00">**110.1 nsec ± 0.7%**</span> |
| 20           | 113.0 nsec ± 0.4% | <span style="color:#00aa00">**110.4 nsec ± 0.3%**</span> | 163.5 nsec ± 0.2%    | 121.7 nsec ± 0.8%           | 111.2 nsec ± 0.2%                                        |
| 50           | 125.3 nsec ± 0.2% | <span style="color:#00aa00">**111.9 nsec ± 0.5%**</span> | 224.4 nsec ± 0.2%    | 128.6 nsec ± 1.0%           | 114.5 nsec ± 0.2%                                        |
| 100          | 149.5 nsec ± 0.3% | <span style="color:#00aa00">**112.6 nsec ± 0.2%**</span> | 319.0 nsec ± 0.1%    | 144.4 nsec ± 0.5%           | 124.3 nsec ± 0.3%                                        |
| 200          | 203.2 nsec ± 0.2% | <span style="color:#00aa00">**117.9 nsec ± 0.5%**</span> | 508.5 nsec ± 0.1%    | 172.8 nsec ± 0.2%           | 136.5 nsec ± 0.3%                                        |
| 500          | 367.6 nsec ± 0.4% | <span style="color:#00aa00">**136.4 nsec ± 0.2%**</span> | 1.080 μsec ± 0.1%    | 256.7 nsec ± 0.4%           | 182.7 nsec ± 0.3%                                        |
| 1000         | 628.1 nsec ± 1.5% | <span style="color:#00aa00">**159.8 nsec ± 0.3%**</span> | 2.027 μsec ± 0.1%    | 395.0 nsec ± 0.4%           | 250.1 nsec ± 0.1%                                        |
| 2000         | 1.116 μsec ± 1.3% | <span style="color:#00aa00">**197.8 nsec ± 0.4%**</span> | 3.918 μsec ± 0.1%    | 656.2 nsec ± 0.2%           | 383.6 nsec ± 0.1%                                        |
| 5000         | 2.579 μsec ± 0.2% | <span style="color:#00aa00">**334.9 nsec ± 0.3%**</span> | 9.598 μsec ± 0.1%    | 1.462 μsec ± 0.1%           | 782.3 nsec ± 0.1%                                        |
| 10000        | 5.066 μsec ± 0.3% | <span style="color:#00aa00">**560.6 nsec ± 0.3%**</span> | 19.06 μsec ± 0.2%    | 2.796 μsec ± 0.1%           | 1.446 μsec ± 0.1%                                        |
| 20000        | 10.03 μsec ± 0.1% | <span style="color:#00aa00">**1.027 μsec ± 1.6%**</span> | 38.00 μsec ± 0.1%    | 5.479 μsec ± 0.2%           | 2.778 μsec ± 0.1%                                        |
| **Geomean:** | 557.5 nsec ± 0.5% | <span style="color:#00aa00">**194.5 nsec ± 0.5%**</span> | 1.432 μsec ± 0.1%    | 418.9 nsec ± 0.4%           | 291.9 nsec ± 0.2%                                        |

