## DiversityMetric Performance


| `size`       | `min_separation`  | `mean_separation`                                        | `geomean_separation` | `approx_geomean_separation` | `non_zero_separation_frac`                               |
| ------------ | ----------------- | -------------------------------------------------------- | -------------------- | --------------------------- | -------------------------------------------------------- |
| 2            | 107.0 nsec ± 1.9% | 109.3 nsec ± 1.5%                                        | 122.0 nsec ± 1.5%    | 111.4 nsec ± 1.5%           | <span style="color:#00aa00">**103.8 nsec ± 2.0%**</span> |
| 4            | 105.7 nsec ± 0.5% | 107.9 nsec ± 0.2%                                        | 124.9 nsec ± 0.3%    | 113.7 nsec ± 0.3%           | <span style="color:#00aa00">**102.5 nsec ± 0.3%**</span> |
| 8            | 106.5 nsec ± 0.4% | 108.1 nsec ± 0.5%                                        | 132.0 nsec ± 0.3%    | 118.0 nsec ± 0.2%           | <span style="color:#00aa00">**102.4 nsec ± 1.0%**</span> |
| 16           | 109.0 nsec ± 0.4% | 108.3 nsec ± 0.3%                                        | 147.1 nsec ± 0.4%    | 128.5 nsec ± 0.2%           | <span style="color:#00aa00">**103.2 nsec ± 0.3%**</span> |
| 32           | 114.3 nsec ± 0.6% | 107.7 nsec ± 0.7%                                        | 184.2 nsec ± 0.1%    | 151.8 nsec ± 0.1%           | <span style="color:#00aa00">**105.2 nsec ± 0.2%**</span> |
| 64           | 128.7 nsec ± 0.6% | <span style="color:#00aa00">**112.8 nsec ± 4.8%**</span> | 244.0 nsec ± 0.6%    | 193.8 nsec ± 1.0%           | <span style="color:#00aa00">**112.5 nsec ± 2.2%**</span> |
| 128          | 163.0 nsec ± 1.1% | <span style="color:#00aa00">**111.7 nsec ± 1.0%**</span> | 364.4 nsec ± 0.6%    | 280.5 nsec ± 0.6%           | 118.2 nsec ± 1.9%                                        |
| 256          | 220.2 nsec ± 0.4% | <span style="color:#00aa00">**115.7 nsec ± 1.7%**</span> | 604.1 nsec ± 0.4%    | 444.5 nsec ± 0.3%           | 137.4 nsec ± 1.3%                                        |
| 512          | 348.8 nsec ± 0.5% | <span style="color:#00aa00">**132.2 nsec ± 0.8%**</span> | 1.084 μsec ± 0.2%    | 795.2 nsec ± 0.2%           | 177.2 nsec ± 1.1%                                        |
| 1024         | 610.3 nsec ± 1.4% | <span style="color:#00aa00">**155.1 nsec ± 0.6%**</span> | 2.037 μsec ± 0.1%    | 1.487 μsec ± 0.1%           | 239.9 nsec ± 0.8%                                        |
| 2048         | 1.115 μsec ± 0.5% | <span style="color:#00aa00">**197.1 nsec ± 0.9%**</span> | 3.938 μsec ± 0.1%    | 2.864 μsec ± 0.1%           | 364.4 nsec ± 0.3%                                        |
| 4096         | 2.131 μsec ± 0.2% | <span style="color:#00aa00">**283.0 nsec ± 0.4%**</span> | 7.752 μsec ± 0.1%    | 5.625 μsec ± 0.1%           | 616.0 nsec ± 0.1%                                        |
| **Geomean:** | 239.2 nsec ± 0.7% | <span style="color:#00aa00">**130.7 nsec ± 1.1%**</span> | 491.2 nsec ± 0.4%    | 392.6 nsec ± 0.4%           | 155.9 nsec ± 1.0%                                        |

