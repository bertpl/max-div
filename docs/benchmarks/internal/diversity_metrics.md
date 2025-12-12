# `DiversityMetric`

Command:
```bash
uv tool install max-div
max-div benchmark --markdown diversity_metrics
```
or 
```bash
uv run max-div benchmark --markdown diversity_metrics
```

We compare speed of computing the different diversity metrics for different vector selection sizes.

## DiversityMetric Performance

                                                                                                                                                                                                           
| `size`       | `min_separation`                                         | `mean_separation`                                        | `geomean_separation` | `approx_geomean_separation` | `non_zero_separation_frac`                               |
| ------------ | -------------------------------------------------------- | -------------------------------------------------------- | -------------------- | --------------------------- | -------------------------------------------------------- |
| 2            | <span style="color:#00aa00">**109.1 nsec ± 2.4%**</span> | 107.4 nsec ± 1.7%                                        | 125.6 nsec ± 2.3%    | 117.3 nsec ± 1.9%           | <span style="color:#00aa00">**106.5 nsec ± 2.4%**</span> |
| 4            | <span style="color:#00aa00">**106.0 nsec ± 0.4%**</span> | 108.2 nsec ± 0.1%                                        | 128.0 nsec ± 0.7%    | 115.7 nsec ± 0.3%           | <span style="color:#00aa00">**106.0 nsec ± 0.2%**</span> |
| 8            | 107.5 nsec ± 0.5%                                        | 108.3 nsec ± 0.1%                                        | 133.0 nsec ± 0.6%    | 118.6 nsec ± 0.2%           | <span style="color:#00aa00">**105.5 nsec ± 0.6%**</span> |
| 16           | 109.2 nsec ± 0.4%                                        | 108.2 nsec ± 0.2%                                        | 149.1 nsec ± 0.4%    | 130.2 nsec ± 0.3%           | <span style="color:#00aa00">**107.8 nsec ± 0.3%**</span> |
| 32           | 113.7 nsec ± 0.2%                                        | <span style="color:#00aa00">**108.5 nsec ± 0.2%**</span> | 178.4 nsec ± 0.3%    | 153.1 nsec ± 0.2%           | 109.8 nsec ± 0.5%                                        |
| 64           | 129.1 nsec ± 2.2%                                        | <span style="color:#00aa00">**110.1 nsec ± 2.4%**</span> | 240.2 nsec ± 1.2%    | 193.5 nsec ± 1.5%           | 116.9 nsec ± 2.3%                                        |
| 128          | 159.4 nsec ± 1.6%                                        | <span style="color:#00aa00">**110.2 nsec ± 2.0%**</span> | 356.9 nsec ± 0.9%    | 278.2 nsec ± 1.0%           | 127.3 nsec ± 2.8%                                        |
| 256          | 220.4 nsec ± 1.5%                                        | <span style="color:#00aa00">**116.4 nsec ± 1.2%**</span> | 596.3 nsec ± 0.5%    | 445.8 nsec ± 0.6%           | 140.5 nsec ± 2.5%                                        |
| 512          | 350.7 nsec ± 0.2%                                        | <span style="color:#00aa00">**134.1 nsec ± 1.4%**</span> | 1.083 μsec ± 0.3%    | 799.2 nsec ± 0.2%           | 179.9 nsec ± 1.4%                                        |
| 1024         | 604.0 nsec ± 0.4%                                        | <span style="color:#00aa00">**159.8 nsec ± 0.8%**</span> | 2.031 μsec ± 0.2%    | 1.488 μsec ± 0.2%           | 240.6 nsec ± 1.0%                                        |
| 2048         | 1.114 μsec ± 0.2%                                        | <span style="color:#00aa00">**203.6 nsec ± 0.7%**</span> | 3.940 μsec ± 0.1%    | 2.871 μsec ± 0.1%           | 368.5 nsec ± 0.8%                                        |
| 4096         | 2.134 μsec ± 0.1%                                        | <span style="color:#00aa00">**294.4 nsec ± 0.4%**</span> | 7.751 μsec ± 0.0%    | 5.632 μsec ± 0.2%           | 622.2 nsec ± 0.2%                                        |
| **Geomean:** | 239.3 nsec ± 0.8%                                        | <span style="color:#00aa00">**131.5 nsec ± 0.9%**</span> | 490.8 nsec ± 0.6%    | 395.9 nsec ± 0.6%           | 160.5 nsec ± 1.2%                                        |

