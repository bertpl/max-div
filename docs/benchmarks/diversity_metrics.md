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

                                                                                                                                                                                                           
| `size`       | `min_separation`  | `mean_separation`                                        | `geomean_separation` | `approx_geomean_separation` |
| ------------ | ----------------- | -------------------------------------------------------- | -------------------- | --------------------------- |
| 2            | 119.3 nsec ± 1.2% | <span style="color:#00aa00">**106.9 nsec ± 1.2%**</span> | 120.8 nsec ± 1.3%    | 109.3 nsec ± 2.1%           |
| 4            | 121.6 nsec ± 0.4% | <span style="color:#00aa00">**106.8 nsec ± 0.1%**</span> | 123.5 nsec ± 0.7%    | 111.2 nsec ± 0.3%           |
| 8            | 123.1 nsec ± 0.7% | <span style="color:#00aa00">**108.3 nsec ± 0.8%**</span> | 132.4 nsec ± 0.6%    | 116.8 nsec ± 0.7%           |
| 16           | 126.4 nsec ± 0.6% | <span style="color:#00aa00">**109.4 nsec ± 0.4%**</span> | 145.8 nsec ± 0.3%    | 126.1 nsec ± 0.8%           |
| 32           | 132.3 nsec ± 0.4% | <span style="color:#00aa00">**109.7 nsec ± 0.2%**</span> | 176.3 nsec ± 0.5%    | 145.6 nsec ± 0.7%           |
| 64           | 157.7 nsec ± 0.7% | <span style="color:#00aa00">**112.3 nsec ± 0.6%**</span> | 234.0 nsec ± 1.0%    | 187.6 nsec ± 1.0%           |
| 128          | 216.8 nsec ± 0.9% | <span style="color:#00aa00">**113.9 nsec ± 1.1%**</span> | 349.9 nsec ± 0.2%    | 272.2 nsec ± 0.9%           |
| 256          | 314.0 nsec ± 0.5% | <span style="color:#00aa00">**118.5 nsec ± 0.8%**</span> | 588.9 nsec ± 0.3%    | 447.3 nsec ± 0.5%           |
| 512          | 507.1 nsec ± 0.3% | <span style="color:#00aa00">**135.5 nsec ± 0.4%**</span> | 1.060 μsec ± 0.2%    | 796.1 nsec ± 0.2%           |
| 1024         | 892.1 nsec ± 0.1% | <span style="color:#00aa00">**159.0 nsec ± 1.2%**</span> | 2.009 μsec ± 0.5%    | 1.488 μsec ± 0.2%           |
| 2048         | 1.657 μsec ± 0.2% | <span style="color:#00aa00">**201.3 nsec ± 0.8%**</span> | 3.868 μsec ± 0.3%    | 2.872 μsec ± 0.1%           |
| 4096         | 3.183 μsec ± 0.1% | <span style="color:#00aa00">**288.4 nsec ± 0.5%**</span> | 7.631 μsec ± 0.2%    | 5.638 μsec ± 0.1%           |
| **Geomean:** | 309.3 nsec ± 0.5% | <span style="color:#00aa00">**132.1 nsec ± 0.7%**</span> | 481.2 nsec ± 0.5%    | 387.4 nsec ± 0.6%           |
