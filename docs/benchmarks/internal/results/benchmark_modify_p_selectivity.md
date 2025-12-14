## modify_p_selectivity Performance

Tested methods:

 - `power`: `modify_p_selectivity_power`
 - `pwl2`: `modify_p_selectivity_pwl2`

| `size`       | `power`           | `pwl2`                                                   |
| ------------ | ----------------- | -------------------------------------------------------- |
| 2            | 370.3 nsec ± 1.1% | <span style="color:#00aa00">**363.9 nsec ± 1.9%**</span> |
| 4            | 378.4 nsec ± 1.1% | <span style="color:#00aa00">**354.4 nsec ± 0.9%**</span> |
| 8            | 421.4 nsec ± 0.9% | <span style="color:#00aa00">**342.9 nsec ± 0.9%**</span> |
| 16           | 463.4 nsec ± 1.0% | <span style="color:#00aa00">**359.7 nsec ± 1.3%**</span> |
| 32           | 585.7 nsec ± 0.5% | <span style="color:#00aa00">**368.8 nsec ± 1.4%**</span> |
| 64           | 838.2 nsec ± 0.7% | <span style="color:#00aa00">**424.4 nsec ± 1.7%**</span> |
| 128          | 1.288 μsec ± 0.5% | <span style="color:#00aa00">**471.9 nsec ± 1.2%**</span> |
| 256          | 2.151 μsec ± 0.5% | <span style="color:#00aa00">**570.7 nsec ± 1.2%**</span> |
| 512          | 3.959 μsec ± 0.3% | <span style="color:#00aa00">**777.1 nsec ± 0.5%**</span> |
| 1024         | 7.508 μsec ± 0.6% | <span style="color:#00aa00">**1.178 μsec ± 0.7%**</span> |
| 2048         | 14.61 μsec ± 0.5% | <span style="color:#00aa00">**2.010 μsec ± 0.8%**</span> |
| 4096         | 28.91 μsec ± 0.2% | <span style="color:#00aa00">**3.647 μsec ± 0.8%**</span> |
| 8192         | 57.48 μsec ± 0.2% | <span style="color:#00aa00">**6.917 μsec ± 0.9%**</span> |
| **Geomean:** | 2.191 μsec ± 0.6% | <span style="color:#00aa00">**765.7 nsec ± 1.1%**</span> |

