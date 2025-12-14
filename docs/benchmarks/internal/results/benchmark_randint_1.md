## A. WITH replacement, UNIFORM probabilities

| `k`   | `n`          | `randint_numpy`   | `randint_numba`                                          |
| ----- | ------------ | ----------------- | -------------------------------------------------------- |
| 1     | 10           | 3.990 μsec ± 0.4% | <span style="color:#00aa00">**498.2 nsec ± 0.9%**</span> |
| 10    | 10           | 4.031 μsec ± 0.9% | <span style="color:#00aa00">**501.6 nsec ± 1.2%**</span> |
| 100   | 10           | 4.717 μsec ± 0.6% | <span style="color:#00aa00">**626.7 nsec ± 0.9%**</span> |
| 1000  | 10           | 11.22 μsec ± 0.7% | <span style="color:#00aa00">**1.767 μsec ± 0.2%**</span> |
| 10000 | 10           | 73.66 μsec ± 0.9% | <span style="color:#00aa00">**13.48 μsec ± 0.2%**</span> |
| 1     | 100          | 4.054 μsec ± 0.6% | <span style="color:#00aa00">**499.3 nsec ± 0.7%**</span> |
| 10    | 100          | 4.107 μsec ± 0.4% | <span style="color:#00aa00">**506.1 nsec ± 1.0%**</span> |
| 100   | 100          | 4.454 μsec ± 0.6% | <span style="color:#00aa00">**657.6 nsec ± 1.3%**</span> |
| 1000  | 100          | 8.199 μsec ± 0.7% | <span style="color:#00aa00">**1.784 μsec ± 0.4%**</span> |
| 10000 | 100          | 43.10 μsec ± 0.2% | <span style="color:#00aa00">**13.48 μsec ± 0.2%**</span> |
| 1     | 1000         | 4.117 μsec ± 0.5% | <span style="color:#00aa00">**504.9 nsec ± 2.1%**</span> |
| 10    | 1000         | 4.144 μsec ± 0.8% | <span style="color:#00aa00">**512.2 nsec ± 1.1%**</span> |
| 100   | 1000         | 4.312 μsec ± 0.5% | <span style="color:#00aa00">**665.3 nsec ± 0.8%**</span> |
| 1000  | 1000         | 5.829 μsec ± 0.5% | <span style="color:#00aa00">**1.784 μsec ± 0.3%**</span> |
| 10000 | 1000         | 19.73 μsec ± 0.2% | <span style="color:#00aa00">**13.48 μsec ± 0.2%**</span> |
| 1     | 10000        | 4.122 μsec ± 0.6% | <span style="color:#00aa00">**504.7 nsec ± 1.3%**</span> |
| 10    | 10000        | 4.191 μsec ± 0.5% | <span style="color:#00aa00">**517.2 nsec ± 1.6%**</span> |
| 100   | 10000        | 4.936 μsec ± 0.5% | <span style="color:#00aa00">**659.6 nsec ± 0.3%**</span> |
| 1000  | 10000        | 12.10 μsec ± 0.4% | <span style="color:#00aa00">**1.784 μsec ± 0.5%**</span> |
| 10000 | 10000        | 80.34 μsec ± 0.4% | <span style="color:#00aa00">**13.48 μsec ± 0.1%**</span> |
|       | **Geomean:** | 8.000 μsec ± 0.5% | <span style="color:#00aa00">**1.319 μsec ± 0.8%**</span> |

