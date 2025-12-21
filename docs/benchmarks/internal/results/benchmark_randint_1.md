## A. WITH replacement, UNIFORM probabilities

| `k`   | `n`          | `randint_numpy`   | `randint_numba`                                          |
| ----- | ------------ | ----------------- | -------------------------------------------------------- |
| 1     | 10           | 4.092 μsec ± 1.6% | <span style="color:#00aa00">**492.3 nsec ± 3.2%**</span> |
| 10    | 10           | 4.249 μsec ± 1.0% | <span style="color:#00aa00">**490.1 nsec ± 1.4%**</span> |
| 100   | 10           | 4.956 μsec ± 0.4% | <span style="color:#00aa00">**664.7 nsec ± 1.0%**</span> |
| 1000  | 10           | 11.63 μsec ± 0.5% | <span style="color:#00aa00">**1.805 μsec ± 0.4%**</span> |
| 10000 | 10           | 76.52 μsec ± 0.4% | <span style="color:#00aa00">**13.48 μsec ± 0.2%**</span> |
| 1     | 100          | 4.129 μsec ± 1.1% | <span style="color:#00aa00">**485.1 nsec ± 0.7%**</span> |
| 10    | 100          | 4.171 μsec ± 1.1% | <span style="color:#00aa00">**502.9 nsec ± 1.6%**</span> |
| 100   | 100          | 4.552 μsec ± 0.3% | <span style="color:#00aa00">**659.3 nsec ± 0.9%**</span> |
| 1000  | 100          | 8.239 μsec ± 0.3% | <span style="color:#00aa00">**1.847 μsec ± 0.8%**</span> |
| 10000 | 100          | 43.04 μsec ± 0.1% | <span style="color:#00aa00">**13.58 μsec ± 0.5%**</span> |
| 1     | 1000         | 4.189 μsec ± 0.4% | <span style="color:#00aa00">**492.1 nsec ± 1.9%**</span> |
| 10    | 1000         | 4.160 μsec ± 0.4% | <span style="color:#00aa00">**499.4 nsec ± 0.5%**</span> |
| 100   | 1000         | 4.313 μsec ± 0.7% | <span style="color:#00aa00">**681.2 nsec ± 1.3%**</span> |
| 1000  | 1000         | 5.872 μsec ± 0.6% | <span style="color:#00aa00">**1.877 μsec ± 0.4%**</span> |
| 10000 | 1000         | 19.83 μsec ± 0.3% | <span style="color:#00aa00">**13.82 μsec ± 0.5%**</span> |
| 1     | 10000        | 4.155 μsec ± 0.6% | <span style="color:#00aa00">**482.2 nsec ± 0.8%**</span> |
| 10    | 10000        | 4.260 μsec ± 0.7% | <span style="color:#00aa00">**512.0 nsec ± 1.7%**</span> |
| 100   | 10000        | 5.002 μsec ± 0.6% | <span style="color:#00aa00">**667.5 nsec ± 1.7%**</span> |
| 1000  | 10000        | 12.25 μsec ± 0.6% | <span style="color:#00aa00">**1.900 μsec ± 0.5%**</span> |
| 10000 | 10000        | 80.86 μsec ± 0.7% | <span style="color:#00aa00">**13.94 μsec ± 0.5%**</span> |
|       | **Geomean:** | 8.141 μsec ± 0.6% | <span style="color:#00aa00">**1.330 μsec ± 1.0%**</span> |

