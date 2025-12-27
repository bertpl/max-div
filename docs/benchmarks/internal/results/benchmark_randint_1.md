## A. WITH replacement, UNIFORM probabilities

| `k`   | `n`          | `randint_numpy`   | `randint_numba`                                          |
| ----- | ------------ | ----------------- | -------------------------------------------------------- |
| 1     | 10           | 4.094 μsec ± 0.8% | <span style="color:#00aa00">**496.4 nsec ± 2.2%**</span> |
| 10    | 10           | 4.160 μsec ± 0.5% | <span style="color:#00aa00">**504.0 nsec ± 1.0%**</span> |
| 100   | 10           | 4.839 μsec ± 0.5% | <span style="color:#00aa00">**656.9 nsec ± 0.9%**</span> |
| 1000  | 10           | 11.82 μsec ± 1.3% | <span style="color:#00aa00">**1.878 μsec ± 0.7%**</span> |
| 10000 | 10           | 76.59 μsec ± 0.2% | <span style="color:#00aa00">**14.41 μsec ± 0.5%**</span> |
| 1     | 100          | 4.132 μsec ± 0.5% | <span style="color:#00aa00">**495.5 nsec ± 0.8%**</span> |
| 10    | 100          | 4.188 μsec ± 0.4% | <span style="color:#00aa00">**518.8 nsec ± 1.3%**</span> |
| 100   | 100          | 4.616 μsec ± 0.3% | <span style="color:#00aa00">**652.0 nsec ± 1.2%**</span> |
| 1000  | 100          | 8.255 μsec ± 0.2% | <span style="color:#00aa00">**1.887 μsec ± 0.5%**</span> |
| 10000 | 100          | 43.11 μsec ± 0.2% | <span style="color:#00aa00">**14.40 μsec ± 0.3%**</span> |
| 1     | 1000         | 4.257 μsec ± 0.5% | <span style="color:#00aa00">**508.4 nsec ± 1.7%**</span> |
| 10    | 1000         | 4.233 μsec ± 0.3% | <span style="color:#00aa00">**504.8 nsec ± 0.4%**</span> |
| 100   | 1000         | 4.301 μsec ± 0.4% | <span style="color:#00aa00">**674.1 nsec ± 1.5%**</span> |
| 1000  | 1000         | 5.869 μsec ± 0.3% | <span style="color:#00aa00">**1.902 μsec ± 0.7%**</span> |
| 10000 | 1000         | 19.86 μsec ± 0.7% | <span style="color:#00aa00">**14.46 μsec ± 0.2%**</span> |
| 1     | 10000        | 4.200 μsec ± 0.1% | <span style="color:#00aa00">**507.8 nsec ± 0.8%**</span> |
| 10    | 10000        | 4.271 μsec ± 0.2% | <span style="color:#00aa00">**512.9 nsec ± 2.0%**</span> |
| 100   | 10000        | 4.971 μsec ± 0.3% | <span style="color:#00aa00">**664.0 nsec ± 1.0%**</span> |
| 1000  | 10000        | 12.10 μsec ± 1.2% | <span style="color:#00aa00">**1.895 μsec ± 0.4%**</span> |
| 10000 | 10000        | 81.89 μsec ± 0.9% | <span style="color:#00aa00">**14.42 μsec ± 0.2%**</span> |
|       | **Geomean:** | 8.155 μsec ± 0.5% | <span style="color:#00aa00">**1.358 μsec ± 0.9%**</span> |

