## B. WITHOUT replacement, UNIFORM probabilities

| `k`   | `n`          | `randint_numpy`   | `randint_numba`                                          |
| ----- | ------------ | ----------------- | -------------------------------------------------------- |
| 1     | 10           | 4.140 μsec ± 0.6% | <span style="color:#00aa00">**495.9 nsec ± 1.9%**</span> |
| 10    | 10           | 3.124 μsec ± 0.5% | <span style="color:#00aa00">**550.8 nsec ± 1.9%**</span> |
| 1     | 100          | 4.077 μsec ± 0.4% | <span style="color:#00aa00">**508.0 nsec ± 0.9%**</span> |
| 10    | 100          | 3.793 μsec ± 1.0% | <span style="color:#00aa00">**567.8 nsec ± 1.2%**</span> |
| 100   | 100          | 3.746 μsec ± 0.5% | <span style="color:#00aa00">**864.3 nsec ± 0.9%**</span> |
| 1     | 1000         | 4.152 μsec ± 0.7% | <span style="color:#00aa00">**510.1 nsec ± 2.5%**</span> |
| 10    | 1000         | 9.732 μsec ± 0.2% | <span style="color:#00aa00">**582.8 nsec ± 0.7%**</span> |
| 100   | 1000         | 9.707 μsec ± 0.2% | <span style="color:#00aa00">**838.8 nsec ± 0.5%**</span> |
| 1000  | 1000         | 9.846 μsec ± 0.2% | <span style="color:#00aa00">**3.205 μsec ± 0.4%**</span> |
| 1     | 10000        | 4.165 μsec ± 0.9% | <span style="color:#00aa00">**518.0 nsec ± 0.9%**</span> |
| 10    | 10000        | 76.25 μsec ± 0.3% | <span style="color:#00aa00">**967.7 nsec ± 0.9%**</span> |
| 100   | 10000        | 76.25 μsec ± 0.2% | <span style="color:#00aa00">**1.173 μsec ± 0.5%**</span> |
| 1000  | 10000        | 76.22 μsec ± 0.2% | <span style="color:#00aa00">**3.435 μsec ± 0.3%**</span> |
| 10000 | 10000        | 76.72 μsec ± 0.4% | <span style="color:#00aa00">**26.51 μsec ± 0.1%**</span> |
|       | **Geomean:** | 11.06 μsec ± 0.5% | <span style="color:#00aa00">**1.079 μsec ± 1.0%**</span> |

