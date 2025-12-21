## B. WITHOUT replacement, UNIFORM probabilities

| `k`   | `n`          | `randint_numpy`   | `randint_numba`                                          |
| ----- | ------------ | ----------------- | -------------------------------------------------------- |
| 1     | 10           | 4.121 μsec ± 1.3% | <span style="color:#00aa00">**485.4 nsec ± 2.1%**</span> |
| 10    | 10           | 3.109 μsec ± 0.8% | <span style="color:#00aa00">**518.3 nsec ± 1.7%**</span> |
| 1     | 100          | 4.105 μsec ± 0.9% | <span style="color:#00aa00">**494.0 nsec ± 0.9%**</span> |
| 10    | 100          | 3.811 μsec ± 0.4% | <span style="color:#00aa00">**571.0 nsec ± 0.6%**</span> |
| 100   | 100          | 3.814 μsec ± 0.1% | <span style="color:#00aa00">**857.0 nsec ± 1.0%**</span> |
| 1     | 1000         | 4.175 μsec ± 1.4% | <span style="color:#00aa00">**485.0 nsec ± 0.5%**</span> |
| 10    | 1000         | 9.715 μsec ± 0.4% | <span style="color:#00aa00">**587.1 nsec ± 0.6%**</span> |
| 100   | 1000         | 9.727 μsec ± 0.3% | <span style="color:#00aa00">**831.3 nsec ± 0.4%**</span> |
| 1000  | 1000         | 9.900 μsec ± 0.2% | <span style="color:#00aa00">**3.392 μsec ± 0.3%**</span> |
| 1     | 10000        | 4.171 μsec ± 0.9% | <span style="color:#00aa00">**492.4 nsec ± 1.4%**</span> |
| 10    | 10000        | 75.96 μsec ± 0.5% | <span style="color:#00aa00">**968.6 nsec ± 0.9%**</span> |
| 100   | 10000        | 75.94 μsec ± 0.2% | <span style="color:#00aa00">**1.189 μsec ± 0.5%**</span> |
| 1000  | 10000        | 75.98 μsec ± 0.1% | <span style="color:#00aa00">**3.652 μsec ± 0.3%**</span> |
| 10000 | 10000        | 76.72 μsec ± 0.2% | <span style="color:#00aa00">**28.12 μsec ± 1.1%**</span> |
|       | **Geomean:** | 11.08 μsec ± 0.5% | <span style="color:#00aa00">**1.078 μsec ± 0.9%**</span> |

