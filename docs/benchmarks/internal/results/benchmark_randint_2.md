## B. WITHOUT replacement, UNIFORM probabilities

| `k`   | `n`          | `randint_numpy`   | `randint_numba`                                          |
| ----- | ------------ | ----------------- | -------------------------------------------------------- |
| 1     | 10           | 4.141 μsec ± 0.6% | <span style="color:#00aa00">**488.9 nsec ± 0.5%**</span> |
| 10    | 10           | 3.182 μsec ± 0.3% | <span style="color:#00aa00">**525.5 nsec ± 0.5%**</span> |
| 1     | 100          | 4.144 μsec ± 0.3% | <span style="color:#00aa00">**499.1 nsec ± 1.2%**</span> |
| 10    | 100          | 3.763 μsec ± 0.3% | <span style="color:#00aa00">**558.0 nsec ± 0.6%**</span> |
| 100   | 100          | 3.846 μsec ± 0.2% | <span style="color:#00aa00">**861.0 nsec ± 0.5%**</span> |
| 1     | 1000         | 4.263 μsec ± 0.4% | <span style="color:#00aa00">**503.8 nsec ± 0.4%**</span> |
| 10    | 1000         | 9.778 μsec ± 0.3% | <span style="color:#00aa00">**575.0 nsec ± 0.6%**</span> |
| 100   | 1000         | 9.756 μsec ± 0.4% | <span style="color:#00aa00">**818.5 nsec ± 0.3%**</span> |
| 1000  | 1000         | 9.900 μsec ± 0.4% | <span style="color:#00aa00">**3.385 μsec ± 0.3%**</span> |
| 1     | 10000        | 4.193 μsec ± 1.2% | <span style="color:#00aa00">**500.1 nsec ± 1.0%**</span> |
| 10    | 10000        | 76.38 μsec ± 0.3% | <span style="color:#00aa00">**996.4 nsec ± 1.3%**</span> |
| 100   | 10000        | 76.01 μsec ± 0.2% | <span style="color:#00aa00">**1.199 μsec ± 0.5%**</span> |
| 1000  | 10000        | 76.24 μsec ± 0.2% | <span style="color:#00aa00">**3.638 μsec ± 0.3%**</span> |
| 10000 | 10000        | 76.66 μsec ± 0.2% | <span style="color:#00aa00">**28.34 μsec ± 0.6%**</span> |
|       | **Geomean:** | 11.14 μsec ± 0.4% | <span style="color:#00aa00">**1.083 μsec ± 0.6%**</span> |

