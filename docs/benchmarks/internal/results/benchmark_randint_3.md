## C. WITH replacement, CUSTOM probabilities

| `k`   | `n`          | `randint_numpy`   | `randint_numba`                                          |
| ----- | ------------ | ----------------- | -------------------------------------------------------- |
| 1     | 10           | 8.417 μsec ± 0.5% | <span style="color:#00aa00">**533.5 nsec ± 1.6%**</span> |
| 10    | 10           | 8.610 μsec ± 0.3% | <span style="color:#00aa00">**615.4 nsec ± 0.7%**</span> |
| 100   | 10           | 10.02 μsec ± 0.3% | <span style="color:#00aa00">**1.682 μsec ± 0.2%**</span> |
| 1000  | 10           | 24.82 μsec ± 0.1% | <span style="color:#00aa00">**8.714 μsec ± 0.1%**</span> |
| 10000 | 10           | 165.6 μsec ± 0.4% | <span style="color:#00aa00">**130.3 μsec ± 0.2%**</span> |
| 1     | 100          | 10.40 μsec ± 1.7% | <span style="color:#00aa00">**627.3 nsec ± 1.7%**</span> |
| 10    | 100          | 9.242 μsec ± 0.6% | <span style="color:#00aa00">**756.7 nsec ± 1.2%**</span> |
| 100   | 100          | 12.38 μsec ± 0.2% | <span style="color:#00aa00">**2.176 μsec ± 0.5%**</span> |
| 1000  | 100          | 43.20 μsec ± 0.9% | <span style="color:#00aa00">**14.70 μsec ± 0.5%**</span> |
| 10000 | 100          | 336.1 μsec ± 0.8% | <span style="color:#00aa00">**138.2 μsec ± 0.1%**</span> |
| 1     | 1000         | 13.81 μsec ± 3.4% | <span style="color:#00aa00">**1.275 μsec ± 0.7%**</span> |
| 10    | 1000         | 14.48 μsec ± 2.5% | <span style="color:#00aa00">**1.387 μsec ± 0.5%**</span> |
| 100   | 1000         | 18.95 μsec ± 0.6% | <span style="color:#00aa00">**2.582 μsec ± 0.2%**</span> |
| 1000  | 1000         | 64.75 μsec ± 0.7% | <span style="color:#00aa00">**13.64 μsec ± 0.4%**</span> |
| 10000 | 1000         | 514.7 μsec ± 0.6% | <span style="color:#00aa00">**125.5 μsec ± 0.3%**</span> |
| 1     | 10000        | 57.14 μsec ± 6.8% | <span style="color:#00aa00">**7.398 μsec ± 0.4%**</span> |
| 10    | 10000        | 56.29 μsec ± 5.2% | <span style="color:#00aa00">**7.710 μsec ± 0.6%**</span> |
| 100   | 10000        | 60.45 μsec ± 2.8% | <span style="color:#00aa00">**10.91 μsec ± 0.8%**</span> |
| 1000  | 10000        | 130.2 μsec ± 2.3% | <span style="color:#00aa00">**40.68 μsec ± 1.1%**</span> |
| 10000 | 10000        | 764.6 μsec ± 2.4% | <span style="color:#00aa00">**343.8 μsec ± 0.4%**</span> |
|       | **Geomean:** | 40.27 μsec ± 1.7% | <span style="color:#00aa00">**7.040 μsec ± 0.6%**</span> |

