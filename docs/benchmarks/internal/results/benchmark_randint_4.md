## D. WITHOUT replacement, CUSTOM probabilities

| `k`   | `n`          | `randint_numpy`    | `randint_numba`                                          |
| ----- | ------------ | ------------------ | -------------------------------------------------------- |
| 1     | 10           | 8.573 μsec ± 10.9% | <span style="color:#00aa00">**533.5 nsec ± 2.7%**</span> |
| 10    | 10           | 29.10 μsec ± 0.4%  | <span style="color:#00aa00">**532.8 nsec ± 1.4%**</span> |
| 1     | 100          | 9.037 μsec ± 0.7%  | <span style="color:#00aa00">**640.5 nsec ± 1.2%**</span> |
| 10    | 100          | 18.10 μsec ± 0.6%  | <span style="color:#00aa00">**1.350 μsec ± 0.9%**</span> |
| 100   | 100          | 55.32 μsec ± 0.6%  | <span style="color:#00aa00">**851.9 nsec ± 0.6%**</span> |
| 1     | 1000         | 13.53 μsec ± 0.9%  | <span style="color:#00aa00">**1.229 μsec ± 1.0%**</span> |
| 10    | 1000         | 20.93 μsec ± 0.5%  | <span style="color:#00aa00">**4.667 μsec ± 0.3%**</span> |
| 100   | 1000         | 35.85 μsec ± 0.4%  | <span style="color:#00aa00">**11.26 μsec ± 0.7%**</span> |
| 1000  | 1000         | 241.4 μsec ± 0.7%  | <span style="color:#00aa00">**3.200 μsec ± 0.4%**</span> |
| 1     | 10000        | 56.31 μsec ± 5.6%  | <span style="color:#00aa00">**7.362 μsec ± 0.3%**</span> |
| 10    | 10000        | 63.35 μsec ± 3.2%  | <span style="color:#00aa00">**33.80 μsec ± 0.5%**</span> |
| 100   | 10000        | 82.28 μsec ± 3.0%  | <span style="color:#00aa00">**49.13 μsec ± 0.2%**</span> |
| 1000  | 10000        | 205.5 μsec ± 4.3%  | <span style="color:#00aa00">**101.0 μsec ± 0.6%**</span> |
| 10000 | 10000        | 2.461 msec ± 2.0%  | <span style="color:#00aa00">**26.50 μsec ± 0.1%**</span> |
|       | **Geomean:** | 51.07 μsec ± 2.3%  | <span style="color:#00aa00">**4.505 μsec ± 0.8%**</span> |

