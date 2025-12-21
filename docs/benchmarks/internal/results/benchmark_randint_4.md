## D. WITHOUT replacement, CUSTOM probabilities

| `k`   | `n`          | `randint_numpy`    | `randint_numba`                                          |
| ----- | ------------ | ------------------ | -------------------------------------------------------- |
| 1     | 10           | 8.475 μsec ± 17.8% | <span style="color:#00aa00">**520.7 nsec ± 1.1%**</span> |
| 10    | 10           | 27.90 μsec ± 0.3%  | <span style="color:#00aa00">**514.7 nsec ± 0.9%**</span> |
| 1     | 100          | 8.924 μsec ± 0.5%  | <span style="color:#00aa00">**620.2 nsec ± 0.8%**</span> |
| 10    | 100          | 17.55 μsec ± 0.3%  | <span style="color:#00aa00">**1.345 μsec ± 0.4%**</span> |
| 100   | 100          | 55.54 μsec ± 0.2%  | <span style="color:#00aa00">**849.8 nsec ± 0.8%**</span> |
| 1     | 1000         | 13.49 μsec ± 0.4%  | <span style="color:#00aa00">**1.259 μsec ± 0.5%**</span> |
| 10    | 1000         | 20.49 μsec ± 0.4%  | <span style="color:#00aa00">**4.667 μsec ± 0.2%**</span> |
| 100   | 1000         | 35.89 μsec ± 1.9%  | <span style="color:#00aa00">**11.32 μsec ± 0.3%**</span> |
| 1000  | 1000         | 258.3 μsec ± 0.6%  | <span style="color:#00aa00">**3.381 μsec ± 0.2%**</span> |
| 1     | 10000        | 50.77 μsec ± 5.2%  | <span style="color:#00aa00">**7.528 μsec ± 0.6%**</span> |
| 10    | 10000        | 65.51 μsec ± 1.8%  | <span style="color:#00aa00">**33.83 μsec ± 0.1%**</span> |
| 100   | 10000        | 79.41 μsec ± 4.2%  | <span style="color:#00aa00">**49.02 μsec ± 0.1%**</span> |
| 1000  | 10000        | 210.3 μsec ± 1.3%  | <span style="color:#00aa00">**96.54 μsec ± 0.3%**</span> |
| 10000 | 10000        | 2.566 msec ± 0.7%  | <span style="color:#00aa00">**28.24 μsec ± 0.1%**</span> |
|       | **Geomean:** | 50.75 μsec ± 2.4%  | <span style="color:#00aa00">**4.513 μsec ± 0.4%**</span> |

