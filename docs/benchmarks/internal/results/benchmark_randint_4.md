## D. WITHOUT replacement, CUSTOM probabilities

| `k`   | `n`          | `randint_numpy`   | `randint_numba`                                          |
| ----- | ------------ | ----------------- | -------------------------------------------------------- |
| 1     | 10           | 8.460 μsec ± 0.3% | <span style="color:#00aa00">**531.5 nsec ± 0.7%**</span> |
| 10    | 10           | 26.89 μsec ± 0.5% | <span style="color:#00aa00">**522.9 nsec ± 0.3%**</span> |
| 1     | 100          | 9.007 μsec ± 0.7% | <span style="color:#00aa00">**628.2 nsec ± 1.0%**</span> |
| 10    | 100          | 18.00 μsec ± 0.6% | <span style="color:#00aa00">**1.368 μsec ± 0.4%**</span> |
| 100   | 100          | 59.73 μsec ± 0.4% | <span style="color:#00aa00">**838.8 nsec ± 0.5%**</span> |
| 1     | 1000         | 15.79 μsec ± 0.1% | <span style="color:#00aa00">**1.260 μsec ± 0.8%**</span> |
| 10    | 1000         | 20.68 μsec ± 0.4% | <span style="color:#00aa00">**4.629 μsec ± 0.4%**</span> |
| 100   | 1000         | 36.27 μsec ± 0.4% | <span style="color:#00aa00">**11.88 μsec ± 0.6%**</span> |
| 1000  | 1000         | 279.7 μsec ± 0.3% | <span style="color:#00aa00">**3.361 μsec ± 0.3%**</span> |
| 1     | 10000        | 54.44 μsec ± 1.9% | <span style="color:#00aa00">**7.764 μsec ± 2.6%**</span> |
| 10    | 10000        | 83.76 μsec ± 6.7% | <span style="color:#00aa00">**33.80 μsec ± 0.3%**</span> |
| 100   | 10000        | 107.9 μsec ± 6.8% | <span style="color:#00aa00">**48.97 μsec ± 0.2%**</span> |
| 1000  | 10000        | 252.6 μsec ± 2.0% | <span style="color:#00aa00">**100.1 μsec ± 1.6%**</span> |
| 10000 | 10000        | 2.883 msec ± 0.8% | <span style="color:#00aa00">**28.23 μsec ± 0.3%**</span> |
|       | **Geomean:** | 55.48 μsec ± 1.6% | <span style="color:#00aa00">**4.563 μsec ± 0.7%**</span> |

