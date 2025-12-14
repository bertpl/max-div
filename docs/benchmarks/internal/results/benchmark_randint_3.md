## C. WITH replacement, CUSTOM probabilities

| `k`   | `n`          | `randint_numpy`   | `randint_numba`                                          |
| ----- | ------------ | ----------------- | -------------------------------------------------------- |
| 1     | 10           | 8.458 μsec ± 0.4% | <span style="color:#00aa00">**551.5 nsec ± 1.4%**</span> |
| 10    | 10           | 8.679 μsec ± 0.4% | <span style="color:#00aa00">**643.0 nsec ± 0.8%**</span> |
| 100   | 10           | 10.20 μsec ± 0.7% | <span style="color:#00aa00">**1.664 μsec ± 0.6%**</span> |
| 1000  | 10           | 24.68 μsec ± 0.8% | <span style="color:#00aa00">**12.54 μsec ± 0.5%**</span> |
| 10000 | 10           | 155.1 μsec ± 0.3% | <span style="color:#00aa00">**99.60 μsec ± 0.7%**</span> |
| 1     | 100          | 9.022 μsec ± 0.9% | <span style="color:#00aa00">**647.0 nsec ± 1.9%**</span> |
| 10    | 100          | 9.465 μsec ± 0.6% | <span style="color:#00aa00">**758.9 nsec ± 0.8%**</span> |
| 100   | 100          | 12.27 μsec ± 0.5% | <span style="color:#00aa00">**2.037 μsec ± 0.2%**</span> |
| 1000  | 100          | 41.30 μsec ± 0.3% | <span style="color:#00aa00">**12.13 μsec ± 0.1%**</span> |
| 10000 | 100          | 335.8 μsec ± 0.6% | <span style="color:#00aa00">**129.5 μsec ± 0.5%**</span> |
| 1     | 1000         | 13.31 μsec ± 0.8% | <span style="color:#00aa00">**1.226 μsec ± 0.8%**</span> |
| 10    | 1000         | 13.77 μsec ± 0.4% | <span style="color:#00aa00">**1.317 μsec ± 0.6%**</span> |
| 100   | 1000         | 18.19 μsec ± 1.1% | <span style="color:#00aa00">**2.422 μsec ± 0.2%**</span> |
| 1000  | 1000         | 63.42 μsec ± 0.5% | <span style="color:#00aa00">**12.99 μsec ± 0.1%**</span> |
| 10000 | 1000         | 510.9 μsec ± 0.2% | <span style="color:#00aa00">**119.9 μsec ± 0.1%**</span> |
| 1     | 10000        | 53.94 μsec ± 4.1% | <span style="color:#00aa00">**7.368 μsec ± 0.5%**</span> |
| 10    | 10000        | 57.53 μsec ± 9.3% | <span style="color:#00aa00">**7.645 μsec ± 0.6%**</span> |
| 100   | 10000        | 62.83 μsec ± 1.1% | <span style="color:#00aa00">**10.74 μsec ± 0.3%**</span> |
| 1000  | 10000        | 123.5 μsec ± 2.0% | <span style="color:#00aa00">**40.18 μsec ± 0.2%**</span> |
| 10000 | 10000        | 706.9 μsec ± 0.4% | <span style="color:#00aa00">**335.1 μsec ± 0.4%**</span> |
|       | **Geomean:** | 39.28 μsec ± 1.2% | <span style="color:#00aa00">**6.886 μsec ± 0.6%**</span> |

