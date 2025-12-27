## C. WITH replacement, CUSTOM probabilities

| `k`   | `n`          | `randint_numpy`   | `randint_numba`                                          |
| ----- | ------------ | ----------------- | -------------------------------------------------------- |
| 1     | 10           | 8.470 μsec ± 0.4% | <span style="color:#00aa00">**543.3 nsec ± 1.3%**</span> |
| 10    | 10           | 8.591 μsec ± 0.2% | <span style="color:#00aa00">**603.2 nsec ± 1.8%**</span> |
| 100   | 10           | 10.18 μsec ± 0.3% | <span style="color:#00aa00">**1.753 μsec ± 0.5%**</span> |
| 1000  | 10           | 24.97 μsec ± 0.4% | <span style="color:#00aa00">**10.85 μsec ± 0.3%**</span> |
| 10000 | 10           | 182.0 μsec ± 0.2% | <span style="color:#00aa00">**114.3 μsec ± 0.4%**</span> |
| 1     | 100          | 8.990 μsec ± 0.4% | <span style="color:#00aa00">**624.6 nsec ± 0.4%**</span> |
| 10    | 100          | 9.359 μsec ± 0.3% | <span style="color:#00aa00">**770.6 nsec ± 1.2%**</span> |
| 100   | 100          | 12.54 μsec ± 0.4% | <span style="color:#00aa00">**2.279 μsec ± 0.3%**</span> |
| 1000  | 100          | 44.11 μsec ± 0.3% | <span style="color:#00aa00">**14.44 μsec ± 0.2%**</span> |
| 10000 | 100          | 350.9 μsec ± 1.5% | <span style="color:#00aa00">**140.8 μsec ± 0.5%**</span> |
| 1     | 1000         | 13.57 μsec ± 0.9% | <span style="color:#00aa00">**1.280 μsec ± 0.9%**</span> |
| 10    | 1000         | 13.93 μsec ± 0.7% | <span style="color:#00aa00">**1.387 μsec ± 0.3%**</span> |
| 100   | 1000         | 18.73 μsec ± 3.7% | <span style="color:#00aa00">**2.536 μsec ± 0.2%**</span> |
| 1000  | 1000         | 69.11 μsec ± 1.7% | <span style="color:#00aa00">**13.64 μsec ± 0.2%**</span> |
| 10000 | 1000         | 541.0 μsec ± 1.7% | <span style="color:#00aa00">**124.9 μsec ± 0.3%**</span> |
| 1     | 10000        | 72.43 μsec ± 0.2% | <span style="color:#00aa00">**7.776 μsec ± 1.7%**</span> |
| 10    | 10000        | 61.87 μsec ± 2.0% | <span style="color:#00aa00">**7.958 μsec ± 2.5%**</span> |
| 100   | 10000        | 79.22 μsec ± 0.2% | <span style="color:#00aa00">**11.15 μsec ± 2.7%**</span> |
| 1000  | 10000        | 139.2 μsec ± 0.5% | <span style="color:#00aa00">**41.96 μsec ± 1.1%**</span> |
| 10000 | 10000        | 755.0 μsec ± 1.5% | <span style="color:#00aa00">**348.3 μsec ± 1.9%**</span> |
|       | **Geomean:** | 41.85 μsec ± 0.9% | <span style="color:#00aa00">**7.153 μsec ± 0.9%**</span> |

