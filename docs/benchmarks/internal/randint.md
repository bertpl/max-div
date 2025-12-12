# `randint`

([API reference][max_div.sampling.uncon.randint])

Command:
```bash
uv tool install max-div
max-div benchmark --markdown randint
```
or 
```bash
uv run max-div benchmark --markdown randint
```

## A. WITH replacement, UNIFORM probabilities
                                                                                                                                                                                                                                 
| `k`   | `n`          | `randint_numpy`   | `randint_numba`                                          |
| ----- | ------------ | ----------------- | -------------------------------------------------------- |
| 1     | 10           | 8.312 μsec ± 0.2% | <span style="color:#00aa00">**990.8 nsec ± 0.8%**</span> |
| 10    | 10           | 8.570 μsec ± 0.8% | <span style="color:#00aa00">**985.9 nsec ± 0.1%**</span> |
| 100   | 10           | 9.410 μsec ± 1.5% | <span style="color:#00aa00">**1.116 μsec ± 0.0%**</span> |
| 1000  | 10           | 19.99 μsec ± 0.1% | <span style="color:#00aa00">**2.715 μsec ± 0.0%**</span> |
| 10000 | 10           | 120.8 μsec ± 0.0% | <span style="color:#00aa00">**17.79 μsec ± 0.0%**</span> |
| 1     | 100          | 8.229 μsec ± 0.3% | <span style="color:#00aa00">**973.1 nsec ± 0.0%**</span> |
| 10    | 100          | 8.330 μsec ± 0.4% | <span style="color:#00aa00">**986.3 nsec ± 0.0%**</span> |
| 100   | 100          | 9.007 μsec ± 0.3% | <span style="color:#00aa00">**1.123 μsec ± 0.5%**</span> |
| 1000  | 100          | 15.59 μsec ± 0.1% | <span style="color:#00aa00">**2.712 μsec ± 0.0%**</span> |
| 10000 | 100          | 77.28 μsec ± 0.1% | <span style="color:#00aa00">**17.79 μsec ± 0.0%**</span> |
| 1     | 1000         | 8.258 μsec ± 0.1% | <span style="color:#00aa00">**970.6 nsec ± 0.1%**</span> |
| 10    | 1000         | 8.364 μsec ± 0.2% | <span style="color:#00aa00">**991.2 nsec ± 0.2%**</span> |
| 100   | 1000         | 8.720 μsec ± 0.1% | <span style="color:#00aa00">**1.133 μsec ± 0.0%**</span> |
| 1000  | 1000         | 12.52 μsec ± 0.5% | <span style="color:#00aa00">**2.711 μsec ± 0.0%**</span> |
| 10000 | 1000         | 45.50 μsec ± 0.4% | <span style="color:#00aa00">**17.79 μsec ± 0.0%**</span> |
| 1     | 10000        | 8.348 μsec ± 0.3% | <span style="color:#00aa00">**976.4 nsec ± 0.1%**</span> |
| 10    | 10000        | 8.534 μsec ± 0.4% | <span style="color:#00aa00">**980.9 nsec ± 0.1%**</span> |
| 100   | 10000        | 9.681 μsec ± 0.4% | <span style="color:#00aa00">**1.133 μsec ± 0.1%**</span> |
| 1000  | 10000        | 22.03 μsec ± 0.4% | <span style="color:#00aa00">**2.712 μsec ± 0.1%**</span> |
| 10000 | 10000        | 127.4 μsec ± 0.1% | <span style="color:#00aa00">**17.79 μsec ± 0.0%**</span> |
|       | **Geomean:** | 15.67 μsec ± 0.3% | <span style="color:#00aa00">**2.207 μsec ± 0.1%**</span> |

## B. WITHOUT replacement, UNIFORM probabilities
                                                                                                                                                                                                                                 
| `k`   | `n`          | `randint_numpy`   | `randint_numba`                                          |
| ----- | ------------ | ----------------- | -------------------------------------------------------- |
| 1     | 10           | 8.575 μsec ± 0.6% | <span style="color:#00aa00">**997.1 nsec ± 0.9%**</span> |
| 10    | 10           | 6.686 μsec ± 0.3% | <span style="color:#00aa00">**1.017 μsec ± 0.0%**</span> |
| 1     | 100          | 8.457 μsec ± 0.2% | <span style="color:#00aa00">**984.1 nsec ± 0.1%**</span> |
| 10    | 100          | 7.736 μsec ± 0.2% | <span style="color:#00aa00">**1.018 μsec ± 0.1%**</span> |
| 100   | 100          | 7.815 μsec ± 0.5% | <span style="color:#00aa00">**1.275 μsec ± 0.1%**</span> |
| 1     | 1000         | 8.587 μsec ± 0.7% | <span style="color:#00aa00">**983.3 nsec ± 0.2%**</span> |
| 10    | 1000         | 17.70 μsec ± 0.4% | <span style="color:#00aa00">**1.143 μsec ± 0.4%**</span> |
| 100   | 1000         | 17.89 μsec ± 0.4% | <span style="color:#00aa00">**1.365 μsec ± 0.5%**</span> |
| 1000  | 1000         | 18.26 μsec ± 0.3% | <span style="color:#00aa00">**3.734 μsec ± 0.0%**</span> |
| 1     | 10000        | 8.474 μsec ± 0.4% | <span style="color:#00aa00">**988.2 nsec ± 0.9%**</span> |
| 10    | 10000        | 122.1 μsec ± 0.1% | <span style="color:#00aa00">**1.499 μsec ± 0.7%**</span> |
| 100   | 10000        | 123.5 μsec ± 0.0% | <span style="color:#00aa00">**1.716 μsec ± 0.6%**</span> |
| 1000  | 10000        | 123.1 μsec ± 0.3% | <span style="color:#00aa00">**3.841 μsec ± 0.1%**</span> |
| 10000 | 10000        | 124.5 μsec ± 0.1% | <span style="color:#00aa00">**25.68 μsec ± 0.0%**</span> |
|       | **Geomean:** | 20.81 μsec ± 0.3% | <span style="color:#00aa00">**1.712 μsec ± 0.3%**</span> |

## C. WITH replacement, CUSTOM probabilities
                                                                                                                                                                                                                                 
| `k`   | `n`          | `randint_numpy`   | `randint_numba`                                          |
| ----- | ------------ | ----------------- | -------------------------------------------------------- |
| 1     | 10           | 22.56 μsec ± 1.5% | <span style="color:#00aa00">**1.039 μsec ± 0.1%**</span> |
| 10    | 10           | 22.76 μsec ± 0.6% | <span style="color:#00aa00">**1.212 μsec ± 0.8%**</span> |
| 100   | 10           | 24.64 μsec ± 0.6% | <span style="color:#00aa00">**2.881 μsec ± 0.3%**</span> |
| 1000  | 10           | 46.60 μsec ± 0.1% | <span style="color:#00aa00">**19.57 μsec ± 0.1%**</span> |
| 10000 | 10           | 253.4 μsec ± 0.1% | <span style="color:#00aa00">**185.9 μsec ± 0.0%**</span> |
| 1     | 100          | 22.92 μsec ± 0.5% | <span style="color:#00aa00">**1.126 μsec ± 0.2%**</span> |
| 10    | 100          | 23.55 μsec ± 0.4% | <span style="color:#00aa00">**1.468 μsec ± 0.1%**</span> |
| 100   | 100          | 27.85 μsec ± 0.4% | <span style="color:#00aa00">**4.804 μsec ± 0.0%**</span> |
| 1000  | 100          | 70.56 μsec ± 0.1% | <span style="color:#00aa00">**38.35 μsec ± 0.0%**</span> |
| 10000 | 100          | 479.6 μsec ± 0.0% | <span style="color:#00aa00">**366.8 μsec ± 0.0%**</span> |
| 1     | 1000         | 32.82 μsec ± 0.3% | <span style="color:#00aa00">**1.972 μsec ± 0.0%**</span> |
| 10    | 1000         | 33.51 μsec ± 0.2% | <span style="color:#00aa00">**2.537 μsec ± 0.0%**</span> |
| 100   | 1000         | 40.02 μsec ± 0.1% | <span style="color:#00aa00">**7.941 μsec ± 0.0%**</span> |
| 1000  | 1000         | 102.1 μsec ± 0.0% | <span style="color:#00aa00">**61.86 μsec ± 0.0%**</span> |
| 10000 | 1000         | 710.6 μsec ± 0.1% | <span style="color:#00aa00">**601.2 μsec ± 0.0%**</span> |
| 1     | 10000        | 103.9 μsec ± 0.2% | <span style="color:#00aa00">**9.535 μsec ± 0.0%**</span> |
| 10    | 10000        | 105.2 μsec ± 0.2% | <span style="color:#00aa00">**10.30 μsec ± 0.0%**</span> |
| 100   | 10000        | 114.6 μsec ± 0.1% | <span style="color:#00aa00">**17.57 μsec ± 0.0%**</span> |
| 1000  | 10000        | 205.1 μsec ± 0.0% | <span style="color:#00aa00">**89.60 μsec ± 0.0%**</span> |
| 10000 | 10000        | 1.090 msec ± 0.0% | <span style="color:#00aa00">**809.5 μsec ± 0.0%**</span> |
|       | **Geomean:** | 77.91 μsec ± 0.3% | <span style="color:#00aa00">**14.85 μsec ± 0.1%**</span> |

## D. WITHOUT replacement, CUSTOM probabilities
                                                                                                                                                                                                                                 
| `k`   | `n`          | `randint_numpy`   | `randint_numba`                                          |
| ----- | ------------ | ----------------- | -------------------------------------------------------- |
| 1     | 10           | 23.15 μsec ± 2.2% | <span style="color:#00aa00">**1.052 μsec ± 0.4%**</span> |
| 10    | 10           | 79.11 μsec ± 0.6% | <span style="color:#00aa00">**1.061 μsec ± 0.6%**</span> |
| 1     | 100          | 23.15 μsec ± 0.4% | <span style="color:#00aa00">**1.136 μsec ± 0.5%**</span> |
| 10    | 100          | 50.92 μsec ± 0.4% | <span style="color:#00aa00">**2.130 μsec ± 0.5%**</span> |
| 100   | 100          | 150.0 μsec ± 0.5% | <span style="color:#00aa00">**1.267 μsec ± 1.2%**</span> |
| 1     | 1000         | 33.02 μsec ± 0.3% | <span style="color:#00aa00">**1.981 μsec ± 0.3%**</span> |
| 10    | 1000         | 56.14 μsec ± 0.3% | <span style="color:#00aa00">**6.196 μsec ± 0.2%**</span> |
| 100   | 1000         | 86.44 μsec ± 0.2% | <span style="color:#00aa00">**16.06 μsec ± 0.2%**</span> |
| 1000  | 1000         | 450.5 μsec ± 0.5% | <span style="color:#00aa00">**3.759 μsec ± 0.2%**</span> |
| 1     | 10000        | 103.8 μsec ± 0.1% | <span style="color:#00aa00">**9.524 μsec ± 0.1%**</span> |
| 10    | 10000        | 130.1 μsec ± 0.2% | <span style="color:#00aa00">**40.48 μsec ± 0.0%**</span> |
| 100   | 10000        | 170.9 μsec ± 0.5% | <span style="color:#00aa00">**63.01 μsec ± 0.0%**</span> |
| 1000  | 10000        | 364.1 μsec ± 0.4% | <span style="color:#00aa00">**147.0 μsec ± 0.5%**</span> |
| 10000 | 10000        | 3.883 msec ± 0.2% | <span style="color:#00aa00">**25.83 μsec ± 0.0%**</span> |
|       | **Geomean:** | 115.6 μsec ± 0.5% | <span style="color:#00aa00">**6.488 μsec ± 0.3%**</span> |