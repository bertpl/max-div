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
                                                                                                                                                                                              
| `k`   | `n`          | `randint_numpy`    | `randint_numba`                                          |
| ----- | ------------ |--------------------|----------------------------------------------------------|
| 1     | 10           | 8.488 μsec ± 0.2%  | <span style="color:#00aa00">**966.9 nsec ± 0.8%**</span> |
| 10    | 10           | 8.758 μsec ± 0.3%  | <span style="color:#00aa00">**980.9 nsec ± 0.2%**</span> |
| 100   | 10           | 9.843 μsec ± 0.2%  | <span style="color:#00aa00">**1.118 μsec ± 0.1%**</span> |
| 1000  | 10           | 20.37 μsec ± 0.1%  | <span style="color:#00aa00">**2.666 μsec ± 0.0%**</span> |
| 10000 | 10           | 121.0 μsec ± 0.0%  | <span style="color:#00aa00">**17.82 μsec ± 0.0%**</span> |
| 1     | 100          | 8.753 μsec ± 0.6%  | <span style="color:#00aa00">**983.5 nsec ± 0.8%**</span> |
| 10    | 100          | 8.825 μsec ± 0.3%  | <span style="color:#00aa00">**977.8 nsec ± 0.7%**</span> |
| 100   | 100          | 9.511 μsec ± 0.2%  | <span style="color:#00aa00">**1.131 μsec ± 0.8%**</span> |
| 1000  | 100          | 16.07 μsec ± 0.3%  | <span style="color:#00aa00">**2.680 μsec ± 0.3%**</span> |
| 10000 | 100          | 77.52 μsec ± 0.0%  | <span style="color:#00aa00">**17.82 μsec ± 0.0%**</span> |
| 1     | 1000         | 8.703 μsec ± 0.3%  | <span style="color:#00aa00">**976.9 nsec ± 0.9%**</span> |
| 10    | 1000         | 8.739 μsec ± 0.2%  | <span style="color:#00aa00">**980.7 nsec ± 0.8%**</span> |
| 100   | 1000         | 9.183 μsec ± 0.3%  | <span style="color:#00aa00">**1.143 μsec ± 0.8%**</span> |
| 1000  | 1000         | 12.86 μsec ± 0.2%  | <span style="color:#00aa00">**2.666 μsec ± 0.3%**</span> |
| 10000 | 1000         | 45.52 μsec ± 0.1%  | <span style="color:#00aa00">**17.82 μsec ± 0.0%**</span> |
| 1     | 10000        | 8.818 μsec ± 0.6%  | <span style="color:#00aa00">**980.7 nsec ± 0.7%**</span> |
| 10    | 10000        | 8.833 μsec ± 0.3%  | <span style="color:#00aa00">**983.7 nsec ± 0.7%**</span> |
| 100   | 10000        | 9.978 μsec ± 0.1%  | <span style="color:#00aa00">**1.126 μsec ± 0.5%**</span> |
| 1000  | 10000        | 20.98 μsec ± 0.1%  | <span style="color:#00aa00">**2.669 μsec ± 0.3%**</span> |
| 10000 | 10000        | 126.1 μsec ± 0.1%  | <span style="color:#00aa00">**17.82 μsec ± 0.0%**</span> |
|       | **Geomean:** | 16.11 μsec ± 0.2%  | <span style="color:#00aa00">**2.200 μsec ± 0.4%**</span> |

## B. WITHOUT replacement, UNIFORM probabilities
                                                                                                                                                                                              
| `k`   | `n`          | `randint_numpy`    | `randint_numba`                                          |
| ----- | ------------ |--------------------|----------------------------------------------------------|
| 1     | 10           | 8.669 μsec ± 0.5%  | <span style="color:#00aa00">**972.7 nsec ± 0.2%**</span> |
| 10    | 10           | 6.827 μsec ± 0.5%  | <span style="color:#00aa00">**1.017 μsec ± 0.1%**</span> |
| 1     | 100          | 8.595 μsec ± 0.2%  | <span style="color:#00aa00">**967.2 nsec ± 0.3%**</span> |
| 10    | 100          | 7.902 μsec ± 0.3%  | <span style="color:#00aa00">**1.011 μsec ± 0.2%**</span> |
| 100   | 100          | 7.936 μsec ± 0.2%  | <span style="color:#00aa00">**1.262 μsec ± 0.1%**</span> |
| 1     | 1000         | 8.724 μsec ± 0.3%  | <span style="color:#00aa00">**972.4 nsec ± 0.4%**</span> |
| 10    | 1000         | 17.48 μsec ± 0.2%  | <span style="color:#00aa00">**1.063 μsec ± 0.1%**</span> |
| 100   | 1000         | 17.52 μsec ± 11.2% | <span style="color:#00aa00">**1.301 μsec ± 0.1%**</span> |
| 1000  | 1000         | 17.91 μsec ± 0.1%  | <span style="color:#00aa00">**3.672 μsec ± 0.0%**</span> |
| 1     | 10000        | 8.732 μsec ± 0.2%  | <span style="color:#00aa00">**983.6 nsec ± 0.2%**</span> |
| 10    | 10000        | 121.7 μsec ± 0.0%  | <span style="color:#00aa00">**1.487 μsec ± 0.1%**</span> |
| 100   | 10000        | 121.5 μsec ± 0.1%  | <span style="color:#00aa00">**1.728 μsec ± 0.2%**</span> |
| 1000  | 10000        | 122.4 μsec ± 0.0%  | <span style="color:#00aa00">**3.873 μsec ± 0.1%**</span> |
| 10000 | 10000        | 123.2 μsec ± 0.0%  | <span style="color:#00aa00">**25.73 μsec ± 0.0%**</span> |
|       | **Geomean:** | 20.87 μsec ± 0.9%  | <span style="color:#00aa00">**1.687 μsec ± 0.2%**</span> |

## C. WITH replacement, CUSTOM probabilities
                                                                                                                                                                                              
| `k`   | `n`          | `randint_numpy`    | `randint_numba`                                          |
| ----- | ------------ |--------------------|----------------------------------------------------------|
| 1     | 10           | 16.34 μsec ± 0.2%  | <span style="color:#00aa00">**1.022 μsec ± 0.2%**</span> |
| 10    | 10           | 16.58 μsec ± 0.2%  | <span style="color:#00aa00">**1.214 μsec ± 0.8%**</span> |
| 100   | 10           | 18.52 μsec ± 0.2%  | <span style="color:#00aa00">**2.702 μsec ± 0.4%**</span> |
| 1000  | 10           | 40.53 μsec ± 0.1%  | <span style="color:#00aa00">**18.64 μsec ± 0.0%**</span> |
| 10000 | 10           | 246.2 μsec ± 0.0%  | <span style="color:#00aa00">**164.8 μsec ± 0.0%**</span> |
| 1     | 100          | 16.98 μsec ± 0.2%  | <span style="color:#00aa00">**1.110 μsec ± 0.7%**</span> |
| 10    | 100          | 17.51 μsec ± 0.2%  | <span style="color:#00aa00">**1.443 μsec ± 0.8%**</span> |
| 100   | 100          | 21.56 μsec ± 0.2%  | <span style="color:#00aa00">**4.694 μsec ± 0.2%**</span> |
| 1000  | 100          | 63.49 μsec ± 0.0%  | <span style="color:#00aa00">**37.36 μsec ± 0.0%**</span> |
| 10000 | 100          | 474.4 μsec ± 0.1%  | <span style="color:#00aa00">**363.6 μsec ± 0.0%**</span> |
| 1     | 1000         | 24.80 μsec ± 0.1%  | <span style="color:#00aa00">**1.905 μsec ± 0.1%**</span> |
| 10    | 1000         | 25.50 μsec ± 0.2%  | <span style="color:#00aa00">**2.467 μsec ± 0.0%**</span> |
| 100   | 1000         | 31.74 μsec ± 0.1%  | <span style="color:#00aa00">**7.824 μsec ± 0.1%**</span> |
| 1000  | 1000         | 93.53 μsec ± 0.0%  | <span style="color:#00aa00">**61.49 μsec ± 0.0%**</span> |
| 10000 | 1000         | 703.3 μsec ± 0.0%  | <span style="color:#00aa00">**596.4 μsec ± 0.0%**</span> |
| 1     | 10000        | 93.00 μsec ± 0.1%  | <span style="color:#00aa00">**9.515 μsec ± 0.1%**</span> |
| 10    | 10000        | 94.36 μsec ± 0.2%  | <span style="color:#00aa00">**10.29 μsec ± 0.0%**</span> |
| 100   | 10000        | 103.5 μsec ± 0.2%  | <span style="color:#00aa00">**17.57 μsec ± 0.1%**</span> |
| 1000  | 10000        | 194.5 μsec ± 0.0%  | <span style="color:#00aa00">**89.57 μsec ± 0.0%**</span> |
| 10000 | 10000        | 1.079 msec ± 0.0%  | <span style="color:#00aa00">**808.1 μsec ± 0.0%**</span> |
|       | **Geomean:** | 65.96 μsec ± 0.1%  | <span style="color:#00aa00">**14.53 μsec ± 0.2%**</span> |

## D. WITHOUT replacement, CUSTOM probabilities
                                                                                                                                                                                              
| `k`   | `n`          | `randint_numpy`    | `randint_numba`                                          |
| ----- | ------------ |--------------------|----------------------------------------------------------|
| 1     | 10           | 16.40 μsec ± 0.3%  | <span style="color:#00aa00">**1.016 μsec ± 0.2%**</span> |
| 10    | 10           | 66.43 μsec ± 0.3%  | <span style="color:#00aa00">**1.008 μsec ± 0.2%**</span> |
| 1     | 100          | 17.14 μsec ± 0.3%  | <span style="color:#00aa00">**1.088 μsec ± 0.1%**</span> |
| 10    | 100          | 42.90 μsec ± 0.3%  | <span style="color:#00aa00">**2.139 μsec ± 0.0%**</span> |
| 100   | 100          | 131.5 μsec ± 0.4%  | <span style="color:#00aa00">**1.250 μsec ± 0.2%**</span> |
| 1     | 1000         | 24.99 μsec ± 0.2%  | <span style="color:#00aa00">**1.896 μsec ± 0.0%**</span> |
| 10    | 1000         | 45.74 μsec ± 0.2%  | <span style="color:#00aa00">**6.200 μsec ± 0.1%**</span> |
| 100   | 1000         | 76.32 μsec ± 0.2%  | <span style="color:#00aa00">**16.16 μsec ± 0.2%**</span> |
| 1000  | 1000         | 427.2 μsec ± 0.3%  | <span style="color:#00aa00">**3.711 μsec ± 0.0%**</span> |
| 1     | 10000        | 92.99 μsec ± 0.1%  | <span style="color:#00aa00">**9.519 μsec ± 0.0%**</span> |
| 10    | 10000        | 119.0 μsec ± 0.1%  | <span style="color:#00aa00">**40.12 μsec ± 0.0%**</span> |
| 100   | 10000        | 158.1 μsec ± 0.4%  | <span style="color:#00aa00">**62.24 μsec ± 0.0%**</span> |
| 1000  | 10000        | 348.3 μsec ± 0.3%  | <span style="color:#00aa00">**126.2 μsec ± 0.5%**</span> |
| 10000 | 10000        | 3.813 msec ± 0.2%  | <span style="color:#00aa00">**26.00 μsec ± 0.0%**</span> |
|       | **Geomean:** | 99.35 μsec ± 0.2%  | <span style="color:#00aa00">**6.324 μsec ± 0.1%**</span> |