# `sample_int`

([API reference][max_div.sampling.discrete.sample_int])

Command:
```bash
uv tool install max-div
max-div benchmark --markdown sample-int
```
or 
```bash
uv run max-div benchmark --markdown sample-int
```

## A. WITH replacement, UNIFORM probabilities
                                                                                                                                                                                              
| `k`   | `n`          | `sample_int_numpy` | `sample_int_numba`                                       |
| ----- | ------------ | ------------------ | -------------------------------------------------------- |
| 1     | 10           | 8.727 μsec ± 10.0% | <span style="color:#00aa00">**1.014 μsec ± 1.1%**</span> |
| 10    | 10           | 10.85 μsec ± 0.4%  | <span style="color:#00aa00">**1.014 μsec ± 0.1%**</span> |
| 100   | 10           | 12.18 μsec ± 0.9%  | <span style="color:#00aa00">**1.151 μsec ± 0.2%**</span> |
| 1000  | 10           | 22.39 μsec ± 0.8%  | <span style="color:#00aa00">**2.784 μsec ± 0.1%**</span> |
| 10000 | 10           | 124.2 μsec ± 0.1%  | <span style="color:#00aa00">**17.86 μsec ± 0.0%**</span> |
| 1     | 100          | 10.79 μsec ± 1.3%  | <span style="color:#00aa00">**1.002 μsec ± 0.1%**</span> |
| 10    | 100          | 10.47 μsec ± 0.8%  | <span style="color:#00aa00">**1.005 μsec ± 1.1%**</span> |
| 100   | 100          | 11.03 μsec ± 0.9%  | <span style="color:#00aa00">**1.147 μsec ± 1.3%**</span> |
| 1000  | 100          | 17.61 μsec ± 0.4%  | <span style="color:#00aa00">**2.778 μsec ± 0.4%**</span> |
| 10000 | 100          | 80.83 μsec ± 0.3%  | <span style="color:#00aa00">**17.86 μsec ± 0.1%**</span> |
| 1     | 1000         | 10.34 μsec ± 1.2%  | <span style="color:#00aa00">**1.009 μsec ± 1.3%**</span> |
| 10    | 1000         | 10.42 μsec ± 2.6%  | <span style="color:#00aa00">**1.008 μsec ± 1.3%**</span> |
| 100   | 1000         | 11.19 μsec ± 2.1%  | <span style="color:#00aa00">**1.183 μsec ± 0.6%**</span> |
| 1000  | 1000         | 13.83 μsec ± 1.2%  | <span style="color:#00aa00">**2.798 μsec ± 0.1%**</span> |
| 10000 | 1000         | 48.01 μsec ± 0.1%  | <span style="color:#00aa00">**17.87 μsec ± 0.0%**</span> |
| 1     | 10000        | 10.13 μsec ± 0.5%  | <span style="color:#00aa00">**1.020 μsec ± 0.7%**</span> |
| 10    | 10000        | 10.88 μsec ± 3.3%  | <span style="color:#00aa00">**1.023 μsec ± 0.9%**</span> |
| 100   | 10000        | 10.10 μsec ± 0.2%  | <span style="color:#00aa00">**1.176 μsec ± 0.5%**</span> |
| 1000  | 10000        | 20.97 μsec ± 0.1%  | <span style="color:#00aa00">**2.796 μsec ± 0.2%**</span> |
| 10000 | 10000        | 126.6 μsec ± 0.1%  | <span style="color:#00aa00">**17.87 μsec ± 0.0%**</span> |
|       | **Geomean:** | 18.05 μsec ± 1.3%  | <span style="color:#00aa00">**2.263 μsec ± 0.5%**</span> |

## B. WITHOUT replacement, UNIFORM probabilities
                                                                                                                                                                                              
| `k`   | `n`          | `sample_int_numpy` | `sample_int_numba`                                       |
| ----- | ------------ | ------------------ | -------------------------------------------------------- |
| 1     | 10           | 8.747 μsec ± 0.6%  | <span style="color:#00aa00">**1.003 μsec ± 0.1%**</span> |
| 10    | 10           | 6.832 μsec ± 0.4%  | <span style="color:#00aa00">**1.044 μsec ± 0.2%**</span> |
| 1     | 100          | 8.789 μsec ± 0.1%  | <span style="color:#00aa00">**1.010 μsec ± 0.2%**</span> |
| 10    | 100          | 7.889 μsec ± 0.3%  | <span style="color:#00aa00">**1.058 μsec ± 0.1%**</span> |
| 100   | 100          | 7.844 μsec ± 0.2%  | <span style="color:#00aa00">**1.300 μsec ± 0.2%**</span> |
| 1     | 1000         | 8.885 μsec ± 0.2%  | <span style="color:#00aa00">**1.028 μsec ± 0.8%**</span> |
| 10    | 1000         | 17.53 μsec ± 0.1%  | <span style="color:#00aa00">**1.208 μsec ± 0.6%**</span> |
| 100   | 1000         | 17.51 μsec ± 0.2%  | <span style="color:#00aa00">**1.437 μsec ± 0.7%**</span> |
| 1000  | 1000         | 17.89 μsec ± 0.1%  | <span style="color:#00aa00">**3.803 μsec ± 0.1%**</span> |
| 1     | 10000        | 8.902 μsec ± 0.2%  | <span style="color:#00aa00">**1.024 μsec ± 0.8%**</span> |
| 10    | 10000        | 121.5 μsec ± 0.1%  | <span style="color:#00aa00">**1.573 μsec ± 0.3%**</span> |
| 100   | 10000        | 121.5 μsec ± 0.1%  | <span style="color:#00aa00">**1.769 μsec ± 0.2%**</span> |
| 1000  | 10000        | 122.0 μsec ± 0.1%  | <span style="color:#00aa00">**3.912 μsec ± 0.3%**</span> |
| 10000 | 10000        | 125.4 μsec ± 0.1%  | <span style="color:#00aa00">**25.78 μsec ± 0.1%**</span> |
|       | **Geomean:** | 20.97 μsec ± 0.2%  | <span style="color:#00aa00">**1.765 μsec ± 0.3%**</span> |

## C. WITH replacement, CUSTOM probabilities
                                                                                                                                                                                              
| `k`   | `n`          | `sample_int_numpy` | `sample_int_numba`                                       |
| ----- | ------------ | ------------------ | -------------------------------------------------------- |
| 1     | 10           | 17.69 μsec ± 0.3%  | <span style="color:#00aa00">**1.064 μsec ± 0.4%**</span> |
| 10    | 10           | 17.76 μsec ± 0.2%  | <span style="color:#00aa00">**1.240 μsec ± 0.8%**</span> |
| 100   | 10           | 20.14 μsec ± 0.2%  | <span style="color:#00aa00">**2.820 μsec ± 0.4%**</span> |
| 1000  | 10           | 40.94 μsec ± 0.1%  | <span style="color:#00aa00">**16.25 μsec ± 0.0%**</span> |
| 10000 | 10           | 235.1 μsec ± 0.0%  | <span style="color:#00aa00">**176.5 μsec ± 0.0%**</span> |
| 1     | 100          | 18.26 μsec ± 0.2%  | <span style="color:#00aa00">**1.137 μsec ± 0.7%**</span> |
| 10    | 100          | 18.69 μsec ± 0.5%  | <span style="color:#00aa00">**1.456 μsec ± 0.2%**</span> |
| 100   | 100          | 22.97 μsec ± 0.1%  | <span style="color:#00aa00">**4.723 μsec ± 0.1%**</span> |
| 1000  | 100          | 65.09 μsec ± 0.1%  | <span style="color:#00aa00">**37.39 μsec ± 0.0%**</span> |
| 10000 | 100          | 475.1 μsec ± 0.0%  | <span style="color:#00aa00">**364.1 μsec ± 0.0%**</span> |
| 1     | 1000         | 26.09 μsec ± 0.2%  | <span style="color:#00aa00">**2.028 μsec ± 0.2%**</span> |
| 10    | 1000         | 26.77 μsec ± 0.2%  | <span style="color:#00aa00">**2.567 μsec ± 0.4%**</span> |
| 100   | 1000         | 33.13 μsec ± 0.4%  | <span style="color:#00aa00">**7.931 μsec ± 0.2%**</span> |
| 1000  | 1000         | 95.12 μsec ± 0.1%  | <span style="color:#00aa00">**61.43 μsec ± 0.0%**</span> |
| 10000 | 1000         | 705.4 μsec ± 0.0%  | <span style="color:#00aa00">**596.0 μsec ± 0.0%**</span> |
| 1     | 10000        | 95.94 μsec ± 0.3%  | <span style="color:#00aa00">**9.592 μsec ± 0.1%**</span> |
| 10    | 10000        | 97.01 μsec ± 0.2%  | <span style="color:#00aa00">**10.35 μsec ± 0.1%**</span> |
| 100   | 10000        | 106.5 μsec ± 0.1%  | <span style="color:#00aa00">**17.58 μsec ± 0.1%**</span> |
| 1000  | 10000        | 195.9 μsec ± 0.1%  | <span style="color:#00aa00">**89.26 μsec ± 0.0%**</span> |
| 10000 | 10000        | 1.080 msec ± 0.0%  | <span style="color:#00aa00">**804.2 μsec ± 0.0%**</span> |
|       | **Geomean:** | 68.25 μsec ± 0.2%  | <span style="color:#00aa00">**14.68 μsec ± 0.2%**</span> |

## D. WITHOUT replacement, CUSTOM probabilities
                                                                                                                                                                                              
| `k`   | `n`          | `sample_int_numpy`                                       | `sample_int_numba`                                       |
| ----- | ------------ | -------------------------------------------------------- | -------------------------------------------------------- |
| 1     | 10           | 17.69 μsec ± 0.3%                                        | <span style="color:#00aa00">**1.064 μsec ± 0.2%**</span> |
| 10    | 10           | 72.87 μsec ± 0.4%                                        | <span style="color:#00aa00">**1.043 μsec ± 0.3%**</span> |
| 1     | 100          | 18.44 μsec ± 0.7%                                        | <span style="color:#00aa00">**1.135 μsec ± 0.1%**</span> |
| 10    | 100          | 46.33 μsec ± 0.3%                                        | <span style="color:#00aa00">**3.073 μsec ± 0.1%**</span> |
| 100   | 100          | 131.4 μsec ± 0.5%                                        | <span style="color:#00aa00">**1.300 μsec ± 0.1%**</span> |
| 1     | 1000         | 26.16 μsec ± 0.3%                                        | <span style="color:#00aa00">**2.020 μsec ± 0.2%**</span> |
| 10    | 1000         | 49.54 μsec ± 0.2%                                        | <span style="color:#00aa00">**14.38 μsec ± 0.1%**</span> |
| 100   | 1000         | 79.23 μsec ± 0.2%                                        | <span style="color:#00aa00">**15.54 μsec ± 0.1%**</span> |
| 1000  | 1000         | 436.0 μsec ± 0.2%                                        | <span style="color:#00aa00">**3.742 μsec ± 0.2%**</span> |
| 1     | 10000        | 96.30 μsec ± 0.2%                                        | <span style="color:#00aa00">**9.584 μsec ± 0.1%**</span> |
| 10    | 10000        | <span style="color:#00aa00">**122.3 μsec ± 0.2%**</span> | 129.3 μsec ± 0.4%                                        |
| 100   | 10000        | 161.4 μsec ± 0.6%                                        | <span style="color:#00aa00">**126.8 μsec ± 0.5%**</span> |
| 1000  | 10000        | 352.5 μsec ± 0.2%                                        | <span style="color:#00aa00">**131.6 μsec ± 0.4%**</span> |
| 10000 | 10000        | 3.921 msec ± 0.2%                                        | <span style="color:#00aa00">**26.06 μsec ± 0.0%**</span> |
|       | **Geomean:** | 103.9 μsec ± 0.3%                                        | <span style="color:#00aa00">**8.022 μsec ± 0.2%**</span> |