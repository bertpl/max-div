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
| 1     | 10           | 8.808 μsec ± 0.3%  | <span style="color:#00aa00">**1.002 μsec ± 1.0%**</span> |
| 10    | 10           | 8.953 μsec ± 0.2%  | <span style="color:#00aa00">**1.025 μsec ± 0.2%**</span> |
| 100   | 10           | 10.11 μsec ± 0.2%  | <span style="color:#00aa00">**1.237 μsec ± 0.2%**</span> |
| 1000  | 10           | 20.61 μsec ± 0.1%  | <span style="color:#00aa00">**3.559 μsec ± 0.0%**</span> |
| 10000 | 10           | 123.2 μsec ± 0.0%  | <span style="color:#00aa00">**26.15 μsec ± 0.0%**</span> |
| 1     | 100          | 8.849 μsec ± 0.2%  | <span style="color:#00aa00">**1.024 μsec ± 1.0%**</span> |
| 10    | 100          | 8.853 μsec ± 0.3%  | <span style="color:#00aa00">**1.042 μsec ± 1.0%**</span> |
| 100   | 100          | 9.643 μsec ± 0.2%  | <span style="color:#00aa00">**1.248 μsec ± 0.8%**</span> |
| 1000  | 100          | 16.09 μsec ± 0.1%  | <span style="color:#00aa00">**3.540 μsec ± 0.2%**</span> |
| 10000 | 100          | 78.74 μsec ± 1.0%  | <span style="color:#00aa00">**25.92 μsec ± 0.0%**</span> |
| 1     | 1000         | 8.913 μsec ± 0.2%  | <span style="color:#00aa00">**1.013 μsec ± 1.0%**</span> |
| 10    | 1000         | 8.940 μsec ± 0.2%  | <span style="color:#00aa00">**1.038 μsec ± 0.9%**</span> |
| 100   | 1000         | 9.329 μsec ± 0.2%  | <span style="color:#00aa00">**1.249 μsec ± 0.7%**</span> |
| 1000  | 1000         | 12.99 μsec ± 0.1%  | <span style="color:#00aa00">**3.318 μsec ± 0.3%**</span> |
| 10000 | 1000         | 45.92 μsec ± 0.1%  | <span style="color:#00aa00">**23.47 μsec ± 0.0%**</span> |
| 1     | 10000        | 8.914 μsec ± 0.2%  | <span style="color:#00aa00">**1.004 μsec ± 1.0%**</span> |
| 10    | 10000        | 8.982 μsec ± 0.2%  | <span style="color:#00aa00">**1.027 μsec ± 1.0%**</span> |
| 100   | 10000        | 10.21 μsec ± 0.2%  | <span style="color:#00aa00">**1.230 μsec ± 0.7%**</span> |
| 1000  | 10000        | 21.14 μsec ± 0.1%  | <span style="color:#00aa00">**3.296 μsec ± 0.3%**</span> |
| 10000 | 10000        | 127.0 μsec ± 0.1%  | <span style="color:#00aa00">**23.36 μsec ± 0.0%**</span> |
|       | **Geomean:** | 16.36 μsec ± 0.2%  | <span style="color:#00aa00">**2.558 μsec ± 0.5%**</span> |

## B. WITHOUT replacement, UNIFORM probabilities
                                                                                                                                                                                                                                 
| `k`   | `n`          | `sample_int_numpy` | `sample_int_numba`                                       |
| ----- | ------------ | ------------------ | -------------------------------------------------------- |
| 1     | 10           | 8.847 μsec ± 0.2%  | <span style="color:#00aa00">**1.005 μsec ± 0.2%**</span> |
| 10    | 10           | 7.200 μsec ± 0.2%  | <span style="color:#00aa00">**1.039 μsec ± 0.1%**</span> |
| 1     | 100          | 8.949 μsec ± 0.3%  | <span style="color:#00aa00">**1.008 μsec ± 0.1%**</span> |
| 10    | 100          | 8.324 μsec ± 0.4%  | <span style="color:#00aa00">**1.041 μsec ± 0.1%**</span> |
| 100   | 100          | 8.386 μsec ± 0.2%  | <span style="color:#00aa00">**1.292 μsec ± 0.1%**</span> |
| 1     | 1000         | 8.962 μsec ± 0.2%  | <span style="color:#00aa00">**1.010 μsec ± 0.3%**</span> |
| 10    | 1000         | 18.45 μsec ± 0.2%  | <span style="color:#00aa00">**1.115 μsec ± 0.1%**</span> |
| 100   | 1000         | 18.41 μsec ± 0.1%  | <span style="color:#00aa00">**1.342 μsec ± 0.1%**</span> |
| 1000  | 1000         | 18.83 μsec ± 0.1%  | <span style="color:#00aa00">**3.712 μsec ± 0.0%**</span> |
| 1     | 10000        | 8.964 μsec ± 0.2%  | <span style="color:#00aa00">**1.007 μsec ± 0.1%**</span> |
| 10    | 10000        | 124.2 μsec ± 0.0%  | <span style="color:#00aa00">**1.515 μsec ± 0.1%**</span> |
| 100   | 10000        | 124.2 μsec ± 0.0%  | <span style="color:#00aa00">**1.749 μsec ± 0.1%**</span> |
| 1000  | 10000        | 124.7 μsec ± 0.0%  | <span style="color:#00aa00">**3.908 μsec ± 0.0%**</span> |
| 10000 | 10000        | 126.2 μsec ± 0.0%  | <span style="color:#00aa00">**25.68 μsec ± 0.0%**</span> |
|       | **Geomean:** | 21.65 μsec ± 0.2%  | <span style="color:#00aa00">**1.729 μsec ± 0.1%**</span> |

## C. WITH replacement, CUSTOM probabilities
                                                                                                                                                                                                                                 
| `k`   | `n`          | `sample_int_numpy` | `sample_int_numba`                                       |
| ----- | ------------ | ------------------ | -------------------------------------------------------- |
| 1     | 10           | 16.79 μsec ± 0.2%  | <span style="color:#00aa00">**1.069 μsec ± 0.7%**</span> |
| 10    | 10           | 17.11 μsec ± 0.2%  | <span style="color:#00aa00">**1.233 μsec ± 0.3%**</span> |
| 100   | 10           | 19.16 μsec ± 0.2%  | <span style="color:#00aa00">**2.975 μsec ± 0.1%**</span> |
| 1000  | 10           | 40.87 μsec ± 0.3%  | <span style="color:#00aa00">**18.46 μsec ± 0.0%**</span> |
| 10000 | 10           | 247.8 μsec ± 0.1%  | <span style="color:#00aa00">**176.4 μsec ± 0.0%**</span> |
| 1     | 100          | 17.46 μsec ± 0.3%  | <span style="color:#00aa00">**1.145 μsec ± 0.6%**</span> |
| 10    | 100          | 18.07 μsec ± 0.2%  | <span style="color:#00aa00">**1.480 μsec ± 0.6%**</span> |
| 100   | 100          | 22.27 μsec ± 0.2%  | <span style="color:#00aa00">**4.719 μsec ± 0.3%**</span> |
| 1000  | 100          | 63.60 μsec ± 0.1%  | <span style="color:#00aa00">**36.98 μsec ± 0.0%**</span> |
| 10000 | 100          | 475.0 μsec ± 0.0%  | <span style="color:#00aa00">**358.1 μsec ± 0.0%**</span> |
| 1     | 1000         | 25.45 μsec ± 0.2%  | <span style="color:#00aa00">**1.965 μsec ± 0.4%**</span> |
| 10    | 1000         | 26.09 μsec ± 0.1%  | <span style="color:#00aa00">**2.506 μsec ± 0.4%**</span> |
| 100   | 1000         | 32.49 μsec ± 0.2%  | <span style="color:#00aa00">**7.832 μsec ± 0.2%**</span> |
| 1000  | 1000         | 94.02 μsec ± 0.1%  | <span style="color:#00aa00">**61.08 μsec ± 0.0%**</span> |
| 10000 | 1000         | 704.4 μsec ± 0.0%  | <span style="color:#00aa00">**591.8 μsec ± 0.0%**</span> |
| 1     | 10000        | 93.87 μsec ± 0.1%  | <span style="color:#00aa00">**9.568 μsec ± 0.1%**</span> |
| 10    | 10000        | 94.62 μsec ± 0.2%  | <span style="color:#00aa00">**10.33 μsec ± 0.1%**</span> |
| 100   | 10000        | 104.0 μsec ± 0.1%  | <span style="color:#00aa00">**17.55 μsec ± 0.1%**</span> |
| 1000  | 10000        | 193.9 μsec ± 0.0%  | <span style="color:#00aa00">**88.95 μsec ± 0.0%**</span> |
| 10000 | 10000        | 1.080 msec ± 0.0%  | <span style="color:#00aa00">**802.0 μsec ± 0.0%**</span> |
|       | **Geomean:** | 66.95 μsec ± 0.1%  | <span style="color:#00aa00">**14.74 μsec ± 0.2%**</span> |

## D. WITHOUT replacement, CUSTOM probabilities
                                                                                                                                                                                                                                 
| `k`   | `n`          | `sample_int_numpy` | `sample_int_numba`                                       |
| ----- | ------------ | ------------------ | -------------------------------------------------------- |
| 1     | 10           | 24.81 μsec ± 4.5%  | <span style="color:#00aa00">**1.053 μsec ± 0.2%**</span> |
| 10    | 10           | 74.68 μsec ± 0.6%  | <span style="color:#00aa00">**1.040 μsec ± 0.1%**</span> |
| 1     | 100          | 17.85 μsec ± 0.3%  | <span style="color:#00aa00">**1.124 μsec ± 0.2%**</span> |
| 10    | 100          | 48.55 μsec ± 0.5%  | <span style="color:#00aa00">**3.120 μsec ± 0.2%**</span> |
| 100   | 100          | 137.0 μsec ± 0.5%  | <span style="color:#00aa00">**1.273 μsec ± 0.4%**</span> |
| 1     | 1000         | 27.02 μsec ± 0.5%  | <span style="color:#00aa00">**1.945 μsec ± 0.1%**</span> |
| 10    | 1000         | 50.58 μsec ± 0.3%  | <span style="color:#00aa00">**15.13 μsec ± 0.1%**</span> |
| 100   | 1000         | 79.43 μsec ± 0.1%  | <span style="color:#00aa00">**15.01 μsec ± 0.2%**</span> |
| 1000  | 1000         | 419.9 μsec ± 1.3%  | <span style="color:#00aa00">**3.722 μsec ± 0.1%**</span> |
| 1     | 10000        | 94.06 μsec ± 0.1%  | <span style="color:#00aa00">**9.552 μsec ± 0.0%**</span> |
| 10    | 10000        | 119.8 μsec ± 0.1%  | <span style="color:#00aa00">**119.5 μsec ± 0.7%**</span> |
| 100   | 10000        | 159.3 μsec ± 0.5%  | <span style="color:#00aa00">**133.1 μsec ± 0.4%**</span> |
| 1000  | 10000        | 349.2 μsec ± 0.2%  | <span style="color:#00aa00">**141.9 μsec ± 0.6%**</span> |
| 10000 | 10000        | 3.818 msec ± 0.2%  | <span style="color:#00aa00">**26.05 μsec ± 0.0%**</span> |
|       | **Geomean:** | 106.4 μsec ± 0.7%  | <span style="color:#00aa00">**8.016 μsec ± 0.2%**</span> |