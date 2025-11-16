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
                                                                                                                                                                                                                                 
| `k`   | `n`          | `sample_int_numpy`                                       | `sample_int_numba`                                       |
| ----- | ------------ | -------------------------------------------------------- | -------------------------------------------------------- |
| 1     | 10           | 10.91 μsec ± 4.1%                                        | <span style="color:#00aa00">**946.7 nsec ± 0.4%**</span> |
| 10    | 10           | 9.067 μsec ± 0.1%                                        | <span style="color:#00aa00">**1.096 μsec ± 0.2%**</span> |
| 100   | 10           | 10.12 μsec ± 0.1%                                        | <span style="color:#00aa00">**2.457 μsec ± 0.2%**</span> |
| 1000  | 10           | 20.99 μsec ± 0.1%                                        | <span style="color:#00aa00">**16.05 μsec ± 0.4%**</span> |
| 10000 | 10           | <span style="color:#00aa00">**123.0 μsec ± 0.0%**</span> | 151.1 μsec ± 0.1%                                        |
| 1     | 100          | 8.923 μsec ± 0.3%                                        | <span style="color:#00aa00">**938.3 nsec ± 0.1%**</span> |
| 10    | 100          | 8.987 μsec ± 0.1%                                        | <span style="color:#00aa00">**1.050 μsec ± 0.1%**</span> |
| 100   | 100          | 9.651 μsec ± 0.3%                                        | <span style="color:#00aa00">**2.043 μsec ± 0.1%**</span> |
| 1000  | 100          | 16.30 μsec ± 0.1%                                        | <span style="color:#00aa00">**11.80 μsec ± 0.1%**</span> |
| 10000 | 100          | <span style="color:#00aa00">**78.90 μsec ± 0.0%**</span> | 108.7 μsec ± 0.0%                                        |
| 1     | 1000         | 8.900 μsec ± 0.1%                                        | <span style="color:#00aa00">**949.7 nsec ± 0.2%**</span> |
| 10    | 1000         | 8.951 μsec ± 0.1%                                        | <span style="color:#00aa00">**1.011 μsec ± 0.0%**</span> |
| 100   | 1000         | 9.325 μsec ± 0.2%                                        | <span style="color:#00aa00">**1.717 μsec ± 0.1%**</span> |
| 1000  | 1000         | 13.18 μsec ± 0.3%                                        | <span style="color:#00aa00">**8.565 μsec ± 0.0%**</span> |
| 10000 | 1000         | <span style="color:#00aa00">**47.24 μsec ± 0.1%**</span> | 76.15 μsec ± 0.1%                                        |
| 1     | 10000        | 8.955 μsec ± 0.2%                                        | <span style="color:#00aa00">**957.5 nsec ± 0.3%**</span> |
| 10    | 10000        | 9.053 μsec ± 0.1%                                        | <span style="color:#00aa00">**1.105 μsec ± 0.2%**</span> |
| 100   | 10000        | 10.10 μsec ± 0.1%                                        | <span style="color:#00aa00">**2.518 μsec ± 0.0%**</span> |
| 1000  | 10000        | 21.22 μsec ± 0.2%                                        | <span style="color:#00aa00">**16.55 μsec ± 0.0%**</span> |
| 10000 | 10000        | <span style="color:#00aa00">**128.2 μsec ± 0.1%**</span> | 156.3 μsec ± 0.1%                                        |
|       | **Geomean:** | 16.64 μsec ± 0.4%                                        | <span style="color:#00aa00">**5.054 μsec ± 0.1%**</span> |

## B. WITHOUT replacement, UNIFORM probabilities
                                                                                                                                                                                                                                 
| `k`   | `n`          | `sample_int_numpy`                                       | `sample_int_numba`                                       |
| ----- | ------------ | -------------------------------------------------------- | -------------------------------------------------------- |
| 1     | 10           | 8.821 μsec ± 0.1%                                        | <span style="color:#00aa00">**951.6 nsec ± 0.2%**</span> |
| 10    | 10           | 6.854 μsec ± 0.1%                                        | <span style="color:#00aa00">**1.085 μsec ± 0.4%**</span> |
| 1     | 100          | 8.924 μsec ± 0.4%                                        | <span style="color:#00aa00">**959.3 nsec ± 0.6%**</span> |
| 10    | 100          | 8.022 μsec ± 0.2%                                        | <span style="color:#00aa00">**1.088 μsec ± 0.6%**</span> |
| 100   | 100          | 7.975 μsec ± 0.2%                                        | <span style="color:#00aa00">**2.312 μsec ± 0.2%**</span> |
| 1     | 1000         | 8.884 μsec ± 0.2%                                        | <span style="color:#00aa00">**948.3 nsec ± 0.5%**</span> |
| 10    | 1000         | 17.70 μsec ± 0.1%                                        | <span style="color:#00aa00">**1.126 μsec ± 0.1%**</span> |
| 100   | 1000         | 17.68 μsec ± 0.1%                                        | <span style="color:#00aa00">**2.008 μsec ± 0.1%**</span> |
| 1000  | 1000         | 18.26 μsec ± 0.1%                                        | <span style="color:#00aa00">**13.85 μsec ± 0.0%**</span> |
| 1     | 10000        | 8.920 μsec ± 0.2%                                        | <span style="color:#00aa00">**959.3 nsec ± 0.2%**</span> |
| 10    | 10000        | 123.4 μsec ± 0.1%                                        | <span style="color:#00aa00">**1.640 μsec ± 0.1%**</span> |
| 100   | 10000        | 123.4 μsec ± 0.0%                                        | <span style="color:#00aa00">**3.099 μsec ± 0.1%**</span> |
| 1000  | 10000        | 124.3 μsec ± 0.0%                                        | <span style="color:#00aa00">**18.49 μsec ± 0.0%**</span> |
| 10000 | 10000        | <span style="color:#00aa00">**125.2 μsec ± 0.0%**</span> | 137.5 μsec ± 0.0%                                        |
|       | **Geomean:** | 21.20 μsec ± 0.1%                                        | <span style="color:#00aa00">**2.666 μsec ± 0.2%**</span> |

## C. WITH replacement, CUSTOM probabilities
                                                                                                                                                                                                                                 
| `k`   | `n`          | `sample_int_numpy`                                       | `sample_int_numba`                                       |
| ----- | ------------ | -------------------------------------------------------- | -------------------------------------------------------- |
| 1     | 10           | 17.35 μsec ± 0.7%                                        | <span style="color:#00aa00">**1.050 μsec ± 0.7%**</span> |
| 10    | 10           | 17.81 μsec ± 0.6%                                        | <span style="color:#00aa00">**1.352 μsec ± 0.3%**</span> |
| 100   | 10           | 19.43 μsec ± 0.4%                                        | <span style="color:#00aa00">**3.936 μsec ± 0.1%**</span> |
| 1000  | 10           | 40.74 μsec ± 0.2%                                        | <span style="color:#00aa00">**31.73 μsec ± 0.0%**</span> |
| 10000 | 10           | <span style="color:#00aa00">**233.5 μsec ± 0.0%**</span> | 304.0 μsec ± 0.0%                                        |
| 1     | 100          | 17.87 μsec ± 21.1%                                       | <span style="color:#00aa00">**1.122 μsec ± 0.2%**</span> |
| 10    | 100          | 18.42 μsec ± 0.4%                                        | <span style="color:#00aa00">**1.582 μsec ± 0.7%**</span> |
| 100   | 100          | 22.24 μsec ± 0.3%                                        | <span style="color:#00aa00">**6.119 μsec ± 0.2%**</span> |
| 1000  | 100          | 63.75 μsec ± 0.2%                                        | <span style="color:#00aa00">**51.84 μsec ± 0.0%**</span> |
| 10000 | 100          | <span style="color:#00aa00">**473.5 μsec ± 0.0%**</span> | 505.6 μsec ± 0.0%                                        |
| 1     | 1000         | 25.97 μsec ± 0.8%                                        | <span style="color:#00aa00">**1.963 μsec ± 0.4%**</span> |
| 10    | 1000         | 26.88 μsec ± 0.4%                                        | <span style="color:#00aa00">**2.635 μsec ± 0.5%**</span> |
| 100   | 1000         | 32.16 μsec ± 0.6%                                        | <span style="color:#00aa00">**9.171 μsec ± 0.5%**</span> |
| 1000  | 1000         | 94.18 μsec ± 0.1%                                        | <span style="color:#00aa00">**74.10 μsec ± 0.1%**</span> |
| 10000 | 1000         | <span style="color:#00aa00">**703.7 μsec ± 0.0%**</span> | 732.2 μsec ± 0.0%                                        |
| 1     | 10000        | 94.22 μsec ± 0.2%                                        | <span style="color:#00aa00">**9.525 μsec ± 0.0%**</span> |
| 10    | 10000        | 95.45 μsec ± 0.1%                                        | <span style="color:#00aa00">**10.43 μsec ± 0.0%**</span> |
| 100   | 10000        | 104.1 μsec ± 0.1%                                        | <span style="color:#00aa00">**19.06 μsec ± 0.0%**</span> |
| 1000  | 10000        | 193.3 μsec ± 0.1%                                        | <span style="color:#00aa00">**105.0 μsec ± 0.0%**</span> |
| 10000 | 10000        | 1.078 msec ± 0.1%                                        | <span style="color:#00aa00">**959.7 μsec ± 0.0%**</span> |
|       | **Geomean:** | 67.33 μsec ± 1.2%                                        | <span style="color:#00aa00">**17.53 μsec ± 0.2%**</span> |

## D. WITHOUT replacement, CUSTOM probabilities
                                                                                                                                                                                                                                 
| `k`   | `n`          | `sample_int_numpy`                                       | `sample_int_numba`                                       |
| ----- | ------------ | -------------------------------------------------------- | -------------------------------------------------------- |
| 1     | 10           | 17.91 μsec ± 1.5%                                        | <span style="color:#00aa00">**1.063 μsec ± 1.0%**</span> |
| 10    | 10           | 68.87 μsec ± 0.3%                                        | <span style="color:#00aa00">**1.073 μsec ± 0.1%**</span> |
| 1     | 100          | 17.47 μsec ± 0.3%                                        | <span style="color:#00aa00">**1.105 μsec ± 0.1%**</span> |
| 10    | 100          | 45.57 μsec ± 0.2%                                        | <span style="color:#00aa00">**4.590 μsec ± 0.0%**</span> |
| 100   | 100          | 134.8 μsec ± 0.3%                                        | <span style="color:#00aa00">**2.288 μsec ± 0.1%**</span> |
| 1     | 1000         | 25.35 μsec ± 0.3%                                        | <span style="color:#00aa00">**1.953 μsec ± 0.5%**</span> |
| 10    | 1000         | 48.54 μsec ± 0.1%                                        | <span style="color:#00aa00">**28.23 μsec ± 0.1%**</span> |
| 100   | 1000         | 78.33 μsec ± 0.2%                                        | <span style="color:#00aa00">**30.24 μsec ± 0.1%**</span> |
| 1000  | 1000         | 432.3 μsec ± 0.1%                                        | <span style="color:#00aa00">**14.40 μsec ± 0.1%**</span> |
| 1     | 10000        | 93.94 μsec ± 0.2%                                        | <span style="color:#00aa00">**9.529 μsec ± 0.1%**</span> |
| 10    | 10000        | <span style="color:#00aa00">**119.9 μsec ± 0.1%**</span> | 255.5 μsec ± 0.4%                                        |
| 100   | 10000        | <span style="color:#00aa00">**159.1 μsec ± 0.5%**</span> | 261.4 μsec ± 0.4%                                        |
| 1000  | 10000        | 349.7 μsec ± 0.3%                                        | <span style="color:#00aa00">**266.2 μsec ± 0.3%**</span> |
| 10000 | 10000        | 3.892 msec ± 0.2%                                        | <span style="color:#00aa00">**144.4 μsec ± 0.0%**</span> |
|       | **Geomean:** | 102.1 μsec ± 0.3%                                        | <span style="color:#00aa00">**13.65 μsec ± 0.2%**</span> |