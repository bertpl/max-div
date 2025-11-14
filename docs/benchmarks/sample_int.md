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
                                                                                                                                                                                              
| `k`   | `n`   | `sample_int_numpy`                                       | `sample_int_numba`                                       |
| ----- | ----- | -------------------------------------------------------- | -------------------------------------------------------- |
| 1     | 10    | 8.122 μsec ± 0.1%                                        | <span style="color:#00aa00">**922.5 nsec ± 0.3%**</span> |
| 10    | 10    | 8.323 μsec ± 0.2%                                        | <span style="color:#00aa00">**1.073 μsec ± 0.1%**</span> |
| 100   | 10    | 9.425 μsec ± 0.1%                                        | <span style="color:#00aa00">**2.447 μsec ± 0.1%**</span> |
| 1000  | 10    | 19.66 μsec ± 0.1%                                        | <span style="color:#00aa00">**16.09 μsec ± 0.2%**</span> |
| 10000 | 10    | <span style="color:#00aa00">**119.4 μsec ± 0.1%**</span> | 151.8 μsec ± 0.1%                                        |
| 1     | 100   | 8.196 μsec ± 0.2%                                        | <span style="color:#00aa00">**918.3 nsec ± 0.1%**</span> |
| 10    | 100   | 8.324 μsec ± 0.2%                                        | <span style="color:#00aa00">**1.030 μsec ± 0.1%**</span> |
| 100   | 100   | 8.897 μsec ± 0.2%                                        | <span style="color:#00aa00">**2.033 μsec ± 0.1%**</span> |
| 1000  | 100   | 15.18 μsec ± 0.2%                                        | <span style="color:#00aa00">**11.97 μsec ± 0.1%**</span> |
| 10000 | 100   | <span style="color:#00aa00">**75.14 μsec ± 0.0%**</span> | 110.4 μsec ± 0.1%                                        |
| 1     | 1000  | 8.238 μsec ± 0.2%                                        | <span style="color:#00aa00">**923.4 nsec ± 0.1%**</span> |
| 10    | 1000  | 8.360 μsec ± 0.2%                                        | <span style="color:#00aa00">**990.3 nsec ± 0.1%**</span> |
| 100   | 1000  | 8.701 μsec ± 0.3%                                        | <span style="color:#00aa00">**1.712 μsec ± 0.1%**</span> |
| 1000  | 1000  | 12.11 μsec ± 0.2%                                        | <span style="color:#00aa00">**8.824 μsec ± 0.0%**</span> |
| 10000 | 1000  | <span style="color:#00aa00">**43.56 μsec ± 0.0%**</span> | 78.87 μsec ± 0.0%                                        |
| 1     | 10000 | 8.286 μsec ± 0.3%                                        | <span style="color:#00aa00">**929.2 nsec ± 0.4%**</span> |
| 10    | 10000 | 8.429 μsec ± 0.2%                                        | <span style="color:#00aa00">**1.082 μsec ± 0.3%**</span> |
| 100   | 10000 | 9.547 μsec ± 0.2%                                        | <span style="color:#00aa00">**2.492 μsec ± 0.1%**</span> |
| 1000  | 10000 | 20.20 μsec ± 0.1%                                        | <span style="color:#00aa00">**16.57 μsec ± 0.1%**</span> |
| 10000 | 10000 | <span style="color:#00aa00">**124.7 μsec ± 0.0%**</span> | 156.7 μsec ± 0.2%                                        |

## B. WITHOUT replacement, UNIFORM probabilities
                                                                                                                                                                                              
| `k`   | `n`   | `sample_int_numpy`                                       | `sample_int_numba`                                       |
| ----- | ----- | -------------------------------------------------------- | -------------------------------------------------------- |
| 1     | 10    | 8.189 μsec ± 0.2%                                        | <span style="color:#00aa00">**933.1 nsec ± 0.1%**</span> |
| 10    | 10    | 6.176 μsec ± 0.2%                                        | <span style="color:#00aa00">**1.044 μsec ± 0.1%**</span> |
| 1     | 100   | 8.208 μsec ± 0.2%                                        | <span style="color:#00aa00">**925.4 nsec ± 0.1%**</span> |
| 10    | 100   | 7.269 μsec ± 0.3%                                        | <span style="color:#00aa00">**1.060 μsec ± 0.2%**</span> |
| 100   | 100   | 7.279 μsec ± 0.1%                                        | <span style="color:#00aa00">**2.251 μsec ± 0.1%**</span> |
| 1     | 1000  | 8.262 μsec ± 0.2%                                        | <span style="color:#00aa00">**924.7 nsec ± 0.2%**</span> |
| 10    | 1000  | 17.16 μsec ± 0.1%                                        | <span style="color:#00aa00">**1.136 μsec ± 0.1%**</span> |
| 100   | 1000  | 17.18 μsec ± 0.1%                                        | <span style="color:#00aa00">**2.007 μsec ± 0.1%**</span> |
| 1000  | 1000  | 17.15 μsec ± 0.5%                                        | <span style="color:#00aa00">**13.60 μsec ± 0.0%**</span> |
| 1     | 10000 | 8.299 μsec ± 0.3%                                        | <span style="color:#00aa00">**935.8 nsec ± 0.3%**</span> |
| 10    | 10000 | 124.6 μsec ± 0.0%                                        | <span style="color:#00aa00">**2.039 μsec ± 0.1%**</span> |
| 100   | 10000 | 124.6 μsec ± 0.1%                                        | <span style="color:#00aa00">**3.490 μsec ± 0.0%**</span> |
| 1000  | 10000 | 124.0 μsec ± 0.7%                                        | <span style="color:#00aa00">**18.82 μsec ± 0.0%**</span> |
| 10000 | 10000 | <span style="color:#00aa00">**124.6 μsec ± 0.8%**</span> | 136.6 μsec ± 0.0%                                        |

## C. WITH replacement, CUSTOM probabilities
                                                                                                                                                                                              
| `k`   | `n`   | `sample_int_numpy`                                       | `sample_int_numba`                                       |
| ----- | ----- | -------------------------------------------------------- | -------------------------------------------------------- |
| 1     | 10    | 14.91 μsec ± 0.8%                                        | <span style="color:#00aa00">**985.8 nsec ± 0.1%**</span> |
| 10    | 10    | 14.96 μsec ± 0.2%                                        | <span style="color:#00aa00">**1.275 μsec ± 0.6%**</span> |
| 100   | 10    | 17.29 μsec ± 0.3%                                        | <span style="color:#00aa00">**3.965 μsec ± 0.1%**</span> |
| 1000  | 10    | 37.93 μsec ± 0.2%                                        | <span style="color:#00aa00">**30.62 μsec ± 0.0%**</span> |
| 10000 | 10    | <span style="color:#00aa00">**242.6 μsec ± 0.1%**</span> | 315.0 μsec ± 0.0%                                        |
| 1     | 100   | 15.84 μsec ± 0.4%                                        | <span style="color:#00aa00">**1.057 μsec ± 0.1%**</span> |
| 10    | 100   | 16.28 μsec ± 0.5%                                        | <span style="color:#00aa00">**1.538 μsec ± 0.1%**</span> |
| 100   | 100   | 20.79 μsec ± 1.4%                                        | <span style="color:#00aa00">**6.232 μsec ± 0.2%**</span> |
| 1000  | 100   | 63.39 μsec ± 0.2%                                        | <span style="color:#00aa00">**53.16 μsec ± 0.1%**</span> |
| 10000 | 100   | <span style="color:#00aa00">**467.9 μsec ± 0.0%**</span> | 518.2 μsec ± 0.1%                                        |
| 1     | 1000  | 24.24 μsec ± 0.2%                                        | <span style="color:#00aa00">**1.905 μsec ± 0.1%**</span> |
| 10    | 1000  | 24.73 μsec ± 0.2%                                        | <span style="color:#00aa00">**2.603 μsec ± 0.1%**</span> |
| 100   | 1000  | 31.20 μsec ± 0.1%                                        | <span style="color:#00aa00">**9.448 μsec ± 0.0%**</span> |
| 1000  | 1000  | 92.48 μsec ± 0.1%                                        | <span style="color:#00aa00">**77.78 μsec ± 0.0%**</span> |
| 10000 | 1000  | <span style="color:#00aa00">**700.2 μsec ± 0.0%**</span> | 752.8 μsec ± 0.1%                                        |
| 1     | 10000 | 91.21 μsec ± 0.1%                                        | <span style="color:#00aa00">**9.523 μsec ± 0.0%**</span> |
| 10    | 10000 | 92.33 μsec ± 0.1%                                        | <span style="color:#00aa00">**10.43 μsec ± 0.0%**</span> |
| 100   | 10000 | 101.4 μsec ± 0.0%                                        | <span style="color:#00aa00">**19.27 μsec ± 0.0%**</span> |
| 1000  | 10000 | 190.4 μsec ± 0.0%                                        | <span style="color:#00aa00">**107.2 μsec ± 0.0%**</span> |
| 10000 | 10000 | 1.074 msec ± 0.0%                                        | <span style="color:#00aa00">**982.9 μsec ± 0.1%**</span> |

## D. WITHOUT replacement, CUSTOM probabilities
                                                                                                                                                                                              
| `k`   | `n`   | `sample_int_numpy`                                       | `sample_int_numba`                                       |
| ----- | ----- | -------------------------------------------------------- | -------------------------------------------------------- |
| 1     | 10    | 15.21 μsec ± 2.7%                                        | <span style="color:#00aa00">**985.1 nsec ± 0.2%**</span> |
| 10    | 10    | 82.62 μsec ± 0.2%                                        | <span style="color:#00aa00">**1.050 μsec ± 0.5%**</span> |
| 1     | 100   | 15.55 μsec ± 0.3%                                        | <span style="color:#00aa00">**1.069 μsec ± 0.5%**</span> |
| 10    | 100   | 46.26 μsec ± 0.4%                                        | <span style="color:#00aa00">**4.229 μsec ± 0.3%**</span> |
| 100   | 100   | 136.5 μsec ± 0.3%                                        | <span style="color:#00aa00">**2.322 μsec ± 0.3%**</span> |
| 1     | 1000  | 24.27 μsec ± 0.3%                                        | <span style="color:#00aa00">**1.905 μsec ± 0.1%**</span> |
| 10    | 1000  | 47.95 μsec ± 0.2%                                        | <span style="color:#00aa00">**26.14 μsec ± 0.2%**</span> |
| 100   | 1000  | 79.89 μsec ± 0.2%                                        | <span style="color:#00aa00">**25.97 μsec ± 0.1%**</span> |
| 1000  | 1000  | 460.3 μsec ± 0.2%                                        | <span style="color:#00aa00">**14.36 μsec ± 0.1%**</span> |
| 1     | 10000 | 91.22 μsec ± 0.0%                                        | <span style="color:#00aa00">**9.517 μsec ± 0.0%**</span> |
| 10    | 10000 | <span style="color:#00aa00">**118.5 μsec ± 0.1%**</span> | 236.8 μsec ± 0.3%                                        |
| 100   | 10000 | <span style="color:#00aa00">**157.3 μsec ± 0.4%**</span> | 226.6 μsec ± 1.0%                                        |
| 1000  | 10000 | 349.3 μsec ± 0.3%                                        | <span style="color:#00aa00">**251.4 μsec ± 0.5%**</span> |
| 10000 | 10000 | 3.857 msec ± 0.2%                                        | <span style="color:#00aa00">**144.3 μsec ± 0.0%**</span> |