# Benchmarks

All benchmarks are run on `m6a.2xlarge` AWS EC2 instances.

## max_div.sampling.discrete.sample_int

```bash
uv run --extra numba max-div benchmark --markdown sample-int
```

### A. WITH replacement, UNIFORM probabilities

| `k`   | `n`   | **accelerated=False**<br>*(numpy)*                       | **accelerated=True**<br>*(custom numba)*                 |
| ----- | ----- | -------------------------------------------------------- | -------------------------------------------------------- |
| 1     | 10    | 9.967 μsec ± 4.4%                                        | <span style="color:#00aa00">**1.529 μsec ± 0.2%**</span> |
| 10    | 10    | 8.896 μsec ± 0.7%                                        | <span style="color:#00aa00">**1.676 μsec ± 0.1%**</span> |
| 100   | 10    | 9.879 μsec ± 0.3%                                        | <span style="color:#00aa00">**3.077 μsec ± 1.5%**</span> |
| 1000  | 10    | 20.18 μsec ± 0.3%                                        | <span style="color:#00aa00">**17.65 μsec ± 0.1%**</span> |
| 10000 | 10    | <span style="color:#00aa00">**120.3 μsec ± 0.1%**</span> | 161.7 μsec ± 0.0%                                        |
| 1     | 100   | 8.677 μsec ± 0.8%                                        | <span style="color:#00aa00">**1.530 μsec ± 0.1%**</span> |
| 10    | 100   | 8.731 μsec ± 0.4%                                        | <span style="color:#00aa00">**1.633 μsec ± 0.0%**</span> |
| 100   | 100   | 9.460 μsec ± 1.0%                                        | <span style="color:#00aa00">**2.638 μsec ± 0.0%**</span> |
| 1000  | 100   | 15.63 μsec ± 0.4%                                        | <span style="color:#00aa00">**13.42 μsec ± 0.1%**</span> |
| 10000 | 100   | <span style="color:#00aa00">**76.10 μsec ± 0.1%**</span> | 119.4 μsec ± 0.0%                                        |
| 1     | 1000  | 8.768 μsec ± 0.7%                                        | <span style="color:#00aa00">**1.533 μsec ± 0.1%**</span> |
| 10    | 1000  | 8.952 μsec ± 0.7%                                        | <span style="color:#00aa00">**1.586 μsec ± 0.2%**</span> |
| 100   | 1000  | 9.268 μsec ± 0.8%                                        | <span style="color:#00aa00">**2.316 μsec ± 0.1%**</span> |
| 1000  | 1000  | 12.56 μsec ± 0.5%                                        | <span style="color:#00aa00">**10.26 μsec ± 0.0%**</span> |
| 10000 | 1000  | <span style="color:#00aa00">**44.47 μsec ± 0.2%**</span> | 84.23 μsec ± 4.9%                                        |
| 1     | 10000 | 8.869 μsec ± 0.4%                                        | <span style="color:#00aa00">**1.523 μsec ± 0.1%**</span> |
| 10    | 10000 | 8.923 μsec ± 0.7%                                        | <span style="color:#00aa00">**1.677 μsec ± 0.1%**</span> |
| 100   | 10000 | 10.05 μsec ± 0.3%                                        | <span style="color:#00aa00">**3.094 μsec ± 0.1%**</span> |
| 1000  | 10000 | 20.78 μsec ± 0.2%                                        | <span style="color:#00aa00">**17.22 μsec ± 2.8%**</span> |
| 10000 | 10000 | <span style="color:#00aa00">**125.6 μsec ± 0.1%**</span> | 166.7 μsec ± 0.1%                                        |

### B. WITHOUT replacement, UNIFORM probabilities

| `k`   | `n`   | **accelerated=False**<br>*(numpy)*                       | **accelerated=True**<br>*(custom numba)*                 |
| ----- | ----- | -------------------------------------------------------- | -------------------------------------------------------- |
| 1     | 10    | 8.766 μsec ± 0.9%                                        | <span style="color:#00aa00">**1.538 μsec ± 0.1%**</span> |
| 10    | 10    | 7.114 μsec ± 1.0%                                        | <span style="color:#00aa00">**1.646 μsec ± 0.1%**</span> |
| 1     | 100   | 8.663 μsec ± 0.6%                                        | <span style="color:#00aa00">**1.530 μsec ± 0.1%**</span> |
| 10    | 100   | 8.065 μsec ± 0.7%                                        | <span style="color:#00aa00">**1.675 μsec ± 0.2%**</span> |
| 100   | 100   | 8.077 μsec ± 0.4%                                        | <span style="color:#00aa00">**2.862 μsec ± 0.1%**</span> |
| 1     | 1000  | 8.758 μsec ± 0.4%                                        | <span style="color:#00aa00">**1.524 μsec ± 0.1%**</span> |
| 10    | 1000  | 17.93 μsec ± 0.2%                                        | <span style="color:#00aa00">**1.750 μsec ± 0.1%**</span> |
| 100   | 1000  | 17.90 μsec ± 0.2%                                        | <span style="color:#00aa00">**2.630 μsec ± 0.1%**</span> |
| 1000  | 1000  | 18.22 μsec ± 0.3%                                        | <span style="color:#00aa00">**14.99 μsec ± 0.1%**</span> |
| 1     | 10000 | 8.891 μsec ± 1.0%                                        | <span style="color:#00aa00">**1.538 μsec ± 0.3%**</span> |
| 10    | 10000 | 124.6 μsec ± 0.1%                                        | <span style="color:#00aa00">**2.596 μsec ± 0.3%**</span> |
| 100   | 10000 | 124.5 μsec ± 0.1%                                        | <span style="color:#00aa00">**4.059 μsec ± 0.2%**</span> |
| 1000  | 10000 | 126.4 μsec ± 0.8%                                        | <span style="color:#00aa00">**19.46 μsec ± 0.0%**</span> |
| 10000 | 10000 | <span style="color:#00aa00">**124.6 μsec ± 0.0%**</span> | 145.2 μsec ± 0.0%                                        |

### C. WITH replacement, CUSTOM probabilities

| `k`   | `n`   | **accelerated=False**<br>*(numpy)*                       | **accelerated=True**<br>*(custom numba)*                 |
| ----- | ----- | -------------------------------------------------------- | -------------------------------------------------------- |
| 1     | 10    | 15.91 μsec ± 0.3%                                        | <span style="color:#00aa00">**1.308 μsec ± 0.1%**</span> |
| 10    | 10    | 16.14 μsec ± 0.3%                                        | <span style="color:#00aa00">**1.587 μsec ± 0.5%**</span> |
| 100   | 10    | 18.42 μsec ± 0.5%                                        | <span style="color:#00aa00">**4.263 μsec ± 0.2%**</span> |
| 1000  | 10    | 39.61 μsec ± 0.2%                                        | <span style="color:#00aa00">**31.76 μsec ± 0.0%**</span> |
| 10000 | 10    | <span style="color:#00aa00">**238.0 μsec ± 0.1%**</span> | 308.6 μsec ± 0.1%                                        |
| 1     | 100   | 16.60 μsec ± 0.8%                                        | <span style="color:#00aa00">**1.372 μsec ± 0.1%**</span> |
| 10    | 100   | 17.14 μsec ± 0.7%                                        | <span style="color:#00aa00">**1.858 μsec ± 0.1%**</span> |
| 100   | 100   | 21.22 μsec ± 0.3%                                        | <span style="color:#00aa00">**6.471 μsec ± 0.1%**</span> |
| 1000  | 100   | 62.26 μsec ± 0.1%                                        | <span style="color:#00aa00">**51.96 μsec ± 0.1%**</span> |
| 10000 | 100   | <span style="color:#00aa00">**470.7 μsec ± 0.1%**</span> | 508.8 μsec ± 0.1%                                        |
| 1     | 1000  | 21.81 μsec ± 0.3%                                        | <span style="color:#00aa00">**2.242 μsec ± 0.1%**</span> |
| 10    | 1000  | 22.41 μsec ± 0.2%                                        | <span style="color:#00aa00">**2.932 μsec ± 0.1%**</span> |
| 100   | 1000  | 31.30 μsec ± 0.2%                                        | <span style="color:#00aa00">**9.657 μsec ± 0.1%**</span> |
| 1000  | 1000  | 93.11 μsec ± 0.1%                                        | <span style="color:#00aa00">**76.27 μsec ± 0.0%**</span> |
| 10000 | 1000  | <span style="color:#00aa00">**701.6 μsec ± 0.1%**</span> | 743.8 μsec ± 0.1%                                        |
| 1     | 10000 | 91.54 μsec ± 0.2%                                        | <span style="color:#00aa00">**9.835 μsec ± 0.0%**</span> |
| 10    | 10000 | 93.03 μsec ± 0.4%                                        | <span style="color:#00aa00">**10.78 μsec ± 0.0%**</span> |
| 100   | 10000 | 102.3 μsec ± 0.4%                                        | <span style="color:#00aa00">**19.63 μsec ± 0.1%**</span> |
| 1000  | 10000 | 191.9 μsec ± 0.1%                                        | <span style="color:#00aa00">**107.0 μsec ± 0.2%**</span> |
| 10000 | 10000 | 1.077 msec ± 0.1%                                        | <span style="color:#00aa00">**980.7 μsec ± 0.2%**</span> |

### D. WITHOUT replacement, CUSTOM probabilities

| `k`   | `n`   | **accelerated=False**<br>*(numpy)*                       | **accelerated=True**<br>*(custom numba)*                 |
| ----- | ----- | -------------------------------------------------------- | -------------------------------------------------------- |
| 1     | 10    | 16.07 μsec ± 0.3%                                        | <span style="color:#00aa00">**1.296 μsec ± 0.1%**</span> |
| 10    | 10    | 69.74 μsec ± 0.3%                                        | <span style="color:#00aa00">**1.399 μsec ± 0.4%**</span> |
| 1     | 100   | 16.69 μsec ± 0.5%                                        | <span style="color:#00aa00">**1.387 μsec ± 0.5%**</span> |
| 10    | 100   | 44.00 μsec ± 0.7%                                        | <span style="color:#00aa00">**4.501 μsec ± 0.3%**</span> |
| 100   | 100   | 119.0 μsec ± 0.2%                                        | <span style="color:#00aa00">**2.635 μsec ± 1.0%**</span> |
| 1     | 1000  | 24.29 μsec ± 0.3%                                        | <span style="color:#00aa00">**2.256 μsec ± 0.1%**</span> |
| 10    | 1000  | 46.56 μsec ± 0.2%                                        | <span style="color:#00aa00">**23.75 μsec ± 0.2%**</span> |
| 100   | 1000  | 77.15 μsec ± 0.5%                                        | <span style="color:#00aa00">**26.84 μsec ± 0.1%**</span> |
| 1000  | 1000  | 439.0 μsec ± 0.2%                                        | <span style="color:#00aa00">**14.72 μsec ± 0.1%**</span> |
| 1     | 10000 | 91.23 μsec ± 0.1%                                        | <span style="color:#00aa00">**9.850 μsec ± 0.1%**</span> |
| 10    | 10000 | <span style="color:#00aa00">**118.0 μsec ± 0.2%**</span> | 229.6 μsec ± 0.3%                                        |
| 100   | 10000 | <span style="color:#00aa00">**157.0 μsec ± 0.4%**</span> | 248.2 μsec ± 0.3%                                        |
| 1000  | 10000 | 350.1 μsec ± 0.3%                                        | <span style="color:#00aa00">**243.6 μsec ± 0.4%**</span> |
| 10000 | 10000 | 3.860 msec ± 0.2%                                        | <span style="color:#00aa00">**144.9 μsec ± 0.1%**</span> |