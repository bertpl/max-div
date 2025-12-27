Tested methods:

| `name`          | `method(args)`                     | Type                | Details                                                                    |
| --------------- | ---------------------------------- | ------------------- | -------------------------------------------------------------------------- |
| **method=0**    | `modify_p_selectivity(method=0)`   | Power-based         | p**t                                                                       |
| **method=10**   | `modify_p_selectivity(method=10)`  | Power-based         | fast_exp2(t * fast_log2(p)) (NOT specifically optimized for this use case) |
| **method=20**   | `modify_p_selectivity(method=20)`  | Power-based         | fast_pow(p, t) (specifically optimized for this use case)                  |
| **method=100**  | `modify_p_selectivity(method=100)` | Power-based         | 2-segment PWL approx. of p**t                                              |
| **exponential** | `exponential_selectivity()`        | Exponential mapping | maps [p_min, p_max] to [1.0, low_value**t]                                 |

## Benchmark results


| size          | method=0                                     | method=10         | method=20         | method=100                                               | exponential       |
| ------------- | -------------------------------------------- | ----------------- | ----------------- | -------------------------------------------------------- | ----------------- |
| 10            | 289.0 nsec ± 0.4%                            | 227.1 nsec ± 0.6% | 231.8 nsec ± 1.4% | <span style="color:#00aa00">**221.7 nsec ± 0.8%**</span> | 275.5 nsec ± 0.6% |
| 20            | 358.6 nsec ± 0.7%                            | 257.9 nsec ± 2.1% | 255.2 nsec ± 1.6% | <span style="color:#00aa00">**234.7 nsec ± 0.9%**</span> | 292.3 nsec ± 1.9% |
| 50            | 578.8 nsec ± 1.3%                            | 312.3 nsec ± 1.3% | 304.2 nsec ± 1.1% | <span style="color:#00aa00">**251.2 nsec ± 1.3%**</span> | 347.8 nsec ± 1.4% |
| 100           | 915.6 nsec ± 0.8%                            | 391.5 nsec ± 0.5% | 394.4 nsec ± 1.5% | <span style="color:#00aa00">**287.5 nsec ± 0.9%**</span> | 416.8 nsec ± 1.3% |
| 200           | 1.607 μsec ± 1.2%                            | 559.4 nsec ± 2.4% | 562.9 nsec ± 3.0% | <span style="color:#00aa00">**368.0 nsec ± 1.0%**</span> | 530.4 nsec ± 0.3% |
| 500           | 3.711 μsec ± 0.7%                            | 1.063 μsec ± 0.3% | 1.075 μsec ± 0.3% | <span style="color:#00aa00">**648.3 nsec ± 1.0%**</span> | 945.9 nsec ± 0.9% |
| 1000          | 7.177 μsec ± 0.8%                            | 1.898 μsec ± 0.5% | 1.895 μsec ± 0.2% | <span style="color:#00aa00">**1.013 μsec ± 1.0%**</span> | 1.539 μsec ± 0.8% |
| 2000          | 14.04 μsec ± 0.7%                            | 3.594 μsec ± 0.2% | 3.595 μsec ± 0.3% | <span style="color:#00aa00">**1.783 μsec ± 0.2%**</span> | 2.861 μsec ± 1.4% |
| 5000          | 35.15 μsec ± 1.5%                            | 8.693 μsec ± 0.8% | 8.783 μsec ± 1.5% | <span style="color:#00aa00">**4.123 μsec ± 0.3%**</span> | 6.438 μsec ± 0.2% |
| 10000         | 68.60 μsec ± 0.2%                            | 17.13 μsec ± 0.3% | 17.09 μsec ± 0.1% | <span style="color:#00aa00">**8.077 μsec ± 0.1%**</span> | 12.58 μsec ± 0.1% |
| 20000         | 140.1 μsec ± 0.1%                            | 34.57 μsec ± 0.3% | 34.48 μsec ± 0.1% | <span style="color:#00aa00">**16.28 μsec ± 0.2%**</span> | 24.95 μsec ± 0.2% |
| **Geomean:**  | 4.371 μsec ± 0.8%                            | 1.577 μsec ± 0.9% | 1.578 μsec ± 1.0% | <span style="color:#00aa00">**988.2 nsec ± 0.7%**</span> | 1.432 μsec ± 0.8% |
| **e_approx:** | <span style="color:#00aa00">**0.00%**</span> | 0.89%             | 0.41%             | 16.33%                                                   | N/A               |

