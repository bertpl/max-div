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
| 10            | 282.7 nsec ± 0.3%                            | 223.8 nsec ± 0.9% | 225.1 nsec ± 0.3% | <span style="color:#00aa00">**213.2 nsec ± 0.3%**</span> | 282.1 nsec ± 0.9% |
| 20            | 351.4 nsec ± 0.3%                            | 239.2 nsec ± 0.3% | 236.9 nsec ± 0.2% | <span style="color:#00aa00">**215.8 nsec ± 0.7%**</span> | 291.2 nsec ± 0.7% |
| 50            | 562.4 nsec ± 0.3%                            | 291.7 nsec ± 0.5% | 288.3 nsec ± 0.6% | <span style="color:#00aa00">**236.7 nsec ± 0.4%**</span> | 332.0 nsec ± 1.7% |
| 100           | 901.8 nsec ± 0.2%                            | 370.3 nsec ± 0.6% | 368.0 nsec ± 0.2% | <span style="color:#00aa00">**280.2 nsec ± 0.6%**</span> | 414.2 nsec ± 0.6% |
| 200           | 1.595 μsec ± 0.1%                            | 541.3 nsec ± 0.7% | 541.7 nsec ± 0.2% | <span style="color:#00aa00">**354.2 nsec ± 0.5%**</span> | 523.7 nsec ± 0.2% |
| 500           | 3.673 μsec ± 0.1%                            | 1.048 μsec ± 0.3% | 1.048 μsec ± 0.2% | <span style="color:#00aa00">**596.6 nsec ± 0.3%**</span> | 884.1 nsec ± 0.1% |
| 1000          | 7.128 μsec ± 0.2%                            | 1.889 μsec ± 0.2% | 1.879 μsec ± 0.1% | <span style="color:#00aa00">**983.0 nsec ± 0.2%**</span> | 1.507 μsec ± 0.3% |
| 2000          | 13.96 μsec ± 0.2%                            | 3.569 μsec ± 0.1% | 3.561 μsec ± 0.1% | <span style="color:#00aa00">**1.760 μsec ± 0.2%**</span> | 2.730 μsec ± 0.5% |
| 5000          | 34.58 μsec ± 0.2%                            | 8.603 μsec ± 0.3% | 8.583 μsec ± 0.1% | <span style="color:#00aa00">**3.988 μsec ± 1.6%**</span> | 6.443 μsec ± 0.3% |
| 10000         | 69.09 μsec ± 0.1%                            | 17.14 μsec ± 0.2% | 17.08 μsec ± 0.1% | <span style="color:#00aa00">**7.696 μsec ± 1.9%**</span> | 12.55 μsec ± 0.1% |
| 20000         | 140.4 μsec ± 0.2%                            | 34.56 μsec ± 0.4% | 34.42 μsec ± 0.2% | <span style="color:#00aa00">**15.80 μsec ± 1.4%**</span> | 24.83 μsec ± 0.1% |
| **Geomean:**  | 4.323 μsec ± 0.2%                            | 1.536 μsec ± 0.4% | 1.531 μsec ± 0.2% | <span style="color:#00aa00">**945.7 nsec ± 0.7%**</span> | 1.408 μsec ± 0.5% |
| **e_approx:** | <span style="color:#00aa00">**0.00%**</span> | 0.89%             | 0.41%             | 16.33%                                                   | N/A               |

