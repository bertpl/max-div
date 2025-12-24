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
| 10            | 287.0 nsec ± 0.8%                            | 237.6 nsec ± 0.6% | 239.4 nsec ± 0.6% | <span style="color:#00aa00">**216.1 nsec ± 1.4%**</span> | 288.3 nsec ± 1.1% |
| 20            | 355.3 nsec ± 0.9%                            | 261.7 nsec ± 0.9% | 264.7 nsec ± 0.9% | <span style="color:#00aa00">**219.6 nsec ± 0.5%**</span> | 310.0 nsec ± 0.9% |
| 50            | 553.0 nsec ± 0.4%                            | 356.3 nsec ± 0.5% | 343.6 nsec ± 0.7% | <span style="color:#00aa00">**237.3 nsec ± 0.6%**</span> | 379.5 nsec ± 1.2% |
| 100           | 887.8 nsec ± 0.5%                            | 484.5 nsec ± 0.2% | 475.2 nsec ± 0.8% | <span style="color:#00aa00">**279.7 nsec ± 0.6%**</span> | 474.0 nsec ± 0.4% |
| 200           | 1.589 μsec ± 0.4%                            | 757.2 nsec ± 0.5% | 735.4 nsec ± 0.8% | <span style="color:#00aa00">**353.0 nsec ± 0.5%**</span> | 660.4 nsec ± 0.3% |
| 500           | 3.667 μsec ± 0.5%                            | 1.544 μsec ± 0.3% | 1.526 μsec ± 0.6% | <span style="color:#00aa00">**573.9 nsec ± 0.5%**</span> | 1.227 μsec ± 0.4% |
| 1000          | 7.094 μsec ± 0.5%                            | 2.826 μsec ± 0.4% | 2.814 μsec ± 0.2% | <span style="color:#00aa00">**916.9 nsec ± 0.4%**</span> | 2.158 μsec ± 0.2% |
| 2000          | 13.90 μsec ± 0.2%                            | 5.510 μsec ± 0.5% | 5.503 μsec ± 0.6% | <span style="color:#00aa00">**1.653 μsec ± 0.1%**</span> | 4.022 μsec ± 0.2% |
| 5000          | 34.37 μsec ± 0.1%                            | 13.40 μsec ± 0.5% | 13.40 μsec ± 0.2% | <span style="color:#00aa00">**3.838 μsec ± 0.2%**</span> | 9.604 μsec ± 0.1% |
| 10000         | 68.61 μsec ± 0.3%                            | 26.79 μsec ± 0.2% | 26.71 μsec ± 0.5% | <span style="color:#00aa00">**7.524 μsec ± 0.1%**</span> | 18.93 μsec ± 0.2% |
| 20000         | 139.9 μsec ± 0.2%                            | 54.41 μsec ± 0.4% | 54.11 μsec ± 0.3% | <span style="color:#00aa00">**15.17 μsec ± 0.4%**</span> | 37.54 μsec ± 0.1% |
| **Geomean:**  | 4.308 μsec ± 0.4%                            | 2.117 μsec ± 0.4% | 2.099 μsec ± 0.6% | <span style="color:#00aa00">**924.9 nsec ± 0.5%**</span> | 1.830 μsec ± 0.5% |
| **e_approx:** | <span style="color:#00aa00">**0.00%**</span> | 0.89%             | 0.41%             | 16.33%                                                   | N/A               |

