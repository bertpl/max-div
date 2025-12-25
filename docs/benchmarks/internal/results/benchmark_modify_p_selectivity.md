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
| 10            | 284.4 nsec ± 0.6%                            | 221.2 nsec ± 0.5% | 222.8 nsec ± 0.7% | <span style="color:#00aa00">**213.7 nsec ± 0.7%**</span> | 276.2 nsec ± 0.6% |
| 20            | 357.2 nsec ± 0.8%                            | 239.7 nsec ± 0.3% | 239.6 nsec ± 0.5% | <span style="color:#00aa00">**217.8 nsec ± 0.8%**</span> | 285.9 nsec ± 0.4% |
| 50            | 553.8 nsec ± 0.5%                            | 288.4 nsec ± 0.3% | 287.9 nsec ± 0.3% | <span style="color:#00aa00">**235.8 nsec ± 0.5%**</span> | 321.9 nsec ± 0.3% |
| 100           | 893.3 nsec ± 0.2%                            | 370.3 nsec ± 0.4% | 370.0 nsec ± 0.2% | <span style="color:#00aa00">**277.5 nsec ± 0.2%**</span> | 398.0 nsec ± 0.5% |
| 200           | 1.583 μsec ± 0.2%                            | 539.5 nsec ± 0.4% | 540.2 nsec ± 0.3% | <span style="color:#00aa00">**352.3 nsec ± 0.1%**</span> | 521.6 nsec ± 0.3% |
| 500           | 3.682 μsec ± 0.4%                            | 1.062 μsec ± 2.2% | 1.089 μsec ± 1.6% | <span style="color:#00aa00">**597.3 nsec ± 1.0%**</span> | 920.6 nsec ± 1.6% |
| 1000          | 7.145 μsec ± 0.8%                            | 1.884 μsec ± 0.2% | 1.880 μsec ± 0.2% | <span style="color:#00aa00">**978.4 nsec ± 0.2%**</span> | 1.512 μsec ± 0.3% |
| 2000          | 13.92 μsec ± 0.4%                            | 3.582 μsec ± 0.3% | 3.564 μsec ± 0.2% | <span style="color:#00aa00">**1.740 μsec ± 0.6%**</span> | 2.731 μsec ± 0.1% |
| 5000          | 34.35 μsec ± 0.2%                            | 8.615 μsec ± 0.1% | 8.602 μsec ± 0.1% | <span style="color:#00aa00">**3.923 μsec ± 1.0%**</span> | 6.426 μsec ± 0.3% |
| 10000         | 69.00 μsec ± 0.1%                            | 17.12 μsec ± 0.1% | 17.09 μsec ± 0.1% | <span style="color:#00aa00">**7.721 μsec ± 0.4%**</span> | 12.61 μsec ± 0.3% |
| 20000         | 140.2 μsec ± 0.1%                            | 34.35 μsec ± 0.1% | 34.45 μsec ± 0.1% | <span style="color:#00aa00">**15.68 μsec ± 0.6%**</span> | 24.85 μsec ± 0.0% |
| **Geomean:**  | 4.316 μsec ± 0.4%                            | 1.534 μsec ± 0.4% | 1.538 μsec ± 0.4% | <span style="color:#00aa00">**942.0 nsec ± 0.6%**</span> | 1.400 μsec ± 0.4% |
| **e_approx:** | <span style="color:#00aa00">**0.00%**</span> | 0.89%             | 0.41%             | 16.33%                                                   | N/A               |

