Tested methods:                                                                                                                                                                                                                                                                                                                                     

| `method` | Description                                                                  |
| -------- | ---------------------------------------------------------------------------- |
| 0        | p**t                                                                         |
| 10       | fast_exp2(t * fast_log2(p))   (NOT specifically optimized for this use case) |
| 20       | fast_pow(p, t)                (specifically optimized for this use case)     |
| 100      | 2-segment PWL approx. of p**t                                                |

## Benchmark results


| size          | method=0                                     | method=10         | method=20         | method=100                                               |
| ------------- | -------------------------------------------- | ----------------- | ----------------- | -------------------------------------------------------- |
| 10            | 285.4 nsec ± 0.7%                            | 236.2 nsec ± 0.4% | 237.1 nsec ± 0.8% | <span style="color:#00aa00">**217.2 nsec ± 0.5%**</span> |
| 20            | 354.3 nsec ± 0.5%                            | 258.9 nsec ± 0.5% | 264.7 nsec ± 0.6% | <span style="color:#00aa00">**224.4 nsec ± 0.8%**</span> |
| 50            | 552.8 nsec ± 0.3%                            | 353.5 nsec ± 0.8% | 346.3 nsec ± 0.6% | <span style="color:#00aa00">**251.8 nsec ± 0.5%**</span> |
| 100           | 895.1 nsec ± 0.3%                            | 490.4 nsec ± 0.6% | 478.8 nsec ± 0.7% | <span style="color:#00aa00">**297.6 nsec ± 0.7%**</span> |
| 200           | 1.586 μsec ± 0.4%                            | 758.5 nsec ± 0.5% | 739.3 nsec ± 0.5% | <span style="color:#00aa00">**388.9 nsec ± 0.6%**</span> |
| 500           | 3.671 μsec ± 0.3%                            | 1.555 μsec ± 0.5% | 1.526 μsec ± 0.7% | <span style="color:#00aa00">**666.4 nsec ± 0.4%**</span> |
| 1000          | 7.069 μsec ± 0.3%                            | 2.827 μsec ± 0.5% | 2.847 μsec ± 1.0% | <span style="color:#00aa00">**1.105 μsec ± 0.3%**</span> |
| 2000          | 13.93 μsec ± 0.1%                            | 5.516 μsec ± 0.2% | 5.499 μsec ± 0.5% | <span style="color:#00aa00">**2.016 μsec ± 0.1%**</span> |
| 5000          | 34.34 μsec ± 0.4%                            | 13.33 μsec ± 0.8% | 13.32 μsec ± 0.2% | <span style="color:#00aa00">**4.724 μsec ± 0.2%**</span> |
| 10000         | 68.81 μsec ± 0.2%                            | 26.91 μsec ± 0.4% | 26.67 μsec ± 0.5% | <span style="color:#00aa00">**9.307 μsec ± 0.1%**</span> |
| 20000         | 139.6 μsec ± 0.2%                            | 53.65 μsec ± 0.6% | 53.73 μsec ± 0.6% | <span style="color:#00aa00">**18.82 μsec ± 0.5%**</span> |
| **Geomean:**  | 4.307 μsec ± 0.3%                            | 2.113 μsec ± 0.5% | 2.100 μsec ± 0.6% | <span style="color:#00aa00">**1.052 μsec ± 0.4%**</span> |
| **e_approx:** | <span style="color:#00aa00">**0.00%**</span> | 0.89%             | 0.41%             | 16.33%                                                   |