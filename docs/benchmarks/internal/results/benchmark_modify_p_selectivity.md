## modify_p_selectivity Performance

Tested methods:

  - 0    --> p**t
  - 10   --> fast_exp2(t * fast_log2(p))   (NOT specifically optimized for this use case)
  - 20   --> fast_pow(p, t)                (specifically optimized for this use case)
  - 100  --> 2-segment PWL approx. of p**t


| size          | method=0                                     | method=10         | method=20         | method=100                                               |
| ------------- | -------------------------------------------- | ----------------- | ----------------- | -------------------------------------------------------- |
| 10            | 284.0 nsec ± 0.6%                            | 235.4 nsec ± 0.3% | 235.3 nsec ± 0.7% | <span style="color:#00aa00">**215.4 nsec ± 1.6%**</span> |
| 20            | 352.5 nsec ± 0.3%                            | 262.4 nsec ± 3.4% | 262.7 nsec ± 0.2% | <span style="color:#00aa00">**222.0 nsec ± 0.4%**</span> |
| 50            | 551.2 nsec ± 0.3%                            | 354.6 nsec ± 0.4% | 356.4 nsec ± 0.5% | <span style="color:#00aa00">**251.8 nsec ± 0.4%**</span> |
| 100           | 895.8 nsec ± 0.2%                            | 490.4 nsec ± 1.2% | 516.6 nsec ± 0.5% | <span style="color:#00aa00">**301.9 nsec ± 1.5%**</span> |
| 200           | 1.621 μsec ± 0.9%                            | 775.4 nsec ± 0.4% | 763.9 nsec ± 0.4% | <span style="color:#00aa00">**395.1 nsec ± 0.5%**</span> |
| 500           | 3.677 μsec ± 0.1%                            | 1.592 μsec ± 0.1% | 1.568 μsec ± 0.3% | <span style="color:#00aa00">**690.2 nsec ± 0.2%**</span> |
| 1000          | 7.126 μsec ± 0.1%                            | 2.934 μsec ± 0.2% | 2.938 μsec ± 0.2% | <span style="color:#00aa00">**1.184 μsec ± 0.3%**</span> |
| 2000          | 13.94 μsec ± 0.3%                            | 5.720 μsec ± 0.3% | 5.656 μsec ± 0.3% | <span style="color:#00aa00">**2.167 μsec ± 0.3%**</span> |
| 5000          | 34.76 μsec ± 0.4%                            | 13.89 μsec ± 0.2% | 13.84 μsec ± 0.3% | <span style="color:#00aa00">**5.062 μsec ± 0.5%**</span> |
| 10000         | 69.83 μsec ± 2.1%                            | 27.59 μsec ± 0.5% | 27.44 μsec ± 0.3% | <span style="color:#00aa00">**9.986 μsec ± 0.6%**</span> |
| 20000         | 142.7 μsec ± 1.5%                            | 55.23 μsec ± 0.2% | 54.89 μsec ± 0.2% | <span style="color:#00aa00">**20.05 μsec ± 0.3%**</span> |
| **Geomean:**  | 4.334 μsec ± 0.6%                            | 2.157 μsec ± 0.7% | 2.158 μsec ± 0.4% | <span style="color:#00aa00">**1.090 μsec ± 0.6%**</span> |
| **e_approx:** | <span style="color:#00aa00">**0.00%**</span> | 0.89%             | 0.48%             | 16.33%                                                   |

