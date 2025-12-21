## modify_p_selectivity Performance                                                                                                                                                                                                                                                                                                                 

Tested methods:

  - 0    --> p**t
  - 10   --> fast_exp2(t * fast_log2(p))   (NOT specifically optimized for this use case)
  - 20   --> fast_pow(p, t)                (specifically optimized for this use case)
  - 100  --> 2-segment PWL approx. of p**t


| size          | method=0                                     | method=10         | method=20         | method=100                                               |
| ------------- | -------------------------------------------- | ----------------- | ----------------- | -------------------------------------------------------- |
| 10            | 283.6 nsec ± 0.7%                            | 235.1 nsec ± 0.8% | 232.6 nsec ± 0.5% | <span style="color:#00aa00">**213.4 nsec ± 0.6%**</span> |
| 20            | 351.1 nsec ± 0.2%                            | 262.4 nsec ± 0.7% | 263.9 nsec ± 0.3% | <span style="color:#00aa00">**220.9 nsec ± 0.2%**</span> |
| 50            | 550.7 nsec ± 0.3%                            | 355.8 nsec ± 0.2% | 345.4 nsec ± 0.5% | <span style="color:#00aa00">**251.0 nsec ± 0.5%**</span> |
| 100           | 895.3 nsec ± 0.3%                            | 490.3 nsec ± 0.2% | 478.1 nsec ± 0.2% | <span style="color:#00aa00">**297.1 nsec ± 0.4%**</span> |
| 200           | 1.588 μsec ± 0.3%                            | 771.1 nsec ± 0.4% | 746.5 nsec ± 0.2% | <span style="color:#00aa00">**392.8 nsec ± 0.3%**</span> |
| 500           | 3.677 μsec ± 0.2%                            | 1.593 μsec ± 0.1% | 1.566 μsec ± 0.0% | <span style="color:#00aa00">**686.5 nsec ± 0.8%**</span> |
| 1000          | 7.108 μsec ± 0.5%                            | 2.926 μsec ± 0.3% | 2.913 μsec ± 0.1% | <span style="color:#00aa00">**1.155 μsec ± 0.8%**</span> |
| 2000          | 13.95 μsec ± 0.3%                            | 5.671 μsec ± 0.5% | 5.616 μsec ± 0.4% | <span style="color:#00aa00">**2.046 μsec ± 0.4%**</span> |
| 5000          | 34.53 μsec ± 0.3%                            | 13.80 μsec ± 0.3% | 13.73 μsec ± 0.3% | <span style="color:#00aa00">**4.816 μsec ± 0.5%**</span> |
| 10000         | 69.14 μsec ± 0.2%                            | 27.52 μsec ± 0.2% | 27.31 μsec ± 0.4% | <span style="color:#00aa00">**9.564 μsec ± 0.2%**</span> |
| 20000         | 140.6 μsec ± 0.3%                            | 55.08 μsec ± 0.3% | 54.86 μsec ± 0.4% | <span style="color:#00aa00">**19.30 μsec ± 0.5%**</span> |
| **Geomean:**  | 4.310 μsec ± 0.3%                            | 2.152 μsec ± 0.4% | 2.125 μsec ± 0.3% | <span style="color:#00aa00">**1.065 μsec ± 0.5%**</span> |
| **e_approx:** | <span style="color:#00aa00">**0.00%**</span> | 0.89%             | 0.41%             | 16.33%                                                   |