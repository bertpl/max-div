Tested Initialization strategies:

 - `OSR(u)` : InitOneShotRandom(uniform=True, constrained=False)
 - `OSR(nu)`: InitOneShotRandom(uniform=False, constrained=False)

### Time Duration

| `d` | `n`  | `k` | `m`          | `OSR(u)`                                                 | `OSR(nu)`                                                |
| --- | ---- | --- | ------------ | -------------------------------------------------------- | -------------------------------------------------------- |
| 1   | 100  | 10  | 0            | <span style="color:#00aa00">**43.94 μsec ± 4.9%**</span> | 45.77 μsec ± 1.4%                                        |
| 2   | 200  | 20  | 0            | <span style="color:#00aa00">**95.35 μsec ± 5.9%**</span> | <span style="color:#00aa00">**91.31 μsec ± 1.2%**</span> |
| 3   | 300  | 30  | 0            | <span style="color:#00aa00">**143.5 μsec ± 1.8%**</span> | 147.4 μsec ± 1.0%                                        |
| 4   | 400  | 40  | 0            | <span style="color:#00aa00">**208.3 μsec ± 3.0%**</span> | <span style="color:#00aa00">**211.7 μsec ± 2.9%**</span> |
| 5   | 500  | 50  | 0            | <span style="color:#00aa00">**273.5 μsec ± 1.5%**</span> | 277.3 μsec ± 4.0%                                        |
| 6   | 600  | 60  | 0            | <span style="color:#00aa00">**343.4 μsec ± 0.9%**</span> | 352.9 μsec ± 1.5%                                        |
| 7   | 700  | 70  | 0            | <span style="color:#00aa00">**428.3 μsec ± 1.0%**</span> | 437.3 μsec ± 0.7%                                        |
| 8   | 800  | 80  | 0            | <span style="color:#00aa00">**514.9 μsec ± 1.0%**</span> | 518.6 μsec ± 1.0%                                        |
| 9   | 900  | 90  | 0            | <span style="color:#00aa00">**612.6 μsec ± 1.1%**</span> | 631.4 μsec ± 1.6%                                        |
| 10  | 1000 | 100 | 0            | <span style="color:#00aa00">**717.9 μsec ± 1.8%**</span> | <span style="color:#00aa00">**723.2 μsec ± 1.3%**</span> |
| 11  | 1100 | 110 | 0            | <span style="color:#00aa00">**830.8 μsec ± 1.2%**</span> | 840.3 μsec ± 1.1%                                        |
| 12  | 1200 | 120 | 0            | <span style="color:#00aa00">**956.4 μsec ± 1.5%**</span> | <span style="color:#00aa00">**962.6 μsec ± 0.9%**</span> |
| 13  | 1300 | 130 | 0            | <span style="color:#00aa00">**1.077 msec ± 1.2%**</span> | 1.102 msec ± 1.3%                                        |
| 14  | 1400 | 140 | 0            | <span style="color:#00aa00">**1.209 msec ± 1.6%**</span> | 1.239 msec ± 1.1%                                        |
| 15  | 1500 | 150 | 0            | 1.510 msec ± 4.9%                                        | <span style="color:#00aa00">**1.388 msec ± 1.1%**</span> |
| 16  | 1600 | 160 | 0            | <span style="color:#00aa00">**1.534 msec ± 2.2%**</span> | <span style="color:#00aa00">**1.565 msec ± 2.3%**</span> |
| 17  | 1700 | 170 | 0            | <span style="color:#00aa00">**1.689 msec ± 1.4%**</span> | <span style="color:#00aa00">**1.691 msec ± 1.6%**</span> |
| 18  | 1800 | 180 | 0            | <span style="color:#00aa00">**1.841 msec ± 1.7%**</span> | <span style="color:#00aa00">**1.856 msec ± 0.9%**</span> |
| 19  | 1900 | 190 | 0            | <span style="color:#00aa00">**2.034 msec ± 1.2%**</span> | 2.071 msec ± 1.1%                                        |
| 20  | 2000 | 200 | 0            | <span style="color:#00aa00">**2.192 msec ± 1.3%**</span> | <span style="color:#00aa00">**2.197 msec ± 0.7%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**601.1 μsec ± 2.1%**</span> | <span style="color:#00aa00">**606.5 μsec ± 1.4%**</span> |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `OSR(u)`      | `OSR(nu)`                                            |
| --- | ---- | --- | ------------ | ------------- | ---------------------------------------------------- |
| 1   | 100  | 10  | 0            | 0.113 ± 37.2% | <span style="color:#00aa00">**0.227 ± 19.3%**</span> |
| 2   | 200  | 20  | 0            | 0.359 ± 13.6% | <span style="color:#00aa00">**0.502 ± 8.6%**</span>  |
| 3   | 300  | 30  | 0            | 0.689 ± 7.8%  | <span style="color:#00aa00">**0.797 ± 6.5%**</span>  |
| 4   | 400  | 40  | 0            | 0.954 ± 3.5%  | <span style="color:#00aa00">**1.090 ± 3.9%**</span>  |
| 5   | 500  | 50  | 0            | 1.182 ± 4.1%  | <span style="color:#00aa00">**1.319 ± 3.5%**</span>  |
| 6   | 600  | 60  | 0            | 1.445 ± 2.0%  | <span style="color:#00aa00">**1.544 ± 2.4%**</span>  |
| 7   | 700  | 70  | 0            | 1.653 ± 2.9%  | <span style="color:#00aa00">**1.739 ± 2.0%**</span>  |
| 8   | 800  | 80  | 0            | 1.895 ± 2.3%  | <span style="color:#00aa00">**1.974 ± 1.8%**</span>  |
| 9   | 900  | 90  | 0            | 2.096 ± 1.9%  | <span style="color:#00aa00">**2.191 ± 1.5%**</span>  |
| 10  | 1000 | 100 | 0            | 2.292 ± 2.0%  | <span style="color:#00aa00">**2.419 ± 2.2%**</span>  |
| 11  | 1100 | 110 | 0            | 2.476 ± 1.5%  | <span style="color:#00aa00">**2.562 ± 1.8%**</span>  |
| 12  | 1200 | 120 | 0            | 2.644 ± 1.6%  | <span style="color:#00aa00">**2.759 ± 1.2%**</span>  |
| 13  | 1300 | 130 | 0            | 2.840 ± 0.6%  | <span style="color:#00aa00">**2.911 ± 1.6%**</span>  |
| 14  | 1400 | 140 | 0            | 3.023 ± 1.4%  | <span style="color:#00aa00">**3.099 ± 1.0%**</span>  |
| 15  | 1500 | 150 | 0            | 3.187 ± 1.1%  | <span style="color:#00aa00">**3.264 ± 1.0%**</span>  |
| 16  | 1600 | 160 | 0            | 3.326 ± 1.0%  | <span style="color:#00aa00">**3.404 ± 1.1%**</span>  |
| 17  | 1700 | 170 | 0            | 3.507 ± 0.8%  | <span style="color:#00aa00">**3.547 ± 1.1%**</span>  |
| 18  | 1800 | 180 | 0            | 3.654 ± 1.2%  | <span style="color:#00aa00">**3.723 ± 0.8%**</span>  |
| 19  | 1900 | 190 | 0            | 3.793 ± 0.9%  | <span style="color:#00aa00">**3.868 ± 1.0%**</span>  |
| 20  | 2000 | 200 | 0            | 3.959 ± 0.9%  | <span style="color:#00aa00">**4.027 ± 0.8%**</span>  |
|     |      |     | **Geomean:** | 1.757 ± 4.5%  | <span style="color:#00aa00">**1.934 ± 3.2%**</span>  |

