Tested Initialization strategies:

 - `OSR(u)` : InitOneShotRandom(uniform=True, constrained=False)
 - `OSR(nu)`: InitOneShotRandom(uniform=False, constrained=False)

### Time Duration

| `d` | `n`  | `k` | `m`          | `OSR(u)`                                                 | `OSR(nu)`                                                |
| --- | ---- | --- | ------------ | -------------------------------------------------------- | -------------------------------------------------------- |
| 1   | 100  | 10  | 0            | <span style="color:#00aa00">**43.42 μsec ± 3.5%**</span> | 44.46 μsec ± 1.9%                                        |
| 2   | 200  | 20  | 0            | <span style="color:#00aa00">**94.00 μsec ± 9.0%**</span> | <span style="color:#00aa00">**91.52 μsec ± 3.1%**</span> |
| 3   | 300  | 30  | 0            | <span style="color:#00aa00">**149.5 μsec ± 3.1%**</span> | <span style="color:#00aa00">**146.9 μsec ± 2.5%**</span> |
| 4   | 400  | 40  | 0            | <span style="color:#00aa00">**205.8 μsec ± 3.9%**</span> | <span style="color:#00aa00">**206.2 μsec ± 1.5%**</span> |
| 5   | 500  | 50  | 0            | <span style="color:#00aa00">**270.1 μsec ± 2.1%**</span> | 277.4 μsec ± 3.3%                                        |
| 6   | 600  | 60  | 0            | <span style="color:#00aa00">**347.4 μsec ± 3.0%**</span> | 372.6 μsec ± 7.4%                                        |
| 7   | 700  | 70  | 0            | <span style="color:#00aa00">**474.6 μsec ± 8.2%**</span> | <span style="color:#00aa00">**465.9 μsec ± 7.7%**</span> |
| 8   | 800  | 80  | 0            | <span style="color:#00aa00">**529.2 μsec ± 4.4%**</span> | <span style="color:#00aa00">**523.7 μsec ± 0.9%**</span> |
| 9   | 900  | 90  | 0            | <span style="color:#00aa00">**630.1 μsec ± 4.9%**</span> | <span style="color:#00aa00">**636.2 μsec ± 1.8%**</span> |
| 10  | 1000 | 100 | 0            | <span style="color:#00aa00">**744.2 μsec ± 2.3%**</span> | <span style="color:#00aa00">**757.7 μsec ± 5.5%**</span> |
| 11  | 1100 | 110 | 0            | 935.0 μsec ± 5.6%                                        | <span style="color:#00aa00">**842.0 μsec ± 1.6%**</span> |
| 12  | 1200 | 120 | 0            | <span style="color:#00aa00">**965.3 μsec ± 1.7%**</span> | <span style="color:#00aa00">**977.5 μsec ± 3.1%**</span> |
| 13  | 1300 | 130 | 0            | 1.180 msec ± 3.1%                                        | <span style="color:#00aa00">**1.107 msec ± 2.6%**</span> |
| 14  | 1400 | 140 | 0            | <span style="color:#00aa00">**1.293 msec ± 4.2%**</span> | <span style="color:#00aa00">**1.260 msec ± 1.0%**</span> |
| 15  | 1500 | 150 | 0            | <span style="color:#00aa00">**1.381 msec ± 0.9%**</span> | <span style="color:#00aa00">**1.390 msec ± 1.9%**</span> |
| 16  | 1600 | 160 | 0            | 1.937 msec ± 10.0%                                       | <span style="color:#00aa00">**1.615 msec ± 4.1%**</span> |
| 17  | 1700 | 170 | 0            | <span style="color:#00aa00">**1.711 msec ± 2.4%**</span> | <span style="color:#00aa00">**1.707 msec ± 1.3%**</span> |
| 18  | 1800 | 180 | 0            | <span style="color:#00aa00">**1.912 msec ± 3.7%**</span> | <span style="color:#00aa00">**1.865 msec ± 3.1%**</span> |
| 19  | 1900 | 190 | 0            | <span style="color:#00aa00">**1.994 msec ± 1.2%**</span> | 2.058 msec ± 1.4%                                        |
| 20  | 2000 | 200 | 0            | <span style="color:#00aa00">**2.205 msec ± 0.9%**</span> | <span style="color:#00aa00">**2.186 msec ± 1.3%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**621.3 μsec ± 3.9%**</span> | <span style="color:#00aa00">**612.5 μsec ± 2.8%**</span> |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `OSR(u)`                                             | `OSR(nu)`                                            |
| --- | ---- | --- | ------------ | ---------------------------------------------------- | ---------------------------------------------------- |
| 1   | 100  | 10  | 0            | <span style="color:#00aa00">**0.035 ± 30.6%**</span> | <span style="color:#00aa00">**0.044 ± 19.6%**</span> |
| 2   | 200  | 20  | 0            | <span style="color:#00aa00">**0.109 ± 10.1%**</span> | <span style="color:#00aa00">**0.114 ± 6.5%**</span>  |
| 3   | 300  | 30  | 0            | 0.190 ± 5.8%                                         | <span style="color:#00aa00">**0.205 ± 6.2%**</span>  |
| 4   | 400  | 40  | 0            | <span style="color:#00aa00">**0.259 ± 5.1%**</span>  | <span style="color:#00aa00">**0.276 ± 4.3%**</span>  |
| 5   | 500  | 50  | 0            | 0.338 ± 4.0%                                         | <span style="color:#00aa00">**0.356 ± 3.8%**</span>  |
| 6   | 600  | 60  | 0            | <span style="color:#00aa00">**0.412 ± 3.2%**</span>  | <span style="color:#00aa00">**0.421 ± 1.9%**</span>  |
| 7   | 700  | 70  | 0            | <span style="color:#00aa00">**0.472 ± 2.4%**</span>  | <span style="color:#00aa00">**0.484 ± 2.0%**</span>  |
| 8   | 800  | 80  | 0            | <span style="color:#00aa00">**0.545 ± 2.4%**</span>  | <span style="color:#00aa00">**0.553 ± 1.6%**</span>  |
| 9   | 900  | 90  | 0            | <span style="color:#00aa00">**0.606 ± 2.6%**</span>  | <span style="color:#00aa00">**0.612 ± 1.8%**</span>  |
| 10  | 1000 | 100 | 0            | <span style="color:#00aa00">**0.657 ± 2.0%**</span>  | <span style="color:#00aa00">**0.668 ± 1.8%**</span>  |
| 11  | 1100 | 110 | 0            | <span style="color:#00aa00">**0.719 ± 1.4%**</span>  | <span style="color:#00aa00">**0.724 ± 1.5%**</span>  |
| 12  | 1200 | 120 | 0            | <span style="color:#00aa00">**0.777 ± 1.1%**</span>  | <span style="color:#00aa00">**0.782 ± 1.2%**</span>  |
| 13  | 1300 | 130 | 0            | <span style="color:#00aa00">**0.826 ± 0.9%**</span>  | <span style="color:#00aa00">**0.831 ± 1.0%**</span>  |
| 14  | 1400 | 140 | 0            | <span style="color:#00aa00">**0.881 ± 1.0%**</span>  | <span style="color:#00aa00">**0.888 ± 0.9%**</span>  |
| 15  | 1500 | 150 | 0            | <span style="color:#00aa00">**0.928 ± 1.0%**</span>  | <span style="color:#00aa00">**0.936 ± 0.6%**</span>  |
| 16  | 1600 | 160 | 0            | 0.971 ± 0.9%                                         | <span style="color:#00aa00">**0.987 ± 0.8%**</span>  |
| 17  | 1700 | 170 | 0            | 1.023 ± 0.7%                                         | <span style="color:#00aa00">**1.031 ± 0.7%**</span>  |
| 18  | 1800 | 180 | 0            | <span style="color:#00aa00">**1.062 ± 1.1%**</span>  | <span style="color:#00aa00">**1.075 ± 0.9%**</span>  |
| 19  | 1900 | 190 | 0            | 1.108 ± 0.7%                                         | <span style="color:#00aa00">**1.118 ± 0.6%**</span>  |
| 20  | 2000 | 200 | 0            | 1.156 ± 0.8%                                         | <span style="color:#00aa00">**1.168 ± 0.7%**</span>  |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**0.509 ± 3.9%**</span>  | <span style="color:#00aa00">**0.526 ± 3.0%**</span>  |

