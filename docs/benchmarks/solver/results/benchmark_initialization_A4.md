Tested Initialization strategies:

 - `OSR(u)`     : InitOneShotRandom(uniform=True, constrained=False)
 - `OSR(nu)`    : InitOneShotRandom(uniform=False, constrained=False)
 - `OSR(u,con)` : InitOneShotRandom(uniform=True, constrained=True)
 - `OSR(nu,con)`: InitOneShotRandom(uniform=False, constrained=True)

### Time Duration

| `d` | `n`  | `k` | `m`          | `OSR(u)`                                                  | `OSR(nu)`                                                | `OSR(u,con)`       | `OSR(nu,con)`      |
| --- | ---- | --- | ------------ | --------------------------------------------------------- | -------------------------------------------------------- | ------------------ | ------------------ |
| 1   | 150  | 10  | 2            | <span style="color:#00aa00">**52.21 μsec ± 10.0%**</span> | <span style="color:#00aa00">**50.23 μsec ± 6.8%**</span> | 89.56 μsec ± 4.5%  | 54.75 μsec ± 5.6%  |
| 2   | 300  | 20  | 4            | <span style="color:#00aa00">**119.2 μsec ± 1.9%**</span>  | <span style="color:#00aa00">**121.1 μsec ± 2.7%**</span> | 179.7 μsec ± 2.2%  | 144.3 μsec ± 3.0%  |
| 3   | 450  | 30  | 6            | <span style="color:#00aa00">**198.1 μsec ± 4.4%**</span>  | <span style="color:#00aa00">**200.5 μsec ± 2.3%**</span> | 293.6 μsec ± 1.9%  | 258.8 μsec ± 2.0%  |
| 4   | 600  | 40  | 8            | <span style="color:#00aa00">**298.0 μsec ± 1.1%**</span>  | 303.2 μsec ± 1.0%                                        | 491.4 μsec ± 10.0% | 410.5 μsec ± 12.6% |
| 5   | 750  | 50  | 10           | <span style="color:#00aa00">**409.5 μsec ± 0.9%**</span>  | 413.0 μsec ± 1.0%                                        | 792.0 μsec ± 10.6% | 746.1 μsec ± 11.7% |
| 6   | 900  | 60  | 12           | <span style="color:#00aa00">**534.7 μsec ± 0.8%**</span>  | 545.3 μsec ± 0.8%                                        | 1.119 msec ± 11.7% | 1.072 msec ± 12.3% |
| 7   | 1050 | 70  | 14           | <span style="color:#00aa00">**685.9 μsec ± 1.2%**</span>  | 696.7 μsec ± 0.6%                                        | 1.569 msec ± 2.2%  | 1.519 msec ± 1.5%  |
| 8   | 1200 | 80  | 16           | <span style="color:#00aa00">**857.4 μsec ± 2.2%**</span>  | <span style="color:#00aa00">**850.6 μsec ± 1.5%**</span> | 2.075 msec ± 1.5%  | 2.017 msec ± 1.4%  |
| 9   | 1350 | 90  | 18           | <span style="color:#00aa00">**1.037 msec ± 1.1%**</span>  | <span style="color:#00aa00">**1.044 msec ± 1.2%**</span> | 2.661 msec ± 1.5%  | 2.656 msec ± 1.9%  |
| 10  | 1500 | 100 | 20           | <span style="color:#00aa00">**1.216 msec ± 1.6%**</span>  | 1.229 msec ± 1.0%                                        | 3.374 msec ± 1.8%  | 3.348 msec ± 1.7%  |
| 11  | 1650 | 110 | 22           | <span style="color:#00aa00">**1.432 msec ± 1.3%**</span>  | <span style="color:#00aa00">**1.436 msec ± 1.1%**</span> | 4.160 msec ± 2.0%  | 4.167 msec ± 1.6%  |
| 12  | 1800 | 120 | 24           | <span style="color:#00aa00">**1.648 msec ± 1.1%**</span>  | 1.657 msec ± 0.7%                                        | 5.182 msec ± 1.5%  | 5.174 msec ± 1.4%  |
| 13  | 1950 | 130 | 26           | 1.941 msec ± 1.2%                                         | <span style="color:#00aa00">**1.919 msec ± 1.4%**</span> | 6.340 msec ± 1.9%  | 6.331 msec ± 1.3%  |
| 14  | 2100 | 140 | 28           | <span style="color:#00aa00">**2.192 msec ± 1.6%**</span>  | <span style="color:#00aa00">**2.191 msec ± 1.2%**</span> | 7.631 msec ± 1.4%  | 7.630 msec ± 1.7%  |
| 15  | 2250 | 150 | 30           | <span style="color:#00aa00">**2.459 msec ± 1.0%**</span>  | <span style="color:#00aa00">**2.454 msec ± 1.2%**</span> | 9.112 msec ± 1.8%  | 9.104 msec ± 1.7%  |
| 16  | 2400 | 160 | 32           | <span style="color:#00aa00">**2.781 msec ± 2.0%**</span>  | <span style="color:#00aa00">**2.801 msec ± 1.0%**</span> | 10.87 msec ± 4.3%  | 10.95 msec ± 3.4%  |
| 17  | 2550 | 170 | 34           | <span style="color:#00aa00">**3.095 msec ± 2.9%**</span>  | <span style="color:#00aa00">**3.076 msec ± 0.7%**</span> | 16.29 msec ± 19.8% | 12.48 msec ± 20.1% |
| 18  | 2700 | 180 | 36           | <span style="color:#00aa00">**3.487 msec ± 1.3%**</span>  | <span style="color:#00aa00">**3.478 msec ± 1.5%**</span> | 14.33 msec ± 2.3%  | 14.30 msec ± 2.0%  |
| 19  | 2850 | 190 | 38           | <span style="color:#00aa00">**3.911 msec ± 1.9%**</span>  | <span style="color:#00aa00">**3.877 msec ± 1.3%**</span> | 22.88 msec ± 26.6% | 23.17 msec ± 25.6% |
| 20  | 3000 | 200 | 40           | <span style="color:#00aa00">**4.348 msec ± 1.3%**</span>  | <span style="color:#00aa00">**4.315 msec ± 1.0%**</span> | 33.34 msec ± 2.2%  | 33.32 msec ± 1.4%  |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**985.9 μsec ± 2.0%**</span>  | <span style="color:#00aa00">**987.6 μsec ± 1.5%**</span> | 2.833 msec ± 5.7%  | 2.636 msec ± 5.7%  |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `OSR(u)`      | `OSR(nu)`                                            | `OSR(u,con)`                                        | `OSR(nu,con)`                                        |
| --- | ---- | --- | ------------ | ------------- | ---------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------- |
| 1   | 150  | 10  | 2            | 0.127 ± 24.5% | <span style="color:#00aa00">**0.245 ± 16.5%**</span> | 0.115 ± 31.2%                                       | <span style="color:#00aa00">**0.220 ± 22.2%**</span> |
| 2   | 300  | 20  | 4            | 0.373 ± 10.8% | <span style="color:#00aa00">**0.505 ± 7.4%**</span>  | 0.379 ± 5.0%                                        | 0.402 ± 9.1%                                         |
| 3   | 450  | 30  | 6            | 0.647 ± 9.6%  | <span style="color:#00aa00">**0.773 ± 5.8%**</span>  | 0.740 ± 7.6%                                        | <span style="color:#00aa00">**0.747 ± 8.4%**</span>  |
| 4   | 600  | 40  | 8            | 0.945 ± 5.9%  | <span style="color:#00aa00">**1.082 ± 4.4%**</span>  | 0.926 ± 3.9%                                        | 0.963 ± 6.1%                                         |
| 5   | 750  | 50  | 10           | 1.209 ± 4.8%  | <span style="color:#00aa00">**1.328 ± 4.9%**</span>  | 1.236 ± 3.6%                                        | 1.252 ± 3.1%                                         |
| 6   | 900  | 60  | 12           | 1.441 ± 4.0%  | <span style="color:#00aa00">**1.550 ± 3.0%**</span>  | 1.460 ± 3.0%                                        | 1.472 ± 2.9%                                         |
| 7   | 1050 | 70  | 14           | 1.660 ± 2.4%  | <span style="color:#00aa00">**1.751 ± 1.7%**</span>  | <span style="color:#00aa00">**1.748 ± 1.7%**</span> | <span style="color:#00aa00">**1.760 ± 1.1%**</span>  |
| 8   | 1200 | 80  | 16           | 1.910 ± 2.5%  | <span style="color:#00aa00">**2.005 ± 2.8%**</span>  | 1.932 ± 3.2%                                        | 1.957 ± 3.1%                                         |
| 9   | 1350 | 90  | 18           | 2.094 ± 1.5%  | <span style="color:#00aa00">**2.188 ± 2.0%**</span>  | 2.136 ± 1.9%                                        | <span style="color:#00aa00">**2.152 ± 2.5%**</span>  |
| 10  | 1500 | 100 | 20           | 2.303 ± 1.5%  | <span style="color:#00aa00">**2.406 ± 1.5%**</span>  | <span style="color:#00aa00">**2.372 ± 1.9%**</span> | <span style="color:#00aa00">**2.395 ± 1.8%**</span>  |
| 11  | 1650 | 110 | 22           | 2.505 ± 1.5%  | <span style="color:#00aa00">**2.576 ± 1.4%**</span>  | <span style="color:#00aa00">**2.573 ± 1.0%**</span> | <span style="color:#00aa00">**2.572 ± 1.1%**</span>  |
| 12  | 1800 | 120 | 24           | 2.682 ± 1.4%  | <span style="color:#00aa00">**2.782 ± 1.4%**</span>  | 2.683 ± 1.5%                                        | 2.709 ± 1.2%                                         |
| 13  | 1950 | 130 | 26           | 2.832 ± 1.5%  | <span style="color:#00aa00">**2.896 ± 1.3%**</span>  | <span style="color:#00aa00">**2.888 ± 1.5%**</span> | <span style="color:#00aa00">**2.867 ± 1.4%**</span>  |
| 14  | 2100 | 140 | 28           | 2.999 ± 1.3%  | <span style="color:#00aa00">**3.077 ± 1.5%**</span>  | <span style="color:#00aa00">**3.108 ± 1.2%**</span> | <span style="color:#00aa00">**3.126 ± 0.9%**</span>  |
| 15  | 2250 | 150 | 30           | 3.181 ± 1.2%  | 3.258 ± 1.1%                                         | <span style="color:#00aa00">**3.312 ± 1.5%**</span> | <span style="color:#00aa00">**3.329 ± 1.4%**</span>  |
| 16  | 2400 | 160 | 32           | 3.314 ± 0.9%  | 3.412 ± 1.1%                                         | <span style="color:#00aa00">**3.424 ± 1.3%**</span> | <span style="color:#00aa00">**3.441 ± 1.2%**</span>  |
| 17  | 2550 | 170 | 34           | 3.496 ± 0.6%  | <span style="color:#00aa00">**3.573 ± 1.1%**</span>  | <span style="color:#00aa00">**3.600 ± 1.0%**</span> | <span style="color:#00aa00">**3.587 ± 1.1%**</span>  |
| 18  | 2700 | 180 | 36           | 3.630 ± 1.1%  | 3.708 ± 0.8%                                         | <span style="color:#00aa00">**3.762 ± 0.9%**</span> | <span style="color:#00aa00">**3.759 ± 0.8%**</span>  |
| 19  | 2850 | 190 | 38           | 3.788 ± 1.0%  | <span style="color:#00aa00">**3.897 ± 0.9%**</span>  | <span style="color:#00aa00">**3.913 ± 0.7%**</span> | <span style="color:#00aa00">**3.911 ± 0.9%**</span>  |
| 20  | 3000 | 200 | 40           | 3.947 ± 0.8%  | <span style="color:#00aa00">**4.020 ± 0.7%**</span>  | <span style="color:#00aa00">**3.992 ± 0.9%**</span> | <span style="color:#00aa00">**4.005 ± 0.8%**</span>  |
|     |      |     | **Geomean:** | 1.767 ± 4.0%  | <span style="color:#00aa00">**1.942 ± 3.0%**</span>  | 1.807 ± 3.9%                                        | <span style="color:#00aa00">**1.883 ± 3.5%**</span>  |

### Constraint Score

| `d` | `n`  | `k` | `m`       | `OSR(u)`                                            | `OSR(nu)`                                           | `OSR(u,con)`                                        | `OSR(nu,con)`                                       |
| --- | ---- | --- | --------- | --------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- |
| 1   | 150  | 10  | 2         | <span style="color:#00aa00">**0.944 ± 7.4%**</span> | <span style="color:#00aa00">**1.000 ± 5.6%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 300  | 20  | 4         | 0.879 ± 4.3%                                        | 0.939 ± 4.8%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 3   | 450  | 30  | 6         | 0.877 ± 3.1%                                        | 0.890 ± 5.4%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 4   | 600  | 40  | 8         | 0.884 ± 2.6%                                        | 0.891 ± 2.8%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 5   | 750  | 50  | 10        | 0.886 ± 2.5%                                        | 0.905 ± 2.0%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 6   | 900  | 60  | 12        | 0.888 ± 1.4%                                        | 0.893 ± 2.6%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 7   | 1050 | 70  | 14        | 0.884 ± 2.1%                                        | 0.897 ± 1.7%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 8   | 1200 | 80  | 16        | 0.887 ± 1.3%                                        | 0.891 ± 1.8%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 9   | 1350 | 90  | 18        | 0.886 ± 1.4%                                        | 0.896 ± 1.5%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 10  | 1500 | 100 | 20        | 0.883 ± 1.6%                                        | 0.895 ± 1.3%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 11  | 1650 | 110 | 22        | 0.883 ± 1.0%                                        | 0.894 ± 1.0%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 12  | 1800 | 120 | 24        | 0.886 ± 1.0%                                        | 0.894 ± 1.5%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 13  | 1950 | 130 | 26        | 0.887 ± 1.0%                                        | 0.895 ± 0.7%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 14  | 2100 | 140 | 28        | 0.887 ± 1.1%                                        | 0.894 ± 0.6%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 15  | 2250 | 150 | 30        | 0.886 ± 0.6%                                        | 0.895 ± 0.6%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 16  | 2400 | 160 | 32        | 0.888 ± 0.9%                                        | 0.891 ± 0.9%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 17  | 2550 | 170 | 34        | 0.887 ± 0.6%                                        | 0.890 ± 0.8%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 18  | 2700 | 180 | 36        | 0.886 ± 0.4%                                        | 0.887 ± 0.6%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 19  | 2850 | 190 | 38        | 0.887 ± 0.7%                                        | 0.891 ± 0.8%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 20  | 3000 | 200 | 40        | 0.882 ± 0.8%                                        | 0.888 ± 0.7%                                        | <span style="color:#00aa00">**0.999 ± 0.1%**</span> | <span style="color:#00aa00">**0.998 ± 0.1%**</span> |
|     |      |     | **Mean:** | 0.888 ± 1.8%                                        | 0.901 ± 1.9%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |

