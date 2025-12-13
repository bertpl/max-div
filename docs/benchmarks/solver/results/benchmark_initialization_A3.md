Tested Initialization strategies:

 - `OSR(u)`     : InitOneShotRandom(uniform=True, constrained=False)
 - `OSR(nu)`    : InitOneShotRandom(uniform=False, constrained=False)
 - `OSR(u,con)` : InitOneShotRandom(uniform=True, constrained=True)
 - `OSR(nu,con)`: InitOneShotRandom(uniform=False, constrained=True)

### Time Duration

| `d` | `n`  | `k` | `m`          | `OSR(u)`                                                 | `OSR(nu)`                                                | `OSR(u,con)`      | `OSR(nu,con)`     |
| --- | ---- | --- | ------------ | -------------------------------------------------------- | -------------------------------------------------------- | ----------------- | ----------------- |
| 2   | 100  | 10  | 2            | <span style="color:#00aa00">**48.17 μsec ± 2.5%**</span> | 51.52 μsec ± 0.8%                                        | 93.21 μsec ± 6.4% | 55.52 μsec ± 5.2% |
| 2   | 200  | 20  | 4            | <span style="color:#00aa00">**106.5 μsec ± 3.9%**</span> | <span style="color:#00aa00">**107.1 μsec ± 1.3%**</span> | 152.9 μsec ± 4.7% | 117.2 μsec ± 4.5% |
| 2   | 300  | 30  | 6            | <span style="color:#00aa00">**167.3 μsec ± 1.7%**</span> | 171.1 μsec ± 1.7%                                        | 236.7 μsec ± 1.0% | 200.8 μsec ± 1.0% |
| 2   | 400  | 40  | 8            | <span style="color:#00aa00">**235.5 μsec ± 2.7%**</span> | <span style="color:#00aa00">**231.3 μsec ± 3.1%**</span> | 318.7 μsec ± 4.1% | 279.6 μsec ± 3.2% |
| 2   | 500  | 50  | 10           | <span style="color:#00aa00">**308.6 μsec ± 2.9%**</span> | <span style="color:#00aa00">**308.0 μsec ± 3.2%**</span> | 416.3 μsec ± 3.5% | 375.6 μsec ± 2.4% |
| 2   | 600  | 60  | 12           | <span style="color:#00aa00">**389.0 μsec ± 3.1%**</span> | <span style="color:#00aa00">**398.4 μsec ± 2.8%**</span> | 531.4 μsec ± 3.0% | 485.4 μsec ± 3.2% |
| 2   | 700  | 70  | 14           | <span style="color:#00aa00">**473.8 μsec ± 2.9%**</span> | <span style="color:#00aa00">**470.0 μsec ± 2.1%**</span> | 631.1 μsec ± 2.0% | 597.8 μsec ± 1.4% |
| 2   | 800  | 80  | 16           | <span style="color:#00aa00">**560.1 μsec ± 2.2%**</span> | <span style="color:#00aa00">**565.9 μsec ± 3.0%**</span> | 761.7 μsec ± 2.1% | 731.4 μsec ± 1.9% |
| 2   | 900  | 90  | 18           | <span style="color:#00aa00">**667.2 μsec ± 2.2%**</span> | <span style="color:#00aa00">**674.2 μsec ± 1.9%**</span> | 903.0 μsec ± 1.5% | 879.9 μsec ± 2.4% |
| 2   | 1000 | 100 | 20           | <span style="color:#00aa00">**767.4 μsec ± 2.2%**</span> | <span style="color:#00aa00">**776.0 μsec ± 1.2%**</span> | 1.039 msec ± 1.6% | 1.033 msec ± 1.9% |
| 2   | 1100 | 110 | 22           | <span style="color:#00aa00">**913.2 μsec ± 1.9%**</span> | <span style="color:#00aa00">**918.2 μsec ± 1.9%**</span> | 1.247 msec ± 1.8% | 1.210 msec ± 1.5% |
| 2   | 1200 | 120 | 24           | <span style="color:#00aa00">**1.039 msec ± 1.3%**</span> | <span style="color:#00aa00">**1.034 msec ± 1.4%**</span> | 1.426 msec ± 2.0% | 1.396 msec ± 2.4% |
| 2   | 1300 | 130 | 26           | <span style="color:#00aa00">**1.170 msec ± 1.8%**</span> | 1.208 msec ± 2.0%                                        | 1.620 msec ± 1.5% | 1.584 msec ± 1.7% |
| 2   | 1400 | 140 | 28           | <span style="color:#00aa00">**1.295 msec ± 1.4%**</span> | 1.316 msec ± 0.9%                                        | 1.836 msec ± 2.4% | 1.782 msec ± 1.7% |
| 2   | 1500 | 150 | 30           | <span style="color:#00aa00">**1.488 msec ± 1.7%**</span> | 1.517 msec ± 1.4%                                        | 2.048 msec ± 2.1% | 2.016 msec ± 1.8% |
| 2   | 1600 | 160 | 32           | <span style="color:#00aa00">**1.645 msec ± 1.2%**</span> | <span style="color:#00aa00">**1.624 msec ± 1.2%**</span> | 2.274 msec ± 1.8% | 2.265 msec ± 2.0% |
| 2   | 1700 | 170 | 34           | <span style="color:#00aa00">**1.783 msec ± 0.9%**</span> | 1.830 msec ± 2.0%                                        | 2.548 msec ± 1.8% | 2.512 msec ± 1.8% |
| 2   | 1800 | 180 | 36           | <span style="color:#00aa00">**1.959 msec ± 1.4%**</span> | 2.005 msec ± 1.4%                                        | 2.821 msec ± 2.5% | 2.738 msec ± 2.5% |
| 2   | 1900 | 190 | 38           | <span style="color:#00aa00">**2.196 msec ± 1.4%**</span> | <span style="color:#00aa00">**2.224 msec ± 1.9%**</span> | 3.066 msec ± 2.1% | 3.016 msec ± 1.6% |
| 2   | 2000 | 200 | 40           | <span style="color:#00aa00">**2.360 msec ± 1.3%**</span> | <span style="color:#00aa00">**2.352 msec ± 1.0%**</span> | 3.283 msec ± 1.8% | 3.311 msec ± 2.3% |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**654.8 μsec ± 2.0%**</span> | <span style="color:#00aa00">**662.6 μsec ± 1.8%**</span> | 920.6 μsec ± 2.5% | 850.8 μsec ± 2.3% |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `OSR(u)`      | `OSR(nu)`                                            | `OSR(u,con)`  | `OSR(nu,con)`                                        |
| --- | ---- | --- | ------------ | ------------- | ---------------------------------------------------- | ------------- | ---------------------------------------------------- |
| 2   | 100  | 10  | 2            | 0.302 ± 14.8% | <span style="color:#00aa00">**0.376 ± 8.8%**</span>  | 0.314 ± 18.8% | <span style="color:#00aa00">**0.368 ± 13.6%**</span> |
| 2   | 200  | 20  | 4            | 0.216 ± 12.6% | <span style="color:#00aa00">**0.292 ± 10.9%**</span> | 0.214 ± 8.0%  | 0.233 ± 17.6%                                        |
| 2   | 300  | 30  | 6            | 0.163 ± 10.6% | <span style="color:#00aa00">**0.225 ± 5.6%**</span>  | 0.175 ± 9.7%  | 0.199 ± 8.8%                                         |
| 2   | 400  | 40  | 8            | 0.150 ± 11.3% | <span style="color:#00aa00">**0.177 ± 5.5%**</span>  | 0.151 ± 8.7%  | 0.151 ± 6.1%                                         |
| 2   | 500  | 50  | 10           | 0.126 ± 6.7%  | <span style="color:#00aa00">**0.158 ± 7.1%**</span>  | 0.121 ± 8.6%  | 0.129 ± 7.9%                                         |
| 2   | 600  | 60  | 12           | 0.119 ± 6.5%  | <span style="color:#00aa00">**0.142 ± 4.3%**</span>  | 0.120 ± 7.4%  | 0.121 ± 8.0%                                         |
| 2   | 700  | 70  | 14           | 0.108 ± 5.7%  | <span style="color:#00aa00">**0.127 ± 5.7%**</span>  | 0.108 ± 4.9%  | 0.110 ± 5.3%                                         |
| 2   | 800  | 80  | 16           | 0.100 ± 6.7%  | <span style="color:#00aa00">**0.123 ± 5.0%**</span>  | 0.102 ± 5.7%  | 0.106 ± 5.7%                                         |
| 2   | 900  | 90  | 18           | 0.096 ± 5.2%  | <span style="color:#00aa00">**0.110 ± 5.2%**</span>  | 0.095 ± 8.8%  | 0.101 ± 5.6%                                         |
| 2   | 1000 | 100 | 20           | 0.086 ± 6.2%  | <span style="color:#00aa00">**0.109 ± 3.6%**</span>  | 0.093 ± 3.8%  | 0.095 ± 4.0%                                         |
| 2   | 1100 | 110 | 22           | 0.084 ± 3.9%  | <span style="color:#00aa00">**0.104 ± 4.7%**</span>  | 0.086 ± 5.7%  | 0.091 ± 6.4%                                         |
| 2   | 1200 | 120 | 24           | 0.082 ± 3.8%  | <span style="color:#00aa00">**0.105 ± 4.0%**</span>  | 0.084 ± 4.3%  | 0.087 ± 5.0%                                         |
| 2   | 1300 | 130 | 26           | 0.077 ± 5.6%  | <span style="color:#00aa00">**0.098 ± 3.3%**</span>  | 0.078 ± 3.7%  | 0.080 ± 4.8%                                         |
| 2   | 1400 | 140 | 28           | 0.075 ± 6.7%  | <span style="color:#00aa00">**0.095 ± 2.5%**</span>  | 0.078 ± 4.1%  | 0.079 ± 4.9%                                         |
| 2   | 1500 | 150 | 30           | 0.071 ± 3.8%  | <span style="color:#00aa00">**0.089 ± 5.5%**</span>  | 0.077 ± 3.7%  | 0.079 ± 2.8%                                         |
| 2   | 1600 | 160 | 32           | 0.071 ± 3.4%  | <span style="color:#00aa00">**0.086 ± 5.2%**</span>  | 0.072 ± 2.9%  | 0.073 ± 2.8%                                         |
| 2   | 1700 | 170 | 34           | 0.067 ± 3.7%  | <span style="color:#00aa00">**0.085 ± 4.3%**</span>  | 0.070 ± 3.6%  | 0.072 ± 3.2%                                         |
| 2   | 1800 | 180 | 36           | 0.067 ± 3.8%  | <span style="color:#00aa00">**0.083 ± 4.8%**</span>  | 0.071 ± 6.3%  | 0.072 ± 5.1%                                         |
| 2   | 1900 | 190 | 38           | 0.063 ± 6.2%  | <span style="color:#00aa00">**0.080 ± 3.1%**</span>  | 0.065 ± 2.2%  | 0.067 ± 3.4%                                         |
| 2   | 2000 | 200 | 40           | 0.064 ± 3.6%  | <span style="color:#00aa00">**0.076 ± 2.8%**</span>  | 0.064 ± 2.4%  | 0.064 ± 3.2%                                         |
|     |      |     | **Geomean:** | 0.099 ± 6.6%  | <span style="color:#00aa00">**0.123 ± 5.1%**</span>  | 0.101 ± 6.2%  | 0.105 ± 6.2%                                         |

### Constraint Score

| `d` | `n`  | `k` | `m`       | `OSR(u)`                                            | `OSR(nu)`                                           | `OSR(u,con)`                                        | `OSR(nu,con)`                                       |
| --- | ---- | --- | --------- | --------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- |
| 2   | 100  | 10  | 2         | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 5.6%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 200  | 20  | 4         | 0.941 ± 3.1%                                        | <span style="color:#00aa00">**0.941 ± 6.3%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 300  | 30  | 6         | 0.880 ± 4.5%                                        | 0.920 ± 4.9%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 400  | 40  | 8         | 0.909 ± 3.8%                                        | 0.909 ± 5.0%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 500  | 50  | 10        | 0.902 ± 5.4%                                        | 0.902 ± 4.1%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 600  | 60  | 12        | 0.908 ± 4.8%                                        | 0.918 ± 2.5%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 700  | 70  | 14        | 0.895 ± 2.9%                                        | 0.895 ± 2.0%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 800  | 80  | 16        | 0.908 ± 2.5%                                        | 0.900 ± 1.7%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 900  | 90  | 18        | 0.897 ± 1.7%                                        | 0.890 ± 3.1%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 1000 | 100 | 20        | 0.889 ± 1.4%                                        | 0.901 ± 1.4%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 1100 | 110 | 22        | 0.882 ± 2.9%                                        | 0.899 ± 1.9%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 1200 | 120 | 24        | 0.897 ± 2.4%                                        | 0.902 ± 1.7%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 1300 | 130 | 26        | 0.895 ± 2.8%                                        | 0.895 ± 1.3%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 1400 | 140 | 28        | 0.889 ± 2.5%                                        | 0.894 ± 1.5%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 1500 | 150 | 30        | 0.901 ± 1.1%                                        | 0.893 ± 1.6%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 1600 | 160 | 32        | 0.891 ± 1.3%                                        | 0.899 ± 1.7%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 1700 | 170 | 34        | 0.891 ± 1.4%                                        | 0.898 ± 1.7%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 1800 | 180 | 36        | 0.890 ± 1.9%                                        | 0.897 ± 2.0%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 1900 | 190 | 38        | 0.895 ± 1.7%                                        | 0.899 ± 1.1%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 2000 | 200 | 40        | 0.907 ± 2.1%                                        | 0.901 ± 1.7%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
|     |      |     | **Mean:** | 0.903 ± 2.5%                                        | 0.908 ± 2.7%                                        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |

