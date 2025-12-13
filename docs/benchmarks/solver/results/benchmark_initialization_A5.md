Tested Initialization strategies:

 - `OSR(u)`     : InitOneShotRandom(uniform=True, constrained=False)
 - `OSR(nu)`    : InitOneShotRandom(uniform=False, constrained=False)
 - `OSR(u,con)` : InitOneShotRandom(uniform=True, constrained=True)
 - `OSR(nu,con)`: InitOneShotRandom(uniform=False, constrained=True)

### Time Duration

| `d` | `n`  | `k` | `m`          | `OSR(u)`                                                 | `OSR(nu)`                                                | `OSR(u,con)`      | `OSR(nu,con)`     |
| --- | ---- | --- | ------------ | -------------------------------------------------------- | -------------------------------------------------------- | ----------------- | ----------------- |
| 1   | 150  | 10  | 3            | <span style="color:#00aa00">**53.90 μsec ± 6.4%**</span> | <span style="color:#00aa00">**53.02 μsec ± 3.1%**</span> | 102.4 μsec ± 2.4% | 64.48 μsec ± 1.5% |
| 2   | 300  | 20  | 6            | <span style="color:#00aa00">**126.4 μsec ± 1.8%**</span> | <span style="color:#00aa00">**126.7 μsec ± 1.0%**</span> | 198.0 μsec ± 1.9% | 160.7 μsec ± 1.8% |
| 3   | 450  | 30  | 9            | <span style="color:#00aa00">**215.4 μsec ± 1.2%**</span> | <span style="color:#00aa00">**216.7 μsec ± 1.2%**</span> | 407.1 μsec ± 1.9% | 363.5 μsec ± 8.7% |
| 4   | 600  | 40  | 12           | <span style="color:#00aa00">**326.0 μsec ± 0.9%**</span> | <span style="color:#00aa00">**327.6 μsec ± 0.9%**</span> | 666.5 μsec ± 1.5% | 632.4 μsec ± 1.7% |
| 5   | 750  | 50  | 15           | <span style="color:#00aa00">**451.5 μsec ± 1.8%**</span> | 458.6 μsec ± 1.2%                                        | 1.027 msec ± 2.0% | 993.6 μsec ± 2.5% |
| 6   | 900  | 60  | 18           | <span style="color:#00aa00">**594.4 μsec ± 1.1%**</span> | 598.6 μsec ± 1.1%                                        | 1.515 msec ± 1.6% | 1.482 msec ± 1.4% |
| 7   | 1050 | 70  | 21           | <span style="color:#00aa00">**783.0 μsec ± 1.2%**</span> | <span style="color:#00aa00">**791.0 μsec ± 2.0%**</span> | 2.125 msec ± 2.3% | 2.098 msec ± 2.7% |
| 8   | 1200 | 80  | 24           | <span style="color:#00aa00">**968.8 μsec ± 2.6%**</span> | 988.2 μsec ± 1.7%                                        | 2.881 msec ± 1.5% | 2.839 msec ± 1.5% |
| 9   | 1350 | 90  | 27           | 1.207 msec ± 1.9%                                        | <span style="color:#00aa00">**1.180 msec ± 1.6%**</span> | 3.862 msec ± 1.9% | 3.771 msec ± 1.7% |
| 10  | 1500 | 100 | 30           | <span style="color:#00aa00">**1.440 msec ± 2.0%**</span> | <span style="color:#00aa00">**1.424 msec ± 1.1%**</span> | 4.965 msec ± 1.8% | 4.840 msec ± 2.1% |
| 11  | 1650 | 110 | 33           | <span style="color:#00aa00">**1.661 msec ± 0.9%**</span> | <span style="color:#00aa00">**1.671 msec ± 0.8%**</span> | 6.285 msec ± 2.0% | 6.210 msec ± 2.4% |
| 12  | 1800 | 120 | 36           | <span style="color:#00aa00">**1.917 msec ± 1.0%**</span> | <span style="color:#00aa00">**1.916 msec ± 0.7%**</span> | 7.865 msec ± 2.2% | 7.854 msec ± 2.4% |
| 13  | 1950 | 130 | 39           | 2.239 msec ± 1.0%                                        | <span style="color:#00aa00">**2.218 msec ± 1.4%**</span> | 9.714 msec ± 2.3% | 9.702 msec ± 2.5% |
| 14  | 2100 | 140 | 42           | <span style="color:#00aa00">**2.563 msec ± 2.8%**</span> | <span style="color:#00aa00">**2.569 msec ± 1.7%**</span> | 11.74 msec ± 1.8% | 11.71 msec ± 1.7% |
| 15  | 2250 | 150 | 45           | 2.946 msec ± 3.9%                                        | <span style="color:#00aa00">**2.883 msec ± 2.0%**</span> | 13.95 msec ± 1.9% | 13.84 msec ± 1.9% |
| 16  | 2400 | 160 | 48           | <span style="color:#00aa00">**3.282 msec ± 1.5%**</span> | <span style="color:#00aa00">**3.277 msec ± 1.8%**</span> | 16.66 msec ± 1.9% | 16.56 msec ± 1.9% |
| 17  | 2550 | 170 | 51           | 3.753 msec ± 1.8%                                        | <span style="color:#00aa00">**3.685 msec ± 1.5%**</span> | 19.37 msec ± 2.4% | 19.34 msec ± 2.4% |
| 18  | 2700 | 180 | 54           | <span style="color:#00aa00">**4.170 msec ± 2.3%**</span> | <span style="color:#00aa00">**4.208 msec ± 1.8%**</span> | 22.93 msec ± 2.2% | 22.82 msec ± 2.3% |
| 19  | 2850 | 190 | 57           | <span style="color:#00aa00">**4.724 msec ± 1.2%**</span> | <span style="color:#00aa00">**4.672 msec ± 1.6%**</span> | 26.82 msec ± 2.9% | 26.60 msec ± 2.1% |
| 20  | 3000 | 200 | 60           | <span style="color:#00aa00">**5.188 msec ± 1.5%**</span> | <span style="color:#00aa00">**5.177 msec ± 1.2%**</span> | 30.75 msec ± 2.5% | 30.88 msec ± 2.4% |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**1.130 msec ± 1.9%**</span> | <span style="color:#00aa00">**1.128 msec ± 1.5%**</span> | 3.839 msec ± 2.1% | 3.649 msec ± 2.4% |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `OSR(u)`      | `OSR(nu)`                                            | `OSR(u,con)`  | `OSR(nu,con)` |
| --- | ---- | --- | ------------ | ------------- | ---------------------------------------------------- | ------------- | ------------- |
| 1   | 150  | 10  | 3            | 0.127 ± 24.5% | <span style="color:#00aa00">**0.245 ± 16.5%**</span> | 0.106 ± 39.2% | 0.132 ± 31.1% |
| 2   | 300  | 20  | 6            | 0.373 ± 10.8% | <span style="color:#00aa00">**0.505 ± 7.4%**</span>  | 0.343 ± 10.9% | 0.348 ± 7.7%  |
| 3   | 450  | 30  | 9            | 0.647 ± 9.6%  | <span style="color:#00aa00">**0.773 ± 5.8%**</span>  | 0.511 ± 11.9% | 0.547 ± 16.3% |
| 4   | 600  | 40  | 12           | 0.945 ± 5.9%  | <span style="color:#00aa00">**1.082 ± 4.4%**</span>  | 0.613 ± 4.7%  | 0.634 ± 2.5%  |
| 5   | 750  | 50  | 15           | 1.209 ± 4.8%  | <span style="color:#00aa00">**1.328 ± 4.9%**</span>  | 0.829 ± 1.4%  | 0.838 ± 2.7%  |
| 6   | 900  | 60  | 18           | 1.441 ± 4.0%  | <span style="color:#00aa00">**1.550 ± 3.0%**</span>  | 0.993 ± 2.0%  | 0.995 ± 1.9%  |
| 7   | 1050 | 70  | 21           | 1.660 ± 2.4%  | <span style="color:#00aa00">**1.751 ± 1.7%**</span>  | 1.222 ± 1.3%  | 1.231 ± 1.5%  |
| 8   | 1200 | 80  | 24           | 1.910 ± 2.5%  | <span style="color:#00aa00">**2.005 ± 2.8%**</span>  | 1.460 ± 2.7%  | 1.470 ± 1.7%  |
| 9   | 1350 | 90  | 27           | 2.094 ± 1.5%  | <span style="color:#00aa00">**2.188 ± 2.0%**</span>  | 1.641 ± 1.2%  | 1.643 ± 0.7%  |
| 10  | 1500 | 100 | 30           | 2.303 ± 1.5%  | <span style="color:#00aa00">**2.406 ± 1.5%**</span>  | 1.875 ± 1.1%  | 1.876 ± 0.8%  |
| 11  | 1650 | 110 | 33           | 2.505 ± 1.5%  | <span style="color:#00aa00">**2.576 ± 1.4%**</span>  | 2.066 ± 1.4%  | 2.087 ± 1.0%  |
| 12  | 1800 | 120 | 36           | 2.682 ± 1.4%  | <span style="color:#00aa00">**2.782 ± 1.4%**</span>  | 2.179 ± 1.2%  | 2.187 ± 0.6%  |
| 13  | 1950 | 130 | 39           | 2.832 ± 1.5%  | <span style="color:#00aa00">**2.896 ± 1.3%**</span>  | 2.309 ± 0.7%  | 2.313 ± 0.8%  |
| 14  | 2100 | 140 | 42           | 2.999 ± 1.3%  | <span style="color:#00aa00">**3.077 ± 1.5%**</span>  | 2.499 ± 0.7%  | 2.504 ± 0.5%  |
| 15  | 2250 | 150 | 45           | 3.181 ± 1.2%  | <span style="color:#00aa00">**3.258 ± 1.1%**</span>  | 2.692 ± 0.5%  | 2.695 ± 0.4%  |
| 16  | 2400 | 160 | 48           | 3.314 ± 0.9%  | <span style="color:#00aa00">**3.412 ± 1.1%**</span>  | 2.819 ± 0.6%  | 2.820 ± 0.6%  |
| 17  | 2550 | 170 | 51           | 3.496 ± 0.6%  | <span style="color:#00aa00">**3.573 ± 1.1%**</span>  | 3.010 ± 0.4%  | 3.007 ± 0.7%  |
| 18  | 2700 | 180 | 54           | 3.630 ± 1.1%  | <span style="color:#00aa00">**3.708 ± 0.8%**</span>  | 3.130 ± 0.6%  | 3.133 ± 0.5%  |
| 19  | 2850 | 190 | 57           | 3.788 ± 1.0%  | <span style="color:#00aa00">**3.897 ± 0.9%**</span>  | 3.255 ± 0.4%  | 3.267 ± 0.5%  |
| 20  | 3000 | 200 | 60           | 3.947 ± 0.8%  | <span style="color:#00aa00">**4.020 ± 0.7%**</span>  | 3.398 ± 0.3%  | 3.408 ± 0.4%  |
|     |      |     | **Geomean:** | 1.767 ± 4.0%  | <span style="color:#00aa00">**1.942 ± 3.0%**</span>  | 1.416 ± 4.1%  | 1.444 ± 3.5%  |

### Constraint Score

| `d` | `n`  | `k` | `m`       | `OSR(u)`                                             | `OSR(nu)`     | `OSR(u,con)`                                        | `OSR(nu,con)`                                       |
| --- | ---- | --- | --------- | ---------------------------------------------------- | ------------- | --------------------------------------------------- | --------------------------------------------------- |
| 1   | 150  | 10  | 3         | <span style="color:#00aa00">**0.938 ± 10.0%**</span> | 0.812 ± 11.5% | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 300  | 20  | 6         | 0.893 ± 5.7%                                         | 0.852 ± 5.3%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 3   | 450  | 30  | 9         | 0.890 ± 3.5%                                         | 0.857 ± 4.9%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 4   | 600  | 40  | 12        | 0.892 ± 2.2%                                         | 0.861 ± 2.9%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 5   | 750  | 50  | 15        | 0.888 ± 1.8%                                         | 0.868 ± 1.6%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 6   | 900  | 60  | 18        | 0.886 ± 1.4%                                         | 0.880 ± 1.5%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 7   | 1050 | 70  | 21        | 0.887 ± 1.9%                                         | 0.881 ± 1.1%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 8   | 1200 | 80  | 24        | 0.891 ± 1.2%                                         | 0.873 ± 1.1%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 9   | 1350 | 90  | 27        | 0.895 ± 1.1%                                         | 0.891 ± 1.1%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 10  | 1500 | 100 | 30        | 0.890 ± 1.4%                                         | 0.885 ± 0.8%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 11  | 1650 | 110 | 33        | 0.888 ± 0.8%                                         | 0.885 ± 1.2%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 12  | 1800 | 120 | 36        | 0.886 ± 1.1%                                         | 0.884 ± 1.1%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 13  | 1950 | 130 | 39        | 0.892 ± 0.8%                                         | 0.891 ± 0.9%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 14  | 2100 | 140 | 42        | 0.892 ± 0.8%                                         | 0.886 ± 0.5%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 15  | 2250 | 150 | 45        | 0.888 ± 1.0%                                         | 0.888 ± 1.0%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 16  | 2400 | 160 | 48        | 0.892 ± 0.8%                                         | 0.890 ± 0.6%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 17  | 2550 | 170 | 51        | 0.891 ± 0.5%                                         | 0.890 ± 0.7%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 18  | 2700 | 180 | 54        | 0.894 ± 0.6%                                         | 0.890 ± 0.4%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 19  | 2850 | 190 | 57        | 0.892 ± 0.9%                                         | 0.887 ± 0.6%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 20  | 3000 | 200 | 60        | 0.887 ± 0.3%                                         | 0.885 ± 0.6%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
|     |      |     | **Mean:** | 0.893 ± 1.9%                                         | 0.877 ± 1.9%  | <span style="color:#00aa00">**1.000 ± 0.0%**</span> | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |

