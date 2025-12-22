Tested Initialization strategies:

 - `ROS(u)`     : InitRandomOneShot(uniform=True, constrained=False)
 - `ROS(nu)`    : InitRandomOneShot(uniform=False, constrained=False)
 - `ROS(u,con)` : InitRandomOneShot(uniform=True, constrained=True)
 - `ROS(nu,con)`: InitRandomOneShot(uniform=False, constrained=True)

### Time Duration

| `d` | `n`  | `k` | `m`          | `ROS(u)`                                                 | `ROS(nu)`                                                | `ROS(u,con)`      | `ROS(nu,con)`      |
| --- | ---- | --- | ------------ | -------------------------------------------------------- | -------------------------------------------------------- | ----------------- | ------------------ |
| 1   | 150  | 10  | 3            | <span style="color:#00aa00">**59.21 μsec ± 8.5%**</span> | <span style="color:#00aa00">**57.02 μsec ± 4.6%**</span> | 107.0 μsec ± 8.5% | 65.33 μsec ± 3.4%  |
| 2   | 300  | 20  | 6            | <span style="color:#00aa00">**130.4 μsec ± 4.5%**</span> | <span style="color:#00aa00">**127.9 μsec ± 2.5%**</span> | 210.1 μsec ± 8.0% | 165.8 μsec ± 7.2%  |
| 3   | 450  | 30  | 9            | <span style="color:#00aa00">**219.7 μsec ± 3.5%**</span> | <span style="color:#00aa00">**220.6 μsec ± 1.1%**</span> | 405.3 μsec ± 2.4% | 363.2 μsec ± 10.3% |
| 4   | 600  | 40  | 12           | <span style="color:#00aa00">**327.9 μsec ± 1.6%**</span> | <span style="color:#00aa00">**331.0 μsec ± 1.6%**</span> | 678.3 μsec ± 1.5% | 630.2 μsec ± 1.7%  |
| 5   | 750  | 50  | 15           | <span style="color:#00aa00">**460.2 μsec ± 1.5%**</span> | <span style="color:#00aa00">**456.0 μsec ± 1.1%**</span> | 1.044 msec ± 2.3% | 986.1 μsec ± 2.1%  |
| 6   | 900  | 60  | 18           | <span style="color:#00aa00">**606.8 μsec ± 1.4%**</span> | <span style="color:#00aa00">**607.3 μsec ± 1.0%**</span> | 1.502 msec ± 1.7% | 1.467 msec ± 1.9%  |
| 7   | 1050 | 70  | 21           | <span style="color:#00aa00">**775.8 μsec ± 1.4%**</span> | <span style="color:#00aa00">**781.0 μsec ± 1.2%**</span> | 2.122 msec ± 1.9% | 2.068 msec ± 2.2%  |
| 8   | 1200 | 80  | 24           | <span style="color:#00aa00">**956.1 μsec ± 1.1%**</span> | 964.4 μsec ± 1.2%                                        | 2.825 msec ± 2.0% | 2.800 msec ± 1.8%  |
| 9   | 1350 | 90  | 27           | <span style="color:#00aa00">**1.215 msec ± 2.3%**</span> | <span style="color:#00aa00">**1.192 msec ± 1.0%**</span> | 3.764 msec ± 1.6% | 3.719 msec ± 1.7%  |
| 10  | 1500 | 100 | 30           | 1.420 msec ± 1.2%                                        | <span style="color:#00aa00">**1.392 msec ± 1.7%**</span> | 4.824 msec ± 2.0% | 5.071 msec ± 4.7%  |
| 11  | 1650 | 110 | 33           | <span style="color:#00aa00">**1.641 msec ± 0.8%**</span> | <span style="color:#00aa00">**1.632 msec ± 1.9%**</span> | 6.307 msec ± 2.0% | 6.545 msec ± 4.1%  |
| 12  | 1800 | 120 | 36           | <span style="color:#00aa00">**2.094 msec ± 9.6%**</span> | <span style="color:#00aa00">**2.100 msec ± 2.3%**</span> | 7.976 msec ± 2.6% | 7.856 msec ± 2.3%  |
| 13  | 1950 | 130 | 39           | 2.411 msec ± 2.5%                                        | <span style="color:#00aa00">**2.291 msec ± 2.7%**</span> | 9.595 msec ± 2.1% | 9.585 msec ± 2.5%  |
| 14  | 2100 | 140 | 42           | <span style="color:#00aa00">**2.550 msec ± 2.1%**</span> | <span style="color:#00aa00">**2.542 msec ± 1.1%**</span> | 11.61 msec ± 1.5% | 11.65 msec ± 1.6%  |
| 15  | 2250 | 150 | 45           | <span style="color:#00aa00">**3.052 msec ± 4.4%**</span> | 3.251 msec ± 3.4%                                        | 14.29 msec ± 2.6% | 14.12 msec ± 2.0%  |
| 16  | 2400 | 160 | 48           | 3.347 msec ± 2.3%                                        | <span style="color:#00aa00">**3.282 msec ± 1.1%**</span> | 16.75 msec ± 1.9% | 16.65 msec ± 2.2%  |
| 17  | 2550 | 170 | 51           | 3.850 msec ± 2.4%                                        | <span style="color:#00aa00">**3.754 msec ± 1.5%**</span> | 19.33 msec ± 2.0% | 19.38 msec ± 1.6%  |
| 18  | 2700 | 180 | 54           | <span style="color:#00aa00">**4.239 msec ± 2.6%**</span> | <span style="color:#00aa00">**4.182 msec ± 1.0%**</span> | 22.66 msec ± 2.1% | 22.57 msec ± 2.2%  |
| 19  | 2850 | 190 | 57           | 4.678 msec ± 1.7%                                        | <span style="color:#00aa00">**4.632 msec ± 1.2%**</span> | 26.66 msec ± 3.4% | 26.58 msec ± 2.2%  |
| 20  | 3000 | 200 | 60           | 5.305 msec ± 0.9%                                        | <span style="color:#00aa00">**5.148 msec ± 1.6%**</span> | 30.59 msec ± 2.3% | 30.92 msec ± 2.4%  |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**1.153 msec ± 2.8%**</span> | <span style="color:#00aa00">**1.144 msec ± 1.7%**</span> | 3.847 msec ± 2.7% | 3.664 msec ± 3.0%  |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `ROS(u)`      | `ROS(nu)`                                            | `ROS(u,con)`  | `ROS(nu,con)` |
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

| `d` | `n`  | `k` | `m`       | `ROS(u)`                                             | `ROS(nu)`     | `ROS(u,con)`                                        | `ROS(nu,con)`                                       |
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

