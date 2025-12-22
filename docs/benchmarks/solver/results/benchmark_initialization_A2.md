Tested Initialization strategies:

 - `ROS(u)` : InitRandomOneShot(uniform=True, constrained=False)
 - `ROS(nu)`: InitRandomOneShot(uniform=False, constrained=False)

### Time Duration

| `d` | `n`  | `k` | `m`          | `ROS(u)`                                                 | `ROS(nu)`                                                |
| --- | ---- | --- | ------------ | -------------------------------------------------------- | -------------------------------------------------------- |
| 1   | 100  | 10  | 0            | <span style="color:#00aa00">**45.79 μsec ± 4.3%**</span> | <span style="color:#00aa00">**45.71 μsec ± 1.6%**</span> |
| 2   | 200  | 20  | 0            | <span style="color:#00aa00">**94.90 μsec ± 1.0%**</span> | 96.54 μsec ± 1.2%                                        |
| 3   | 300  | 30  | 0            | <span style="color:#00aa00">**151.9 μsec ± 3.1%**</span> | 156.6 μsec ± 1.1%                                        |
| 4   | 400  | 40  | 0            | <span style="color:#00aa00">**214.5 μsec ± 0.8%**</span> | 219.1 μsec ± 1.5%                                        |
| 5   | 500  | 50  | 0            | <span style="color:#00aa00">**288.5 μsec ± 0.9%**</span> | 293.3 μsec ± 0.6%                                        |
| 6   | 600  | 60  | 0            | <span style="color:#00aa00">**367.9 μsec ± 1.7%**</span> | <span style="color:#00aa00">**368.1 μsec ± 3.0%**</span> |
| 7   | 700  | 70  | 0            | <span style="color:#00aa00">**454.2 μsec ± 2.5%**</span> | <span style="color:#00aa00">**464.6 μsec ± 2.0%**</span> |
| 8   | 800  | 80  | 0            | <span style="color:#00aa00">**530.2 μsec ± 1.9%**</span> | 541.3 μsec ± 2.0%                                        |
| 9   | 900  | 90  | 0            | <span style="color:#00aa00">**634.6 μsec ± 1.8%**</span> | <span style="color:#00aa00">**628.2 μsec ± 1.7%**</span> |
| 10  | 1000 | 100 | 0            | <span style="color:#00aa00">**737.2 μsec ± 1.6%**</span> | 764.8 μsec ± 4.6%                                        |
| 11  | 1100 | 110 | 0            | 933.1 μsec ± 4.6%                                        | <span style="color:#00aa00">**875.6 μsec ± 2.2%**</span> |
| 12  | 1200 | 120 | 0            | <span style="color:#00aa00">**991.5 μsec ± 1.7%**</span> | <span style="color:#00aa00">**994.0 μsec ± 1.3%**</span> |
| 13  | 1300 | 130 | 0            | <span style="color:#00aa00">**1.131 msec ± 1.4%**</span> | <span style="color:#00aa00">**1.139 msec ± 2.0%**</span> |
| 14  | 1400 | 140 | 0            | <span style="color:#00aa00">**1.272 msec ± 1.8%**</span> | <span style="color:#00aa00">**1.268 msec ± 1.1%**</span> |
| 15  | 1500 | 150 | 0            | <span style="color:#00aa00">**1.387 msec ± 1.8%**</span> | 1.433 msec ± 1.7%                                        |
| 16  | 1600 | 160 | 0            | <span style="color:#00aa00">**1.559 msec ± 1.3%**</span> | <span style="color:#00aa00">**1.573 msec ± 1.4%**</span> |
| 17  | 1700 | 170 | 0            | <span style="color:#00aa00">**1.701 msec ± 2.0%**</span> | 1.731 msec ± 1.2%                                        |
| 18  | 1800 | 180 | 0            | <span style="color:#00aa00">**1.891 msec ± 1.3%**</span> | <span style="color:#00aa00">**1.904 msec ± 1.7%**</span> |
| 19  | 1900 | 190 | 0            | <span style="color:#00aa00">**2.034 msec ± 1.0%**</span> | 2.064 msec ± 1.6%                                        |
| 20  | 2000 | 200 | 0            | <span style="color:#00aa00">**2.278 msec ± 1.7%**</span> | <span style="color:#00aa00">**2.278 msec ± 1.2%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**620.9 μsec ± 1.9%**</span> | <span style="color:#00aa00">**626.5 μsec ± 1.7%**</span> |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `ROS(u)`      | `ROS(nu)`                                            |
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

