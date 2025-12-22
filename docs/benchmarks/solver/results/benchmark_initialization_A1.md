Tested Initialization strategies:

 - `ROS(u)` : InitRandomOneShot(uniform=True, constrained=False)
 - `ROS(nu)`: InitRandomOneShot(uniform=False, constrained=False)

### Time Duration

| `d` | `n`  | `k` | `m`          | `ROS(u)`                                                 | `ROS(nu)`                                                |
| --- | ---- | --- | ------------ | -------------------------------------------------------- | -------------------------------------------------------- |
| 1   | 100  | 10  | 0            | <span style="color:#00aa00">**45.94 μsec ± 1.7%**</span> | 47.33 μsec ± 2.3%                                        |
| 2   | 200  | 20  | 0            | <span style="color:#00aa00">**97.12 μsec ± 2.7%**</span> | <span style="color:#00aa00">**96.33 μsec ± 1.3%**</span> |
| 3   | 300  | 30  | 0            | <span style="color:#00aa00">**152.5 μsec ± 1.1%**</span> | <span style="color:#00aa00">**154.5 μsec ± 2.7%**</span> |
| 4   | 400  | 40  | 0            | <span style="color:#00aa00">**215.6 μsec ± 1.3%**</span> | 217.9 μsec ± 0.8%                                        |
| 5   | 500  | 50  | 0            | <span style="color:#00aa00">**287.9 μsec ± 1.1%**</span> | 295.0 μsec ± 1.0%                                        |
| 6   | 600  | 60  | 0            | <span style="color:#00aa00">**352.2 μsec ± 3.5%**</span> | 363.6 μsec ± 3.8%                                        |
| 7   | 700  | 70  | 0            | <span style="color:#00aa00">**453.2 μsec ± 3.1%**</span> | <span style="color:#00aa00">**439.1 μsec ± 2.3%**</span> |
| 8   | 800  | 80  | 0            | <span style="color:#00aa00">**526.0 μsec ± 3.6%**</span> | <span style="color:#00aa00">**528.9 μsec ± 2.3%**</span> |
| 9   | 900  | 90  | 0            | <span style="color:#00aa00">**628.6 μsec ± 2.0%**</span> | <span style="color:#00aa00">**627.2 μsec ± 1.4%**</span> |
| 10  | 1000 | 100 | 0            | <span style="color:#00aa00">**730.0 μsec ± 2.0%**</span> | 742.3 μsec ± 1.7%                                        |
| 11  | 1100 | 110 | 0            | <span style="color:#00aa00">**854.5 μsec ± 1.5%**</span> | <span style="color:#00aa00">**862.4 μsec ± 2.3%**</span> |
| 12  | 1200 | 120 | 0            | <span style="color:#00aa00">**985.5 μsec ± 1.4%**</span> | <span style="color:#00aa00">**977.5 μsec ± 1.1%**</span> |
| 13  | 1300 | 130 | 0            | <span style="color:#00aa00">**1.115 msec ± 1.3%**</span> | 1.137 msec ± 1.3%                                        |
| 14  | 1400 | 140 | 0            | <span style="color:#00aa00">**1.248 msec ± 1.5%**</span> | <span style="color:#00aa00">**1.259 msec ± 1.3%**</span> |
| 15  | 1500 | 150 | 0            | <span style="color:#00aa00">**1.414 msec ± 0.8%**</span> | 1.430 msec ± 0.9%                                        |
| 16  | 1600 | 160 | 0            | <span style="color:#00aa00">**1.549 msec ± 0.6%**</span> | <span style="color:#00aa00">**1.550 msec ± 1.0%**</span> |
| 17  | 1700 | 170 | 0            | <span style="color:#00aa00">**1.698 msec ± 2.1%**</span> | 1.719 msec ± 1.1%                                        |
| 18  | 1800 | 180 | 0            | <span style="color:#00aa00">**1.842 msec ± 1.1%**</span> | 1.919 msec ± 1.4%                                        |
| 19  | 1900 | 190 | 0            | <span style="color:#00aa00">**2.138 msec ± 2.7%**</span> | <span style="color:#00aa00">**2.097 msec ± 1.8%**</span> |
| 20  | 2000 | 200 | 0            | 2.286 msec ± 1.0%                                        | <span style="color:#00aa00">**2.254 msec ± 1.4%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**616.9 μsec ± 1.8%**</span> | <span style="color:#00aa00">**621.7 μsec ± 1.7%**</span> |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `ROS(u)`                                             | `ROS(nu)`                                            |
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

