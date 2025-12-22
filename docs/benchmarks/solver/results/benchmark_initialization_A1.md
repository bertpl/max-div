Tested Initialization strategies:

 - `ROS(u)` : InitRandomOneShot(uniform=True, constrained=False)
 - `ROS(nu)`: InitRandomOneShot(uniform=False, constrained=False)
 - `RB(2)`  : InitRandomBatched(b=2, constrained=False)
 - `RB(10)` : InitRandomBatched(b=10, constrained=False)

### Time Duration

| `d` | `n`  | `k` | `m`          | `ROS(u)`                                                  | `ROS(nu)`                                                | `RB(2)`           | `RB(10)`          |
| --- | ---- | --- | ------------ | --------------------------------------------------------- | -------------------------------------------------------- | ----------------- | ----------------- |
| 1   | 100  | 10  | 0            | <span style="color:#00aa00">**45.42 μsec ± 3.0%**</span>  | 46.52 μsec ± 2.0%                                        | 70.96 μsec ± 2.0% | 194.5 μsec ± 1.7% |
| 2   | 200  | 20  | 0            | <span style="color:#00aa00">**94.85 μsec ± 1.4%**</span>  | 97.15 μsec ± 3.4%                                        | 132.0 μsec ± 0.8% | 332.1 μsec ± 3.7% |
| 3   | 300  | 30  | 0            | <span style="color:#00aa00">**146.0 μsec ± 3.0%**</span>  | 154.1 μsec ± 1.6%                                        | 205.9 μsec ± 0.9% | 489.2 μsec ± 1.7% |
| 4   | 400  | 40  | 0            | <span style="color:#00aa00">**207.1 μsec ± 6.2%**</span>  | <span style="color:#00aa00">**210.6 μsec ± 4.2%**</span> | 276.6 μsec ± 3.4% | 672.6 μsec ± 0.9% |
| 5   | 500  | 50  | 0            | <span style="color:#00aa00">**287.4 μsec ± 3.3%**</span>  | <span style="color:#00aa00">**277.3 μsec ± 3.3%**</span> | 371.4 μsec ± 0.8% | 834.0 μsec ± 1.0% |
| 6   | 600  | 60  | 0            | <span style="color:#00aa00">**356.9 μsec ± 3.3%**</span>  | <span style="color:#00aa00">**363.2 μsec ± 3.0%**</span> | 461.2 μsec ± 3.0% | 1.012 msec ± 0.6% |
| 7   | 700  | 70  | 0            | <span style="color:#00aa00">**434.4 μsec ± 2.5%**</span>  | 442.5 μsec ± 2.5%                                        | 563.9 μsec ± 3.2% | 1.207 msec ± 6.6% |
| 8   | 800  | 80  | 0            | <span style="color:#00aa00">**535.5 μsec ± 3.4%**</span>  | <span style="color:#00aa00">**551.0 μsec ± 6.2%**</span> | 764.3 μsec ± 4.8% | 1.461 msec ± 5.3% |
| 9   | 900  | 90  | 0            | <span style="color:#00aa00">**707.6 μsec ± 14.0%**</span> | <span style="color:#00aa00">**670.1 μsec ± 3.7%**</span> | 770.4 μsec ± 5.0% | 1.587 msec ± 1.5% |
| 10  | 1000 | 100 | 0            | <span style="color:#00aa00">**725.8 μsec ± 3.5%**</span>  | 750.8 μsec ± 2.0%                                        | 892.2 μsec ± 2.2% | 1.800 msec ± 1.5% |
| 11  | 1100 | 110 | 0            | <span style="color:#00aa00">**885.1 μsec ± 11.3%**</span> | <span style="color:#00aa00">**892.6 μsec ± 4.6%**</span> | 1.055 msec ± 2.5% | 2.036 msec ± 2.6% |
| 12  | 1200 | 120 | 0            | <span style="color:#00aa00">**997.6 μsec ± 2.3%**</span>  | <span style="color:#00aa00">**1.009 msec ± 1.8%**</span> | 1.235 msec ± 6.5% | 2.396 msec ± 7.0% |
| 13  | 1300 | 130 | 0            | <span style="color:#00aa00">**1.167 msec ± 9.0%**</span>  | <span style="color:#00aa00">**1.151 msec ± 3.8%**</span> | 1.367 msec ± 3.8% | 2.565 msec ± 2.6% |
| 14  | 1400 | 140 | 0            | <span style="color:#00aa00">**1.297 msec ± 3.8%**</span>  | <span style="color:#00aa00">**1.330 msec ± 6.0%**</span> | 1.528 msec ± 8.4% | 2.803 msec ± 3.1% |
| 15  | 1500 | 150 | 0            | <span style="color:#00aa00">**1.686 msec ± 8.4%**</span>  | <span style="color:#00aa00">**1.588 msec ± 5.0%**</span> | 1.773 msec ± 8.6% | 3.095 msec ± 5.4% |
| 16  | 1600 | 160 | 0            | <span style="color:#00aa00">**1.602 msec ± 3.0%**</span>  | <span style="color:#00aa00">**1.649 msec ± 4.8%**</span> | 1.854 msec ± 2.2% | 3.342 msec ± 4.7% |
| 17  | 1700 | 170 | 0            | <span style="color:#00aa00">**1.770 msec ± 3.1%**</span>  | 1.867 msec ± 6.5%                                        | 2.101 msec ± 9.3% | 3.549 msec ± 1.8% |
| 18  | 1800 | 180 | 0            | <span style="color:#00aa00">**2.022 msec ± 8.6%**</span>  | <span style="color:#00aa00">**2.001 msec ± 1.9%**</span> | 2.286 msec ± 1.8% | 4.114 msec ± 5.6% |
| 19  | 1900 | 190 | 0            | <span style="color:#00aa00">**2.533 msec ± 10.3%**</span> | <span style="color:#00aa00">**2.414 msec ± 7.1%**</span> | 2.586 msec ± 8.7% | 4.200 msec ± 4.1% |
| 20  | 2000 | 200 | 0            | <span style="color:#00aa00">**2.434 msec ± 8.0%**</span>  | <span style="color:#00aa00">**2.672 msec ± 9.0%**</span> | 2.823 msec ± 4.6% | 4.621 msec ± 4.0% |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**638.6 μsec ± 5.5%**</span>  | <span style="color:#00aa00">**645.5 μsec ± 4.1%**</span> | 787.8 μsec ± 4.1% | 1.579 msec ± 3.3% |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `ROS(u)`      | `ROS(nu)`     | `RB(2)`       | `RB(10)`                                            |
| --- | ---- | --- | ------------ | ------------- | ------------- | ------------- | --------------------------------------------------- |
| 1   | 100  | 10  | 0            | 0.035 ± 30.6% | 0.044 ± 19.6% | 0.037 ± 30.8% | <span style="color:#00aa00">**0.088 ± 4.5%**</span> |
| 2   | 200  | 20  | 0            | 0.109 ± 10.1% | 0.114 ± 6.5%  | 0.115 ± 12.0% | <span style="color:#00aa00">**0.153 ± 8.7%**</span> |
| 3   | 300  | 30  | 0            | 0.190 ± 5.8%  | 0.205 ± 6.2%  | 0.212 ± 4.8%  | <span style="color:#00aa00">**0.238 ± 4.3%**</span> |
| 4   | 400  | 40  | 0            | 0.259 ± 5.1%  | 0.276 ± 4.3%  | 0.289 ± 2.6%  | <span style="color:#00aa00">**0.336 ± 3.2%**</span> |
| 5   | 500  | 50  | 0            | 0.338 ± 4.0%  | 0.356 ± 3.8%  | 0.372 ± 3.4%  | <span style="color:#00aa00">**0.404 ± 2.9%**</span> |
| 6   | 600  | 60  | 0            | 0.412 ± 3.2%  | 0.421 ± 1.9%  | 0.439 ± 2.4%  | <span style="color:#00aa00">**0.478 ± 1.9%**</span> |
| 7   | 700  | 70  | 0            | 0.472 ± 2.4%  | 0.484 ± 2.0%  | 0.514 ± 2.0%  | <span style="color:#00aa00">**0.542 ± 1.7%**</span> |
| 8   | 800  | 80  | 0            | 0.545 ± 2.4%  | 0.553 ± 1.6%  | 0.563 ± 1.2%  | <span style="color:#00aa00">**0.598 ± 1.6%**</span> |
| 9   | 900  | 90  | 0            | 0.606 ± 2.6%  | 0.612 ± 1.8%  | 0.634 ± 1.4%  | <span style="color:#00aa00">**0.668 ± 1.7%**</span> |
| 10  | 1000 | 100 | 0            | 0.657 ± 2.0%  | 0.668 ± 1.8%  | 0.686 ± 0.8%  | <span style="color:#00aa00">**0.722 ± 1.2%**</span> |
| 11  | 1100 | 110 | 0            | 0.719 ± 1.4%  | 0.724 ± 1.5%  | 0.742 ± 1.3%  | <span style="color:#00aa00">**0.777 ± 0.8%**</span> |
| 12  | 1200 | 120 | 0            | 0.777 ± 1.1%  | 0.782 ± 1.2%  | 0.799 ± 1.3%  | <span style="color:#00aa00">**0.835 ± 0.8%**</span> |
| 13  | 1300 | 130 | 0            | 0.826 ± 0.9%  | 0.831 ± 1.0%  | 0.849 ± 0.7%  | <span style="color:#00aa00">**0.886 ± 0.6%**</span> |
| 14  | 1400 | 140 | 0            | 0.881 ± 1.0%  | 0.888 ± 0.9%  | 0.904 ± 1.0%  | <span style="color:#00aa00">**0.929 ± 0.9%**</span> |
| 15  | 1500 | 150 | 0            | 0.928 ± 1.0%  | 0.936 ± 0.6%  | 0.949 ± 0.8%  | <span style="color:#00aa00">**0.978 ± 1.0%**</span> |
| 16  | 1600 | 160 | 0            | 0.971 ± 0.9%  | 0.987 ± 0.8%  | 0.993 ± 0.9%  | <span style="color:#00aa00">**1.024 ± 0.5%**</span> |
| 17  | 1700 | 170 | 0            | 1.023 ± 0.7%  | 1.031 ± 0.7%  | 1.042 ± 0.3%  | <span style="color:#00aa00">**1.072 ± 0.6%**</span> |
| 18  | 1800 | 180 | 0            | 1.062 ± 1.1%  | 1.075 ± 0.9%  | 1.084 ± 0.5%  | <span style="color:#00aa00">**1.114 ± 0.7%**</span> |
| 19  | 1900 | 190 | 0            | 1.108 ± 0.7%  | 1.118 ± 0.6%  | 1.129 ± 0.5%  | <span style="color:#00aa00">**1.159 ± 0.6%**</span> |
| 20  | 2000 | 200 | 0            | 1.156 ± 0.8%  | 1.168 ± 0.7%  | 1.173 ± 0.5%  | <span style="color:#00aa00">**1.201 ± 0.6%**</span> |
|     |      |     | **Geomean:** | 0.509 ± 3.9%  | 0.526 ± 3.0%  | 0.533 ± 3.3%  | <span style="color:#00aa00">**0.593 ± 1.9%**</span> |

