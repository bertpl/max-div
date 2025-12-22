Tested Initialization strategies:

 - `ROS(u)` : InitRandomOneShot(uniform=True, constrained=False)
 - `ROS(nu)`: InitRandomOneShot(uniform=False, constrained=False)
 - `RB(2)`  : InitRandomBatched(b=2, constrained=False)
 - `RB(10)` : InitRandomBatched(b=10, constrained=False)

### Time Duration

| `d` | `n`  | `k` | `m`          | `ROS(u)`                                                  | `ROS(nu)`                                                 | `RB(2)`            | `RB(10)`          |
| --- | ---- | --- | ------------ | --------------------------------------------------------- | --------------------------------------------------------- | ------------------ | ----------------- |
| 1   | 100  | 10  | 0            | 47.65 μsec ± 6.4%                                         | <span style="color:#00aa00">**43.04 μsec ± 2.0%**</span>  | 70.00 μsec ± 2.3%  | 191.6 μsec ± 3.0% |
| 2   | 200  | 20  | 0            | <span style="color:#00aa00">**95.88 μsec ± 2.0%**</span>  | <span style="color:#00aa00">**96.81 μsec ± 2.4%**</span>  | 131.4 μsec ± 3.5%  | 337.6 μsec ± 1.8% |
| 3   | 300  | 30  | 0            | <span style="color:#00aa00">**147.7 μsec ± 2.9%**</span>  | 155.8 μsec ± 2.9%                                         | 205.6 μsec ± 2.4%  | 488.7 μsec ± 2.7% |
| 4   | 400  | 40  | 0            | <span style="color:#00aa00">**214.7 μsec ± 3.1%**</span>  | 225.3 μsec ± 2.3%                                         | 267.9 μsec ± 5.3%  | 660.8 μsec ± 3.9% |
| 5   | 500  | 50  | 0            | <span style="color:#00aa00">**290.0 μsec ± 4.0%**</span>  | <span style="color:#00aa00">**286.5 μsec ± 2.8%**</span>  | 364.4 μsec ± 3.7%  | 835.6 μsec ± 1.0% |
| 6   | 600  | 60  | 0            | <span style="color:#00aa00">**363.4 μsec ± 5.6%**</span>  | 398.6 μsec ± 4.5%                                         | 499.8 μsec ± 6.9%  | 1.041 msec ± 2.9% |
| 7   | 700  | 70  | 0            | <span style="color:#00aa00">**460.5 μsec ± 6.8%**</span>  | <span style="color:#00aa00">**456.8 μsec ± 4.9%**</span>  | 561.5 μsec ± 4.5%  | 1.204 msec ± 3.0% |
| 8   | 800  | 80  | 0            | <span style="color:#00aa00">**526.5 μsec ± 3.8%**</span>  | <span style="color:#00aa00">**533.8 μsec ± 5.1%**</span>  | 657.2 μsec ± 3.5%  | 1.403 msec ± 2.4% |
| 9   | 900  | 90  | 0            | <span style="color:#00aa00">**650.2 μsec ± 3.3%**</span>  | <span style="color:#00aa00">**643.2 μsec ± 1.7%**</span>  | 788.0 μsec ± 4.1%  | 1.610 msec ± 1.4% |
| 10  | 1000 | 100 | 0            | <span style="color:#00aa00">**735.1 μsec ± 5.0%**</span>  | 760.0 μsec ± 3.2%                                         | 919.4 μsec ± 3.4%  | 1.857 msec ± 2.4% |
| 11  | 1100 | 110 | 0            | <span style="color:#00aa00">**862.8 μsec ± 4.0%**</span>  | 894.0 μsec ± 2.7%                                         | 1.044 msec ± 3.3%  | 2.153 msec ± 9.5% |
| 12  | 1200 | 120 | 0            | <span style="color:#00aa00">**1.061 msec ± 10.5%**</span> | <span style="color:#00aa00">**1.022 msec ± 2.7%**</span>  | 1.200 msec ± 2.4%  | 2.297 msec ± 3.4% |
| 13  | 1300 | 130 | 0            | <span style="color:#00aa00">**1.160 msec ± 1.8%**</span>  | <span style="color:#00aa00">**1.172 msec ± 2.8%**</span>  | 1.396 msec ± 3.7%  | 2.580 msec ± 2.7% |
| 14  | 1400 | 140 | 0            | <span style="color:#00aa00">**1.355 msec ± 6.2%**</span>  | <span style="color:#00aa00">**1.383 msec ± 11.6%**</span> | 1.597 msec ± 4.2%  | 2.830 msec ± 1.8% |
| 15  | 1500 | 150 | 0            | <span style="color:#00aa00">**1.489 msec ± 2.3%**</span>  | 1.550 msec ± 9.0%                                         | 1.928 msec ± 11.2% | 3.116 msec ± 4.3% |
| 16  | 1600 | 160 | 0            | <span style="color:#00aa00">**1.734 msec ± 8.6%**</span>  | <span style="color:#00aa00">**1.684 msec ± 4.0%**</span>  | 1.917 msec ± 3.4%  | 3.358 msec ± 2.9% |
| 17  | 1700 | 170 | 0            | <span style="color:#00aa00">**1.919 msec ± 9.5%**</span>  | <span style="color:#00aa00">**1.960 msec ± 8.9%**</span>  | 2.131 msec ± 6.9%  | 3.664 msec ± 2.3% |
| 18  | 1800 | 180 | 0            | <span style="color:#00aa00">**2.027 msec ± 6.2%**</span>  | <span style="color:#00aa00">**2.094 msec ± 6.0%**</span>  | 2.571 msec ± 8.4%  | 3.941 msec ± 2.4% |
| 19  | 1900 | 190 | 0            | <span style="color:#00aa00">**2.247 msec ± 3.8%**</span>  | <span style="color:#00aa00">**2.295 msec ± 7.3%**</span>  | 2.549 msec ± 3.5%  | 4.477 msec ± 5.9% |
| 20  | 2000 | 200 | 0            | <span style="color:#00aa00">**2.465 msec ± 8.1%**</span>  | <span style="color:#00aa00">**2.517 msec ± 7.7%**</span>  | 2.820 msec ± 4.1%  | 4.940 msec ± 5.8% |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**642.0 μsec ± 5.2%**</span>  | <span style="color:#00aa00">**650.2 μsec ± 4.7%**</span>  | 794.5 μsec ± 4.6%  | 1.593 msec ± 3.3% |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `ROS(u)`      | `ROS(nu)`     | `RB(2)`       | `RB(10)`                                            |
| --- | ---- | --- | ------------ | ------------- | ------------- | ------------- | --------------------------------------------------- |
| 1   | 100  | 10  | 0            | 0.113 ± 37.2% | 0.227 ± 19.3% | 0.241 ± 25.9% | <span style="color:#00aa00">**0.426 ± 3.8%**</span> |
| 2   | 200  | 20  | 0            | 0.359 ± 13.6% | 0.502 ± 8.6%  | 0.648 ± 6.5%  | <span style="color:#00aa00">**0.742 ± 8.5%**</span> |
| 3   | 300  | 30  | 0            | 0.689 ± 7.8%  | 0.797 ± 6.5%  | 0.900 ± 9.0%  | <span style="color:#00aa00">**1.122 ± 5.4%**</span> |
| 4   | 400  | 40  | 0            | 0.954 ± 3.5%  | 1.090 ± 3.9%  | 1.223 ± 4.5%  | <span style="color:#00aa00">**1.407 ± 4.3%**</span> |
| 5   | 500  | 50  | 0            | 1.182 ± 4.1%  | 1.319 ± 3.5%  | 1.429 ± 4.3%  | <span style="color:#00aa00">**1.580 ± 2.4%**</span> |
| 6   | 600  | 60  | 0            | 1.445 ± 2.0%  | 1.544 ± 2.4%  | 1.674 ± 2.2%  | <span style="color:#00aa00">**1.857 ± 3.0%**</span> |
| 7   | 700  | 70  | 0            | 1.653 ± 2.9%  | 1.739 ± 2.0%  | 1.892 ± 2.4%  | <span style="color:#00aa00">**2.054 ± 2.0%**</span> |
| 8   | 800  | 80  | 0            | 1.895 ± 2.3%  | 1.974 ± 1.8%  | 2.097 ± 1.7%  | <span style="color:#00aa00">**2.271 ± 1.8%**</span> |
| 9   | 900  | 90  | 0            | 2.096 ± 1.9%  | 2.191 ± 1.5%  | 2.311 ± 2.0%  | <span style="color:#00aa00">**2.490 ± 1.5%**</span> |
| 10  | 1000 | 100 | 0            | 2.292 ± 2.0%  | 2.419 ± 2.2%  | 2.508 ± 1.4%  | <span style="color:#00aa00">**2.684 ± 1.6%**</span> |
| 11  | 1100 | 110 | 0            | 2.476 ± 1.5%  | 2.562 ± 1.8%  | 2.687 ± 1.4%  | <span style="color:#00aa00">**2.865 ± 1.2%**</span> |
| 12  | 1200 | 120 | 0            | 2.644 ± 1.6%  | 2.759 ± 1.2%  | 2.844 ± 1.2%  | <span style="color:#00aa00">**3.015 ± 1.0%**</span> |
| 13  | 1300 | 130 | 0            | 2.840 ± 0.6%  | 2.911 ± 1.6%  | 3.019 ± 1.1%  | <span style="color:#00aa00">**3.223 ± 1.0%**</span> |
| 14  | 1400 | 140 | 0            | 3.023 ± 1.4%  | 3.099 ± 1.0%  | 3.223 ± 1.2%  | <span style="color:#00aa00">**3.366 ± 1.6%**</span> |
| 15  | 1500 | 150 | 0            | 3.187 ± 1.1%  | 3.264 ± 1.0%  | 3.362 ± 1.0%  | <span style="color:#00aa00">**3.538 ± 0.9%**</span> |
| 16  | 1600 | 160 | 0            | 3.326 ± 1.0%  | 3.404 ± 1.1%  | 3.482 ± 1.0%  | <span style="color:#00aa00">**3.665 ± 0.7%**</span> |
| 17  | 1700 | 170 | 0            | 3.507 ± 0.8%  | 3.547 ± 1.1%  | 3.638 ± 0.7%  | <span style="color:#00aa00">**3.797 ± 1.1%**</span> |
| 18  | 1800 | 180 | 0            | 3.654 ± 1.2%  | 3.723 ± 0.8%  | 3.795 ± 1.0%  | <span style="color:#00aa00">**3.954 ± 0.6%**</span> |
| 19  | 1900 | 190 | 0            | 3.793 ± 0.9%  | 3.868 ± 1.0%  | 3.952 ± 0.8%  | <span style="color:#00aa00">**4.095 ± 0.7%**</span> |
| 20  | 2000 | 200 | 0            | 3.959 ± 0.9%  | 4.027 ± 0.8%  | 4.100 ± 0.8%  | <span style="color:#00aa00">**4.226 ± 0.5%**</span> |
|     |      |     | **Geomean:** | 1.757 ± 4.5%  | 1.934 ± 3.2%  | 2.057 ± 3.6%  | <span style="color:#00aa00">**2.280 ± 2.2%**</span> |

