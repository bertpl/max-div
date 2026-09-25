Tested Initialization strategies:

| `name`                 | `class`             | `params`                            | Note                                                                  |
| ---------------------- | ------------------- | ----------------------------------- | --------------------------------------------------------------------- |
| `rsel`                 | InitRandomSelection | ignore_constraints=False            |                                                                       |
| `fps(1)`               | InitFarthestPoint   | top_k=1                             |                                                                       |
| `fps(8)`               | InitFarthestPoint   | top_k=8                             | = the SMART/THOROUGH presets' initialization (unconstrained problems) |
| `fps(8,one-at-a-time)` | InitFarthestPoint   | top_k=8<br>candidate_pool_size=None |                                                                       |

### Time Duration

| `d` | `n`   | `k`  | `m`          | `rsel`                                                    | `fps(1)`                                                 | `fps(8)`          | `fps(8,one-at-a-time)`                                   |
| --- | ----- | ---- | ------------ | --------------------------------------------------------- | -------------------------------------------------------- | ----------------- | -------------------------------------------------------- |
| 1   | 100   | 10   | 0            | <span style="color:#00aa00">**18.29 μsec ± 12.2%**</span> | 39.85 μsec ± 4.4%                                        | 43.31 μsec ± 9.5% | <span style="color:#dd0000">**157.6 μsec ± 6.3%**</span> |
| 2   | 200   | 20   | 0            | <span style="color:#00aa00">**24.65 μsec ± 5.8%**</span>  | 54.19 μsec ± 4.6%                                        | 70.54 μsec ± 4.1% | <span style="color:#dd0000">**320.8 μsec ± 3.9%**</span> |
| 5   | 500   | 50   | 0            | <span style="color:#00aa00">**54.79 μsec ± 9.1%**</span>  | 156.1 μsec ± 5.9%                                        | 220.8 μsec ± 5.4% | <span style="color:#dd0000">**919.6 μsec ± 2.6%**</span> |
| 10  | 1000  | 100  | 0            | <span style="color:#00aa00">**162.8 μsec ± 4.2%**</span>  | 329.0 μsec ± 5.9%                                        | 456.4 μsec ± 6.8% | <span style="color:#dd0000">**2.044 msec ± 0.9%**</span> |
| 20  | 2000  | 200  | 0            | <span style="color:#00aa00">**524.8 μsec ± 1.3%**</span>  | 808.1 μsec ± 1.8%                                        | 1.089 msec ± 2.8% | <span style="color:#dd0000">**4.916 msec ± 0.9%**</span> |
| 50  | 5000  | 500  | 0            | <span style="color:#00aa00">**2.760 msec ± 1.4%**</span>  | 3.724 msec ± 1.4%                                        | 4.471 msec ± 2.8% | <span style="color:#dd0000">**17.23 msec ± 0.7%**</span> |
| 100 | 10000 | 1000 | 0            | <span style="color:#00aa00">**11.53 msec ± 1.1%**</span>  | 12.27 msec ± 1.9%                                        | 13.18 msec ± 1.2% | <span style="color:#dd0000">**55.54 msec ± 8.8%**</span> |
| 200 | 20000 | 2000 | 0            | 47.88 msec ± 0.2%                                         | <span style="color:#00aa00">**37.49 msec ± 0.4%**</span> | 39.41 msec ± 0.2% | <span style="color:#dd0000">**185.8 msec ± 4.9%**</span> |
|     |       |      | **Geomean:** | <span style="color:#00aa00">**487.9 μsec ± 4.4%**</span>  | 791.2 μsec ± 3.3%                                        | 969.3 μsec ± 4.1% | <span style="color:#dd0000">**4.121 msec ± 3.6%**</span> |

### Diversity Score

| `d` | `n`   | `k`  | `m`          | `rsel`                                               | `fps(1)`                                            | `fps(8)`                                             | `fps(8,one-at-a-time)`                               |
| --- | ----- | ---- | ------------ | ---------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| 1   | 100   | 10   | 0            | <span style="color:#dd0000">**0.139 ± 26.4%**</span> | <span style="color:#00aa00">**0.451 ± 2.9%**</span> | 0.396 ± 6.0%                                         | 0.388 ± 5.8%                                         |
| 2   | 200   | 20   | 0            | <span style="color:#dd0000">**0.391 ± 9.2%**</span>  | <span style="color:#00aa00">**1.013 ± 1.7%**</span> | 0.966 ± 1.4%                                         | 0.962 ± 3.2%                                         |
| 5   | 500   | 50   | 0            | <span style="color:#dd0000">**1.167 ± 4.0%**</span>  | <span style="color:#00aa00">**2.244 ± 1.0%**</span> | 2.181 ± 0.5%                                         | 2.186 ± 0.8%                                         |
| 10  | 1000  | 100  | 0            | <span style="color:#dd0000">**2.282 ± 1.8%**</span>  | <span style="color:#00aa00">**3.674 ± 0.4%**</span> | 3.653 ± 0.3%                                         | 3.649 ± 0.4%                                         |
| 20  | 2000  | 200  | 0            | <span style="color:#dd0000">**3.963 ± 1.4%**</span>  | <span style="color:#00aa00">**5.464 ± 0.1%**</span> | 5.451 ± 0.2%                                         | 5.446 ± 0.3%                                         |
| 50  | 5000  | 500  | 0            | <span style="color:#dd0000">**7.389 ± 0.6%**</span>  | <span style="color:#00aa00">**9.148 ± 0.0%**</span> | <span style="color:#00aa00">**9.150 ± 0.1%**</span>  | 9.146 ± 0.1%                                         |
| 100 | 10000 | 1000 | 0            | <span style="color:#dd0000">**11.366 ± 0.1%**</span> | 13.249 ± 0.0%                                       | <span style="color:#00aa00">**13.254 ± 0.0%**</span> | <span style="color:#00aa00">**13.254 ± 0.0%**</span> |
| 200 | 20000 | 2000 | 0            | <span style="color:#dd0000">**17.072 ± 0.1%**</span> | 19.036 ± 0.0%                                       | <span style="color:#00aa00">**19.037 ± 0.0%**</span> | 19.035 ± 0.0%                                        |
|     |       |      | **Geomean:** | <span style="color:#dd0000">**2.313 ± 5.7%**</span>  | <span style="color:#00aa00">**3.842 ± 0.8%**</span> | 3.741 ± 1.1%                                         | 3.729 ± 1.3%                                         |
