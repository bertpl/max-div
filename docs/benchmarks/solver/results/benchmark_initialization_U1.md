Tested Initialization strategies:

| `name`                 | `class`             | `params`                            | Note                                                                  |
| ---------------------- | ------------------- | ----------------------------------- | --------------------------------------------------------------------- |
| `rsel`                 | InitRandomSelection | ignore_constraints=False            |                                                                       |
| `fps(1)`               | InitFarthestPoint   | top_k=1<br>candidate_pool_size=256  |                                                                       |
| `fps(8)`               | InitFarthestPoint   | top_k=8<br>candidate_pool_size=256  | = the SMART/THOROUGH presets' initialization (unconstrained problems) |
| `fps(8,one-at-a-time)` | InitFarthestPoint   | top_k=8<br>candidate_pool_size=None |                                                                       |

### Time Duration

| `d` | `n`   | `k`  | `m`          | `rsel`                                                   | `fps(1)`                                                 | `fps(8)`          | `fps(8,one-at-a-time)`                                   |
| --- | ----- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------------------------- | ----------------- | -------------------------------------------------------- |
| 2   | 100   | 10   | 0            | <span style="color:#00aa00">**18.06 μsec ± 8.5%**</span> | 41.52 μsec ± 6.3%                                        | 45.02 μsec ± 6.4% | <span style="color:#dd0000">**159.2 μsec ± 7.1%**</span> |
| 2   | 200   | 20   | 0            | <span style="color:#00aa00">**25.62 μsec ± 7.2%**</span> | 53.10 μsec ± 4.6%                                        | 67.98 μsec ± 5.4% | <span style="color:#dd0000">**324.1 μsec ± 4.4%**</span> |
| 2   | 500   | 50   | 0            | <span style="color:#00aa00">**52.52 μsec ± 3.0%**</span> | 164.6 μsec ± 3.6%                                        | 212.6 μsec ± 5.5% | <span style="color:#dd0000">**857.8 μsec ± 1.0%**</span> |
| 2   | 1000  | 100  | 0            | <span style="color:#00aa00">**164.5 μsec ± 5.1%**</span> | 390.7 μsec ± 7.0%                                        | 514.5 μsec ± 5.7% | <span style="color:#dd0000">**1.980 msec ± 1.0%**</span> |
| 2   | 2000  | 200  | 0            | <span style="color:#00aa00">**532.5 μsec ± 1.2%**</span> | 915.6 μsec ± 6.0%                                        | 1.171 msec ± 1.4% | <span style="color:#dd0000">**4.500 msec ± 0.7%**</span> |
| 2   | 5000  | 500  | 0            | <span style="color:#00aa00">**2.674 msec ± 0.4%**</span> | 3.596 msec ± 0.8%                                        | 4.160 msec ± 0.9% | <span style="color:#dd0000">**15.44 msec ± 0.8%**</span> |
| 2   | 10000 | 1000 | 0            | 11.50 msec ± 0.8%                                        | <span style="color:#00aa00">**11.28 msec ± 0.7%**</span> | 12.44 msec ± 0.8% | <span style="color:#dd0000">**45.87 msec ± 1.5%**</span> |
| 2   | 20000 | 2000 | 0            | 46.50 msec ± 1.3%                                        | <span style="color:#00aa00">**35.30 msec ± 1.4%**</span> | 35.98 msec ± 0.6% | <span style="color:#dd0000">**149.4 msec ± 3.3%**</span> |
|     |       |      | **Geomean:** | <span style="color:#00aa00">**484.7 μsec ± 3.4%**</span> | 810.3 μsec ± 3.8%                                        | 961.4 μsec ± 3.3% | <span style="color:#dd0000">**3.780 msec ± 2.5%**</span> |

### Diversity Score

| `d` | `n`   | `k`  | `m`          | `rsel`                                               | `fps(1)`                                            | `fps(8)`                                            | `fps(8,one-at-a-time)`                              |
| --- | ----- | ---- | ------------ | ---------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- |
| 2   | 100   | 10   | 0            | <span style="color:#dd0000">**0.069 ± 25.5%**</span> | <span style="color:#00aa00">**0.345 ± 2.5%**</span> | 0.309 ± 4.6%                                        | 0.304 ± 5.8%                                        |
| 2   | 200   | 20   | 0            | <span style="color:#dd0000">**0.044 ± 13.9%**</span> | <span style="color:#00aa00">**0.266 ± 3.5%**</span> | 0.244 ± 3.5%                                        | 0.242 ± 1.6%                                        |
| 2   | 500   | 50   | 0            | <span style="color:#dd0000">**0.029 ± 16.2%**</span> | <span style="color:#00aa00">**0.160 ± 0.8%**</span> | 0.157 ± 1.5%                                        | 0.157 ± 1.6%                                        |
| 2   | 1000  | 100  | 0            | <span style="color:#dd0000">**0.020 ± 10.9%**</span> | <span style="color:#00aa00">**0.110 ± 0.3%**</span> | 0.106 ± 1.1%                                        | 0.107 ± 0.7%                                        |
| 2   | 2000  | 200  | 0            | <span style="color:#dd0000">**0.015 ± 9.1%**</span>  | <span style="color:#00aa00">**0.071 ± 0.5%**</span> | 0.071 ± 0.6%                                        | 0.071 ± 0.5%                                        |
| 2   | 5000  | 500  | 0            | <span style="color:#dd0000">**0.009 ± 4.1%**</span>  | <span style="color:#00aa00">**0.045 ± 0.4%**</span> | <span style="color:#00aa00">**0.045 ± 0.3%**</span> | <span style="color:#00aa00">**0.045 ± 0.2%**</span> |
| 2   | 10000 | 1000 | 0            | <span style="color:#dd0000">**0.006 ± 2.0%**</span>  | 0.031 ± 0.2%                                        | <span style="color:#00aa00">**0.031 ± 0.5%**</span> | 0.031 ± 0.2%                                        |
| 2   | 20000 | 2000 | 0            | <span style="color:#dd0000">**0.005 ± 2.0%**</span>  | <span style="color:#00aa00">**0.022 ± 0.0%**</span> | 0.022 ± 0.1%                                        | <span style="color:#00aa00">**0.022 ± 0.2%**</span> |
|     |       |      | **Geomean:** | <span style="color:#dd0000">**0.017 ± 10.3%**</span> | <span style="color:#00aa00">**0.088 ± 1.0%**</span> | 0.085 ± 1.5%                                        | 0.085 ± 1.3%                                        |
