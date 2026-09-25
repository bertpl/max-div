Tested Initialization strategies:

| `name`                 | `class`             | `params`                            | Note                                                                  |
| ---------------------- | ------------------- | ----------------------------------- | --------------------------------------------------------------------- |
| `rsel`                 | InitRandomSelection | ignore_constraints=False            |                                                                       |
| `fps(1)`               | InitFarthestPoint   | top_k=1<br>candidate_pool_size=256  |                                                                       |
| `fps(8)`               | InitFarthestPoint   | top_k=8<br>candidate_pool_size=256  | = the SMART/THOROUGH presets' initialization (unconstrained problems) |
| `fps(8,one-at-a-time)` | InitFarthestPoint   | top_k=8<br>candidate_pool_size=None |                                                                       |

### Time Duration

| `d` | `n`   | `k`  | `m`          | `rsel`                                                    | `fps(1)`                                                 | `fps(8)`          | `fps(8,one-at-a-time)`                                    |
| --- | ----- | ---- | ------------ | --------------------------------------------------------- | -------------------------------------------------------- | ----------------- | --------------------------------------------------------- |
| 1   | 100   | 10   | 0            | <span style="color:#00aa00">**18.73 μsec ± 10.8%**</span> | 41.71 μsec ± 10.5%                                       | 46.35 μsec ± 9.2% | <span style="color:#dd0000">**169.3 μsec ± 4.4%**</span>  |
| 2   | 200   | 20   | 0            | <span style="color:#00aa00">**26.73 μsec ± 2.1%**</span>  | 57.60 μsec ± 3.8%                                        | 72.67 μsec ± 3.9% | <span style="color:#dd0000">**336.5 μsec ± 2.5%**</span>  |
| 5   | 500   | 50   | 0            | <span style="color:#00aa00">**56.90 μsec ± 1.7%**</span>  | 181.7 μsec ± 4.1%                                        | 242.0 μsec ± 3.6% | <span style="color:#dd0000">**945.4 μsec ± 1.2%**</span>  |
| 10  | 1000  | 100  | 0            | <span style="color:#00aa00">**169.6 μsec ± 4.4%**</span>  | 427.3 μsec ± 6.8%                                        | 552.0 μsec ± 4.1% | <span style="color:#dd0000">**2.081 msec ± 0.6%**</span>  |
| 20  | 2000  | 200  | 0            | <span style="color:#00aa00">**533.2 μsec ± 1.8%**</span>  | 918.2 μsec ± 1.9%                                        | 1.166 msec ± 2.0% | <span style="color:#dd0000">**4.842 msec ± 0.8%**</span>  |
| 50  | 5000  | 500  | 0            | <span style="color:#00aa00">**2.757 msec ± 1.0%**</span>  | 3.671 msec ± 1.8%                                        | 4.347 msec ± 2.6% | <span style="color:#dd0000">**17.13 msec ± 0.4%**</span>  |
| 100 | 10000 | 1000 | 0            | <span style="color:#00aa00">**11.51 msec ± 1.8%**</span>  | 12.08 msec ± 1.0%                                        | 13.13 msec ± 0.6% | <span style="color:#dd0000">**55.50 msec ± 18.8%**</span> |
| 200 | 20000 | 2000 | 0            | 47.83 msec ± 0.9%                                         | <span style="color:#00aa00">**37.72 msec ± 0.5%**</span> | 39.53 msec ± 0.5% | <span style="color:#dd0000">**163.4 msec ± 8.4%**</span>  |
|     |       |      | **Geomean:** | <span style="color:#00aa00">**500.0 μsec ± 3.0%**</span>  | 855.4 μsec ± 3.7%                                        | 1.021 msec ± 3.3% | <span style="color:#dd0000">**4.128 msec ± 4.5%**</span>  |

### Diversity Score

| `d` | `n`   | `k`  | `m`          | `rsel`                                               | `fps(1)`                                            | `fps(8)`                                            | `fps(8,one-at-a-time)`                              |
| --- | ----- | ---- | ------------ | ---------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- |
| 1   | 100   | 10   | 0            | <span style="color:#dd0000">**0.030 ± 45.8%**</span> | <span style="color:#00aa00">**0.105 ± 2.5%**</span> | 0.096 ± 4.1%                                        | 0.097 ± 3.6%                                        |
| 2   | 200   | 20   | 0            | <span style="color:#dd0000">**0.055 ± 9.4%**</span>  | <span style="color:#00aa00">**0.125 ± 3.2%**</span> | 0.118 ± 1.1%                                        | 0.118 ± 1.6%                                        |
| 5   | 500   | 50   | 0            | <span style="color:#dd0000">**0.088 ± 10.0%**</span> | <span style="color:#00aa00">**0.199 ± 0.9%**</span> | 0.195 ± 0.8%                                        | 0.195 ± 0.9%                                        |
| 10  | 1000  | 100  | 0            | <span style="color:#dd0000">**0.134 ± 5.8%**</span>  | <span style="color:#00aa00">**0.296 ± 0.4%**</span> | <span style="color:#00aa00">**0.296 ± 0.3%**</span> | <span style="color:#00aa00">**0.296 ± 0.4%**</span> |
| 20  | 2000  | 200  | 0            | <span style="color:#dd0000">**0.202 ± 6.7%**</span>  | <span style="color:#00aa00">**0.444 ± 0.2%**</span> | <span style="color:#00aa00">**0.444 ± 0.3%**</span> | 0.443 ± 0.2%                                        |
| 50  | 5000  | 500  | 0            | <span style="color:#dd0000">**0.324 ± 4.5%**</span>  | <span style="color:#00aa00">**0.765 ± 0.1%**</span> | <span style="color:#00aa00">**0.765 ± 0.0%**</span> | <span style="color:#00aa00">**0.764 ± 0.0%**</span> |
| 100 | 10000 | 1000 | 0            | <span style="color:#dd0000">**0.483 ± 1.4%**</span>  | <span style="color:#00aa00">**1.145 ± 0.0%**</span> | <span style="color:#00aa00">**1.145 ± 0.0%**</span> | 1.145 ± 0.0%                                        |
| 200 | 20000 | 2000 | 0            | <span style="color:#dd0000">**0.715 ± 1.9%**</span>  | <span style="color:#00aa00">**1.689 ± 0.0%**</span> | 1.689 ± 0.0%                                        | <span style="color:#00aa00">**1.689 ± 0.0%**</span> |
|     |       |      | **Geomean:** | <span style="color:#dd0000">**0.160 ± 10.1%**</span> | <span style="color:#00aa00">**0.388 ± 0.9%**</span> | 0.379 ± 0.8%                                        | 0.380 ± 0.8%                                        |
