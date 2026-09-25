Tested Initialization strategies:

| `name`                 | `class`             | `params`                            | Note                                                                  |
| ---------------------- | ------------------- | ----------------------------------- | --------------------------------------------------------------------- |
| `rsel`                 | InitRandomSelection | ignore_constraints=False            |                                                                       |
| `fps(1)`               | InitFarthestPoint   | top_k=1<br>candidate_pool_size=256  |                                                                       |
| `fps(8)`               | InitFarthestPoint   | top_k=8<br>candidate_pool_size=256  | = the SMART/THOROUGH presets' initialization (unconstrained problems) |
| `fps(8,one-at-a-time)` | InitFarthestPoint   | top_k=8<br>candidate_pool_size=None |                                                                       |

### Time Duration

| `d` | `n`   | `k`  | `m`          | `rsel`                                                    | `fps(1)`                                                 | `fps(8)`           | `fps(8,one-at-a-time)`                                   |
| --- | ----- | ---- | ------------ | --------------------------------------------------------- | -------------------------------------------------------- | ------------------ | -------------------------------------------------------- |
| 1   | 100   | 10   | 0            | <span style="color:#00aa00">**19.29 μsec ± 14.2%**</span> | 41.06 μsec ± 8.4%                                        | 44.48 μsec ± 10.0% | <span style="color:#dd0000">**160.0 μsec ± 8.5%**</span> |
| 2   | 200   | 20   | 0            | <span style="color:#00aa00">**24.83 μsec ± 1.6%**</span>  | 51.25 μsec ± 2.6%                                        | 67.08 μsec ± 3.8%  | <span style="color:#dd0000">**308.4 μsec ± 4.0%**</span> |
| 5   | 500   | 50   | 0            | <span style="color:#00aa00">**52.10 μsec ± 4.6%**</span>  | 165.4 μsec ± 3.9%                                        | 220.4 μsec ± 4.1%  | <span style="color:#dd0000">**873.4 μsec ± 1.7%**</span> |
| 10  | 1000  | 100  | 0            | <span style="color:#00aa00">**162.1 μsec ± 4.2%**</span>  | 358.4 μsec ± 2.3%                                        | 509.4 μsec ± 1.8%  | <span style="color:#dd0000">**1.954 msec ± 1.0%**</span> |
| 20  | 2000  | 200  | 0            | <span style="color:#00aa00">**510.7 μsec ± 3.9%**</span>  | 771.1 μsec ± 1.8%                                        | 1.068 msec ± 2.1%  | <span style="color:#dd0000">**4.488 msec ± 0.8%**</span> |
| 50  | 5000  | 500  | 0            | <span style="color:#00aa00">**2.672 msec ± 1.9%**</span>  | 3.380 msec ± 2.7%                                        | 3.993 msec ± 2.7%  | <span style="color:#dd0000">**16.27 msec ± 1.3%**</span> |
| 100 | 10000 | 1000 | 0            | 11.52 msec ± 1.3%                                         | <span style="color:#00aa00">**11.17 msec ± 0.7%**</span> | 12.05 msec ± 1.1%  | <span style="color:#dd0000">**47.14 msec ± 5.0%**</span> |
| 200 | 20000 | 2000 | 0            | 47.63 msec ± 0.2%                                         | <span style="color:#00aa00">**35.64 msec ± 0.4%**</span> | 36.87 msec ± 0.6%  | <span style="color:#dd0000">**149.8 msec ± 2.0%**</span> |
|     |       |      | **Geomean:** | <span style="color:#00aa00">**484.4 μsec ± 3.9%**</span>  | 774.5 μsec ± 2.8%                                        | 944.7 μsec ± 3.2%  | <span style="color:#dd0000">**3.799 msec ± 3.0%**</span> |

### Diversity Score

| `d` | `n`   | `k`  | `m`          | `rsel`                                               | `fps(1)`                                            | `fps(8)`                                            | `fps(8,one-at-a-time)`                              |
| --- | ----- | ---- | ------------ | ---------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- |
| 1   | 100   | 10   | 0            | <span style="color:#dd0000">**0.031 ± 20.5%**</span> | <span style="color:#00aa00">**0.098 ± 1.8%**</span> | 0.087 ± 2.7%                                        | 0.091 ± 3.3%                                        |
| 2   | 200   | 20   | 0            | <span style="color:#dd0000">**0.104 ± 4.8%**</span>  | <span style="color:#00aa00">**0.223 ± 2.9%**</span> | 0.209 ± 1.7%                                        | 0.210 ± 1.0%                                        |
| 5   | 500   | 50   | 0            | <span style="color:#dd0000">**0.347 ± 3.3%**</span>  | <span style="color:#00aa00">**0.551 ± 0.5%**</span> | 0.535 ± 1.0%                                        | 0.542 ± 1.0%                                        |
| 10  | 1000  | 100  | 0            | <span style="color:#dd0000">**0.666 ± 1.6%**</span>  | <span style="color:#00aa00">**0.940 ± 0.4%**</span> | <span style="color:#00aa00">**0.938 ± 0.4%**</span> | 0.937 ± 0.4%                                        |
| 20  | 2000  | 200  | 0            | <span style="color:#dd0000">**1.154 ± 0.5%**</span>  | <span style="color:#00aa00">**1.467 ± 0.2%**</span> | <span style="color:#00aa00">**1.464 ± 0.2%**</span> | 1.463 ± 0.2%                                        |
| 50  | 5000  | 500  | 0            | <span style="color:#dd0000">**2.168 ± 0.4%**</span>  | 2.508 ± 0.0%                                        | <span style="color:#00aa00">**2.509 ± 0.1%**</span> | <span style="color:#00aa00">**2.509 ± 0.1%**</span> |
| 100 | 10000 | 1000 | 0            | <span style="color:#dd0000">**3.320 ± 0.2%**</span>  | <span style="color:#00aa00">**3.686 ± 0.0%**</span> | <span style="color:#00aa00">**3.686 ± 0.0%**</span> | <span style="color:#00aa00">**3.686 ± 0.0%**</span> |
| 200 | 20000 | 2000 | 0            | <span style="color:#dd0000">**4.978 ± 0.0%**</span>  | <span style="color:#00aa00">**5.356 ± 0.0%**</span> | 5.355 ± 0.0%                                        | 5.356 ± 0.0%                                        |
|     |       |      | **Geomean:** | <span style="color:#dd0000">**0.647 ± 3.8%**</span>  | <span style="color:#00aa00">**0.976 ± 0.7%**</span> | 0.950 ± 0.8%                                        | 0.957 ± 0.7%                                        |
