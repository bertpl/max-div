# `randint_constrained`

([API reference][max_div.sampling.con.randint_constrained])

Command:
```bash
uv tool install max-div
max-div benchmark --markdown randint_constrained
```
or 
```bash
uv run max-div benchmark --markdown randint_constrained
```

## Scenario A1

Varying n & k with 10 non-overlapping constraints spanning equal portions of the n range (uniform sampling).

| `k` | `n`  | `n_cons`     | `randint_numba`<br>(time)                                | `randint_numba`<br>(accuracy %)               | `randint_constrained_numba`<br>(time) | `randint_constrained_numba`<br>(accuracy %)    |
| --- | ---- | ------------ | -------------------------------------------------------- | --------------------------------------------- | ------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**1.103 μsec ± 0.1%**</span> | <span style="color:#00aa00">**100.0%**</span> | 2.656 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10           | <span style="color:#00aa00">**1.132 μsec ± 0.1%**</span> | <span style="color:#00aa00">**100.0%**</span> | 3.321 μsec ± 0.4%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10           | <span style="color:#00aa00">**1.138 μsec ± 0.1%**</span> | <span style="color:#00aa00">**100.0%**</span> | 4.992 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10           | <span style="color:#00aa00">**1.123 μsec ± 0.1%**</span> | 90.6%                                         | 3.193 μsec ± 0.2%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10           | <span style="color:#00aa00">**1.132 μsec ± 0.2%**</span> | 51.1%                                         | 4.448 μsec ± 0.2%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10           | <span style="color:#00aa00">**1.148 μsec ± 0.1%**</span> | 2.4%                                          | 7.304 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10           | <span style="color:#00aa00">**1.174 μsec ± 0.8%**</span> | 1.7%                                          | 13.04 μsec ± 0.2%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10           | <span style="color:#00aa00">**1.220 μsec ± 0.2%**</span> | 4.1%                                          | 24.60 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10           | <span style="color:#00aa00">**1.319 μsec ± 0.7%**</span> | 18.2%                                         | 48.38 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10           | <span style="color:#00aa00">**1.253 μsec ± 0.1%**</span> | 90.0%                                         | 8.011 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10           | <span style="color:#00aa00">**1.208 μsec ± 0.2%**</span> | 46.2%                                         | 14.00 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10           | <span style="color:#00aa00">**1.265 μsec ± 0.1%**</span> | 1.4%                                          | 27.47 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10           | <span style="color:#00aa00">**1.245 μsec ± 0.3%**</span> | 0.8%                                          | 53.27 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10           | <span style="color:#00aa00">**1.328 μsec ± 0.1%**</span> | 1.7%                                          | 106.1 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10           | <span style="color:#00aa00">**1.383 μsec ± 1.0%**</span> | 1.0%                                          | 212.6 μsec ± 0.2%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10           | <span style="color:#00aa00">**1.558 μsec ± 0.1%**</span> | 0.4%                                          | 439.2 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10           | <span style="color:#00aa00">**1.813 μsec ± 0.3%**</span> | 0.7%                                          | 886.7 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Geomean:** | <span style="color:#00aa00">**1.256 μsec ± 0.3%**</span> |                                               | 22.57 μsec ± 0.1%                     |                                                |
|     |      | **Mean:**    |                                                          | 35.90%                                        |                                       | <span style="color:#00aa00">**100.00%**</span> |

## Scenario A2

Identical to Scenario A1, but with custom probabilities p provided, favoring larger values.

| `k` | `n`  | `n_cons`     | `randint_numba`<br>(time)                                | `randint_numba`<br>(accuracy %)               | `randint_constrained_numba`<br>(time) | `randint_constrained_numba`<br>(accuracy %)    |
| --- | ---- | ------------ | -------------------------------------------------------- | --------------------------------------------- | ------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**1.321 μsec ± 0.2%**</span> | <span style="color:#00aa00">**100.0%**</span> | 2.311 μsec ± 0.6%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10           | <span style="color:#00aa00">**1.359 μsec ± 0.1%**</span> | <span style="color:#00aa00">**100.0%**</span> | 3.005 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10           | <span style="color:#00aa00">**1.350 μsec ± 0.1%**</span> | <span style="color:#00aa00">**100.0%**</span> | 4.546 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10           | <span style="color:#00aa00">**1.810 μsec ± 0.1%**</span> | 86.7%                                         | 2.904 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10           | <span style="color:#00aa00">**2.057 μsec ± 0.1%**</span> | 40.3%                                         | 4.220 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10           | <span style="color:#00aa00">**2.506 μsec ± 0.1%**</span> | 0.5%                                          | 7.000 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10           | <span style="color:#00aa00">**3.275 μsec ± 0.1%**</span> | 0.2%                                          | 12.85 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10           | <span style="color:#00aa00">**3.411 μsec ± 0.3%**</span> | 0.0%                                          | 24.48 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10           | <span style="color:#00aa00">**3.370 μsec ± 0.1%**</span> | 0.0%                                          | 48.06 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10           | <span style="color:#00aa00">**5.327 μsec ± 0.0%**</span> | 79.2%                                         | 8.226 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10           | <span style="color:#00aa00">**5.856 μsec ± 0.3%**</span> | 28.9%                                         | 14.87 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10           | <span style="color:#00aa00">**6.899 μsec ± 0.0%**</span> | 0.2%                                          | 29.01 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10           | <span style="color:#00aa00">**9.088 μsec ± 0.2%**</span> | 0.0%                                          | 57.39 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10           | <span style="color:#00aa00">**12.98 μsec ± 0.0%**</span> | 0.0%                                          | 113.5 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10           | <span style="color:#00aa00">**14.72 μsec ± 0.2%**</span> | 0.0%                                          | 229.3 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10           | <span style="color:#00aa00">**15.38 μsec ± 0.2%**</span> | 0.0%                                          | 463.8 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10           | <span style="color:#00aa00">**16.43 μsec ± 0.1%**</span> | 0.0%                                          | 942.3 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Geomean:** | <span style="color:#00aa00">**4.380 μsec ± 0.1%**</span> |                                               | 22.47 μsec ± 0.1%                     |                                                |
|     |      | **Mean:**    |                                                          | 31.53%                                        |                                       | <span style="color:#00aa00">**100.00%**</span> |

## Scenario B1

Fixed n=1000 & k=100 with varying number of constraints spanning random 1% portions of the n range (uniform sampling).

| `k` | `n`  | `n_cons`     | `randint_numba`<br>(time)                                | `randint_numba`<br>(accuracy %) | `randint_constrained_numba`<br>(time) | `randint_constrained_numba`<br>(accuracy %)   |               
| --- | ---- | ------------ | -------------------------------------------------------- | ------------------------------- | ------------------------------------- | --------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**1.559 μsec ± 0.1%**</span> | 0.0%                            | 257.8 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 4            | <span style="color:#00aa00">**1.557 μsec ± 0.1%**</span> | 0.2%                            | 260.1 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 8            | <span style="color:#00aa00">**1.553 μsec ± 0.2%**</span> | 4.2%                            | 266.9 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 16           | <span style="color:#00aa00">**1.557 μsec ± 0.2%**</span> | 98.4%                           | 273.9 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 32           | <span style="color:#00aa00">**1.550 μsec ± 0.1%**</span> | 92.6%                           | 296.9 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 64           | <span style="color:#00aa00">**1.555 μsec ± 0.2%**</span> | 1.1%                            | 340.9 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 128          | <span style="color:#00aa00">**1.575 μsec ± 0.1%**</span> | 0.0%                            | 493.8 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 256          | <span style="color:#00aa00">**1.552 μsec ± 0.1%**</span> | 0.0%                            | 693.7 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512          | <span style="color:#00aa00">**1.541 μsec ± 0.1%**</span> | 0.0%                            | 1.048 msec ± 0.1%                     | <span style="color:#00aa00">**72.9%**</span>  |
|     |      | **Geomean:** | <span style="color:#00aa00">**1.555 μsec ± 0.1%**</span> |                                 | 383.2 μsec ± 0.1%                     |                                               |
|     |      | **Mean:**    |                                                          | 21.83%                          |                                       | <span style="color:#00aa00">**96.99%**</span> |

## Scenario B2

Identical to Scenario B1, but with custom probabilities p provided, favoring larger values.

| `k` | `n`  | `n_cons`     | `randint_numba`<br>(time)                                | `randint_numba`<br>(accuracy %)               | `randint_constrained_numba`<br>(time) | `randint_constrained_numba`<br>(accuracy %)   | 
| --- | ---- | ------------ | -------------------------------------------------------- | --------------------------------------------- | ------------------------------------- | --------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**15.10 μsec ± 0.2%**</span> | 0.0%                                          | 281.3 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 4            | <span style="color:#00aa00">**15.10 μsec ± 0.1%**</span> | 0.7%                                          | 284.1 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 8            | <span style="color:#00aa00">**15.11 μsec ± 0.1%**</span> | 2.4%                                          | 290.7 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 16           | <span style="color:#00aa00">**15.10 μsec ± 0.2%**</span> | <span style="color:#00aa00">**100.0%**</span> | 297.5 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 32           | <span style="color:#00aa00">**15.06 μsec ± 0.3%**</span> | 96.0%                                         | 320.8 μsec ± 0.0%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 64           | <span style="color:#00aa00">**15.10 μsec ± 0.1%**</span> | 0.9%                                          | 365.3 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 128          | <span style="color:#00aa00">**15.13 μsec ± 0.3%**</span> | 0.0%                                          | 517.3 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 256          | <span style="color:#00aa00">**15.13 μsec ± 0.2%**</span> | 0.0%                                          | 717.6 μsec ± 0.1%                     | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512          | <span style="color:#00aa00">**15.11 μsec ± 0.2%**</span> | 0.0%                                          | 1.071 msec ± 0.1%                     | <span style="color:#00aa00">**68.2%**</span>  |
|     |      | **Geomean:** | <span style="color:#00aa00">**15.10 μsec ± 0.2%**</span> |                                               | 409.2 μsec ± 0.0%                     |                                               |
|     |      | **Mean:**    |                                                          | 22.22%                                        |                                       | <span style="color:#00aa00">**96.47%**</span> |