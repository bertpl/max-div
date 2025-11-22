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

| `k` | `n`  | `n_cons`     | `randint_numba`<br>(time)                                | `randint_numba`<br>(accuracy %)             | `randint_constrained`<br>(time) | `randint_constrained`<br>(accuracy %)         |         
| --- | ---- | ------------ | -------------------------------------------------------- | ------------------------------------------- | ------------------------------- | --------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**1.077 μsec ± 1.0%**</span> | <span style="color:#00aa00">**100%**</span> | 98.82 μsec ± 0.3%               | <span style="color:#00aa00">**100%**</span>   |
| 4   | 10   | 10           | <span style="color:#00aa00">**1.104 μsec ± 0.1%**</span> | <span style="color:#00aa00">**100%**</span> | 124.6 μsec ± 0.1%               | <span style="color:#00aa00">**100%**</span>   |
| 8   | 10   | 10           | <span style="color:#00aa00">**1.114 μsec ± 0.2%**</span> | <span style="color:#00aa00">**100%**</span> | 179.6 μsec ± 0.2%               | <span style="color:#00aa00">**100%**</span>   |
| 2   | 100  | 10           | <span style="color:#00aa00">**1.101 μsec ± 0.2%**</span> | 91%                                         | 149.1 μsec ± 0.2%               | <span style="color:#00aa00">**100%**</span>   |
| 4   | 100  | 10           | <span style="color:#00aa00">**1.109 μsec ± 0.2%**</span> | 52%                                         | 216.1 μsec ± 0.1%               | <span style="color:#00aa00">**100%**</span>   |
| 8   | 100  | 10           | <span style="color:#00aa00">**1.116 μsec ± 0.2%**</span> | 2%                                          | 377.7 μsec ± 0.3%               | <span style="color:#00aa00">**100%**</span>   |
| 16  | 100  | 10           | <span style="color:#00aa00">**1.136 μsec ± 0.1%**</span> | 0%                                          | 697.0 μsec ± 0.2%               | <span style="color:#00aa00">**100%**</span>   |
| 32  | 100  | 10           | <span style="color:#00aa00">**1.192 μsec ± 0.2%**</span> | 8%                                          | 1.352 msec ± 0.2%               | <span style="color:#00aa00">**100%**</span>   |
| 64  | 100  | 10           | <span style="color:#00aa00">**1.286 μsec ± 0.2%**</span> | 13%                                         | 2.978 msec ± 0.3%               | <span style="color:#00aa00">**100%**</span>   |
| 2   | 1000 | 10           | <span style="color:#00aa00">**1.210 μsec ± 0.1%**</span> | 93%                                         | 667.0 μsec ± 0.2%               | <span style="color:#00aa00">**100%**</span>   |
| 4   | 1000 | 10           | <span style="color:#00aa00">**1.247 μsec ± 0.1%**</span> | 43%                                         | 1.190 msec ± 0.5%               | <span style="color:#00aa00">**100%**</span>   |
| 8   | 1000 | 10           | <span style="color:#00aa00">**1.246 μsec ± 0.1%**</span> | 2%                                          | 2.400 msec ± 0.3%               | <span style="color:#00aa00">**100%**</span>   |
| 16  | 1000 | 10           | <span style="color:#00aa00">**1.267 μsec ± 0.3%**</span> | 2%                                          | 4.810 msec ± 0.4%               | <span style="color:#00aa00">**100%**</span>   |
| 32  | 1000 | 10           | <span style="color:#00aa00">**1.303 μsec ± 0.1%**</span> | 1%                                          | 9.197 msec ± 0.5%               | <span style="color:#00aa00">**100%**</span>   |
| 64  | 1000 | 10           | <span style="color:#00aa00">**1.548 μsec ± 0.3%**</span> | 0%                                          | 20.05 msec ± 0.8%               | <span style="color:#00aa00">**100%**</span>   |
| 128 | 1000 | 10           | <span style="color:#00aa00">**1.695 μsec ± 0.1%**</span> | 0%                                          | 42.31 msec ± 0.6%               | <span style="color:#00aa00">**100%**</span>   |
| 256 | 1000 | 10           | <span style="color:#00aa00">**1.994 μsec ± 0.2%**</span> | 1%                                          | 92.11 msec ± 0.4%               | <span style="color:#00aa00">**100%**</span>   |
|     |      | **Geomean:** | <span style="color:#00aa00">**1.260 μsec ± 0.2%**</span> |                                             | 1.441 msec ± 0.3%               |                                               |
|     |      | **Mean:**    |                                                          | 35.8%                                       |                                 | <span style="color:#00aa00">**100.0%**</span> |

## Scenario A2

Identical to Scenario A1, but with custom probabilities p provided, favoring larger values.

| `k` | `n`  | `n_cons`     | `randint_numba`<br>(time)                                | `randint_numba`<br>(accuracy %)             | `randint_constrained`<br>(time) | `randint_constrained`<br>(accuracy %)         |         
| --- | ---- | ------------ | -------------------------------------------------------- | ------------------------------------------- | ------------------------------- | --------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**1.290 μsec ± 0.1%**</span> | <span style="color:#00aa00">**100%**</span> | 94.51 μsec ± 0.4%               | <span style="color:#00aa00">**100%**</span>   |
| 4   | 10   | 10           | <span style="color:#00aa00">**1.321 μsec ± 0.1%**</span> | <span style="color:#00aa00">**100%**</span> | 117.0 μsec ± 0.2%               | <span style="color:#00aa00">**100%**</span>   |
| 8   | 10   | 10           | <span style="color:#00aa00">**1.313 μsec ± 0.1%**</span> | <span style="color:#00aa00">**100%**</span> | 170.2 μsec ± 0.2%               | <span style="color:#00aa00">**100%**</span>   |
| 2   | 100  | 10           | <span style="color:#00aa00">**1.763 μsec ± 0.1%**</span> | 93%                                         | 142.2 μsec ± 0.2%               | <span style="color:#00aa00">**100%**</span>   |
| 4   | 100  | 10           | <span style="color:#00aa00">**2.015 μsec ± 0.1%**</span> | 38%                                         | 205.0 μsec ± 0.2%               | <span style="color:#00aa00">**100%**</span>   |
| 8   | 100  | 10           | <span style="color:#00aa00">**2.457 μsec ± 0.1%**</span> | 0%                                          | 363.7 μsec ± 0.3%               | <span style="color:#00aa00">**100%**</span>   |
| 16  | 100  | 10           | <span style="color:#00aa00">**3.218 μsec ± 0.1%**</span> | 0%                                          | 664.5 μsec ± 0.3%               | <span style="color:#00aa00">**100%**</span>   |
| 32  | 100  | 10           | <span style="color:#00aa00">**3.329 μsec ± 0.1%**</span> | 0%                                          | 1.293 msec ± 0.4%               | <span style="color:#00aa00">**100%**</span>   |
| 64  | 100  | 10           | <span style="color:#00aa00">**3.297 μsec ± 0.1%**</span> | 0%                                          | 2.797 msec ± 0.3%               | <span style="color:#00aa00">**100%**</span>   |
| 2   | 1000 | 10           | <span style="color:#00aa00">**5.286 μsec ± 0.1%**</span> | 76%                                         | 663.1 μsec ± 0.4%               | <span style="color:#00aa00">**100%**</span>   |
| 4   | 1000 | 10           | <span style="color:#00aa00">**5.747 μsec ± 0.0%**</span> | 29%                                         | 1.138 msec ± 0.4%               | <span style="color:#00aa00">**100%**</span>   |
| 8   | 1000 | 10           | <span style="color:#00aa00">**6.830 μsec ± 0.2%**</span> | 0%                                          | 2.387 msec ± 0.3%               | <span style="color:#00aa00">**100%**</span>   |
| 16  | 1000 | 10           | <span style="color:#00aa00">**8.988 μsec ± 0.1%**</span> | 0%                                          | 4.599 msec ± 0.3%               | <span style="color:#00aa00">**100%**</span>   |
| 32  | 1000 | 10           | <span style="color:#00aa00">**12.91 μsec ± 0.0%**</span> | 0%                                          | 9.260 msec ± 0.5%               | <span style="color:#00aa00">**100%**</span>   |
| 64  | 1000 | 10           | <span style="color:#00aa00">**14.60 μsec ± 0.2%**</span> | 0%                                          | 19.13 msec ± 0.5%               | <span style="color:#00aa00">**100%**</span>   |
| 128 | 1000 | 10           | <span style="color:#00aa00">**15.26 μsec ± 0.2%**</span> | 0%                                          | 40.74 msec ± 0.4%               | <span style="color:#00aa00">**100%**</span>   |
| 256 | 1000 | 10           | <span style="color:#00aa00">**16.34 μsec ± 0.1%**</span> | 0%                                          | 86.51 msec ± 0.7%               | <span style="color:#00aa00">**100%**</span>   |
|     |      | **Geomean:** | <span style="color:#00aa00">**4.307 μsec ± 0.1%**</span> |                                             | 1.383 msec ± 0.3%               |                                               |
|     |      | **Mean:**    |                                                          | 31.5%                                       |                                 | <span style="color:#00aa00">**100.0%**</span> |

## Scenario B1

Fixed n=1000 & k=100 with varying number of constraints spanning random 1% portions of the n range (uniform sampling).

| `k` | `n`  | `n_cons`     | `randint_numba`<br>(time)                                | `randint_numba`<br>(accuracy %) | `randint_constrained`<br>(time) | `randint_constrained`<br>(accuracy %)        |                      
| --- | ---- | ------------ | -------------------------------------------------------- | ------------------------------- | ------------------------------- | -------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**1.456 μsec ± 0.1%**</span> | 0%                              | 19.96 msec ± 0.2%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 4            | <span style="color:#00aa00">**1.457 μsec ± 0.2%**</span> | 0%                              | 20.02 msec ± 0.6%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 8            | <span style="color:#00aa00">**1.454 μsec ± 0.1%**</span> | 5%                              | 20.09 msec ± 0.1%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 16           | <span style="color:#00aa00">**1.459 μsec ± 0.3%**</span> | 84%                             | 19.96 msec ± 0.4%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 32           | <span style="color:#00aa00">**1.451 μsec ± 0.1%**</span> | 60%                             | 20.30 msec ± 0.5%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 64           | <span style="color:#00aa00">**1.456 μsec ± 0.1%**</span> | 0%                              | 23.56 msec ± 1.1%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 128          | <span style="color:#00aa00">**1.457 μsec ± 0.2%**</span> | 0%                              | 36.04 msec ± 2.1%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 256          | <span style="color:#00aa00">**1.458 μsec ± 0.4%**</span> | 0%                              | 52.32 msec ± 2.1%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 512          | <span style="color:#00aa00">**1.462 μsec ± 0.1%**</span> | 0%                              | 83.79 msec ± 1.9%               | <span style="color:#00aa00">**79%**</span>   |
|     |      | **Geomean:** | <span style="color:#00aa00">**1.457 μsec ± 0.2%**</span> |                                 | 28.42 msec ± 1.0%               |                                              |
|     |      | **Mean:**    |                                                          | 16.6%                           |                                 | <span style="color:#00aa00">**97.7%**</span> |

## Scenario B2

Identical to Scenario B1, but with custom probabilities p provided, favoring larger values.

| `k` | `n`  | `n_cons`     | `randint_numba`<br>(time)                                | `randint_numba`<br>(accuracy %)             | `randint_constrained`<br>(time) | `randint_constrained`<br>(accuracy %)        |          
| --- | ---- | ------------ | -------------------------------------------------------- | ------------------------------------------- | ------------------------------- | -------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**15.01 μsec ± 0.2%**</span> | 0%                                          | 20.23 msec ± 0.6%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 4            | <span style="color:#00aa00">**15.03 μsec ± 0.2%**</span> | 0%                                          | 20.14 msec ± 0.5%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 8            | <span style="color:#00aa00">**15.04 μsec ± 0.2%**</span> | 1%                                          | 20.40 msec ± 0.7%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 16           | <span style="color:#00aa00">**15.01 μsec ± 0.2%**</span> | <span style="color:#00aa00">**100%**</span> | 19.98 msec ± 0.4%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 32           | <span style="color:#00aa00">**15.01 μsec ± 0.1%**</span> | 96%                                         | 20.74 msec ± 0.3%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 64           | <span style="color:#00aa00">**15.04 μsec ± 0.2%**</span> | 0%                                          | 22.81 msec ± 1.6%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 128          | <span style="color:#00aa00">**15.00 μsec ± 0.2%**</span> | 0%                                          | 35.79 msec ± 1.5%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 256          | <span style="color:#00aa00">**15.03 μsec ± 0.2%**</span> | 0%                                          | 52.15 msec ± 2.1%               | <span style="color:#00aa00">**100%**</span>  |
| 100 | 1000 | 512          | <span style="color:#00aa00">**15.02 μsec ± 0.2%**</span> | 0%                                          | 82.36 msec ± 1.8%               | <span style="color:#00aa00">**60%**</span>   |
|     |      | **Geomean:** | <span style="color:#00aa00">**15.02 μsec ± 0.2%**</span> |                                             | 28.42 msec ± 1.0%               |                                              |
|     |      | **Mean:**    |                                                          | 21.9%                                       |                                 | <span style="color:#00aa00">**95.6%**</span> |