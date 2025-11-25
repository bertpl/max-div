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

We compare the following situations:
- use `randint_numba`, i.e. we simply sample without taking constraints into account
- use `randint_constrained_numba`, i.e. we sample taking constraints into account

We test both speed (absolute time of a single call) and accuracy (% of cases the generated samples satisfied the constraints).


## Scenario A

Varying n & k with 10 non-overlapping constraints spanning equal portions of the n-range

### Uniform sampling.

#### Timing Results                                                                                                                                                                                              

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained_numba`<br>(eager=False) | `randint_constrained_numba`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------------- | ------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**502.3 nsec ± 2.3%**</span> | 1.367 μsec ± 1.2%                            | 1.404 μsec ± 1.4%                           |
| 4   | 10   | 10           | <span style="color:#00aa00">**520.7 nsec ± 1.2%**</span> | 1.854 μsec ± 0.7%                            | 1.954 μsec ± 6.0%                           |
| 8   | 10   | 10           | <span style="color:#00aa00">**576.6 nsec ± 4.3%**</span> | 3.354 μsec ± 2.6%                            | 3.096 μsec ± 3.6%                           |
| 2   | 100  | 10           | <span style="color:#00aa00">**554.2 nsec ± 2.9%**</span> | 1.953 μsec ± 0.6%                            | 1.958 μsec ± 0.8%                           |
| 4   | 100  | 10           | <span style="color:#00aa00">**559.5 nsec ± 0.7%**</span> | 3.045 μsec ± 0.8%                            | 3.034 μsec ± 0.7%                           |
| 8   | 100  | 10           | <span style="color:#00aa00">**594.2 nsec ± 1.6%**</span> | 5.278 μsec ± 0.5%                            | 5.233 μsec ± 0.7%                           |
| 16  | 100  | 10           | <span style="color:#00aa00">**584.7 nsec ± 0.9%**</span> | 9.782 μsec ± 0.4%                            | 9.792 μsec ± 0.7%                           |
| 32  | 100  | 10           | <span style="color:#00aa00">**636.8 nsec ± 1.2%**</span> | 19.54 μsec ± 1.2%                            | 19.54 μsec ± 1.0%                           |
| 64  | 100  | 10           | <span style="color:#00aa00">**717.2 nsec ± 0.6%**</span> | 38.73 μsec ± 0.6%                            | 38.49 μsec ± 0.6%                           |
| 2   | 1000 | 10           | <span style="color:#00aa00">**567.1 nsec ± 1.2%**</span> | 5.988 μsec ± 0.7%                            | 5.977 μsec ± 0.3%                           |
| 4   | 1000 | 10           | <span style="color:#00aa00">**568.8 nsec ± 0.7%**</span> | 11.32 μsec ± 0.7%                            | 11.27 μsec ± 0.6%                           |
| 8   | 1000 | 10           | <span style="color:#00aa00">**586.0 nsec ± 2.8%**</span> | 22.43 μsec ± 0.7%                            | 22.40 μsec ± 0.7%                           |
| 16  | 1000 | 10           | <span style="color:#00aa00">**603.2 nsec ± 1.8%**</span> | 44.56 μsec ± 0.4%                            | 44.43 μsec ± 0.1%                           |
| 32  | 1000 | 10           | <span style="color:#00aa00">**646.8 nsec ± 2.5%**</span> | 88.75 μsec ± 0.5%                            | 88.99 μsec ± 0.8%                           |
| 64  | 1000 | 10           | <span style="color:#00aa00">**717.4 nsec ± 0.5%**</span> | 181.0 μsec ± 0.5%                            | 180.7 μsec ± 0.5%                           |
| 128 | 1000 | 10           | <span style="color:#00aa00">**886.6 nsec ± 2.0%**</span> | 366.6 μsec ± 0.5%                            | 367.1 μsec ± 0.3%                           |
| 256 | 1000 | 10           | <span style="color:#00aa00">**1.215 μsec ± 1.1%**</span> | 748.7 μsec ± 0.3%                            | 754.4 μsec ± 0.8%                           |
|     |      | **Geomean:** | <span style="color:#00aa00">**633.2 nsec ± 1.7%**</span> | 16.66 μsec ± 0.8%                            | 16.65 μsec ± 1.1%                           |

#### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                               | `randint_constrained_numba`<br>(eager=False)   | `randint_constrained_numba`<br>(eager=True)    |
| --- | ---- | --------- | --------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 88.6%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 56.8%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 6.8%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 15.9%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 93.2%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 47.7%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 2.3%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 35.96%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |

### Non-uniform sampling (custom p).

#### Timing Results                                                                                                                                                                                              

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained_numba`<br>(eager=False) | `randint_constrained_numba`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------------- | ------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**629.9 nsec ± 1.5%**</span> | 1.256 μsec ± 2.1%                            | 1.243 μsec ± 0.9%                           |
| 4   | 10   | 10           | <span style="color:#00aa00">**663.8 nsec ± 1.4%**</span> | 1.748 μsec ± 0.7%                            | 1.768 μsec ± 2.0%                           |
| 8   | 10   | 10           | <span style="color:#00aa00">**662.9 nsec ± 2.9%**</span> | 2.795 μsec ± 0.8%                            | 2.779 μsec ± 1.5%                           |
| 2   | 100  | 10           | <span style="color:#00aa00">**1.066 μsec ± 1.3%**</span> | 1.844 μsec ± 0.6%                            | 1.859 μsec ± 1.5%                           |
| 4   | 100  | 10           | <span style="color:#00aa00">**1.199 μsec ± 0.7%**</span> | 2.957 μsec ± 0.7%                            | 2.990 μsec ± 2.0%                           |
| 8   | 100  | 10           | <span style="color:#00aa00">**1.465 μsec ± 1.9%**</span> | 5.325 μsec ± 0.8%                            | 5.336 μsec ± 1.3%                           |
| 16  | 100  | 10           | <span style="color:#00aa00">**2.097 μsec ± 0.5%**</span> | 10.17 μsec ± 0.9%                            | 10.09 μsec ± 0.6%                           |
| 32  | 100  | 10           | <span style="color:#00aa00">**2.181 μsec ± 1.6%**</span> | 19.57 μsec ± 0.6%                            | 19.74 μsec ± 0.6%                           |
| 64  | 100  | 10           | <span style="color:#00aa00">**2.158 μsec ± 2.2%**</span> | 39.26 μsec ± 0.7%                            | 39.41 μsec ± 0.5%                           |
| 2   | 1000 | 10           | <span style="color:#00aa00">**3.694 μsec ± 1.5%**</span> | 6.105 μsec ± 0.8%                            | 6.075 μsec ± 0.6%                           |
| 4   | 1000 | 10           | <span style="color:#00aa00">**3.972 μsec ± 0.5%**</span> | 11.53 μsec ± 0.6%                            | 11.56 μsec ± 0.8%                           |
| 8   | 1000 | 10           | <span style="color:#00aa00">**4.688 μsec ± 2.1%**</span> | 22.95 μsec ± 0.6%                            | 22.98 μsec ± 0.7%                           |
| 16  | 1000 | 10           | <span style="color:#00aa00">**5.951 μsec ± 0.8%**</span> | 46.03 μsec ± 0.9%                            | 45.95 μsec ± 0.7%                           |
| 32  | 1000 | 10           | <span style="color:#00aa00">**8.404 μsec ± 1.7%**</span> | 92.47 μsec ± 0.8%                            | 91.94 μsec ± 0.7%                           |
| 64  | 1000 | 10           | <span style="color:#00aa00">**9.988 μsec ± 1.0%**</span> | 185.3 μsec ± 0.3%                            | 185.8 μsec ± 0.5%                           |
| 128 | 1000 | 10           | <span style="color:#00aa00">**10.78 μsec ± 4.5%**</span> | 374.2 μsec ± 0.8%                            | 374.5 μsec ± 0.8%                           |
| 256 | 1000 | 10           | <span style="color:#00aa00">**11.09 μsec ± 0.6%**</span> | 768.4 μsec ± 0.7%                            | 770.2 μsec ± 0.7%                           |
|     |      | **Geomean:** | <span style="color:#00aa00">**2.696 μsec ± 1.6%**</span> | 16.52 μsec ± 0.8%                            | 16.54 μsec ± 1.0%                           |

#### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                               | `randint_constrained_numba`<br>(eager=False)   | `randint_constrained_numba`<br>(eager=True)    |
| --- | ---- | --------- | --------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 90.9%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 45.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 72.7%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 20.5%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 31.15%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |

## Scenario B

Fixed n=1000 & k=100 with varying number of constraints spanning random 1% portions of the n-range

### Uniform sampling.

#### Timing Results                                                                                                                                                                                              

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained_numba`<br>(eager=False) | `randint_constrained_numba`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------------- | ------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**848.5 nsec ± 2.1%**</span> | 226.0 μsec ± 1.0%                            | 225.9 μsec ± 0.8%                           |
| 100 | 1000 | 4            | <span style="color:#00aa00">**838.5 nsec ± 0.7%**</span> | 227.1 μsec ± 0.3%                            | 227.1 μsec ± 0.2%                           |
| 100 | 1000 | 8            | <span style="color:#00aa00">**821.2 nsec ± 3.3%**</span> | 230.5 μsec ± 0.4%                            | 233.1 μsec ± 0.8%                           |
| 100 | 1000 | 16           | <span style="color:#00aa00">**833.5 nsec ± 0.5%**</span> | 238.0 μsec ± 0.8%                            | 236.3 μsec ± 0.6%                           |
| 100 | 1000 | 32           | <span style="color:#00aa00">**811.9 nsec ± 2.4%**</span> | 254.8 μsec ± 0.8%                            | 246.6 μsec ± 0.8%                           |
| 100 | 1000 | 64           | <span style="color:#00aa00">**830.7 nsec ± 1.6%**</span> | 297.2 μsec ± 1.7%                            | 272.6 μsec ± 0.6%                           |
| 100 | 1000 | 128          | <span style="color:#00aa00">**859.9 nsec ± 2.4%**</span> | 368.5 μsec ± 0.5%                            | 319.2 μsec ± 0.5%                           |
| 100 | 1000 | 256          | <span style="color:#00aa00">**811.8 nsec ± 1.0%**</span> | 477.3 μsec ± 1.2%                            | 421.7 μsec ± 0.7%                           |
| 100 | 1000 | 384          | <span style="color:#00aa00">**824.4 nsec ± 2.0%**</span> | 596.3 μsec ± 0.6%                            | 529.2 μsec ± 0.8%                           |
| 100 | 1000 | 512          | <span style="color:#00aa00">**829.5 nsec ± 1.5%**</span> | 725.1 μsec ± 0.8%                            | 659.0 μsec ± 0.8%                           |
| 100 | 1000 | 768          | <span style="color:#00aa00">**820.2 nsec ± 2.2%**</span> | 989.0 μsec ± 0.4%                            | 936.3 μsec ± 0.8%                           |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**828.3 nsec ± 2.7%**</span> | 1.374 msec ± 1.0%                            | 1.327 msec ± 0.9%                           |
|     |      | **Geomean:** | <span style="color:#00aa00">**829.8 nsec ± 1.9%**</span> | 409.2 μsec ± 0.8%                            | 386.4 μsec ± 0.7%                           |

#### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                             | `randint_constrained_numba`<br>(eager=False)  | `randint_constrained_numba`<br>(eager=True)   |
| --- | ---- | --------- | ------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| 100 | 1000 | 2         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 4         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 8         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 16        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 32        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 64        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 128       | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 256       | 0.0%                                        | 97.7%                                         | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 384       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**31.8%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 66.48%                                        | <span style="color:#00aa00">**77.65%**</span> |

### Non-uniform sampling (custom p).

#### Timing Results                                                                                                                                                                                              

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained_numba`<br>(eager=False) | `randint_constrained_numba`<br>(eager=True) |
| --- | ---- | ------------ | -------------------------------------------------------- | -------------------------------------------- | ------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**10.32 μsec ± 1.6%**</span> | 240.4 μsec ± 1.0%                            | 238.6 μsec ± 0.7%                           |
| 100 | 1000 | 4            | <span style="color:#00aa00">**10.29 μsec ± 1.3%**</span> | 239.8 μsec ± 0.8%                            | 239.4 μsec ± 0.7%                           |
| 100 | 1000 | 8            | <span style="color:#00aa00">**10.37 μsec ± 1.8%**</span> | 241.5 μsec ± 0.7%                            | 245.4 μsec ± 1.7%                           |
| 100 | 1000 | 16           | <span style="color:#00aa00">**10.37 μsec ± 3.0%**</span> | 252.6 μsec ± 0.7%                            | 248.2 μsec ± 0.6%                           |
| 100 | 1000 | 32           | <span style="color:#00aa00">**10.34 μsec ± 1.7%**</span> | 267.4 μsec ± 0.6%                            | 257.5 μsec ± 0.8%                           |
| 100 | 1000 | 64           | <span style="color:#00aa00">**10.05 μsec ± 1.4%**</span> | 306.2 μsec ± 0.8%                            | 282.9 μsec ± 0.4%                           |
| 100 | 1000 | 128          | <span style="color:#00aa00">**10.40 μsec ± 2.1%**</span> | 376.6 μsec ± 0.8%                            | 328.7 μsec ± 0.9%                           |
| 100 | 1000 | 256          | <span style="color:#00aa00">**10.29 μsec ± 1.6%**</span> | 486.4 μsec ± 0.6%                            | 430.4 μsec ± 0.5%                           |
| 100 | 1000 | 384          | <span style="color:#00aa00">**10.29 μsec ± 2.1%**</span> | 610.4 μsec ± 0.7%                            | 542.4 μsec ± 0.8%                           |
| 100 | 1000 | 512          | <span style="color:#00aa00">**10.32 μsec ± 1.4%**</span> | 747.4 μsec ± 3.0%                            | 671.7 μsec ± 1.0%                           |
| 100 | 1000 | 768          | <span style="color:#00aa00">**10.41 μsec ± 2.4%**</span> | 998.1 μsec ± 0.9%                            | 930.9 μsec ± 0.6%                           |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**10.24 μsec ± 2.7%**</span> | 1.379 msec ± 1.1%                            | 1.333 msec ± 1.1%                           |
|     |      | **Geomean:** | <span style="color:#00aa00">**10.31 μsec ± 1.9%**</span> | 423.3 μsec ± 1.0%                            | 398.8 μsec ± 0.8%                           |

#### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                             | `randint_constrained_numba`<br>(eager=False)  | `randint_constrained_numba`<br>(eager=True)   |
| --- | ---- | --------- | ------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| 100 | 1000 | 2         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 4         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 8         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 16        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 32        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 64        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 128       | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 256       | 0.0%                                        | 86.4%                                         | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 384       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**38.6%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 65.53%                                        | <span style="color:#00aa00">**78.22%**</span> |