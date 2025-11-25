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


## Scenario A1

Varying n & k with 10 non-overlapping constraints spanning equal portions of the n range (uniform sampling).

### Timing Results                                                                                                                                                                                               

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained_numba` |
| --- | ---- | ------------ | -------------------------------------------------------- | --------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**539.7 nsec ± 1.0%**</span> | 1.367 μsec ± 29.8%          |
| 4   | 10   | 10           | <span style="color:#00aa00">**531.9 nsec ± 0.5%**</span> | 1.855 μsec ± 0.4%           |
| 8   | 10   | 10           | <span style="color:#00aa00">**553.3 nsec ± 0.5%**</span> | 2.946 μsec ± 0.2%           |
| 2   | 100  | 10           | <span style="color:#00aa00">**564.6 nsec ± 0.5%**</span> | 1.917 μsec ± 1.3%           |
| 4   | 100  | 10           | <span style="color:#00aa00">**570.5 nsec ± 1.4%**</span> | 2.970 μsec ± 2.1%           |
| 8   | 100  | 10           | <span style="color:#00aa00">**586.6 nsec ± 1.8%**</span> | 5.319 μsec ± 1.5%           |
| 16  | 100  | 10           | <span style="color:#00aa00">**615.1 nsec ± 0.6%**</span> | 9.830 μsec ± 0.6%           |
| 32  | 100  | 10           | <span style="color:#00aa00">**651.2 nsec ± 1.3%**</span> | 19.42 μsec ± 1.1%           |
| 64  | 100  | 10           | <span style="color:#00aa00">**747.8 nsec ± 0.7%**</span> | 38.73 μsec ± 0.6%           |
| 2   | 1000 | 10           | <span style="color:#00aa00">**588.7 nsec ± 1.1%**</span> | 5.952 μsec ± 0.3%           |
| 4   | 1000 | 10           | <span style="color:#00aa00">**598.2 nsec ± 1.4%**</span> | 11.31 μsec ± 0.4%           |
| 8   | 1000 | 10           | <span style="color:#00aa00">**603.8 nsec ± 0.4%**</span> | 22.42 μsec ± 0.3%           |
| 16  | 1000 | 10           | <span style="color:#00aa00">**624.3 nsec ± 1.1%**</span> | 44.47 μsec ± 0.3%           |
| 32  | 1000 | 10           | <span style="color:#00aa00">**664.1 nsec ± 1.2%**</span> | 88.75 μsec ± 0.2%           |
| 64  | 1000 | 10           | <span style="color:#00aa00">**754.6 nsec ± 1.1%**</span> | 181.2 μsec ± 0.2%           |
| 128 | 1000 | 10           | <span style="color:#00aa00">**909.6 nsec ± 0.3%**</span> | 368.6 μsec ± 0.1%           |
| 256 | 1000 | 10           | <span style="color:#00aa00">**1.221 μsec ± 0.5%**</span> | 750.0 μsec ± 0.1%           |
|     |      | **Geomean:** | <span style="color:#00aa00">**650.2 nsec ± 0.9%**</span> | 16.50 μsec ± 2.0%           |

### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                               | `randint_constrained_numba`                    |
| --- | ---- | --------- | --------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 90.6%                                         | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 51.2%                                         | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 2.4%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 1.7%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 4.1%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 18.3%                                         | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 90.0%                                         | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 46.2%                                         | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 1.4%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.8%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 1.7%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 1.0%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.4%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.7%                                          | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 35.91%                                        | <span style="color:#00aa00">**100.00%**</span> |

## Scenario A2

Identical to Scenario A1, but with custom probabilities p provided, favoring larger values.

### Timing Results                                                                                                                                                                                               

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained_numba` |
| --- | ---- | ------------ | -------------------------------------------------------- | --------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**664.7 nsec ± 0.6%**</span> | 1.248 μsec ± 0.5%           |
| 4   | 10   | 10           | <span style="color:#00aa00">**683.3 nsec ± 0.7%**</span> | 1.781 μsec ± 0.3%           |
| 8   | 10   | 10           | <span style="color:#00aa00">**674.0 nsec ± 0.6%**</span> | 2.873 μsec ± 1.0%           |
| 2   | 100  | 10           | <span style="color:#00aa00">**1.052 μsec ± 0.4%**</span> | 1.821 μsec ± 1.1%           |
| 4   | 100  | 10           | <span style="color:#00aa00">**1.211 μsec ± 1.0%**</span> | 2.944 μsec ± 0.5%           |
| 8   | 100  | 10           | <span style="color:#00aa00">**1.473 μsec ± 1.0%**</span> | 5.197 μsec ± 0.7%           |
| 16  | 100  | 10           | <span style="color:#00aa00">**2.147 μsec ± 0.2%**</span> | 9.765 μsec ± 1.1%           |
| 32  | 100  | 10           | <span style="color:#00aa00">**2.233 μsec ± 0.4%**</span> | 19.36 μsec ± 1.3%           |
| 64  | 100  | 10           | <span style="color:#00aa00">**2.247 μsec ± 1.7%**</span> | 39.05 μsec ± 1.1%           |
| 2   | 1000 | 10           | <span style="color:#00aa00">**3.745 μsec ± 0.5%**</span> | 6.053 μsec ± 0.5%           |
| 4   | 1000 | 10           | <span style="color:#00aa00">**4.104 μsec ± 0.4%**</span> | 11.47 μsec ± 0.3%           |
| 8   | 1000 | 10           | <span style="color:#00aa00">**4.741 μsec ± 0.4%**</span> | 23.05 μsec ± 0.3%           |
| 16  | 1000 | 10           | <span style="color:#00aa00">**6.008 μsec ± 0.7%**</span> | 46.05 μsec ± 0.2%           |
| 32  | 1000 | 10           | <span style="color:#00aa00">**8.488 μsec ± 0.8%**</span> | 91.74 μsec ± 0.2%           |
| 64  | 1000 | 10           | <span style="color:#00aa00">**10.10 μsec ± 0.6%**</span> | 186.0 μsec ± 0.2%           |
| 128 | 1000 | 10           | <span style="color:#00aa00">**10.59 μsec ± 0.4%**</span> | 376.4 μsec ± 0.3%           |
| 256 | 1000 | 10           | <span style="color:#00aa00">**11.17 μsec ± 0.7%**</span> | 764.9 μsec ± 0.1%           |
|     |      | **Geomean:** | <span style="color:#00aa00">**2.739 μsec ± 0.6%**</span> | 16.46 μsec ± 0.6%           |

### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                               | `randint_constrained_numba`                    |
| --- | ---- | --------- | --------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 86.7%                                         | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 40.2%                                         | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 0.2%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 79.2%                                         | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 28.9%                                         | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 0.2%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 31.52%                                        | <span style="color:#00aa00">**100.00%**</span> |

## Scenario B1

Fixed n=1000 & k=100 with varying number of constraints spanning random 1% portions of the n range (uniform sampling).

### Timing Results                                                                                                                                                                                               

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained_numba` |
| --- | ---- | ------------ | -------------------------------------------------------- | --------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**876.1 nsec ± 0.7%**</span> | 228.4 μsec ± 0.2%           |
| 100 | 1000 | 4            | <span style="color:#00aa00">**892.2 nsec ± 1.0%**</span> | 230.2 μsec ± 0.3%           |
| 100 | 1000 | 8            | <span style="color:#00aa00">**891.0 nsec ± 1.7%**</span> | 232.8 μsec ± 0.1%           |
| 100 | 1000 | 16           | <span style="color:#00aa00">**887.3 nsec ± 0.5%**</span> | 237.4 μsec ± 0.1%           |
| 100 | 1000 | 32           | <span style="color:#00aa00">**892.0 nsec ± 1.0%**</span> | 247.0 μsec ± 0.2%           |
| 100 | 1000 | 64           | <span style="color:#00aa00">**888.7 nsec ± 0.7%**</span> | 276.2 μsec ± 0.2%           |
| 100 | 1000 | 128          | <span style="color:#00aa00">**871.3 nsec ± 0.7%**</span> | 360.9 μsec ± 0.3%           |
| 100 | 1000 | 256          | <span style="color:#00aa00">**890.5 nsec ± 0.7%**</span> | 471.6 μsec ± 0.3%           |
| 100 | 1000 | 512          | <span style="color:#00aa00">**879.6 nsec ± 0.8%**</span> | 682.7 μsec ± 1.6%           |
| 100 | 1000 | 768          | <span style="color:#00aa00">**889.1 nsec ± 1.2%**</span> | 995.0 μsec ± 0.6%           |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**867.2 nsec ± 1.2%**</span> | 1.283 msec ± 0.4%           |
|     |      | **Geomean:** | <span style="color:#00aa00">**884.1 nsec ± 0.9%**</span> | 387.3 μsec ± 0.4%           |

### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                             | `randint_constrained_numba`                   |
| --- | ---- | --------- | ------------------------------------------- | --------------------------------------------- |
| 100 | 1000 | 2         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 4         | 0.2%                                        | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 8         | 4.2%                                        | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 16        | 98.4%                                       | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 32        | 92.6%                                       | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 64        | 1.1%                                        | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 128       | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 256       | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | <span style="color:#00aa00">**72.8%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 17.86%                                      | <span style="color:#00aa00">**79.35%**</span> |

## Scenario B2

Identical to Scenario B1, but with custom probabilities p provided, favoring larger values.

### Timing Results                                                                                                                                                                                               

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained_numba` |
| --- | ---- | ------------ | -------------------------------------------------------- | --------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**10.41 μsec ± 0.8%**</span> | 239.7 μsec ± 0.4%           |
| 100 | 1000 | 4            | <span style="color:#00aa00">**10.40 μsec ± 0.4%**</span> | 241.4 μsec ± 0.3%           |
| 100 | 1000 | 8            | <span style="color:#00aa00">**10.38 μsec ± 0.3%**</span> | 244.4 μsec ± 0.4%           |
| 100 | 1000 | 16           | <span style="color:#00aa00">**10.40 μsec ± 0.4%**</span> | 249.0 μsec ± 0.1%           |
| 100 | 1000 | 32           | <span style="color:#00aa00">**10.38 μsec ± 0.5%**</span> | 258.5 μsec ± 0.2%           |
| 100 | 1000 | 64           | <span style="color:#00aa00">**10.44 μsec ± 0.4%**</span> | 287.8 μsec ± 0.3%           |
| 100 | 1000 | 128          | <span style="color:#00aa00">**10.46 μsec ± 0.4%**</span> | 372.7 μsec ± 0.2%           |
| 100 | 1000 | 256          | <span style="color:#00aa00">**10.43 μsec ± 0.3%**</span> | 481.8 μsec ± 0.3%           |
| 100 | 1000 | 512          | <span style="color:#00aa00">**10.39 μsec ± 0.6%**</span> | 692.8 μsec ± 0.9%           |
| 100 | 1000 | 768          | <span style="color:#00aa00">**10.60 μsec ± 0.6%**</span> | 1.012 msec ± 0.3%           |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**10.50 μsec ± 0.6%**</span> | 1.301 msec ± 0.3%           |
|     |      | **Geomean:** | <span style="color:#00aa00">**10.43 μsec ± 0.5%**</span> | 400.9 μsec ± 0.3%           |

### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                               | `randint_constrained_numba`                   |
| --- | ---- | --------- | --------------------------------------------- | --------------------------------------------- |
| 100 | 1000 | 2         | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 4         | 0.7%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 8         | 2.4%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 16        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 32        | 96.0%                                         | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 64        | 0.9%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 128       | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 256       | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                          | <span style="color:#00aa00">**68.2%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 18.18%                                        | <span style="color:#00aa00">**78.93%**</span> |