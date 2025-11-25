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
- use `randint_constrained`, i.e. we sample taking constraints into account

We test both speed (absolute time of a single call) and accuracy (% of cases the generated samples satisfied the constraints).


## Scenario A

Varying n & k with 10 non-overlapping constraints spanning equal portions of the n-range

### Uniform sampling.

#### Timing Results                                                                                                                                                                                              

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ |----------------------------------------------------------| -------------------------------------------- | ------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**544.0 nsec ± 0.4%**</span> | 1.420 μsec ± 0.4%                            | 1.414 μsec ± 0.4%                           |
| 4   | 10   | 10           | <span style="color:#00aa00">**548.6 nsec ± 1.0%**</span> | 1.959 μsec ± 0.5%                            | 1.925 μsec ± 0.4%                           |
| 8   | 10   | 10           | <span style="color:#00aa00">**573.8 nsec ± 0.4%**</span> | 3.074 μsec ± 0.3%                            | 3.078 μsec ± 0.2%                           |
| 2   | 100  | 10           | <span style="color:#00aa00">**597.3 nsec ± 0.4%**</span> | 1.994 μsec ± 0.6%                            | 2.002 μsec ± 0.3%                           |
| 4   | 100  | 10           | <span style="color:#00aa00">**665.9 nsec ± 5.9%**</span> | 3.036 μsec ± 1.0%                            | 3.078 μsec ± 0.6%                           |
| 8   | 100  | 10           | <span style="color:#00aa00">**609.2 nsec ± 0.7%**</span> | 5.311 μsec ± 0.6%                            | 5.331 μsec ± 0.8%                           |
| 16  | 100  | 10           | <span style="color:#00aa00">**639.2 nsec ± 0.4%**</span> | 9.890 μsec ± 0.7%                            | 10.08 μsec ± 0.1%                           |
| 32  | 100  | 10           | <span style="color:#00aa00">**681.5 nsec ± 0.8%**</span> | 19.51 μsec ± 0.8%                            | 19.31 μsec ± 1.0%                           |
| 64  | 100  | 10           | <span style="color:#00aa00">**770.8 nsec ± 0.5%**</span> | 38.90 μsec ± 0.8%                            | 38.88 μsec ± 0.5%                           |
| 2   | 1000 | 10           | <span style="color:#00aa00">**625.7 nsec ± 1.1%**</span> | 6.013 μsec ± 0.4%                            | 6.088 μsec ± 0.2%                           |
| 4   | 1000 | 10           | <span style="color:#00aa00">**626.4 nsec ± 1.0%**</span> | 11.35 μsec ± 0.2%                            | 11.30 μsec ± 0.4%                           |
| 8   | 1000 | 10           | <span style="color:#00aa00">**643.7 nsec ± 1.2%**</span> | 22.34 μsec ± 0.2%                            | 22.30 μsec ± 0.3%                           |
| 16  | 1000 | 10           | <span style="color:#00aa00">**667.9 nsec ± 1.5%**</span> | 44.84 μsec ± 0.2%                            | 44.94 μsec ± 0.7%                           |
| 32  | 1000 | 10           | <span style="color:#00aa00">**695.3 nsec ± 1.2%**</span> | 89.31 μsec ± 0.1%                            | 89.07 μsec ± 0.2%                           |
| 64  | 1000 | 10           | <span style="color:#00aa00">**788.8 nsec ± 0.7%**</span> | 181.7 μsec ± 0.2%                            | 181.4 μsec ± 0.5%                           |
| 128 | 1000 | 10           | <span style="color:#00aa00">**960.3 nsec ± 0.4%**</span> | 369.7 μsec ± 0.2%                            | 369.2 μsec ± 0.2%                           |
| 256 | 1000 | 10           | <span style="color:#00aa00">**1.271 μsec ± 0.4%**</span> | 751.5 μsec ± 0.2%                            | 751.9 μsec ± 0.1%                           |
|     |      | **Geomean:** | <span style="color:#00aa00">**684.0 nsec ± 1.1%**</span> | 16.74 μsec ± 0.4%                            | 16.75 μsec ± 0.4%                           |

#### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                               | `randint_constrained`<br>(eager=False)   | `randint_constrained`<br>(eager=True)    |
| --- | ---- | --------- |-----------------------------------------------| ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 90.6%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 51.1%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 2.4%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 1.7%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 4.1%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 18.3%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 89.9%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 46.3%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 1.4%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.8%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 1.7%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 1.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.4%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.7%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 35.91%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |

### Non-uniform sampling (custom p).

#### Timing Results                                                                                                                                                                                              

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ |----------------------------------------------------------| -------------------------------------------- | ------------------------------------------- |
| 2   | 10   | 10           | <span style="color:#00aa00">**680.3 nsec ± 0.7%**</span> | 1.309 μsec ± 0.6%                            | 1.323 μsec ± 0.5%                           |
| 4   | 10   | 10           | <span style="color:#00aa00">**694.7 nsec ± 1.2%**</span> | 1.866 μsec ± 0.5%                            | 1.843 μsec ± 0.6%                           |
| 8   | 10   | 10           | <span style="color:#00aa00">**678.6 nsec ± 0.7%**</span> | 2.975 μsec ± 0.3%                            | 2.978 μsec ± 0.3%                           |
| 2   | 100  | 10           | <span style="color:#00aa00">**1.091 μsec ± 0.5%**</span> | 1.907 μsec ± 0.2%                            | 1.924 μsec ± 1.6%                           |
| 4   | 100  | 10           | <span style="color:#00aa00">**1.228 μsec ± 0.7%**</span> | 3.078 μsec ± 0.6%                            | 3.050 μsec ± 0.1%                           |
| 8   | 100  | 10           | <span style="color:#00aa00">**1.516 μsec ± 0.2%**</span> | 5.444 μsec ± 1.6%                            | 5.313 μsec ± 0.2%                           |
| 16  | 100  | 10           | <span style="color:#00aa00">**2.201 μsec ± 0.2%**</span> | 10.09 μsec ± 0.3%                            | 10.15 μsec ± 0.3%                           |
| 32  | 100  | 10           | <span style="color:#00aa00">**2.266 μsec ± 0.3%**</span> | 19.57 μsec ± 0.4%                            | 19.30 μsec ± 0.4%                           |
| 64  | 100  | 10           | <span style="color:#00aa00">**2.260 μsec ± 0.6%**</span> | 38.80 μsec ± 4.7%                            | 39.74 μsec ± 0.4%                           |
| 2   | 1000 | 10           | <span style="color:#00aa00">**3.882 μsec ± 0.5%**</span> | 6.118 μsec ± 0.2%                            | 6.176 μsec ± 1.3%                           |
| 4   | 1000 | 10           | <span style="color:#00aa00">**4.170 μsec ± 0.5%**</span> | 11.62 μsec ± 0.8%                            | 11.59 μsec ± 0.2%                           |
| 8   | 1000 | 10           | <span style="color:#00aa00">**4.868 μsec ± 0.5%**</span> | 23.08 μsec ± 0.1%                            | 23.12 μsec ± 0.2%                           |
| 16  | 1000 | 10           | <span style="color:#00aa00">**6.254 μsec ± 0.6%**</span> | 46.41 μsec ± 0.2%                            | 46.60 μsec ± 2.1%                           |
| 32  | 1000 | 10           | <span style="color:#00aa00">**8.743 μsec ± 0.6%**</span> | 92.35 μsec ± 0.2%                            | 92.17 μsec ± 0.3%                           |
| 64  | 1000 | 10           | <span style="color:#00aa00">**10.27 μsec ± 0.3%**</span> | 188.4 μsec ± 0.7%                            | 187.8 μsec ± 0.5%                           |
| 128 | 1000 | 10           | <span style="color:#00aa00">**10.70 μsec ± 0.4%**</span> | 376.4 μsec ± 0.2%                            | 374.9 μsec ± 0.3%                           |
| 256 | 1000 | 10           | <span style="color:#00aa00">**11.39 μsec ± 0.3%**</span> | 764.4 μsec ± 0.1%                            | 766.1 μsec ± 0.2%                           |
|     |      | **Geomean:** | <span style="color:#00aa00">**2.798 μsec ± 0.5%**</span> | 16.80 μsec ± 0.7%                            | 16.80 μsec ± 0.6%                           |

#### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                               | `randint_constrained`<br>(eager=False)   | `randint_constrained`<br>(eager=True)    |
| --- | ---- | --------- |-----------------------------------------------| ---------------------------------------------- | ---------------------------------------------- |
| 2   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 10   | 10        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 100  | 10        | 86.7%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 100  | 10        | 40.2%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 100  | 10        | 0.5%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 100  | 10        | 0.2%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 100  | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 2   | 1000 | 10        | 79.1%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 4   | 1000 | 10        | 28.9%                                         | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 8   | 1000 | 10        | 0.2%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 16  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 32  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 64  | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 128 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
| 256 | 1000 | 10        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span>  | <span style="color:#00aa00">**100.0%**</span>  |
|     |      | **Mean:** | 31.52%                                        | <span style="color:#00aa00">**100.00%**</span> | <span style="color:#00aa00">**100.00%**</span> |

## Scenario B

Fixed n=1000 & k=100 with varying number of constraints spanning random 1% portions of the n-range

### Uniform sampling.

#### Timing Results                                                                                                                                                                                              

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ |----------------------------------------------------------| -------------------------------------------- | ------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**877.3 nsec ± 2.0%**</span> | 228.7 μsec ± 0.2%                            | 227.5 μsec ± 0.2%                           |
| 100 | 1000 | 4            | <span style="color:#00aa00">**883.8 nsec ± 0.7%**</span> | 230.5 μsec ± 0.2%                            | 230.4 μsec ± 0.2%                           |
| 100 | 1000 | 8            | <span style="color:#00aa00">**878.2 nsec ± 1.2%**</span> | 234.5 μsec ± 0.2%                            | 233.4 μsec ± 0.2%                           |
| 100 | 1000 | 16           | <span style="color:#00aa00">**884.2 nsec ± 1.5%**</span> | 242.0 μsec ± 0.2%                            | 238.6 μsec ± 0.1%                           |
| 100 | 1000 | 32           | <span style="color:#00aa00">**861.6 nsec ± 1.7%**</span> | 260.5 μsec ± 0.1%                            | 250.8 μsec ± 0.2%                           |
| 100 | 1000 | 64           | <span style="color:#00aa00">**874.6 nsec ± 1.2%**</span> | 300.9 μsec ± 0.1%                            | 275.5 μsec ± 0.2%                           |
| 100 | 1000 | 128          | <span style="color:#00aa00">**868.8 nsec ± 2.0%**</span> | 370.1 μsec ± 0.3%                            | 323.2 μsec ± 0.4%                           |
| 100 | 1000 | 256          | <span style="color:#00aa00">**882.6 nsec ± 1.2%**</span> | 478.7 μsec ± 0.2%                            | 425.2 μsec ± 0.2%                           |
| 100 | 1000 | 384          | <span style="color:#00aa00">**883.9 nsec ± 1.3%**</span> | 606.3 μsec ± 0.8%                            | 537.2 μsec ± 0.2%                           |
| 100 | 1000 | 512          | <span style="color:#00aa00">**900.7 nsec ± 0.6%**</span> | 746.5 μsec ± 0.5%                            | 663.5 μsec ± 0.6%                           |
| 100 | 1000 | 768          | <span style="color:#00aa00">**884.1 nsec ± 1.3%**</span> | 1.043 msec ± 0.6%                            | 973.9 μsec ± 0.3%                           |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**873.1 nsec ± 0.7%**</span> | 1.462 msec ± 0.5%                            | 1.393 msec ± 0.3%                           |
|     |      | **Geomean:** | <span style="color:#00aa00">**879.4 nsec ± 1.3%**</span> | 418.2 μsec ± 0.3%                            | 392.6 μsec ± 0.3%                           |

#### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                             | `randint_constrained`<br>(eager=False)  | `randint_constrained`<br>(eager=True)   |
| --- | ---- | --------- |---------------------------------------------| --------------------------------------------- | --------------------------------------------- |
| 100 | 1000 | 2         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 4         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 8         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 16        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 32        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 64        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 128       | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 256       | 0.0%                                        | 90.6%                                         | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 384       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**46.6%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 65.88%                                        | <span style="color:#00aa00">**78.88%**</span> |

### Non-uniform sampling (custom p).

#### Timing Results                                                                                                                                                                                              

| `k` | `n`  | `n_cons`     | `randint_numba`                                          | `randint_constrained`<br>(eager=False) | `randint_constrained`<br>(eager=True) |
| --- | ---- | ------------ |----------------------------------------------------------| -------------------------------------------- | ------------------------------------------- |
| 100 | 1000 | 2            | <span style="color:#00aa00">**10.68 μsec ± 0.6%**</span> | 238.8 μsec ± 0.2%                            | 239.6 μsec ± 0.9%                           |
| 100 | 1000 | 4            | <span style="color:#00aa00">**10.63 μsec ± 0.5%**</span> | 241.2 μsec ± 0.2%                            | 241.0 μsec ± 0.3%                           |
| 100 | 1000 | 8            | <span style="color:#00aa00">**10.61 μsec ± 0.7%**</span> | 245.1 μsec ± 0.2%                            | 243.9 μsec ± 0.2%                           |
| 100 | 1000 | 16           | <span style="color:#00aa00">**10.66 μsec ± 0.4%**</span> | 253.0 μsec ± 0.3%                            | 250.2 μsec ± 0.3%                           |
| 100 | 1000 | 32           | <span style="color:#00aa00">**10.59 μsec ± 0.5%**</span> | 271.4 μsec ± 0.2%                            | 261.7 μsec ± 0.2%                           |
| 100 | 1000 | 64           | <span style="color:#00aa00">**10.64 μsec ± 0.6%**</span> | 312.4 μsec ± 0.2%                            | 286.8 μsec ± 0.2%                           |
| 100 | 1000 | 128          | <span style="color:#00aa00">**10.66 μsec ± 0.2%**</span> | 381.2 μsec ± 0.2%                            | 334.1 μsec ± 0.5%                           |
| 100 | 1000 | 256          | <span style="color:#00aa00">**10.72 μsec ± 0.5%**</span> | 492.3 μsec ± 0.4%                            | 435.4 μsec ± 0.2%                           |
| 100 | 1000 | 384          | <span style="color:#00aa00">**10.66 μsec ± 0.7%**</span> | 624.0 μsec ± 0.3%                            | 550.0 μsec ± 0.5%                           |
| 100 | 1000 | 512          | <span style="color:#00aa00">**10.69 μsec ± 0.4%**</span> | 775.2 μsec ± 0.4%                            | 684.4 μsec ± 0.4%                           |
| 100 | 1000 | 768          | <span style="color:#00aa00">**10.77 μsec ± 0.6%**</span> | 1.051 msec ± 0.2%                            | 984.5 μsec ± 0.3%                           |
| 100 | 1000 | 1024         | <span style="color:#00aa00">**10.68 μsec ± 0.2%**</span> | 1.479 msec ± 0.2%                            | 1.432 msec ± 0.4%                           |
|     |      | **Geomean:** | <span style="color:#00aa00">**10.67 μsec ± 0.5%**</span> | 432.4 μsec ± 0.2%                            | 406.6 μsec ± 0.4%                           |

#### Accuracy Results

| `k` | `n`  | `n_cons`  | `randint_numba`                             | `randint_constrained`<br>(eager=False)  | `randint_constrained`<br>(eager=True)   |
| --- | ---- | --------- |---------------------------------------------| --------------------------------------------- | --------------------------------------------- |
| 100 | 1000 | 2         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 4         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 8         | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 16        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 32        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 64        | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 128       | 0.0%                                        | <span style="color:#00aa00">**100.0%**</span> | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 256       | 0.0%                                        | 89.5%                                         | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 384       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**100.0%**</span> |
| 100 | 1000 | 512       | 0.0%                                        | 0.0%                                          | <span style="color:#00aa00">**48.1%**</span>  |
| 100 | 1000 | 768       | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
| 100 | 1000 | 1024      | <span style="color:#00aa00">**0.0%**</span> | <span style="color:#00aa00">**0.0%**</span>   | <span style="color:#00aa00">**0.0%**</span>   |
|     |      | **Mean:** | 0.00%                                       | 65.79%                                        | <span style="color:#00aa00">**79.01%**</span> |