Tested Optimization strategies (1000 iterations):

| `name` | `class`          | `params` | Constraint-aware |
| ------ | ---------------- | -------- | ---------------- |
| `REF`  | OptimRandomSwaps | /        | False            |

### Time Duration

| `d` | `n`  | `k` | `m`          | `REF`                                                    |
| --- | ---- | --- | ------------ | -------------------------------------------------------- |
| 1   | 150  | 10  | 2            | <span style="color:#00aa00">**37.47 msec ± 0.6%**</span> |
| 2   | 300  | 20  | 4            | <span style="color:#00aa00">**52.55 msec ± 0.8%**</span> |
| 3   | 450  | 30  | 6            | <span style="color:#00aa00">**68.09 msec ± 0.6%**</span> |
| 4   | 600  | 40  | 8            | <span style="color:#00aa00">**83.98 msec ± 0.5%**</span> |
| 6   | 900  | 60  | 12           | <span style="color:#00aa00">**113.3 msec ± 0.5%**</span> |
| 8   | 1200 | 80  | 16           | <span style="color:#00aa00">**148.0 msec ± 0.5%**</span> |
| 12  | 1800 | 120 | 24           | <span style="color:#00aa00">**207.8 msec ± 0.8%**</span> |
| 16  | 2400 | 160 | 32           | <span style="color:#00aa00">**277.5 msec ± 0.8%**</span> |
| 24  | 3600 | 240 | 48           | <span style="color:#00aa00">**400.4 msec ± 0.4%**</span> |
| 32  | 4800 | 320 | 64           | <span style="color:#00aa00">**544.6 msec ± 1.0%**</span> |
| 48  | 7200 | 480 | 96           | <span style="color:#00aa00">**825.7 msec ± 1.1%**</span> |
| 64  | 9600 | 640 | 128          | <span style="color:#00aa00">**1.129 sec  ± 3.8%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**190.0 msec ± 0.9%**</span> |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `REF`                                               |
| --- | ---- | --- | ------------ | --------------------------------------------------- |
| 1   | 150  | 10  | 2            | <span style="color:#00aa00">**0.518 ± 2.2%**</span> |
| 2   | 300  | 20  | 4            | <span style="color:#00aa00">**1.040 ± 2.2%**</span> |
| 3   | 450  | 30  | 6            | <span style="color:#00aa00">**1.384 ± 1.3%**</span> |
| 4   | 600  | 40  | 8            | <span style="color:#00aa00">**1.711 ± 2.7%**</span> |
| 6   | 900  | 60  | 12           | <span style="color:#00aa00">**2.129 ± 2.8%**</span> |
| 8   | 1200 | 80  | 16           | <span style="color:#00aa00">**2.459 ± 2.4%**</span> |
| 12  | 1800 | 120 | 24           | <span style="color:#00aa00">**3.081 ± 1.6%**</span> |
| 16  | 2400 | 160 | 32           | <span style="color:#00aa00">**3.645 ± 0.6%**</span> |
| 24  | 3600 | 240 | 48           | <span style="color:#00aa00">**4.710 ± 0.9%**</span> |
| 32  | 4800 | 320 | 64           | <span style="color:#00aa00">**5.722 ± 0.5%**</span> |
| 48  | 7200 | 480 | 96           | <span style="color:#00aa00">**7.389 ± 0.1%**</span> |
| 64  | 9600 | 640 | 128          | <span style="color:#00aa00">**8.802 ± 0.1%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**2.670 ± 1.4%**</span> |

### Constraint Score

| `d` | `n`  | `k` | `m`       | `REF`                                               |
| --- | ---- | --- | --------- | --------------------------------------------------- |
| 1   | 150  | 10  | 2         | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 300  | 20  | 4         | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 3   | 450  | 30  | 6         | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 4   | 600  | 40  | 8         | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 6   | 900  | 60  | 12        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 8   | 1200 | 80  | 16        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 12  | 1800 | 120 | 24        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 16  | 2400 | 160 | 32        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 24  | 3600 | 240 | 48        | <span style="color:#00aa00">**0.991 ± 0.1%**</span> |
| 32  | 4800 | 320 | 64        | <span style="color:#00aa00">**0.976 ± 0.3%**</span> |
| 48  | 7200 | 480 | 96        | <span style="color:#00aa00">**0.948 ± 0.1%**</span> |
| 64  | 9600 | 640 | 128       | <span style="color:#00aa00">**0.934 ± 0.1%**</span> |
|     |      |     | **Mean:** | <span style="color:#00aa00">**0.987 ± 0.1%**</span> |

