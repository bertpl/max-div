Tested Optimization strategies (1000 iterations):

| `name` | `class`          | `params` | Constraint-aware |
| ------ | ---------------- | -------- | ---------------- |
| `REF`  | OptimRandomSwaps | /        | False            |

### Time Duration

| `d` | `n`  | `k` | `m`          | `REF`                                                    |
| --- | ---- | --- | ------------ | -------------------------------------------------------- |
| 1   | 150  | 10  | 2            | <span style="color:#00aa00">**38.06 msec ± 0.7%**</span> |
| 2   | 300  | 20  | 4            | <span style="color:#00aa00">**53.97 msec ± 0.8%**</span> |
| 3   | 450  | 30  | 6            | <span style="color:#00aa00">**69.19 msec ± 0.7%**</span> |
| 4   | 600  | 40  | 8            | <span style="color:#00aa00">**85.13 msec ± 0.4%**</span> |
| 6   | 900  | 60  | 12           | <span style="color:#00aa00">**115.9 msec ± 0.8%**</span> |
| 8   | 1200 | 80  | 16           | <span style="color:#00aa00">**151.7 msec ± 3.8%**</span> |
| 12  | 1800 | 120 | 24           | <span style="color:#00aa00">**214.6 msec ± 2.3%**</span> |
| 16  | 2400 | 160 | 32           | <span style="color:#00aa00">**281.6 msec ± 3.4%**</span> |
| 24  | 3600 | 240 | 48           | <span style="color:#00aa00">**409.1 msec ± 0.9%**</span> |
| 32  | 4800 | 320 | 64           | <span style="color:#00aa00">**571.5 msec ± 3.0%**</span> |
| 48  | 7200 | 480 | 96           | <span style="color:#00aa00">**832.8 msec ± 1.7%**</span> |
| 64  | 9600 | 640 | 128          | <span style="color:#00aa00">**1.129 sec  ± 0.9%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**193.9 msec ± 1.6%**</span> |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `REF`                                               |
| --- | ---- | --- | ------------ | --------------------------------------------------- |
| 1   | 150  | 10  | 2            | <span style="color:#00aa00">**0.530 ± 2.3%**</span> |
| 2   | 300  | 20  | 4            | <span style="color:#00aa00">**1.052 ± 2.3%**</span> |
| 3   | 450  | 30  | 6            | <span style="color:#00aa00">**1.381 ± 2.0%**</span> |
| 4   | 600  | 40  | 8            | <span style="color:#00aa00">**1.648 ± 2.1%**</span> |
| 6   | 900  | 60  | 12           | <span style="color:#00aa00">**2.131 ± 2.2%**</span> |
| 8   | 1200 | 80  | 16           | <span style="color:#00aa00">**2.401 ± 2.8%**</span> |
| 12  | 1800 | 120 | 24           | <span style="color:#00aa00">**2.851 ± 2.2%**</span> |
| 16  | 2400 | 160 | 32           | <span style="color:#00aa00">**3.230 ± 2.7%**</span> |
| 24  | 3600 | 240 | 48           | <span style="color:#00aa00">**4.236 ± 0.3%**</span> |
| 32  | 4800 | 320 | 64           | <span style="color:#00aa00">**5.240 ± 0.4%**</span> |
| 48  | 7200 | 480 | 96           | <span style="color:#00aa00">**6.747 ± 0.0%**</span> |
| 64  | 9600 | 640 | 128          | <span style="color:#00aa00">**8.061 ± 0.1%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**2.540 ± 1.6%**</span> |

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
| 24  | 3600 | 240 | 48        | <span style="color:#00aa00">**0.999 ± 0.1%**</span> |
| 32  | 4800 | 320 | 64        | <span style="color:#00aa00">**0.986 ± 0.1%**</span> |
| 48  | 7200 | 480 | 96        | <span style="color:#00aa00">**0.962 ± 0.1%**</span> |
| 64  | 9600 | 640 | 128       | <span style="color:#00aa00">**0.949 ± 0.0%**</span> |
|     |      |     | **Mean:** | <span style="color:#00aa00">**0.991 ± 0.0%**</span> |

