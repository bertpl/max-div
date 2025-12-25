Tested Optimization strategies (1000 iterations):

| `name` | `class`          | `params` | Constraint-aware |
| ------ | ---------------- | -------- | ---------------- |
| `REF`  | OptimRandomSwaps | /        | False            |

### Time Duration

| `d` | `n`  | `k` | `m`          | `REF`                                                    |
| --- | ---- | --- | ------------ | -------------------------------------------------------- |
| 1   | 150  | 10  | 2            | <span style="color:#00aa00">**37.38 msec ± 0.5%**</span> |
| 2   | 300  | 20  | 4            | <span style="color:#00aa00">**52.61 msec ± 0.6%**</span> |
| 3   | 450  | 30  | 6            | <span style="color:#00aa00">**68.57 msec ± 0.4%**</span> |
| 4   | 600  | 40  | 8            | <span style="color:#00aa00">**84.56 msec ± 0.4%**</span> |
| 6   | 900  | 60  | 12           | <span style="color:#00aa00">**114.7 msec ± 0.9%**</span> |
| 8   | 1200 | 80  | 16           | <span style="color:#00aa00">**148.7 msec ± 0.8%**</span> |
| 12  | 1800 | 120 | 24           | <span style="color:#00aa00">**210.3 msec ± 1.3%**</span> |
| 16  | 2400 | 160 | 32           | <span style="color:#00aa00">**277.9 msec ± 0.6%**</span> |
| 24  | 3600 | 240 | 48           | <span style="color:#00aa00">**408.3 msec ± 1.0%**</span> |
| 32  | 4800 | 320 | 64           | <span style="color:#00aa00">**542.3 msec ± 0.9%**</span> |
| 48  | 7200 | 480 | 96           | <span style="color:#00aa00">**806.6 msec ± 0.8%**</span> |
| 64  | 9600 | 640 | 128          | <span style="color:#00aa00">**1.209 sec  ± 5.5%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**191.6 msec ± 1.1%**</span> |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `REF`                                               |
| --- | ---- | --- | ------------ | --------------------------------------------------- |
| 1   | 150  | 10  | 2            | <span style="color:#00aa00">**0.524 ± 1.6%**</span> |
| 2   | 300  | 20  | 4            | <span style="color:#00aa00">**1.066 ± 3.2%**</span> |
| 3   | 450  | 30  | 6            | <span style="color:#00aa00">**1.408 ± 2.9%**</span> |
| 4   | 600  | 40  | 8            | <span style="color:#00aa00">**1.699 ± 2.1%**</span> |
| 6   | 900  | 60  | 12           | <span style="color:#00aa00">**2.166 ± 2.0%**</span> |
| 8   | 1200 | 80  | 16           | <span style="color:#00aa00">**2.531 ± 2.1%**</span> |
| 12  | 1800 | 120 | 24           | <span style="color:#00aa00">**3.048 ± 1.0%**</span> |
| 16  | 2400 | 160 | 32           | <span style="color:#00aa00">**3.602 ± 0.7%**</span> |
| 24  | 3600 | 240 | 48           | <span style="color:#00aa00">**4.723 ± 0.2%**</span> |
| 32  | 4800 | 320 | 64           | <span style="color:#00aa00">**5.732 ± 0.3%**</span> |
| 48  | 7200 | 480 | 96           | <span style="color:#00aa00">**7.363 ± 0.4%**</span> |
| 64  | 9600 | 640 | 128          | <span style="color:#00aa00">**8.775 ± 0.1%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**2.686 ± 1.4%**</span> |

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
| 24  | 3600 | 240 | 48        | <span style="color:#00aa00">**0.993 ± 0.1%**</span> |
| 32  | 4800 | 320 | 64        | <span style="color:#00aa00">**0.976 ± 0.2%**</span> |
| 48  | 7200 | 480 | 96        | <span style="color:#00aa00">**0.949 ± 0.2%**</span> |
| 64  | 9600 | 640 | 128       | <span style="color:#00aa00">**0.934 ± 0.1%**</span> |
|     |      |     | **Mean:** | <span style="color:#00aa00">**0.988 ± 0.0%**</span> |

