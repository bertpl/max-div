Tested Optimization strategies (1000 iterations):

| `name` | `class`          | `params` | Constraint-aware |
| ------ | ---------------- | -------- | ---------------- |
| `REF`  | OptimRandomSwaps | /        | False            |

### Time Duration

| `d` | `n`  | `k` | `m`          | `REF`                                                    |
| --- | ---- | --- | ------------ | -------------------------------------------------------- |
| 2   | 100  | 10  | 2            | <span style="color:#00aa00">**32.63 msec ± 0.4%**</span> |
| 2   | 200  | 20  | 4            | <span style="color:#00aa00">**42.85 msec ± 1.1%**</span> |
| 2   | 300  | 30  | 6            | <span style="color:#00aa00">**54.57 msec ± 1.0%**</span> |
| 2   | 400  | 40  | 8            | <span style="color:#00aa00">**64.94 msec ± 0.8%**</span> |
| 2   | 600  | 60  | 12           | <span style="color:#00aa00">**89.57 msec ± 2.3%**</span> |
| 2   | 800  | 80  | 16           | <span style="color:#00aa00">**108.7 msec ± 1.1%**</span> |
| 2   | 1200 | 120 | 24           | <span style="color:#00aa00">**152.4 msec ± 0.6%**</span> |
| 2   | 1600 | 160 | 32           | <span style="color:#00aa00">**191.6 msec ± 0.6%**</span> |
| 2   | 2400 | 240 | 48           | <span style="color:#00aa00">**275.5 msec ± 0.7%**</span> |
| 2   | 3200 | 320 | 64           | <span style="color:#00aa00">**363.1 msec ± 1.4%**</span> |
| 2   | 4800 | 480 | 96           | <span style="color:#00aa00">**543.3 msec ± 1.7%**</span> |
| 2   | 6400 | 640 | 128          | <span style="color:#00aa00">**708.4 msec ± 0.5%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**139.5 msec ± 1.0%**</span> |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `REF`                                               |
| --- | ---- | --- | ------------ | --------------------------------------------------- |
| 2   | 100  | 10  | 2            | <span style="color:#00aa00">**0.836 ± 1.9%**</span> |
| 2   | 200  | 20  | 4            | <span style="color:#00aa00">**0.550 ± 1.2%**</span> |
| 2   | 300  | 30  | 6            | <span style="color:#00aa00">**0.391 ± 2.1%**</span> |
| 2   | 400  | 40  | 8            | <span style="color:#00aa00">**0.317 ± 2.0%**</span> |
| 2   | 600  | 60  | 12           | <span style="color:#00aa00">**0.243 ± 1.0%**</span> |
| 2   | 800  | 80  | 16           | <span style="color:#00aa00">**0.198 ± 2.1%**</span> |
| 2   | 1200 | 120 | 24           | <span style="color:#00aa00">**0.153 ± 2.6%**</span> |
| 2   | 1600 | 160 | 32           | <span style="color:#00aa00">**0.121 ± 0.6%**</span> |
| 2   | 2400 | 240 | 48           | <span style="color:#00aa00">**0.089 ± 2.3%**</span> |
| 2   | 3200 | 320 | 64           | <span style="color:#00aa00">**0.071 ± 1.4%**</span> |
| 2   | 4800 | 480 | 96           | <span style="color:#00aa00">**0.049 ± 1.1%**</span> |
| 2   | 6400 | 640 | 128          | <span style="color:#00aa00">**0.037 ± 0.9%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**0.170 ± 1.6%**</span> |

### Constraint Score

| `d` | `n`  | `k` | `m`       | `REF`                                               |
| --- | ---- | --- | --------- | --------------------------------------------------- |
| 2   | 100  | 10  | 2         | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 200  | 20  | 4         | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 300  | 30  | 6         | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 400  | 40  | 8         | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 600  | 60  | 12        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 800  | 80  | 16        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 1200 | 120 | 24        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 1600 | 160 | 32        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 2400 | 240 | 48        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 3200 | 320 | 64        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 4800 | 480 | 96        | <span style="color:#00aa00">**0.995 ± 0.4%**</span> |
| 2   | 6400 | 640 | 128       | <span style="color:#00aa00">**0.977 ± 0.4%**</span> |
|     |      |     | **Mean:** | <span style="color:#00aa00">**0.998 ± 0.1%**</span> |

