Tested Optimization strategies (1000 iterations):

| `name` | `class`          | `params` | Constraint-aware |
| ------ | ---------------- | -------- | ---------------- |
| `REF`  | OptimRandomSwaps | /        | False            |

### Time Duration

| `d` | `n`  | `k` | `m`          | `REF`                                                    |
| --- | ---- | --- | ------------ | -------------------------------------------------------- |
| 2   | 100  | 10  | 2            | <span style="color:#00aa00">**32.56 msec ± 0.6%**</span> |
| 2   | 200  | 20  | 4            | <span style="color:#00aa00">**42.49 msec ± 0.5%**</span> |
| 2   | 300  | 30  | 6            | <span style="color:#00aa00">**53.31 msec ± 0.7%**</span> |
| 2   | 400  | 40  | 8            | <span style="color:#00aa00">**64.41 msec ± 0.5%**</span> |
| 2   | 600  | 60  | 12           | <span style="color:#00aa00">**85.85 msec ± 0.7%**</span> |
| 2   | 800  | 80  | 16           | <span style="color:#00aa00">**106.7 msec ± 0.4%**</span> |
| 2   | 1200 | 120 | 24           | <span style="color:#00aa00">**151.4 msec ± 0.4%**</span> |
| 2   | 1600 | 160 | 32           | <span style="color:#00aa00">**192.4 msec ± 0.7%**</span> |
| 2   | 2400 | 240 | 48           | <span style="color:#00aa00">**278.8 msec ± 0.7%**</span> |
| 2   | 3200 | 320 | 64           | <span style="color:#00aa00">**365.3 msec ± 1.3%**</span> |
| 2   | 4800 | 480 | 96           | <span style="color:#00aa00">**548.7 msec ± 0.4%**</span> |
| 2   | 6400 | 640 | 128          | <span style="color:#00aa00">**726.6 msec ± 0.5%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**138.9 msec ± 0.6%**</span> |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `REF`                                               |
| --- | ---- | --- | ------------ | --------------------------------------------------- |
| 2   | 100  | 10  | 2            | <span style="color:#00aa00">**0.821 ± 5.2%**</span> |
| 2   | 200  | 20  | 4            | <span style="color:#00aa00">**0.545 ± 2.3%**</span> |
| 2   | 300  | 30  | 6            | <span style="color:#00aa00">**0.383 ± 1.0%**</span> |
| 2   | 400  | 40  | 8            | <span style="color:#00aa00">**0.317 ± 2.0%**</span> |
| 2   | 600  | 60  | 12           | <span style="color:#00aa00">**0.247 ± 1.5%**</span> |
| 2   | 800  | 80  | 16           | <span style="color:#00aa00">**0.204 ± 1.5%**</span> |
| 2   | 1200 | 120 | 24           | <span style="color:#00aa00">**0.159 ± 3.0%**</span> |
| 2   | 1600 | 160 | 32           | <span style="color:#00aa00">**0.130 ± 1.4%**</span> |
| 2   | 2400 | 240 | 48           | <span style="color:#00aa00">**0.099 ± 1.3%**</span> |
| 2   | 3200 | 320 | 64           | <span style="color:#00aa00">**0.080 ± 1.3%**</span> |
| 2   | 4800 | 480 | 96           | <span style="color:#00aa00">**0.061 ± 1.1%**</span> |
| 2   | 6400 | 640 | 128          | <span style="color:#00aa00">**0.048 ± 1.2%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**0.182 ± 1.9%**</span> |

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
| 2   | 4800 | 480 | 96        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 6400 | 640 | 128       | <span style="color:#00aa00">**0.997 ± 0.2%**</span> |
|     |      |     | **Mean:** | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |

