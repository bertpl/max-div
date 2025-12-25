Tested Optimization strategies (1000 iterations):

| `name` | `class`          | `params` | Constraint-aware |
| ------ | ---------------- | -------- | ---------------- |
| `REF`  | OptimRandomSwaps | /        | False            |

### Time Duration

| `d` | `n`  | `k` | `m`          | `REF`                                                    |
| --- | ---- | --- | ------------ | -------------------------------------------------------- |
| 1   | 150  | 10  | 3            | <span style="color:#00aa00">**37.55 msec ± 0.6%**</span> |
| 2   | 300  | 20  | 6            | <span style="color:#00aa00">**52.78 msec ± 0.5%**</span> |
| 3   | 450  | 30  | 9            | <span style="color:#00aa00">**69.15 msec ± 0.4%**</span> |
| 4   | 600  | 40  | 12           | <span style="color:#00aa00">**84.92 msec ± 0.3%**</span> |
| 6   | 900  | 60  | 18           | <span style="color:#00aa00">**116.1 msec ± 0.5%**</span> |
| 8   | 1200 | 80  | 24           | <span style="color:#00aa00">**150.7 msec ± 0.6%**</span> |
| 12  | 1800 | 120 | 36           | <span style="color:#00aa00">**217.2 msec ± 2.8%**</span> |
| 16  | 2400 | 160 | 48           | <span style="color:#00aa00">**307.1 msec ± 8.5%**</span> |
| 24  | 3600 | 240 | 72           | <span style="color:#00aa00">**423.5 msec ± 5.6%**</span> |
| 32  | 4800 | 320 | 96           | <span style="color:#00aa00">**560.9 msec ± 2.2%**</span> |
| 48  | 7200 | 480 | 144          | <span style="color:#00aa00">**881.0 msec ± 3.8%**</span> |
| 64  | 9600 | 640 | 192          | <span style="color:#00aa00">**1.173 sec  ± 4.8%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**196.6 msec ± 2.5%**</span> |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `REF`                                               |
| --- | ---- | --- | ------------ | --------------------------------------------------- |
| 1   | 150  | 10  | 3            | <span style="color:#00aa00">**0.425 ± 2.5%**</span> |
| 2   | 300  | 20  | 6            | <span style="color:#00aa00">**0.776 ± 2.8%**</span> |
| 3   | 450  | 30  | 9            | <span style="color:#00aa00">**1.056 ± 2.3%**</span> |
| 4   | 600  | 40  | 12           | <span style="color:#00aa00">**1.294 ± 1.4%**</span> |
| 6   | 900  | 60  | 18           | <span style="color:#00aa00">**1.665 ± 1.8%**</span> |
| 8   | 1200 | 80  | 24           | <span style="color:#00aa00">**1.979 ± 1.6%**</span> |
| 12  | 1800 | 120 | 36           | <span style="color:#00aa00">**2.685 ± 0.9%**</span> |
| 16  | 2400 | 160 | 48           | <span style="color:#00aa00">**3.275 ± 0.6%**</span> |
| 24  | 3600 | 240 | 72           | <span style="color:#00aa00">**4.397 ± 0.6%**</span> |
| 32  | 4800 | 320 | 96           | <span style="color:#00aa00">**5.388 ± 0.4%**</span> |
| 48  | 7200 | 480 | 144          | <span style="color:#00aa00">**7.083 ± 0.4%**</span> |
| 64  | 9600 | 640 | 192          | <span style="color:#00aa00">**8.535 ± 0.1%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**2.270 ± 1.3%**</span> |

### Constraint Score

| `d` | `n`  | `k` | `m`       | `REF`                                               |
| --- | ---- | --- | --------- | --------------------------------------------------- |
| 1   | 150  | 10  | 3         | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 2   | 300  | 20  | 6         | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 3   | 450  | 30  | 9         | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 4   | 600  | 40  | 12        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 6   | 900  | 60  | 18        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 8   | 1200 | 80  | 24        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 12  | 1800 | 120 | 36        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 16  | 2400 | 160 | 48        | <span style="color:#00aa00">**0.997 ± 0.1%**</span> |
| 24  | 3600 | 240 | 72        | <span style="color:#00aa00">**0.984 ± 0.2%**</span> |
| 32  | 4800 | 320 | 96        | <span style="color:#00aa00">**0.966 ± 0.2%**</span> |
| 48  | 7200 | 480 | 144       | <span style="color:#00aa00">**0.944 ± 0.1%**</span> |
| 64  | 9600 | 640 | 192       | <span style="color:#00aa00">**0.928 ± 0.1%**</span> |
|     |      |     | **Mean:** | <span style="color:#00aa00">**0.985 ± 0.1%**</span> |

