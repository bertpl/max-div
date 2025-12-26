Tested Optimization strategies (1000 iterations):

| `name` | `class`          | `params` | Constraint-aware |
| ------ | ---------------- | -------- | ---------------- |
| `REF`  | OptimRandomSwaps | /        | False            |

### Time Duration

| `d` | `n`  | `k` | `m`          | `REF`                                                    |
| --- | ---- | --- | ------------ | -------------------------------------------------------- |
| 1   | 150  | 10  | 3            | <span style="color:#00aa00">**37.51 msec ± 0.5%**</span> |
| 2   | 300  | 20  | 6            | <span style="color:#00aa00">**53.18 msec ± 0.5%**</span> |
| 3   | 450  | 30  | 9            | <span style="color:#00aa00">**69.13 msec ± 0.7%**</span> |
| 4   | 600  | 40  | 12           | <span style="color:#00aa00">**85.08 msec ± 0.4%**</span> |
| 6   | 900  | 60  | 18           | <span style="color:#00aa00">**115.6 msec ± 0.6%**</span> |
| 8   | 1200 | 80  | 24           | <span style="color:#00aa00">**150.1 msec ± 1.1%**</span> |
| 12  | 1800 | 120 | 36           | <span style="color:#00aa00">**212.5 msec ± 1.0%**</span> |
| 16  | 2400 | 160 | 48           | <span style="color:#00aa00">**282.7 msec ± 1.6%**</span> |
| 24  | 3600 | 240 | 72           | <span style="color:#00aa00">**415.1 msec ± 1.1%**</span> |
| 32  | 4800 | 320 | 96           | <span style="color:#00aa00">**569.0 msec ± 1.2%**</span> |
| 48  | 7200 | 480 | 144          | <span style="color:#00aa00">**829.3 msec ± 0.2%**</span> |
| 64  | 9600 | 640 | 192          | <span style="color:#00aa00">**1.242 sec  ± 1.1%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**194.7 msec ± 0.8%**</span> |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `REF`                                               |
| --- | ---- | --- | ------------ | --------------------------------------------------- |
| 1   | 150  | 10  | 3            | <span style="color:#00aa00">**0.432 ± 1.8%**</span> |
| 2   | 300  | 20  | 6            | <span style="color:#00aa00">**0.760 ± 4.1%**</span> |
| 3   | 450  | 30  | 9            | <span style="color:#00aa00">**1.049 ± 2.7%**</span> |
| 4   | 600  | 40  | 12           | <span style="color:#00aa00">**1.309 ± 1.8%**</span> |
| 6   | 900  | 60  | 18           | <span style="color:#00aa00">**1.683 ± 1.9%**</span> |
| 8   | 1200 | 80  | 24           | <span style="color:#00aa00">**1.982 ± 1.5%**</span> |
| 12  | 1800 | 120 | 36           | <span style="color:#00aa00">**2.663 ± 0.8%**</span> |
| 16  | 2400 | 160 | 48           | <span style="color:#00aa00">**3.287 ± 0.6%**</span> |
| 24  | 3600 | 240 | 72           | <span style="color:#00aa00">**4.392 ± 0.3%**</span> |
| 32  | 4800 | 320 | 96           | <span style="color:#00aa00">**5.422 ± 0.1%**</span> |
| 48  | 7200 | 480 | 144          | <span style="color:#00aa00">**7.098 ± 0.2%**</span> |
| 64  | 9600 | 640 | 192          | <span style="color:#00aa00">**8.528 ± 0.2%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**2.273 ± 1.3%**</span> |

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
| 16  | 2400 | 160 | 48        | <span style="color:#00aa00">**0.998 ± 0.1%**</span> |
| 24  | 3600 | 240 | 72        | <span style="color:#00aa00">**0.984 ± 0.1%**</span> |
| 32  | 4800 | 320 | 96        | <span style="color:#00aa00">**0.966 ± 0.2%**</span> |
| 48  | 7200 | 480 | 144       | <span style="color:#00aa00">**0.944 ± 0.1%**</span> |
| 64  | 9600 | 640 | 192       | <span style="color:#00aa00">**0.929 ± 0.1%**</span> |
|     |      |     | **Mean:** | <span style="color:#00aa00">**0.985 ± 0.0%**</span> |

