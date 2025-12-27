Tested Optimization strategies (1000 iterations):

| `name` | `class`          | `params` | Constraint-aware |
| ------ | ---------------- | -------- | ---------------- |
| `REF`  | OptimRandomSwaps | /        | False            |

### Time Duration

| `d` | `n`  | `k` | `m`          | `REF`                                                    |
| --- | ---- | --- | ------------ | -------------------------------------------------------- |
| 1   | 150  | 10  | 3            | <span style="color:#00aa00">**37.54 msec ± 0.8%**</span> |
| 2   | 300  | 20  | 6            | <span style="color:#00aa00">**53.07 msec ± 0.5%**</span> |
| 3   | 450  | 30  | 9            | <span style="color:#00aa00">**68.78 msec ± 0.4%**</span> |
| 4   | 600  | 40  | 12           | <span style="color:#00aa00">**85.27 msec ± 0.4%**</span> |
| 6   | 900  | 60  | 18           | <span style="color:#00aa00">**115.8 msec ± 0.6%**</span> |
| 8   | 1200 | 80  | 24           | <span style="color:#00aa00">**150.7 msec ± 1.4%**</span> |
| 12  | 1800 | 120 | 36           | <span style="color:#00aa00">**216.0 msec ± 0.9%**</span> |
| 16  | 2400 | 160 | 48           | <span style="color:#00aa00">**286.7 msec ± 0.8%**</span> |
| 24  | 3600 | 240 | 72           | <span style="color:#00aa00">**428.4 msec ± 0.8%**</span> |
| 32  | 4800 | 320 | 96           | <span style="color:#00aa00">**566.3 msec ± 0.4%**</span> |
| 48  | 7200 | 480 | 144          | <span style="color:#00aa00">**855.1 msec ± 0.3%**</span> |
| 64  | 9600 | 640 | 192          | <span style="color:#00aa00">**1.181 sec  ± 3.2%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**195.4 msec ± 0.9%**</span> |

### Diversity Score

| `d` | `n`  | `k` | `m`          | `REF`                                               |
| --- | ---- | --- | ------------ | --------------------------------------------------- |
| 1   | 150  | 10  | 3            | <span style="color:#00aa00">**0.429 ± 3.5%**</span> |
| 2   | 300  | 20  | 6            | <span style="color:#00aa00">**0.783 ± 2.1%**</span> |
| 3   | 450  | 30  | 9            | <span style="color:#00aa00">**1.035 ± 1.6%**</span> |
| 4   | 600  | 40  | 12           | <span style="color:#00aa00">**1.275 ± 3.0%**</span> |
| 6   | 900  | 60  | 18           | <span style="color:#00aa00">**1.645 ± 2.3%**</span> |
| 8   | 1200 | 80  | 24           | <span style="color:#00aa00">**1.959 ± 1.5%**</span> |
| 12  | 1800 | 120 | 36           | <span style="color:#00aa00">**2.494 ± 1.6%**</span> |
| 16  | 2400 | 160 | 48           | <span style="color:#00aa00">**3.086 ± 0.7%**</span> |
| 24  | 3600 | 240 | 72           | <span style="color:#00aa00">**4.105 ± 0.4%**</span> |
| 32  | 4800 | 320 | 96           | <span style="color:#00aa00">**5.027 ± 0.3%**</span> |
| 48  | 7200 | 480 | 144          | <span style="color:#00aa00">**6.518 ± 0.3%**</span> |
| 64  | 9600 | 640 | 192          | <span style="color:#00aa00">**7.846 ± 0.1%**</span> |
|     |      |     | **Geomean:** | <span style="color:#00aa00">**2.181 ± 1.5%**</span> |

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
| 16  | 2400 | 160 | 48        | <span style="color:#00aa00">**1.000 ± 0.0%**</span> |
| 24  | 3600 | 240 | 72        | <span style="color:#00aa00">**0.996 ± 0.1%**</span> |
| 32  | 4800 | 320 | 96        | <span style="color:#00aa00">**0.988 ± 0.1%**</span> |
| 48  | 7200 | 480 | 144       | <span style="color:#00aa00">**0.975 ± 0.0%**</span> |
| 64  | 9600 | 640 | 192       | <span style="color:#00aa00">**0.967 ± 0.0%**</span> |
|     |      |     | **Mean:** | <span style="color:#00aa00">**0.994 ± 0.0%**</span> |

