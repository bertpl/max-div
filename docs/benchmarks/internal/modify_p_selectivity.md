# `modify_p_selectivity_*`

Command:
```bash
uv tool install max-div
max-div benchmark --markdown modify_p_selectivity
```
or 
```bash
uv run max-div benchmark --markdown modify_p_selectivity
```

We compare speed of modifying the selectivity of a float32-array of probabilities using different methods:

- `power` --> `modify_p_selectivity_power`
- `pwl2` --> `modify_p_selectivity_pwl2`

## modify_p_selectivity Performance

                                                                                                                                                                                                           
| `size`       | `power`           | `pwl2`                                                   |
| ------------ | ----------------- | -------------------------------------------------------- |
| 2            | 363.0 nsec ± 1.9% | <span style="color:#00aa00">**349.5 nsec ± 1.5%**</span> |
| 4            | 364.7 nsec ± 0.7% | <span style="color:#00aa00">**343.4 nsec ± 0.4%**</span> |
| 8            | 395.2 nsec ± 0.2% | <span style="color:#00aa00">**346.3 nsec ± 0.7%**</span> |
| 16           | 448.2 nsec ± 0.7% | <span style="color:#00aa00">**355.2 nsec ± 0.8%**</span> |
| 32           | 555.2 nsec ± 0.3% | <span style="color:#00aa00">**362.3 nsec ± 0.3%**</span> |
| 64           | 801.7 nsec ± 0.7% | <span style="color:#00aa00">**427.6 nsec ± 2.6%**</span> |
| 128          | 1.232 μsec ± 0.6% | <span style="color:#00aa00">**471.7 nsec ± 1.1%**</span> |
| 256          | 2.056 μsec ± 0.3% | <span style="color:#00aa00">**572.6 nsec ± 1.1%**</span> |
| 512          | 3.767 μsec ± 0.2% | <span style="color:#00aa00">**792.6 nsec ± 0.5%**</span> |
| 1024         | 7.222 μsec ± 0.3% | <span style="color:#00aa00">**1.263 μsec ± 0.7%**</span> |
| 2048         | 14.11 μsec ± 0.7% | <span style="color:#00aa00">**2.184 μsec ± 0.3%**</span> |
| 4096         | 27.74 μsec ± 0.4% | <span style="color:#00aa00">**4.003 μsec ± 0.3%**</span> |
| 8192         | 55.60 μsec ± 0.9% | <span style="color:#00aa00">**7.679 μsec ± 0.4%**</span> |
| **Geomean:** | 2.102 μsec ± 0.6% | <span style="color:#00aa00">**782.9 nsec ± 0.8%**</span> |
