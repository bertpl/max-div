# Benchmarks

All benchmarks are run on `m6a.2xlarge` AWS EC2 instances.

## max_div.sampling.discrete.sample_int

```bash
uv run --extra numba benchmark_sample_int 
```

Essentially here we compare `np.random.choice` (`use_numba=False`) with an optimized, Numba-accelerated version.

**Result:**
```
Benchmarking `sample_int`...

                                        use_numba=False           use_numba=True
WITH REPLACEMENT, UNIFORM
    1      out of 10                  7.819 μsec ± 0.2%        1.542 μsec ± 0.5%
    1      out of 100                 7.751 μsec ± 0.2%        1.550 μsec ± 0.3%
    1      out of 1_000               7.811 μsec ± 0.3%        1.555 μsec ± 0.1%
    1      out of 5_000               7.849 μsec ± 0.2%        1.550 μsec ± 0.1%
    1      out of 10_000              7.844 μsec ± 0.3%        1.567 μsec ± 0.2%
    10     out of 10_000              8.012 μsec ± 0.2%        1.726 μsec ± 0.1%
    100    out of 10_000              9.099 μsec ± 0.2%        3.151 μsec ± 1.5%
    1_000  out of 10_000              19.73 μsec ± 0.2%        18.19 μsec ± 0.0%
    5_000  out of 10_000              65.89 μsec ± 0.0%        84.32 μsec ± 0.1%
    10_000 out of 10_000              123.7 μsec ± 0.0%        166.9 μsec ± 0.0%

WITHOUT REPLACEMENT, UNIFORM
    1      out of 10                  7.803 μsec ± 0.2%        1.546 μsec ± 0.1%
    1      out of 100                 7.808 μsec ± 0.1%        1.555 μsec ± 0.3%
    1      out of 1_000               7.832 μsec ± 0.2%        1.548 μsec ± 0.1%
    1      out of 5_000               7.859 μsec ± 0.3%        1.566 μsec ± 0.2%
    1      out of 10_000              7.849 μsec ± 0.1%        1.558 μsec ± 0.1%
    10     out of 10_000              122.4 μsec ± 0.0%        2.621 μsec ± 0.3%
    100    out of 10_000              122.5 μsec ± 0.1%        4.084 μsec ± 0.1%
    1_000  out of 10_000              122.4 μsec ± 0.1%        19.49 μsec ± 0.1%
    5_000  out of 10_000              122.4 μsec ± 0.0%        70.00 μsec ± 0.0%
    10_000 out of 10_000              122.5 μsec ± 0.0%        137.8 μsec ± 0.0%

WITH REPLACEMENT, NON-UNIFORM
    1      out of 10                  14.01 μsec ± 0.2%        1.344 μsec ± 0.2%
    1      out of 100                 14.63 μsec ± 0.2%        1.416 μsec ± 0.3%
    1      out of 1_000               21.97 μsec ± 0.2%        2.302 μsec ± 0.4%
    1      out of 5_000               53.27 μsec ± 0.1%        5.675 μsec ± 0.1%
    1      out of 10_000              89.30 μsec ± 0.1%        9.872 μsec ± 0.0%
    10     out of 10_000              90.29 μsec ± 0.1%        10.81 μsec ± 0.0%
    100    out of 10_000              99.51 μsec ± 0.1%        19.64 μsec ± 0.1%
    1_000  out of 10_000              189.2 μsec ± 0.0%        106.8 μsec ± 0.0%
    5_000  out of 10_000              583.4 μsec ± 0.0%        494.8 μsec ± 0.0%
    10_000 out of 10_000              1.074 msec ± 0.0%        970.6 μsec ± 0.3%

WITHOUT REPLACEMENT, NON-UNIFORM
    1      out of 10                  14.02 μsec ± 0.2%        1.347 μsec ± 0.2%
    1      out of 100                 14.69 μsec ± 0.2%        1.418 μsec ± 0.6%
    1      out of 1_000               21.94 μsec ± 0.2%        2.270 μsec ± 0.1%
    1      out of 5_000               52.98 μsec ± 0.2%        5.697 μsec ± 0.3%
    1      out of 10_000              89.55 μsec ± 0.1%        9.870 μsec ± 0.0%
    10     out of 10_000              115.1 μsec ± 0.2%        234.7 μsec ± 0.6%
    100    out of 10_000              153.7 μsec ± 0.4%        233.6 μsec ± 0.4%
    1_000  out of 10_000              344.5 μsec ± 0.4%        244.8 μsec ± 0.3%
    5_000  out of 10_000              1.321 msec ± 0.1%        271.5 μsec ± 0.1%
    10_000 out of 10_000              3.853 msec ± 0.1%        144.7 μsec ± 0.0%
```