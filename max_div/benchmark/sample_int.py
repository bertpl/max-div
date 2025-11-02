import math

import numpy as np

from max_div.internal.benchmarking import BenchmarkResult, benchmark
from max_div.internal.compat import is_numba_installed
from max_div.sampling.discrete import sample_int


def benchmark_sample_int(turbo: bool = False) -> None:
    """
    Benchmarks the `sample_int` function from `max_div.sampling.discrete`.

    Different scenarios are tested:

     * with & without replacement
     * uniform & non-uniform sampling
     * `use_numba` True and False
     * different sizes of (`n`, `k`):
        * (10, 1), (100, 1), (1000, 1), (5000, 1), (10000, 1)
        * (10000, 10), (10000, 100), (10000, 1000), (10000, 5000), (10000, 10000)

    :param turbo: If `True`, a much shorter (but less reliable) benchmark is run; intended for testing purposes.
    """

    if not is_numba_installed():
        print("====================================================================================")
        print("   WARNING: Numba is not installed!!!")
        print("====================================================================================")

    print("Benchmarking `sample_int`...")
    print()
    print("".ljust(30) + "use_numba=False".rjust(25) + "use_numba=True".rjust(25))

    for replace, use_p, desc in [
        (True, False, "with replacement, uniform"),
        (False, False, "without replacement, uniform"),
        (True, True, "with replacement, non-uniform"),
        (False, True, "without replacement, non-uniform"),
    ]:
        print(desc.upper())

        for n, k in [
            (10, 1),
            (100, 1),
            (1000, 1),
            (5000, 1),
            (10000, 1),
            (10000, 10),
            (10000, 100),
            (10000, 1000),
            (10000, 5000),
            (10000, 10000),
        ]:
            size_str = f"{k:<6_} out of {n:<6_}"

            results: list[BenchmarkResult] = []
            for use_numba in [False, True]:
                if use_p:
                    p = np.random.rand(n)
                    p /= p.sum()
                else:
                    p = None

                def func_to_benchmark():
                    sample_int(n=n, k=k, replace=replace, p=p, use_numba=use_numba)

                results.append(
                    benchmark(
                        f=func_to_benchmark,
                        t_per_run=0.001 if turbo else 0.1,
                        n_warmup=3 if turbo else 10,
                        n_benchmark=3 if turbo else 30,
                        silent=True,
                    )
                )

            print(
                (" " * 4)
                + size_str.ljust(26)
                + results[0].t_sec_with_uncertainty_str.rjust(25)
                + results[1].t_sec_with_uncertainty_str.rjust(25)
            )

        print()
