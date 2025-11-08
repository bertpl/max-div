import numpy as np

from max_div.internal.benchmarking import BenchmarkResult, benchmark
from max_div.internal.compat import is_numba_installed
from max_div.internal.formatting import md_bold, md_italic, md_multiline
from max_div.sampling.discrete import sample_int

from ._formatting import format_as_markdown, format_for_console


def benchmark_sample_int(turbo: bool = True, markdown: bool = False) -> None:
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
    :param markdown: If `True`, outputs the results as a Markdown table.
    """

    if not is_numba_installed():
        print("====================================================================================")
        print("   WARNING: Numba is not installed!!!")
        print("====================================================================================")

    print("Benchmarking `sample_int`...")
    print()

    for replace, use_p, desc in [
        (True, False, "with replacement, uniform probabilities"),
        (False, False, "without replacement, uniform probabilities"),
        (True, True, "with replacement, custom probabilities"),
        (False, True, "without replacement, custom probabilities"),
    ]:
        print(desc.upper() + ": ", end="")

        # --- create headers ------------------------------
        if markdown:
            headers = [
                "`k`",
                "`n`",
                md_multiline([md_bold("accelerated=False"), md_italic("(numpy)")]),
                md_multiline([md_bold("accelerated=True"), md_italic("(custom numba)")]),
            ]
        else:
            headers = ["k", "n", "accelerated=False", "accelerated=True"]

        # --- benchmark ------------------------------------
        data: list[list[str | BenchmarkResult]] = []
        for n in [10, 100, 1000, 10000]:
            for k in [1, 10, 100, 1000, 10000]:
                if (not replace) and (k > n):
                    continue  # skip this combination, since it's not feasible

                data_row: list[str | BenchmarkResult] = [str(k), str(n)]

                for accelerated in [False, True]:
                    if use_p:
                        p = np.random.rand(n)
                        p /= p.sum()
                    else:
                        p = None

                    def func_to_benchmark():
                        sample_int(n=n, k=k, replace=replace, p=p, accelerated=accelerated)

                    data_row.append(
                        benchmark(
                            f=func_to_benchmark,
                            t_per_run=0.001 if turbo else 0.1,
                            n_warmup=3 if turbo else 10,
                            n_benchmark=3 if turbo else 30,
                            silent=True,
                        )
                    )

                data.append(data_row)
                print(".", end="")  # minimalistic progress indicator
        print()

        # --- show results -----------------------------------------
        if markdown:
            display_data = format_as_markdown(headers, data)
        else:
            display_data = format_for_console(headers, data)

        print()
        for line in display_data:
            print(line)
        print()
