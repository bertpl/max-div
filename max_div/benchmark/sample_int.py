import numpy as np
from tqdm import tqdm

from max_div.internal.benchmarking import BenchmarkResult, benchmark
from max_div.sampling.discrete import sample_int_numba, sample_int_numpy

from ._formatting import extend_table_with_aggregate_row, format_as_markdown, format_for_console


def benchmark_sample_int(speed: float = 0.0, markdown: bool = False) -> None:
    """
    Benchmarks the `sample_int` function from `max_div.sampling.discrete`.

    Different scenarios are tested:

     * with & without replacement
     * uniform & non-uniform sampling
     * `use_numba` True and False
     * different sizes of (`n`, `k`):
        * (10, 1), (100, 1), (1000, 1), (5000, 1), (10000, 1)
        * (10000, 10), (10000, 100), (10000, 1000), (10000, 5000), (10000, 10000)

    :param speed: value in [0.0, 1.0] (default=0.0); 0.0=accurate but slow; 1.0=fast but less accurate
    :param markdown: If `True`, outputs the results as a Markdown table.
    """

    print("Benchmarking `sample_int`...")
    print()

    for replace, use_p, desc in [
        (True, False, "A. WITH replacement, UNIFORM probabilities"),
        (False, False, "B. WITHOUT replacement, UNIFORM probabilities"),
        (True, True, "C. WITH replacement, CUSTOM probabilities"),
        (False, True, "D. WITHOUT replacement, CUSTOM probabilities"),
    ]:
        if markdown:
            print(f"## {desc}")
        else:
            print(f"{desc}:")

        # --- create headers ------------------------------
        if markdown:
            headers = [
                "`k`",
                "`n`",
                "`sample_int_numpy`",
                "`sample_int_numba`",
            ]
        else:
            headers = ["k", "n", "sample_int_numpy", "sample_int_numba"]

        # --- benchmark ------------------------------------
        data: list[list[str | BenchmarkResult]] = []
        n_k_values = [(n, k) for n in [10, 100, 1000, 10000] for k in [1, 10, 100, 1000, 10000] if replace or (k <= n)]
        for n, k in tqdm(n_k_values, leave=False):
            data_row: list[str | BenchmarkResult] = [str(k), str(n)]

            for use_numba in [False, True]:
                if use_p:
                    p = np.random.rand(n)
                    p /= p.sum()
                else:
                    p = np.zeros(0)
                p = p.astype(np.float32)

                if use_numba:

                    def func_to_benchmark():
                        sample_int_numba(n=n, k=k, replace=replace, p=p)
                else:

                    def func_to_benchmark():
                        sample_int_numpy(n=n, k=k, replace=replace, p=p)

                data_row.append(
                    benchmark(
                        f=func_to_benchmark,
                        t_per_run=0.1 / (1000.0**speed),
                        n_warmup=int(10 - 5 * speed),
                        n_benchmark=int(30 - 20 * speed),
                        silent=True,
                    )
                )

            data.append(data_row)

        # --- show results -----------------------------------------
        data = extend_table_with_aggregate_row(data, agg="geomean")
        if markdown:
            display_data = format_as_markdown(headers, data)
        else:
            display_data = format_for_console(headers, data)

        print()
        for line in display_data:
            print(line)
        print()
