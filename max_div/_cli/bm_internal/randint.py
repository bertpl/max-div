import numpy as np
from tqdm import tqdm

from max_div._cli.formatting import (
    BoldLabels,
    CellContent,
    FastestBenchmark,
    extend_table_with_aggregate_row,
    format_table_as_markdown,
    format_table_for_console,
)
from max_div.internal.benchmarking import benchmark
from max_div.sampling.uncon import randint_numba, randint_numpy


def benchmark_randint(speed: float = 0.0, markdown: bool = False) -> None:
    """
    Benchmarks the `randint` function from `max_div.sampling.uncon`.

    Different scenarios are tested:

     * with & without replacement
     * uniform & non-uniform sampling
     * `use_numba` True and False
     * different sizes of (`n`, `k`):
        * both `n` & `k` are varied across [1, 10, 100, 1000, 10000]
        * all valid combinations are tested (if `replace==False` we don't test `k`>`n`)

    :param speed: value in [0.0, 1.0] (default=0.0); 0.0=accurate but slow; 1.0=fast but less accurate
    :param markdown: If `True`, outputs the results as a Markdown table.
    """

    print("Benchmarking `randint`...")
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
                "`randint_numpy`",
                "`randint_numba`",
            ]
        else:
            headers = ["k", "n", "randint_numpy", "randint_numba"]

        # --- benchmark ------------------------------------
        data: list[list[CellContent]] = []
        n_k_values = [(n, k) for n in [10, 100, 1000, 10000] for k in [1, 10, 100, 1000, 10000] if replace or (k <= n)]
        for n, k in tqdm(n_k_values, leave=False):
            data_row: list[CellContent] = [str(k), str(n)]

            for use_numba in [False, True]:
                if use_p:
                    p = np.random.rand(n)
                    p /= p.sum()
                else:
                    p = np.zeros(0)
                p = p.astype(np.float32)

                if use_numba:

                    def func_to_benchmark():
                        randint_numba(n=n, k=k, replace=replace, p=p)
                else:

                    def func_to_benchmark():
                        randint_numpy(n=n, k=k, replace=replace, p=p)

                data_row.append(
                    benchmark(
                        f=func_to_benchmark,
                        t_per_run=0.05 / (1000.0**speed),
                        n_warmup=int(8 - 5 * speed),
                        n_benchmark=int(25 - 22 * speed),
                        silent=True,
                    )
                )

            data.append(data_row)

        # --- show results -----------------------------------------
        data = extend_table_with_aggregate_row(data, agg="geomean")
        if markdown:
            display_data = format_table_as_markdown(headers, data, highlighters=[FastestBenchmark(), BoldLabels()])
        else:
            display_data = format_table_for_console(headers, data)

        print()
        for line in display_data:
            print(line)
        print()
