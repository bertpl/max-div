import numpy as np
from tqdm import tqdm

from max_div.internal.benchmarking import benchmark
from max_div.internal.markdown import (
    Report,
    Table,
    TableAggregationType,
    TableElement,
    TableTimeElapsed,
    h2,
)
from max_div.internal.utils import stdout_to_file
from max_div.random import new_rng_state, randint, randint_python


def benchmark_randint(speed: float = 0.0, markdown: bool = False, file: bool = False) -> None:
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
    :param file: If `True`, redirects output to a file instead of console.
    """

    print("Benchmarking `randint`...")

    # --- speed-dependent settings --------------------
    t_per_run = 0.01 / (1000.0**speed)
    n_warmup = int(8 - 6 * speed)
    n_benchmark = int(25 - 24 * speed)
    max_size = round(10_000 / (100**speed))

    # --- benchmark scenarios -------------------------
    i_file = 0
    for replace, use_p, letter, desc in [
        (True, False, "A", "WITH replacement, UNIFORM probabilities"),
        (False, False, "B", "WITHOUT replacement, UNIFORM probabilities"),
        (True, True, "C", "WITH replacement, CUSTOM probabilities"),
        (False, True, "D", "WITHOUT replacement, CUSTOM probabilities"),
    ]:
        # --- benchmark ------------------------------------
        table = Table(headers=["`k`", "`n`", "`randint_python`", "`randint`"])
        n_k_values = [(n, k) for n in [10, 100, 1000, 10000] for k in [1, 10, 100, 1000, 10000] if replace or (k <= n)]
        for n, k in tqdm(n_k_values, leave=file):
            if n > max_size or k > max_size:
                continue

            table_row: list[TableElement | str] = [str(k), str(n)]

            for use_numba in [False, True]:
                if use_p:
                    p = np.random.rand(n)
                    p /= p.sum()
                else:
                    p = np.zeros(0)
                p = p.astype(np.float32)

                if use_numba:
                    rng_state = new_rng_state(np.int64(42))
                    n = np.int32(n)
                    k = np.int32(k)

                    def func_to_benchmark():
                        randint(n=n, k=k, replace=replace, p=p, rng_state=rng_state)
                else:

                    def func_to_benchmark():
                        randint_python(n=n, k=k, replace=replace, p=p)

                table_row.append(
                    TableTimeElapsed.from_benchmark_result(
                        benchmark(
                            f=func_to_benchmark,
                            t_per_run=t_per_run,
                            n_warmup=n_warmup,
                            n_benchmark=n_benchmark,
                            silent=True,
                        )
                    )
                )

            table.add_row(table_row)

        # --- show results -----------------------------------------

        # --- prepare final report ---
        table.add_aggregate_row(TableAggregationType.GEOMEAN)
        table.highlight_results(TableTimeElapsed, clr_lowest=Table.GREEN)

        report = Report()
        report += [h2(f"{letter}. {desc}"), table]

        # --- output ---
        i_file += 1
        with stdout_to_file(file, f"benchmark_randint_{i_file}.md"):
            report.print(markdown=markdown)
