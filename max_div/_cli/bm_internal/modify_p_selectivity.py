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
from max_div.internal.math.modify_p_selectivity import (
    modify_p_selectivity_power,
    modify_p_selectivity_pwl2,
)


def benchmark_modify_p_selectivity(speed: float = 0.0, markdown: bool = False) -> None:
    """
    Benchmarks the modify_p_selectivity functions from `max_div.internal.math.modify_p_selectivity`.

    Tests both implementations across different sizes of probability arrays:
     * `modify_p_selectivity_power`
     * `modify_p_selectivity_pwl2`

    Array sizes tested: [2, 4, 8, ..., 4096, 8192]

    For each benchmark iteration, a random modifier value in (0.0, 1.0) is chosen from
    100 pre-generated random values to ensure variability.

    :param speed: value in [0.0, 1.0] (default=0.0); 0.0=accurate but slow; 1.0=fast but less accurate
    :param markdown: If `True`, outputs the results as a Markdown table.
    """

    print("Benchmarking `modify_p_selectivity`...")
    print()

    # --- create headers ------------------------------
    if markdown:
        print("## modify_p_selectivity Performance")
        print()
        headers = [
            "`size`",
            "`power`",
            "`pwl2`",
        ]
    else:
        print("modify_p_selectivity Performance:")
        print()
        headers = [
            "size",
            "power",
            "pwl2",
        ]

    # --- prepare random modifier values --------------
    # Generate 100 random modifier values in (0.0, 1.0)
    np.random.seed(42)
    random_modifiers = np.random.uniform(0.0, 1.0, 100).astype(np.float32)

    # --- benchmark ------------------------------------
    data: list[list[CellContent]] = []
    sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    for size in tqdm(sizes, leave=False):
        data_row: list[CellContent] = [str(size)]

        # Generate random p array for benchmarking
        # Use size-dependent seed for reproducibility
        np.random.seed(size + 1000)
        test_p = np.random.rand(size).astype(np.float32)

        for fun in [modify_p_selectivity_power, modify_p_selectivity_pwl2]:
            # define function to be benchmarked
            def benchmark_fun(_idx: int):
                return fun(test_p, random_modifiers[_idx])

            # run benchmark
            data_row.append(
                benchmark(
                    f=benchmark_fun,
                    t_per_run=0.05 / (1000.0**speed),
                    n_warmup=int(8 - 5 * speed),
                    n_benchmark=int(25 - 22 * speed),
                    silent=True,
                    index_range=100,
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
