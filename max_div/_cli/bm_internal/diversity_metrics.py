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
from max_div.internal.utils import stdout_to_file
from max_div.solver._diversity import DiversityMetric


def benchmark_diversity_metrics(speed: float = 0.0, markdown: bool = False, file: bool = False) -> None:
    """
    Benchmarks the 4 DiversityMetric flavors from `max_div.solver._diversity`.

    Tests all 4 metric types across different sizes of separation vectors:
     * `min_separation`
     * `mean_separation`
     * `geomean_separation`
     * `approx_geomean_separation`
     * `non_zero_separation_frac`

    Vector sizes tested: [2, 4, 8, ..., 1024, 2048, 4096]

    :param speed: value in [0.0, 1.0] (default=0.0); 0.0=accurate but slow; 1.0=fast but less accurate
    :param markdown: If `True`, outputs the results as a Markdown table.
    """

    print("Benchmarking `DiversityMetric`...")

    # --- speed-dependent settings --------------------
    max_size = round(100_000 / (1_000**speed))
    t_per_run = 0.05 / (1000.0**speed)
    n_warmup = int(8 - 5 * speed)
    n_benchmark = int(25 - 22 * speed)

    # --- create diversity metrics --------------------
    metrics = [
        DiversityMetric.min_separation(),
        DiversityMetric.mean_separation(),
        DiversityMetric.geomean_separation(),
        DiversityMetric.approx_geomean_separation(),
        DiversityMetric.non_zero_separation_frac(),
    ]

    # --- benchmark ------------------------------------
    data: list[list[CellContent]] = []
    sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    sizes = [size for size in sizes if size <= max_size]

    for size in tqdm(sizes, leave=file):
        data_row: list[CellContent] = [str(size)]

        # Generate random separation vectors for benchmarking
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        test_separations = np.random.rand(size).astype(np.float32)

        for metric in metrics:

            def func_to_benchmark():
                metric.compute(test_separations)

            data_row.append(
                benchmark(
                    f=func_to_benchmark,
                    t_per_run=t_per_run,
                    n_warmup=n_warmup,
                    n_benchmark=n_benchmark,
                    silent=True,
                )
            )

        data.append(data_row)

    # --- show results -----------------------------------------

    # --- prepare table ---
    data = extend_table_with_aggregate_row(data, agg="geomean")
    if markdown:
        headers = [
            "`size`",
            "`min_separation`",
            "`mean_separation`",
            "`geomean_separation`",
            "`approx_geomean_separation`",
            "`non_zero_separation_frac`",
        ]
        display_data = format_table_as_markdown(headers, data, highlighters=[FastestBenchmark(), BoldLabels()])
    else:
        headers = [
            "size",
            "min_separation",
            "mean_separation",
            "geomean_separation",
            "approx_geomean_separation",
            "non_zero_separation_frac",
        ]
        display_data = format_table_for_console(headers, data)

    # --- output ---
    with stdout_to_file(file, "benchmark_diversity_metrics.md"):
        if markdown:
            print("## DiversityMetric Performance")
            print()
        else:
            print("DiversityMetric Performance")
            print()

        print()
        for line in display_data:
            print(line)
        print()
