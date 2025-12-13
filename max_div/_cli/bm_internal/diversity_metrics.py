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
from max_div.solver._diversity import DiversityMetric


def benchmark_diversity_metrics(speed: float = 0.0, markdown: bool = False) -> None:
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
    print()

    # --- create headers ------------------------------
    if markdown:
        print("## DiversityMetric Performance")
        print()
        headers = [
            "`size`",
            "`min_separation`",
            "`mean_separation`",
            "`geomean_separation`",
            "`approx_geomean_separation`",
            "`non_zero_separation_frac`",
        ]
    else:
        print("DiversityMetric Performance:")
        print()
        headers = [
            "size",
            "min_separation",
            "mean_separation",
            "geomean_separation",
            "approx_geomean_separation",
            "non_zero_separation_frac",
        ]

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
    sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    for size in tqdm(sizes, leave=False):
        data_row: list[CellContent] = [str(size)]

        # Generate random separation vectors for benchmarking
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        test_separations = np.random.rand(size).astype(np.float32)

        for metric in metrics:

            def func_to_benchmark():
                return metric.compute(test_separations)

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
