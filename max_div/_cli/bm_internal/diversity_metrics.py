import numpy as np
from tqdm import tqdm

from max_div.internal.benchmarking import benchmark
from max_div.internal.markdown import Report, Table, TableAggregationType, TableElement, TableTimeElapsed, h2
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
    :param file: If `True`, redirects output to a file instead of console.
    """

    print("Benchmarking `DiversityMetric`...")

    # --- speed-dependent settings --------------------
    max_size = round(100_000 / (1_000**speed))
    t_per_run = 0.01 / (1000.0**speed)
    n_warmup = int(8 - 6 * speed)
    n_benchmark = int(25 - 24 * speed)

    # --- create diversity metrics --------------------
    metrics = [
        DiversityMetric.min_separation(),
        DiversityMetric.mean_separation(),
        DiversityMetric.geomean_separation(),
        DiversityMetric.approx_geomean_separation(),
        DiversityMetric.non_zero_separation_frac(),
    ]

    # --- benchmark ------------------------------------
    table = Table(
        headers=[
            "`size`",
            "`min_separation`",
            "`mean_separation`",
            "`geomean_separation`",
            "`approx_geomean_separation`",
            "`non_zero_separation_frac`",
        ]
    )
    sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    sizes = [size for size in sizes if size <= max_size]

    for size in tqdm(sizes, leave=file):
        table_row: list[TableElement | str] = [str(size)]

        # Generate random separation vectors for benchmarking
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        test_separations = np.random.rand(size).astype(np.float32)

        for metric in metrics:

            def func_to_benchmark():
                metric.compute(test_separations)

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

    # --- create final report ---
    table.add_aggregate_row(TableAggregationType.GEOMEAN)
    table.highlight_results(TableTimeElapsed, clr_lowest=Table.GREEN, clr_highest=Table.RED)

    report = Report()
    report += h2("DiversityMetric Performance")
    report += table

    # --- output ---
    with stdout_to_file(file, "benchmark_diversity_metrics.md"):
        report.print(markdown=markdown)
