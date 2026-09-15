import numpy as np
from tqdm import tqdm

from max_div._core._cli.benchmarks._helpers.speed_scaling import SpeedParam
from max_div._core._markdown import Report, Table, TableAggregationType, TableElement, TableTimeElapsed, h2
from max_div._core._utils import benchmark, stdout_to_file
from max_div._core.metrics import DiversityContributionFamily, DiversityMetric

from .run_settings import N_BENCHMARK, N_WARMUP, TIME_PER_RUN_SEC


def benchmark_diversity_metrics(speed: float = 0.0, markdown: bool = False, file: bool = False) -> None:
    """Benchmarks every separation-family `DiversityMetric` across sizes of the separation vector.

    The metrics come from the enum, so a new separation-family member gets its column without an edit
    here; the separation-vector sizes tested are capped by `speed`.

    Args:
        speed: value in [0.0, 1.0] (default=0.0); 0.0=accurate but slow; 1.0=fast but less accurate
        markdown: If `True`, outputs the results as a Markdown table.
        file: If `True`, redirects output to a file instead of console.
    """
    print("Benchmarking `DiversityMetric`...")

    # --- speed-dependent settings ---------------
    max_size = SpeedParam(slow=100_000, fast=100).at(speed)
    t_per_run = TIME_PER_RUN_SEC.at(speed)
    n_warmup = N_WARMUP.at(speed)
    n_benchmark = N_BENCHMARK.at(speed)

    # --- create diversity metrics ---------------
    metrics = [
        metric for metric in DiversityMetric if metric.contribution_family == DiversityContributionFamily.SEPARATION
    ]

    # --- benchmark ------------------------------
    table = Table(headers=["`size`", *(f"`{metric.name.lower()}`" for metric in metrics)])
    sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    sizes = [size for size in sizes if size <= max_size]

    for size in tqdm(sizes, leave=file):
        table_row: list[TableElement | str] = [str(size)]

        # Generate random separation vectors for benchmarking
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        test_separations = np.random.rand(size).astype(np.float32)

        for metric in metrics:

            def func_to_benchmark() -> None:
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

    # --- show results ---------------------------

    # --- create final report --------------------
    table.add_aggregate_row(TableAggregationType.GEOMEAN)
    table.highlight_results(TableTimeElapsed, clr_lowest=Table.GREEN, clr_highest=Table.RED)

    report = Report()
    report += h2("DiversityMetric Performance")
    report += table

    # --- output ---------------------------------
    with stdout_to_file(file, "benchmark_diversity_metrics.md"):
        report.print(markdown=markdown)
