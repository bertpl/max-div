import numpy as np
from tqdm import tqdm

from max_div._cli.formatting import (
    BoldLabels,
    CellContent,
    FastestBenchmark,
    LowestPercentage,
    Percentage,
    extend_table_with_aggregate_row,
    format_table_as_markdown,
    format_table_for_console,
)
from max_div.internal.benchmarking import benchmark
from max_div.internal.math.modify_p_selectivity import exponential_selectivity, modify_p_selectivity
from max_div.internal.utils import stdout_to_file

MODIFY_P_METHODS = [np.int32(0), np.int32(10), np.int32(20), np.int32(100)]


# =================================================================================================
#  Main benchmark
# =================================================================================================
def benchmark_modify_p_selectivity(speed: float = 0.0, markdown: bool = False, file: bool = False) -> None:
    """
    Benchmarks the modify_p_selectivity function from `max_div.internal.math.modify_p_selectivity`,
      for various different 'method'-values across different sizes of probability arrays.

    Array sizes tested: [2, 4, 8, ..., 4096, 8192]

    For each benchmark iteration, a random modifier value in (0.0, 1.0) is chosen from
    100 pre-generated random values to ensure variability.

    :param speed: value in [0.0, 1.0] (default=0.0); 0.0=accurate but slow; 1.0=fast but less accurate
    :param markdown: If `True`, outputs the results as a Markdown table.
    """

    print("Benchmarking `modify_p_selectivity`...")

    # --- speed-dependent settings --------------------
    n_accuracy = round(1000.0 / (100.0**speed))  # 1000 when speed=0, 10 when speed=1
    max_size = round(100_000 / (1_000**speed))
    t_per_run = 0.05 / (1000.0**speed)
    n_warmup = int(8 - 5 * speed)
    n_benchmark = int(25 - 22 * speed)

    # --- compute approximation errors ----------------
    # compute errors by method (by comparing exact power method vs other methods on calibration data)
    error_by_method = {method: compute_accuracy(method, n_accuracy) for method in MODIFY_P_METHODS}

    # --- prepare random modifier values --------------
    # Generate 100 random modifier values in (0.0, 1.0)
    np.random.seed(42)
    random_modifiers = np.random.uniform(0.0, 1.0, 100).astype(np.float32)

    # --- benchmark ------------------------------------
    data: list[list[CellContent]] = []
    sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    sizes = [size for size in sizes if size <= max_size]

    for size in tqdm(sizes, leave=file):
        data_row: list[CellContent] = [str(size)]

        # Generate random p array for benchmarking
        # Use size-dependent seed for reproducibility
        np.random.seed(size + 1000)
        p_in = np.random.rand(size).astype(np.float32)
        p_out = np.empty_like(p_in)

        # test modify_p_selectivity for all defined methods
        for method in MODIFY_P_METHODS:
            # define function to be benchmarked
            def benchmark_fun(_idx: int):
                modify_p_selectivity(p_in, random_modifiers[_idx], method, p_out)

            # run benchmark
            data_row.append(
                benchmark(
                    f=benchmark_fun,
                    t_per_run=t_per_run,
                    n_warmup=n_warmup,
                    n_benchmark=n_benchmark,
                    silent=True,
                    index_range=100,
                )
            )

        # test order_based_selectivity
        def benchmark_fun(_idx: int):
            exponential_selectivity(p_in, p_out, random_modifiers[_idx])

        # run benchmark
        data_row.append(
            benchmark(
                f=benchmark_fun,
                t_per_run=t_per_run,
                n_warmup=n_warmup,
                n_benchmark=n_benchmark,
                silent=True,
                index_range=100,
            )
        )

        data.append(data_row)

    # --- show results -----------------------------------------

    # --- prepare table ---

    # add geomean time + approximation error rows
    data = extend_table_with_aggregate_row(data, agg="geomean")
    extra_data_line = (
        ["e_approx:"] + [Percentage(error_by_method[method], decimals=2) for method in MODIFY_P_METHODS] + ["N/A"]
    )
    data.append(extra_data_line)
    headers = ["size"] + [f"method={method}" for method in MODIFY_P_METHODS] + ["exponential"]

    if markdown:
        display_data = format_table_as_markdown(
            headers,
            data,
            highlighters=[
                FastestBenchmark(),
                LowestPercentage(),
                BoldLabels(),
            ],
        )
    else:
        display_data = format_table_for_console(headers, data)

    # --- output ---
    with stdout_to_file(file, "benchmark_modify_p_selectivity.md"):
        show_methods_table(markdown)

        if markdown:
            print("## Benchmark results")
            print()
        else:
            print("Benchmark results")
            print()

        print()
        for line in display_data:
            print(line)
        print()


# =================================================================================================
#  Helpers
# =================================================================================================
def show_methods_table(markdown: bool) -> None:
    # --- prepare table data ------------------------------
    headers = ["`name`", "`method(args)`", "Type", "Details"]
    data = [
        ["**method=0**", "`modify_p_selectivity(method=0)`", "Power-based", "p**t"],
        [
            "**method=10**",
            "`modify_p_selectivity(method=10)`",
            "Power-based",
            "fast_exp2(t * fast_log2(p)) (NOT specifically optimized for this use case)",
        ],
        [
            "**method=20**",
            "`modify_p_selectivity(method=20)`",
            "Power-based",
            "fast_pow(p, t) (specifically optimized for this use case)",
        ],
        ["**method=100**", "`modify_p_selectivity(method=100)`", "Power-based", "2-segment PWL approx. of p**t"],
        [
            "**exponential**",
            "`exponential_selectivity()`",
            "Exponential mapping",
            "maps [p_min, p_max] to [1.0, low_value**t]",
        ],
    ]

    # --- format appropriately ----------------------------
    if markdown:
        display_data = format_table_as_markdown(headers, data)
    else:
        display_data = format_table_for_console(headers, data)

    # --- display table -----------------------------------
    print("Tested methods:")
    print()
    for line in display_data:
        print(line)
    print()


def compute_accuracy(method: int, n: int) -> float:
    """Computes accuracy of a given method as a fraction in [0.0, 1.0]."""

    total_error = 0.0  # total sum of abs errors
    total_pmod = 0.0  # total sum of target values (wrt which we computed errors)

    for modify in np.linspace(-0.9, 0.9, n):
        t = (1.0 + modify) / (1.0 - modify)

        # construct p_in such that p_out_target is uniformly spaced in [0.0, 1.0],
        # which will focus (for each modify-value) p_in-values in regions where we'll be able to see differences best
        p_out_target = np.linspace(0.0, 1.0, n)
        p_in = (p_out_target ** (1.0 / t)).astype(np.float32)
        p_out = np.empty_like(p_in)
        modify_p_selectivity(p_in, np.float32(modify), np.int32(method), p_out)

        total_error += np.sum(np.abs(p_out_target - p_out))
        total_pmod += np.sum(p_out_target)

    error_frac = total_error / total_pmod
    return error_frac
