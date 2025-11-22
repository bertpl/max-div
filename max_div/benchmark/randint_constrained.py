import math
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from max_div.internal.benchmarking import BenchmarkResult, benchmark
from max_div.internal.formatting import md_multiline
from max_div.sampling import randint_numba
from max_div.sampling.con import Constraint, randint_constrained

from ._formatting import (
    BoldLabels,
    CellContent,
    FastestBenchmark,
    HighestPercentage,
    Percentage,
    extend_table_with_aggregate_row,
    format_as_markdown,
    format_for_console,
)


# =================================================================================================
#  Helper classes
# =================================================================================================
@dataclass
class _Scenario:
    letter: str
    description: str
    n_k_cons_tuples: list[tuple[int, int, int]]  # list of (n, k, n_cons)-tuples
    use_p: bool


# =================================================================================================
#  Main benchmark function
# =================================================================================================
def benchmark_randint_constrained(speed: float = 0.0, markdown: bool = False) -> None:
    """
    Benchmarks the `randint_constrained` function from `max_div.sampling.con`.

    Different scenarios are tested across different values of `k`, `n` & `n_cons` (# of constraints):

     * **SCENARIO A1**
        * all combinations with `k` < `n` with
            * `n` in [10, 100, 1000]
            * `k` in [2, 4, 8, 16, 32, ..., 256]
        * constraints:
            * 10 non-overlapping constraints, each spanning exactly 1/10th of the `n`-range
            * min_count = floor(k/11)
            * max_count = ceil(k/9)
        * no `p` provided (uniform sampling)

     * **SCENARIO A2**
        * same as Scenario A1, but with `p` provided (with p[i] ~ 1+i)

     * **SCENARIO B1**
        * `n` =  1000
        * `k` =   100
        * `n_cons` in [2, 4, 8, 16, ..., 512]
            * each constraint spans a random 1% of the `n` range (=10 values)
            * min_count = floor(10 / n_cons)
            * max_count = ceil(100 / n_cons)
        * no `p` provided (uniform sampling)

     * **SCENARIO B2**
        * same as Scenario B1, but with `p` provided (with p[i] ~ 1+i)

    :param speed: value in [0.0, 1.0] (default=0.0); 0.0=accurate but slow; 1.0=fast but less accurate
    :param markdown: If `True`, outputs the results as a Markdown table.
    """

    # --- build scenarios ---------------------------------
    scenarios = []
    for use_p in [False, True]:
        if not use_p:
            letter = "A1"
            description = "Varying n & k with 10 non-overlapping constraints spanning equal portions of the n range (uniform sampling)."
        else:
            letter = "A2"
            description = "Identical to Scenario A1, but with custom probabilities p provided, favoring larger values."

        scenarios.append(
            _Scenario(
                letter=letter,
                description=description,
                n_k_cons_tuples=[
                    (n, k, 10)
                    for n in [10, 100, 1000]
                    for k in [2**i for i in range(1, 9)]  # 2, 4, 8, ..., 256
                    if k < n
                ],
                use_p=use_p,
            ),
        )

    for use_p in [False, True]:
        if not use_p:
            letter = "B1"
            description = "Fixed n=1000 & k=100 with varying number of constraints spanning random 1% portions of the n range (uniform sampling)."
        else:
            letter = "B2"
            description = "Identical to Scenario B1, but with custom probabilities p provided, favoring larger values."

        scenarios.append(
            _Scenario(
                letter=letter,
                description=description,
                n_k_cons_tuples=[
                    (1000, 100, n_cons)
                    for n_cons in [2**i for i in range(1, 10)]  # 2, 4, 8, ..., 512
                ],
                use_p=use_p,
            ),
        )

    # --- benchmark all scenarios -------------------------
    print("Benchmarking `randint_constrained`...")
    print()
    for s in scenarios:
        if markdown:
            print(f"## Scenario {s.letter}")
        else:
            print(f"Scenario {s.letter}:")

        print()
        print(s.description)
        print()

        # --- create headers --------------------
        if markdown:
            headers = [
                "`k`",
                "`n`",
                "`n_cons`",
                md_multiline(["`randint_numba`", "(time)"]),
                md_multiline(["`randint_numba`", "(accuracy %)"]),
                md_multiline(["`randint_constrained`", "(time)"]),
                md_multiline(["`randint_constrained`", "(accuracy %)"]),
            ]
        else:
            headers = [
                "k",
                "n",
                "n_cons",
                "randint_numba (time)",
                "randint_numba (accuracy %)",
                "randint_constrained (time)",
                "randint_constrained (accuracy %)",
            ]

        # --- benchmark scenario ----------------
        data: list[list[CellContent]] = []
        for n, k, n_cons in tqdm(s.n_k_cons_tuples, leave=False):
            # --- build constraints ---
            if s.letter.startswith("A"):
                cons = [
                    Constraint(
                        int_set=set(range(i * (n // 10), (i + 1) * (n // 10))),
                        min_count=math.floor(k / 11),
                        max_count=math.ceil(k / 9),
                    )
                    for i in range(10)
                ]
            else:
                cons = []
                for i in range(n_cons):
                    cons.append(
                        Constraint(
                            int_set=set(
                                randint_numba(
                                    n=np.int32(n),
                                    k=np.int32(n // 100),  # 1% random samples from n
                                    replace=False,
                                    seed=np.int64(42 + i),
                                )
                            ),
                            min_count=math.floor(10 / n_cons),
                            max_count=math.ceil(100 / n_cons),
                        )
                    )

            if s.use_p:
                p = np.array([1.0 + i for i in range(n)], dtype=np.float32)
                p /= p.sum()
            else:
                p = None

            # --- benchmark & determine precision ---
            data.append(
                [
                    str(k),
                    str(n),
                    str(n_cons),
                    _benchmark(n=n, k=k, cons=cons, ignore_constraints=True, p=p, speed=speed),
                    _determine_precision(n=n, k=k, cons=cons, ignore_constraints=True, p=p, speed=speed),
                    _benchmark(n=n, k=k, cons=cons, ignore_constraints=False, p=p, speed=speed),
                    _determine_precision(n=n, k=k, cons=cons, ignore_constraints=False, p=p, speed=speed),
                ]
            )

        # --- show results -----------------------------------------
        data = extend_table_with_aggregate_row(data, agg="geomean", include_percentage=False)
        data = extend_table_with_aggregate_row(data, agg="mean", include_benchmark_result=False)
        if markdown:
            display_data = format_as_markdown(
                headers,
                data,
                highlighters=[
                    FastestBenchmark(),
                    HighestPercentage(),
                    BoldLabels(),
                ],
            )
        else:
            display_data = format_for_console(headers, data)

        for line in display_data:
            print(line)
        print()


# =================================================================================================
#  Internal helpers
# =================================================================================================
def _benchmark(
    n: int, k: int, cons: list[Constraint], ignore_constraints: bool, p: np.ndarray | None, speed: float
) -> BenchmarkResult:
    """
    Runs a benchmark for given parameters and returns the BenchmarkResult.
    ignore_constraints: if True, benchmarks `randint_numba`, else `randint_constrained`.
    """
    n = np.int32(n)
    k = np.int32(k)

    if ignore_constraints:
        if p is None:

            def benchmark_func():
                return randint_numba(n=n, k=k, replace=False)
        else:
            p = p.astype(np.float32)

            def benchmark_func():
                return randint_numba(n=n, k=k, replace=False, p=p)
    else:
        if p is None:

            def benchmark_func():
                return randint_constrained(n=n, k=k, cons=cons)
        else:
            p = p.astype(np.float32)

            def benchmark_func():
                return randint_constrained(n=n, k=k, cons=cons, p=p)

    return benchmark(
        f=benchmark_func,
        t_per_run=0.05 / (1000.0**speed),
        n_warmup=int(8 - 5 * speed),
        n_benchmark=int(25 - 22 * speed),
        silent=True,
    )


def _determine_precision(
    n: int, k: int, cons: list[Constraint], ignore_constraints: bool, p: np.ndarray | None, speed: float
) -> Percentage:
    """
    Determines how often (%) the constraints are satisfied when sampling with given parameters.
    ignore_constraints: if True, benchmarks `randint_numba`, else `randint_constrained`.
    """

    # Calculate number of runs based on speed (100 at speed=0, 2 at speed=1)
    n_runs = int(100 * (0.02**speed))

    satisfied_count = 0
    for run_idx in range(n_runs):
        # Run the appropriate function with seed equal to run index
        if ignore_constraints:
            if p is None:
                result = randint_numba(n=np.int32(n), k=np.int32(k), replace=False, seed=np.int64(run_idx))
            else:
                result = randint_numba(
                    n=np.int32(n), k=np.int32(k), replace=False, p=p.astype(np.float32), seed=np.int64(run_idx)
                )
        else:
            if p is None:
                result = randint_constrained(n=n, k=k, cons=cons, seed=run_idx)
            else:
                result = randint_constrained(n=n, k=k, cons=cons, p=p.astype(np.float32), seed=run_idx)

        # Check if all constraints are satisfied
        constraints_satisfied = True
        for constraint in cons:
            count = sum(1 for val in result if val in constraint.int_set)
            if count < constraint.min_count or count > constraint.max_count:
                constraints_satisfied = False
                break

        if constraints_satisfied:
            satisfied_count += 1

    return Percentage(frac=satisfied_count / n_runs, decimals=0)
