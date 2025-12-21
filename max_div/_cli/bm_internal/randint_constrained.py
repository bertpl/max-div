from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from max_div._cli.formatting import (
    BoldLabels,
    CellContent,
    FastestBenchmark,
    HighestPercentage,
    Percentage,
    extend_table_with_aggregate_row,
    format_table_as_markdown,
    format_table_for_console,
)
from max_div.internal.benchmarking import BenchmarkResult, benchmark
from max_div.internal.formatting import md_multiline
from max_div.internal.utils import stdout_to_file
from max_div.sampling import randint_numba
from max_div.sampling._constraint_helpers import _build_array_repr
from max_div.sampling.con import randint_constrained, randint_constrained_robust
from max_div.solver import Constraint


# =================================================================================================
#  Main benchmark function
# =================================================================================================
def benchmark_randint_constrained(speed: float = 0.0, markdown: bool = False, file: bool = False) -> None:
    """
    Benchmarks the `randint_constrained` function from `max_div.sampling.con`.

    Different scenarios are tested across different values of `k`, `n` & `m` (# of constraints):

     * **SCENARIO A**
        * all combinations with `k` < `n` with
            * `n` in [10, 100, 1000]
            * `k` in [2, 4, 8, 16, 32, ..., 256]
        * constraints:
            * 10 non-overlapping constraints, each spanning exactly 1/10th of the `n`-range
            * min_count = floor(k/11)
            * max_count = ceil(k/9)

     * **SCENARIO B**
        * `n` =  1000
        * `k` =   100
        * `m` in [2, 4, 8, 16, ..., 256, 384, 512, 768, 1024]
            * each constraint spans a random 1% of the `n` range (=10 values)
            * min_count = 1+floor(10 / m)
            * max_count = 1+ceil(1000 / m)

    Both scenarios are tested with uniform sampling (no custom probabilities p) and with custom probabilities p
     favoring larger values to be sampled.

    :param speed: value in [0.0, 1.0] (default=0.0); 0.0=accurate but slow; 1.0=fast but less accurate
    :param markdown: If `True`, outputs the results as a Markdown table.
    """

    # --- define formatting -------------------------------
    def print_table(_headers: list[str], _data: list[list[CellContent]]):
        if markdown:
            _table = format_table_as_markdown(
                _headers,
                _data,
                highlighters=[
                    FastestBenchmark(),
                    HighestPercentage(),
                    BoldLabels(),
                ],
            )
        else:
            _headers = [h.replace("`", "").replace("<br>", " ") for h in _headers]
            _table = format_table_for_console(_headers, _data)

        for _line in _table:
            print(_line)
        print()

    def print_header(_txt: str, _level: int):
        if markdown:
            print(f"{'#' * _level} {_txt}")
        else:
            print(f"{_txt}:")
        print()

    # --- speed-dependent settings --------------------
    max_count = int(100 * (0.01**speed))  # max_count=100 if speed=0;  max_count=1 at speed=1
    t_per_run = 0.05 / (1000.0**speed)
    n_warmup = int(8 - 5 * speed)
    n_benchmark = int(25 - 22 * speed)

    # --- build scenarios ---------------------------------
    scenarios = [ScenarioA(), ScenarioB()]

    # --- benchmark all scenarios -------------------------
    print("Benchmarking `randint_constrained`...")

    i_file = 0
    for s in scenarios:
        for use_p in [False, True]:
            # --- benchmark scenario ----------------
            timing_data: list[list[CellContent]] = []
            accuracy_data: list[list[CellContent]] = []

            for i, (n, k, m) in enumerate(tqdm(s.n_k_m_tuples(), leave=file)):
                if i >= max_count:
                    continue

                # --- construct p ---
                if use_p:
                    p = np.array([1.0 + i for i in range(n)], dtype=np.float32)
                    p /= p.sum()
                else:
                    p = None

                # --- benchmark & determine precision ---
                timing_data.append(
                    [
                        str(k),
                        str(n),
                        str(m),
                        _benchmark(s, n, k, m, p, speed, "no_cons"),
                        _benchmark(s, n, k, m, p, speed, "non_eager"),
                        _benchmark(s, n, k, m, p, speed, "eager"),
                        _benchmark(s, n, k, m, p, speed, "robust"),
                    ]
                )

                accuracy_data.append(
                    [
                        str(k),
                        str(n),
                        str(m),
                        _determine_precision(s, n, k, m, p, speed, "no_cons"),
                        _determine_precision(s, n, k, m, p, speed, "non_eager"),
                        _determine_precision(s, n, k, m, p, speed, "eager"),
                        _determine_precision(s, n, k, m, p, speed, "robust"),
                    ]
                )

            # --- show all results --------------------------------------------

            # --- prepare tables ---
            headers = [
                "`k`",
                "`n`",
                "`m`",
                "`randint_numba`",
                md_multiline(["`randint_constrained`", "(eager=False)"]),
                md_multiline(["`randint_constrained`", "(eager=True)"]),
                md_multiline(["`randint_constrained_robust`", "(n_trials=5)"]),
            ]
            timing_data = extend_table_with_aggregate_row(timing_data, agg="geomean")
            accuracy_data = extend_table_with_aggregate_row(accuracy_data, agg="mean")

            # --- output ---
            i_file += 1
            with stdout_to_file(file, f"benchmark_randint_constrained_{i_file}.md"):
                # headers
                if i_file in [1, 3]:
                    print_header(s.name, 2)
                    print(s.description)
                    print()

                if use_p:
                    print_header("Non-uniform sampling (custom p).", 3)
                else:
                    print_header("Uniform sampling.", 3)

                # timing results
                print_header("Timing Results", 4)
                print_table(headers, timing_data)

                # accuracy results
                print_header("Accuracy Results", 4)
                print_table(headers, accuracy_data)


# =================================================================================================
#  Internal helpers
# =================================================================================================
def _benchmark(
    s: Scenario,
    n: int,
    k: int,
    m: int,
    p: np.ndarray | None,
    speed: float,
    mode: str,
) -> BenchmarkResult:
    """
    Runs a benchmark and returns the BenchmarkResult.
    """
    n = np.int32(n)
    k = np.int32(k)

    # speed-dependent settings
    index_range = int(100 * (0.02**speed))  # 100 at speed=0, 2 at speed=1
    t_per_run = 0.05 / (1000.0**speed)
    n_warmup = int(8 - 5 * speed)
    n_benchmark = int(25 - 22 * speed)

    # build a <index_range> number of different constraints, to randomize the problems we benchmark
    lst_cons = []
    lst_con_values = []
    lst_con_indices = []
    for i in range(index_range):
        cons = s.build_constraints(n, k, m, seed=424242 * i)
        con_values, con_indices = _build_array_repr(cons)
        lst_cons.append(cons)
        lst_con_values.append(con_values)
        lst_con_indices.append(con_indices)

    if p is None:
        p = np.zeros(0, dtype=np.float32)
    else:
        p = p.astype(np.float32)

    if mode == "no_cons":
        # Benchmark randint_numba
        def benchmark_func(_idx: int):
            return randint_numba(n=n, k=k, replace=False, p=p)

    elif mode in ["non_eager", "eager"]:
        # Benchmark randint_constrained
        def benchmark_func(_idx: int):
            return randint_constrained(
                n=n,
                k=k,
                con_values=lst_con_values[_idx],
                con_indices=lst_con_indices[_idx],
                p=p,
                seed=np.int64(0),
                eager=(mode == "eager"),
            )

    else:
        # Benchmark randint_constrained_robust
        def benchmark_func(_idx: int):
            return randint_constrained_robust(
                n=n,
                k=k,
                con_values=lst_con_values[_idx],
                con_indices=lst_con_indices[_idx],
                p=p,
                seed=np.int64(0),
                n_trials=5,
            )

    return benchmark(
        f=benchmark_func,
        t_per_run=t_per_run,
        n_warmup=n_warmup,
        n_benchmark=n_benchmark,
        silent=True,
        index_range=index_range,
    )


def _determine_precision(
    s: Scenario,
    n: int,
    k: int,
    m: int,
    p: np.ndarray | None,
    speed: float,
    mode: str,
) -> Percentage:
    """
    Determines how often (%) the constraints are satisfied when sampling.
    """

    if p is None:
        p = np.zeros(0, dtype=np.float32)
    else:
        p = p.astype(np.float32)

    # Calculate number of runs based on speed (1000 at speed=0, 2 at speed=1)
    n_runs = int(1000 * (0.002**speed))

    satisfied_count = 0
    for run_idx in range(n_runs):
        # --- build constraints ---
        cons = s.build_constraints(n, k, m, seed=424242 * run_idx)
        con_values, con_indices = _build_array_repr(cons)

        # Run the appropriate function with seed equal to run index
        if mode == "no_cons":
            result = randint_numba(n=np.int32(n), k=np.int32(k), replace=False, p=p, seed=np.int64(run_idx))
        elif mode in ["non_eager", "eager"]:
            # Use randint_constrained_numba
            result = randint_constrained(
                n=np.int32(n),
                k=np.int32(k),
                con_values=con_values,
                con_indices=con_indices,
                p=p,
                seed=np.int64(run_idx),
                eager=(mode == "eager"),
            )
        else:
            # Use randint_constrained_robust
            result = randint_constrained_robust(
                n=np.int32(n),
                k=np.int32(k),
                con_values=con_values,
                con_indices=con_indices,
                p=p,
                seed=np.int64(run_idx),
                n_trials=5,
            )

        # Check if all constraints are satisfied
        constraints_satisfied = True
        for con in cons:
            count = sum(1 for val in result if val in con.int_set)
            if count < con.min_count or count > con.max_count:
                constraints_satisfied = False
                break

        if constraints_satisfied:
            satisfied_count += 1

    return Percentage(frac=satisfied_count / n_runs, decimals=1)


# =================================================================================================
#  Testing Scenarios
# =================================================================================================
class Scenario(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def n_k_m_tuples(self) -> list[tuple[int, int, int]]:
        raise NotImplementedError()

    @abstractmethod
    def build_constraints(self, n: int, k: int, m: int, seed: int) -> list[Constraint]:
        raise NotImplementedError()


class ScenarioA(Scenario):
    def __init__(self):
        super().__init__(
            name="Scenario A",
            description="Varying n & k with 10 non-overlapping constraints spanning equal portions of the n-range",
        )

    def n_k_m_tuples(self) -> list[tuple[int, int, int]]:
        return [
            (n, k, 10)
            for n in [10, 100, 1000]
            for k in [2**i for i in range(1, 9)]  # 2, 4, 8, ..., 256
            if k < n
        ]

    def build_constraints(self, n: int, k: int, m: int, seed: int) -> list[Constraint]:
        return [
            Constraint(
                int_set=set(range(i * (n // 10), (i + 1) * (n // 10))),
                min_count=math.floor(k / 11),
                max_count=math.ceil(k / 9),
            )
            for i in range(10)
        ]


class ScenarioB(Scenario):
    def __init__(self):
        super().__init__(
            name="Scenario B",
            description="Fixed n=1000 & k=100 with varying number of constraints spanning random 1% portions of the n-range",
        )

    def n_k_m_tuples(self) -> list[tuple[int, int, int]]:
        return [
            (1000, 100, m)
            for m in [2**i for i in range(1, 9)] + [384, 512, 768, 1024]  # 2, 4, 8, ..., 256, 384, 512, 768, 1024
        ]

    def build_constraints(self, n: int, k: int, m: int, seed: int) -> list[Constraint]:
        cons = []
        for i in range(m):
            cons.append(
                Constraint(
                    int_set=set(
                        randint_numba(
                            n=np.int32(n),
                            k=np.int32(n // 100),  # 1% random samples from n
                            replace=False,
                            seed=np.int64(seed + i),
                        )
                    ),
                    min_count=1 + math.floor(10 / m),
                    max_count=1 + math.ceil(1000 / m),
                )
            )
        return cons
