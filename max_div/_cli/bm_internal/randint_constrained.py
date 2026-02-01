from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from max_div.internal.benchmarking import benchmark
from max_div.internal.markdown import (
    Report,
    Table,
    TableAggregationType,
    TablePercentage,
    TableTimeElapsed,
    h2,
    h3,
    h4,
)
from max_div.internal.utils import stdout_to_file
from max_div.random import P_UNIFORM, Constraint, ConstraintList, new_rng_state, randint, randint_constrained


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
    :param file: If `True`, redirects output to a file instead of console.
    """

    # --- speed-dependent settings --------------------
    max_count = int(100 * (0.01**speed))  # max_count=100 if speed=0;  max_count=1 at speed=1

    # --- build scenarios ---------------------------------
    scenarios = [ScenarioA(), ScenarioB()]

    # --- benchmark all scenarios -------------------------
    print("Benchmarking `randint_constrained`...")

    i_file = 0
    for s in scenarios:
        for use_p in [False, True]:
            # --- benchmark scenario ----------------
            headers = [
                "`k`",
                "`n`",
                "`m`",
                "`randint`",
                "`randint_constrained`\n(eager=False)",
                "`randint_constrained`\n(eager=True)",
            ]
            timing_table = Table(headers)
            accuracy_table = Table(headers)

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
                timing_table.add_row(
                    [
                        str(k),
                        str(n),
                        str(m),
                        _benchmark(s, n, k, m, p, speed, "no_cons"),
                        _benchmark(s, n, k, m, p, speed, "non_eager"),
                        _benchmark(s, n, k, m, p, speed, "eager"),
                    ]
                )

                accuracy_table.add_row(
                    [
                        str(k),
                        str(n),
                        str(m),
                        _determine_precision(s, n, k, m, p, speed, "no_cons"),
                        _determine_precision(s, n, k, m, p, speed, "non_eager"),
                        _determine_precision(s, n, k, m, p, speed, "eager"),
                    ]
                )

            # --- show all results --------------------------------------------

            # --- prepare final report ---
            timing_table.add_aggregate_row(TableAggregationType.GEOMEAN)
            timing_table.highlight_results(TableTimeElapsed, clr_lowest=Table.GREEN)

            accuracy_table.add_aggregate_row(TableAggregationType.MEAN)
            accuracy_table.highlight_results(TablePercentage, clr_highest=Table.GREEN)

            report = Report()
            if i_file in [1, 3]:
                report += [h2(s.name), s.description]
            if use_p:
                report += h3("Non-uniform sampling (custom p).")
            else:
                report += h3("Uniform sampling.")

            report += [
                h4("Timing Results"),
                timing_table,
                h4("Accuracy Results"),
                accuracy_table,
            ]

            # --- output ---
            i_file += 1
            with stdout_to_file(file, f"benchmark_randint_constrained_{i_file}.md"):
                report.print(markdown=markdown)


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
) -> TableTimeElapsed:
    """
    Runs a benchmark and returns the BenchmarkResult.
    """
    n = np.int32(n)
    k = np.int32(k)

    # speed-dependent settings
    index_range = int(100 * (0.02**speed))  # 100 at speed=0, 2 at speed=1
    t_per_run = 0.01 / (1000.0**speed)
    n_warmup = int(8 - 6 * speed)
    n_benchmark = int(25 - 24 * speed)

    # build a <index_range> number of different constraints, to randomize the problems we benchmark
    lst_cons = []
    lst_con_values = []
    lst_con_indices = []
    for i in range(index_range):
        cons = s.build_constraints(n, k, m, seed=424242 * i)
        con_values, con_indices = ConstraintList(cons).to_numpy()
        lst_cons.append(cons)
        lst_con_values.append(con_values)
        lst_con_indices.append(con_indices)

    if p is None:
        p = np.zeros(0, dtype=np.float32)
    else:
        p = p.astype(np.float32)

    rng_state = new_rng_state(np.int64(42))

    if mode == "no_cons":
        # Benchmark randint
        def benchmark_func(_idx: int):
            return randint(
                n=n,
                k=k,
                replace=False,
                p=p,
                rng_state=rng_state,
            )

    else:
        # Benchmark randint_constrained
        eager = mode == "eager"

        def benchmark_func(_idx: int):
            return randint_constrained(
                n=n,
                k=k,
                con_values=lst_con_values[_idx],
                con_indices=lst_con_indices[_idx],
                p=p,
                rng_state=rng_state,
                eager=eager,
            )

    return TableTimeElapsed.from_benchmark_result(
        benchmark(
            f=benchmark_func,
            t_per_run=t_per_run,
            n_warmup=n_warmup,
            n_benchmark=n_benchmark,
            silent=True,
            index_range=index_range,
        )
    )


def _determine_precision(
    s: Scenario,
    n: int,
    k: int,
    m: int,
    p: np.ndarray | None,
    speed: float,
    mode: str,
) -> TablePercentage:
    """
    Determines how often (%) the constraints are satisfied when sampling.
    """

    if p is None:
        p = np.zeros(0, dtype=np.float32)
    else:
        p = p.astype(np.float32)

    # Calculate number of runs based on speed (200 at speed=0, 1 at speed=1)
    n_runs = round(200 ** (1 - speed))

    satisfied_count = 0
    for run_idx in range(n_runs):
        # --- build constraints ---
        cons = s.build_constraints(n, k, m, seed=424242 * run_idx)
        con_values, con_indices = ConstraintList(cons).to_numpy()

        # Run the appropriate function with seed equal to run index
        rng_state = new_rng_state(np.int64(run_idx))
        if mode == "no_cons":
            result = randint(n=np.int32(n), k=np.int32(k), replace=False, p=p, rng_state=rng_state)
        else:
            # Use randint_constrained_numba
            result = randint_constrained(
                n=np.int32(n),
                k=np.int32(k),
                con_values=con_values,
                con_indices=con_indices,
                p=p,
                rng_state=rng_state,
                eager=(mode == "eager"),
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

    return TablePercentage(frac=satisfied_count / n_runs, decimals=1)


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
                        randint(
                            n=np.int32(n),
                            k=np.int32(n // 100),  # 1% random samples from n
                            replace=False,
                            p=P_UNIFORM,
                            rng_state=new_rng_state(np.int64(seed + i)),
                        )
                    ),
                    min_count=1 + math.floor(10 / m),
                    max_count=1 + math.ceil(1000 / m),
                )
            )
        return cons
