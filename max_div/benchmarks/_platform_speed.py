"""
Module to get a rough indication of processing speed of the platform (CPU, RAM, OS, Python, numba, ...)

Overall setup:
  - composed of 10 low-level micro-benchmark functions
      - 5 generic functions
      - 5 package-specific functions
      - each function takes 1 argument (size: int); no return value.  Time taken is proportional to size.
  - overall setup
      - size-values for all functions are pre-set such that on the ref. platform they take 1msec each
      - we run each of 10 benchmarks once for warm-up purposes
      - then we run each of 10 benchmarks a total of 9 more times (interleaved)
      - we take the median value of each of 9 executions per benchmark
      - we then compute overall speed as 1/geomean(all 10 median times) -> reference platform will have 1000 as result
"""

from collections import defaultdict
from functools import cache, partial
from typing import Callable

import numpy as np

from max_div.internal.benchmarking import Timer
from max_div.internal.formatting._time_duration import format_short_time_duration
from max_div.internal.math.modify_p_selectivity import exponential_selectivity
from max_div.sampling import randint_numba
from max_div.sampling._constraint_helpers import _build_array_repr
from max_div.sampling.con import randint_constrained
from max_div.solver import Constraint, DistanceMetric, DiversityMetric, MaxDivProblem
from max_div.solver._solver_state import SolverState


# =================================================================================================
#  Benchmark functions - GENERIC
# =================================================================================================
def _bm_generic_list_log2(size: int):
    _ = [np.log2(1 + i) for i in range(size)]


def _bm_generic_sort_int32(size: int):
    _ = sorted(np.random.randint(0, 1_000_000, size=size).astype(np.int32))


def _bm_generic_for_loop_conditional(size: int):
    t = 0.0
    for i in range(size):
        if np.random.rand() < 0.5:
            t += 1.0


def _bm_np_sum_float32(size: int):
    _ = np.sum(np.random.randn(size).astype(np.float32))


def _bm_np_addition_float64(size: int):
    x = np.random.randn(size).astype(np.float64)
    y = np.random.randn(size).astype(np.float64)
    _ = x + y


# =================================================================================================
#  Benchmark functions - MAX-DIV-SPECIFIC
# =================================================================================================
def _bm_max_div_diversity(size: int):
    sep = np.linspace(0.1, 1.0, size, dtype=np.float32)
    for metric in DiversityMetric.all_metrics():
        _ = metric.compute(sep)


def _bm_max_div_selectivity(size: int):
    p = np.linspace(0.1, 1.0, size, dtype=np.float32)
    exponential_selectivity(p, p, modifier=np.float32(0.5))


def _bm_max_div_randint(size: int):
    p = np.linspace(0.1, 1.0, size, dtype=np.float32)
    for replace in [True, False]:
        _ = randint_numba(n=np.int32(size), k=np.int32(size // 2), replace=replace, p=p, seed=np.int64(size))


def _bm_max_div_randint_constrained(size: int):
    p = np.linspace(0.1, 1.0, size, dtype=np.float32)
    constraints = [
        Constraint(
            int_set=set(range(size // 2)),
            min_count=size // 4,
            max_count=size // 3,
        )
    ]
    con_values, con_indices = _build_array_repr(constraints)
    for eager in [True, False]:
        _ = randint_constrained(
            n=np.int32(size),
            k=np.int32(size // 2),
            con_values=con_values,
            con_indices=con_indices,
            p=p,
            seed=np.int64(size),
            eager=eager,
        )


def _bm_max_div_solver_state(size: int):
    solver_state = SolverState.new(
        vectors=np.random.randn(size, 10).astype(np.float32),
        k=size // 10,
        distance_metric=DistanceMetric.L2_EUCLIDEAN,
        diversity_metric=DiversityMetric.approx_geomean_separation(),
        diversity_tie_breakers=[],
        constraints=[
            Constraint(
                int_set=set(range(size // 2)),
                min_count=size // 4,
                max_count=size // 3,
            )
        ],
    )
    for i_add in range(size // 10):
        solver_state.add(i_add)
    for i_remove in range(size // 10):
        solver_state.remove(i_remove)


# =================================================================================================
#  Main functionality
# ================================================================================================
_REFERENCE_BENCHMARKS: list[tuple[str, Callable]] = [
    # --- GENERIC ---------------------
    ("gen_list_log2    ", partial(_bm_generic_list_log2, size=2_150)),
    ("gen_sort_int32   ", partial(_bm_generic_sort_int32, size=6_400)),
    ("gen_for_cond     ", partial(_bm_generic_for_loop_conditional, size=7_500)),
    ("gen_np_sum_f32   ", partial(_bm_np_sum_float32, size=85_000)),
    ("gen_np_add_f64   ", partial(_bm_np_addition_float64, size=43_000)),
    # --- MAX-DIV-SPECIFIC ------------
    ("pkg_diversity    ", partial(_bm_max_div_diversity, size=260_000)),
    ("pkg_selectivity  ", partial(_bm_max_div_selectivity, size=470_000)),
    ("pkg_randint      ", partial(_bm_max_div_randint, size=37_500)),
    ("pkg_randint_con  ", partial(_bm_max_div_randint_constrained, size=710)),
    ("pkg_solver_state ", partial(_bm_max_div_solver_state, size=380)),
]


@cache
def estimate_platform_speed(silent: bool = True, fast: bool = False) -> int:
    """Estimate platform speed as a relative number compared to a reference platform.

    The reference platform is defined such that it achieves a speed score of 1000.

    Returns:
        int: Estimated platform speed (higher is faster; reference platform has speed=1000).
    """

    # --- warm-up ----------------------------------------
    for name, bm in _REFERENCE_BENCHMARKS:
        bm(size=10)  # override size just for warm-up; below 10 some tests are partially a no-op

    # --- speed-dependent settings -----------------------
    # NOTE: taking the first 5 benchmarks avoids numba-accelerated functions in fast mode;
    #       this sounds counter-intuitive, but for testing code coverage, we disable numba for improved granularity,
    #       making a) these functions not representative anyway, b) causing massive slowdowns when executing tests,
    #       c) we trigger them anyway with very small size above, for warm-up purposes.
    benchmarks = _REFERENCE_BENCHMARKS[:5] if fast else _REFERENCE_BENCHMARKS
    n_runs = 1 if fast else 9

    # --- benchmark --------------------------------------
    run_times: dict[int, list[float]] = defaultdict(list)
    for _ in range(n_runs):
        for i_bm, (name, bm) in enumerate(benchmarks):
            with Timer() as t:
                bm()
            run_times[i_bm].append(t.t_elapsed_sec())

    # --- compute speed ----------------------------------
    med_run_times = [float(np.median(run_times_lst)) for run_times_lst in run_times.values()]

    if not silent:
        for (bm_name, bm), t in zip(_REFERENCE_BENCHMARKS, med_run_times):
            print(f"{bm_name}: median time = {format_short_time_duration(t, n_chars=7)}")

    geomean_time = np.exp(np.mean(np.log(np.array(med_run_times))))
    speed = int(round(1.0 / geomean_time))

    return speed
