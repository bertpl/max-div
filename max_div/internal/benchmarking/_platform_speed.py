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
      - we then compute overall speed as 1e-3/geomean(all 10 median times) -> reference platform will have 1.0 as result
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from typing import Callable

import numpy as np

from max_div.internal.formatting._time_duration import format_short_time_duration

from ._timer import Timer


# =================================================================================================
#  Main functionality
# ================================================================================================
@cache
def estimate_platform_speed(silent: bool = True, fast: bool = False) -> float:
    """
    Estimate platform speed as a relative number compared to a reference platform.  This number can be used to correct
    estimates of the pre-calibrated time models for initialization & optimization strategies to get time estimates for
    the actual platform being used, rather than the reference platform.

    Returns:
        float: Estimated platform speed (higher is faster) relative to the reference platform (which corresponds to 1.0)
    """

    # --- get reference benchmarks -----------------------
    benchmarks = _get_reference_benchmarks()

    # --- warm-up ----------------------------------------
    for bm in benchmarks:
        bm.run(size=10)  # override size just for warm-up; below 10 some tests are partially a no-op

    # --- speed-dependent settings -----------------------
    n_runs = 1 if fast else 9

    # --- benchmark --------------------------------------
    run_times: dict[int, list[float]] = defaultdict(list)
    for _ in range(n_runs):
        for i_bm, bm in enumerate(benchmarks):
            run_times[i_bm].append(bm.run())

    # --- compute speed ----------------------------------
    med_run_times = [float(np.median(run_times_lst)) for run_times_lst in run_times.values()]

    if not silent:
        for bm, t in zip(benchmarks, med_run_times):
            print(f"{bm.name}: median time = {format_short_time_duration(t, n_chars=7)}")

    # for the reference platform, geomean_time should be very close to 1ms
    # hence we return 1e-3/geomean_time as speed, in order to return 1.0 on the reference platform
    geomean_time = np.exp(np.mean(np.log(np.array(med_run_times))))
    speed = 1e-3 / geomean_time

    if not silent:
        print()
        print(f" --> Estimated platform speed: {speed:.3f}")

    return speed


# =================================================================================================
#  Suite of micro-benchmarks
# =================================================================================================
@dataclass
class PlatformSpeedBenchmark:
    name: str
    function: Callable[[int], None]
    reference_size: int

    def run(self, size: int | None = None) -> float:
        """Run benchmark and return elapsed time in seconds."""
        with Timer() as t:
            self.function(size or self.reference_size)
        return t.t_elapsed_sec()


def _get_reference_benchmarks() -> list[PlatformSpeedBenchmark]:
    # -------------------------------------------------------------------------
    #  Late imports to avoid circular dependencies
    # -------------------------------------------------------------------------
    from max_div.internal.math.modify_p_selectivity import exponential_selectivity
    from max_div.random import Constraint, ConstraintList, new_rng_state, randint, randint_constrained
    from max_div.solver._distance import DistanceMetric
    from max_div.solver._diversity import DiversityMetric
    from max_div.solver._solver_state import SolverState

    # -------------------------------------------------------------------------
    #  Define micro-benchmarks - GENERIC
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    #  Define micro-benchmarks - MAX-DIV-SPECIFIC
    # -------------------------------------------------------------------------
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
            _ = randint(
                n=np.int32(size),
                k=np.int32(size // 2),
                replace=replace,
                p=p,
                rng_state=new_rng_state(np.int64(size)),
            )

    def _bm_max_div_randint_constrained(size: int):
        p = np.linspace(0.1, 1.0, size, dtype=np.float32)
        constraints = [
            Constraint(
                int_set=set(range(size // 2)),
                min_count=size // 4,
                max_count=size // 3,
            )
        ]
        con_values, con_indices = ConstraintList(constraints).to_numpy()
        for eager in [True, False]:
            _ = randint_constrained(
                n=np.int32(size),
                k=np.int32(size // 2),
                con_values=con_values,
                con_indices=con_indices,
                p=p,
                rng_state=new_rng_state(np.int64(size)),
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

    # -------------------------------------------------------------------------
    #  Collect & return
    # -------------------------------------------------------------------------
    return [
        # --- GENERIC ---------------------
        PlatformSpeedBenchmark("gen_list_log2    ", _bm_generic_list_log2, 2_300),
        PlatformSpeedBenchmark("gen_sort_int32   ", _bm_generic_sort_int32, 6_750),
        PlatformSpeedBenchmark("gen_for_cond     ", _bm_generic_for_loop_conditional, 7_750),
        PlatformSpeedBenchmark("gen_np_sum_f32   ", _bm_np_sum_float32, 85_000),
        PlatformSpeedBenchmark("gen_np_add_f64   ", _bm_np_addition_float64, 43_000),
        # --- MAX-DIV-SPECIFIC ------------
        PlatformSpeedBenchmark("pkg_diversity    ", _bm_max_div_diversity, 260_000),
        PlatformSpeedBenchmark("pkg_selectivity  ", _bm_max_div_selectivity, 470_000),
        PlatformSpeedBenchmark("pkg_randint      ", _bm_max_div_randint, 37_250),
        PlatformSpeedBenchmark("pkg_randint_con  ", _bm_max_div_randint_constrained, 710),
        PlatformSpeedBenchmark("pkg_solver_state ", _bm_max_div_solver_state, 380),
    ]
