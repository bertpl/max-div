import random
from pathlib import Path
from typing import Callable

import click

from max_div.internal.formatting import format_time_duration
from max_div.solver import TargetTimeDuration

from ._executors import executor_multi_parallel
from ._models import SolverPresetBenchmarkParams, SolverPresetBenchmarkResult, results_to_json
from ._utils import estimate_execution_time_sec_single, get_n_processes


# =================================================================================================
#  Main function
# =================================================================================================
def execute_solver_presets_benchmark(
    scope: list[SolverPresetBenchmarkParams],
    json_file_name: Path | None,
    executor: Callable[
        [list[SolverPresetBenchmarkParams], int], list[SolverPresetBenchmarkResult]
    ] = executor_multi_parallel,
):
    # --- semi-randomization ------------------------------
    # first we shuffle, and then we sort from long to short durations
    # --> this way we will shuffle presets and problems, but still process short duration runs last
    #     (=more efficient to keep all parallel workers busy as long as possible)
    random.seed(42)
    random.shuffle(scope)
    scope = sorted(scope, key=lambda p: p.duration.value(), reverse=True)

    # --- determine total processes -----------------------
    n_processes = get_n_processes(len(scope))

    # --- execute -----------------------------------------
    results = executor(scope, n_processes)
    results = sorted(
        results,
        key=lambda r: (
            r.params.problem_name,
            r.params.preset.name,
            r.params.duration.value(),
            r.params.seed,
        ),
    )

    # --- save --------------------------------------------
    if json_file_name:
        with open(json_file_name, "w") as f:
            f.write(results_to_json(results))

    # --- return ------------------------------------------
    return results
