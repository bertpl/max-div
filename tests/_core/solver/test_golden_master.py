"""Bit-exact golden master over seeded solves.

Permanent determinism guard: solves a matrix of preset x benchmark problem x seed and compares
the full deterministic output (final selection + all score checkpoints) against committed
expected data, with exact equality. Any drift is a regression, unless it comes from an
intentional numeric change, in which case the expected data must be regenerated in the same
change (see below) so the drift is explicit.

JIT-compiled and NUMBA_DISABLE_JIT execution produce identical selections but slightly different
score floats (float32 register arithmetic vs numpy's float64 scalar promotion), while each
regime is bit-stable across runs. The expected data is therefore committed per regime, and the
test asserts against the dataset matching the active regime.

To regenerate the expected data (both regimes) after an intentional numeric change:

    uv run --all-extras --python 3.13 python -m tests._core.solver.test_golden_master
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from numba import config as numba_config

from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.solver import MaxDivSolverBuilder, SolverPreset
from max_div._core.solver._duration import iterations
from max_div._core.solver._solution import MaxDivSolution

# =================================================================================================
#  Matrix & expected data
# =================================================================================================
PROBLEMS = ["U1", "U2", "U3", "U4", "C1", "C2", "C3", "C4"]
PRESETS = [SolverPreset.RANDOM, SolverPreset.GUIDED, SolverPreset.SMART, SolverPreset.THOROUGH]
SEEDS = [42, 123]

# small problems (size=1 -> n=100..150, k=10) + tiny iteration budget: full matrix must stay
# fast with NUMBA_DISABLE_JIT, where these solves run interpreted
N_ITERATIONS = 30


def _active_regime() -> str:
    return "nojit" if numba_config.DISABLE_JIT else "jit"


def _data_file(regime: str) -> Path:
    return Path(__file__).parent / f"golden_master_data_{regime}.json"


def _solve(problem_name: str, preset: SolverPreset, seed: int) -> MaxDivSolution:
    params = BenchmarkProblemFactory.get_all_benchmark_problems()[problem_name].get_example_parameters()
    params["size"] = 1
    problem = BenchmarkProblemFactory.construct_problem(problem_name, **params)
    solver = MaxDivSolverBuilder(problem).with_preset(iterations(N_ITERATIONS), preset).with_seed(seed).build()
    return solver.solve(verbosity=0)


def _as_record(solution: MaxDivSolution) -> dict[str, Any]:
    """Extract the deterministic part of a solution (selection + score checkpoints, no wall-clock times)."""
    return {
        "i_selected": [int(i) for i in solution.i_selected],
        "score_checkpoints": [
            {
                "step": step_name,
                "size": score.size,
                "constraints": score.constraints,
                "diversity": score.diversity,
                "div_tie_breakers": list(score.div_tie_breakers),
            }
            for step_name, _, score in solution.score_checkpoints
        ],
    }


def _case_key(problem_name: str, preset: SolverPreset, seed: int) -> str:
    return f"{problem_name}|{preset.name}|{seed}"


# =================================================================================================
#  Tests
# =================================================================================================
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("problem_name", PROBLEMS)
def test_golden_master(problem_name: str, preset: SolverPreset, seed: int):
    # --- arrange -----------------------------------------
    expected_data = json.loads(_data_file(_active_regime()).read_text())
    expected = expected_data[_case_key(problem_name, preset, seed)]

    # --- act ---------------------------------------------
    solution = _solve(problem_name, preset, seed)

    # --- assert ------------------------------------------
    assert _as_record(solution) == expected  # exact equality, incl. float bits (see module docstring)


# =================================================================================================
#  Regeneration mode
# =================================================================================================
def regenerate_active_regime() -> None:
    """Recompute and overwrite the expected data file of the active numba regime, for the full matrix."""
    records = {}
    for problem_name in PROBLEMS:
        for preset in PRESETS:
            for seed in SEEDS:
                records[_case_key(problem_name, preset, seed)] = _as_record(_solve(problem_name, preset, seed))
    data_file = _data_file(_active_regime())
    data_file.write_text(json.dumps(records, indent=1) + "\n")
    print(f"wrote {len(records)} cases to {data_file}")  # noqa: T201 -- script mode


if __name__ == "__main__":
    if "--active-regime-only" in sys.argv:
        regenerate_active_regime()
    else:
        # numba reads NUMBA_DISABLE_JIT at import time, so each regime runs in its own interpreter
        for disable_jit in ("0", "1"):
            env = {**os.environ, "NUMBA_DISABLE_JIT": disable_jit}
            subprocess.run([sys.executable, "-m", __spec__.name, "--active-regime-only"], env=env, check=True)  # noqa: S603 -- fixed args, script mode
