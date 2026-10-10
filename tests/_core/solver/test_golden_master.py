"""Bit-exact golden master over seeded solves.

Permanent determinism guard: solves a matrix of preset x benchmark problem x seed and compares
the full deterministic output (final selection + all score checkpoints) against committed
expected data, with exact equality. Any drift is a regression, unless it comes from an
intentional numeric change, in which case the expected data must be regenerated in the same
change (see below) so the drift is explicit.

What it guards is the *search*: the RNG, the scoring, the swap sequence, the tie-breaks. It
deliberately does not guard distance computation, and is built so it cannot. Each case is
solved from a precomputed distance matrix, so no pairwise distance function runs during the
solve, and that matrix is built here with plain numpy, not through the library's own pairwise
distance functions. Distance computation is guarded separately, by the cross-backend agreement
tests.

That separation is what keeps this guard meaningful. A compiler is free to reassociate the
sums inside a pairwise distance function, and reassociation is target- and toolchain-dependent — so a
bit-exact pin over computed distances would go red on a new numba release or a different CPU
for reasons that are not regressions, and the habit of regenerating it to restore a green
build would leave it guarding nothing.

JIT-compiled and NUMBA_DISABLE_JIT execution produce identical selections but slightly different
score floats (float32 register arithmetic vs numpy's float64 scalar promotion), while each
JIT setting is bit-stable across runs. The expected data is therefore committed per JIT setting,
and the test asserts against the dataset of the current JIT setting:

- 'nojit' (interpreted) output is environment-independent (verified across platforms and
  Python versions), so it is asserted unconditionally — including on the jit-off CI jobs.
- 'jit' output depends on numba's codegen, which varies with the Python and numba versions
  (numba compiles from Python bytecode, so two Python minors can round floats differently
  under the same numba) and with the CPU architecture that LLVM generates code for.

  The jit dataset therefore records the fingerprint of the environment it was generated in,
  and only a run in that environment can check the jit dataset: a change in numba's code
  generation caused by a new Python or numba version is not a regression of this codebase.

  So with JIT compilation on, the test runs only when the environment variable
  MAX_DIV_JIT_GOLDEN_MASTER is 1. The `[tool.ci-test-matrix]` table in pyproject.toml decides which
  CI jobs set it, and `JIT_GOLDEN_MASTER_ENV` in the Makefile sets it on the matching `make test`
  invocations.

  On a run with the variable set, every case errors on a fingerprint mismatch, because such a
  mismatch means that uv.lock pins other versions than the ones recorded in the jit dataset.

To regenerate the expected data (both JIT settings) after an intentional numeric change or a numba
upgrade in uv.lock:

    uv run --all-extras --python 3.14 python -m tests._core.solver.test_golden_master
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numba import config as numba_config
from numpy.typing import NDArray

from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.problem import MaxDivProblem
from max_div._core.solver import MaxDivSolverBuilder, SolverPreset, Verbosity
from max_div._core.solver._duration import iterations
from max_div._core.solver._solution import MaxDivSolution

# ==================================================================================================
#  Matrix & expected data
# ==================================================================================================
PROBLEMS = ["U1", "U2", "U3", "U4", "C1", "C2", "C3", "C4"]
PRESETS = [SolverPreset.RANDOM, SolverPreset.GUIDED, SolverPreset.SMART, SolverPreset.THOROUGH]
SEEDS = [42, 123]

# small problems (PROBLEM_N) + tiny iteration budget: full matrix must stay
# fast with NUMBA_DISABLE_JIT, where these solves run interpreted
N_ITERATIONS = 30
PROBLEM_N = 100

# with JIT compilation on, the test runs only on request (see the module docstring)
IS_JIT_GOLDEN_MASTER_REQUESTED = os.environ.get("MAX_DIV_JIT_GOLDEN_MASTER") == "1"


def _is_numba_jit_enabled() -> bool:
    """Return True when numba compiles, and False when NUMBA_DISABLE_JIT=1 runs the code interpreted."""
    return not numba_config.DISABLE_JIT


def _runtime_fingerprint() -> dict[str, str]:
    """Identify the properties of the runtime that jit-compiled numeric output depends on."""
    import numba

    # one name per architecture: macOS reports arm64 where Linux reports aarch64, and Windows
    # reports AMD64 where the others report x86_64
    machine = platform.machine().lower()
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numba": numba.__version__,
        "cpu_architecture": {"aarch64": "arm64", "amd64": "x86_64"}.get(machine, machine),
    }


def _data_file(is_numba_jit_enabled: bool) -> Path:
    """Return the expected data file for numba JIT compilation on or off."""
    suffix = "jit" if is_numba_jit_enabled else "nojit"
    return Path(__file__).parent / f"golden_master_data_{suffix}.json"


def _distances_from(vectors: NDArray[np.float32], metric: DistanceMetric) -> NDArray[np.float32]:
    """Build a distance matrix with plain numpy, independent of the library's own pairwise distance functions.

    The library's pairwise distance functions compile with `fastmath={"reassoc", "contract"}`, so
    their float bits change with the CPU and the numba version without any regression. Computing
    the matrix here keeps those changes out of this guard, which checks the search given fixed
    distances; `tests/_core/metrics/_distance/_store/test_bundle.py` checks the distance functions
    across backends. The result is quantized and symmetrized so the matrix is bit-identical on
    every machine — the same reasoning that quantizes the vectors below, applied one step later.
    """
    diff = vectors[:, None, :].astype(np.float64) - vectors[None, :, :].astype(np.float64)
    if metric == DistanceMetric.l1_manhattan():
        raw = np.abs(diff).sum(axis=-1)
    elif metric == DistanceMetric.linf_chebyshev():
        raw = np.abs(diff).max(axis=-1)
    elif metric == DistanceMetric.l2s_euclidean_squared():
        raw = (diff * diff).sum(axis=-1)
    else:  # euclidean, and cosine's pre-normalized rows reduce to it monotonically
        raw = np.sqrt((diff * diff).sum(axis=-1))
    matrix = np.round(raw, 4).astype(np.float32)
    matrix = np.minimum(matrix, matrix.T)  # exact symmetry, structurally rather than by tolerance
    np.fill_diagonal(matrix, np.float32(0.0))
    return np.ascontiguousarray(matrix)


def _solve(problem_name: str, preset: SolverPreset, seed: int) -> MaxDivSolution:
    generated = BenchmarkProblemFactory.construct_problem(
        problem_name, n=PROBLEM_N, diversity_metric=DiversityMetric.approx_geomean_separation()
    )
    # quantize the vectors: problem generation may involve transcendental functions (e.g. a power
    # mapping) whose SIMD implementations differ ~1 ULP across CPU generations; rounding to a coarse
    # grid absorbs that, so the solver runs on bit-identical inputs on every machine
    vectors = np.round(generated.vectors, 2).astype(np.float32)
    problem = MaxDivProblem.from_distances(
        distances=_distances_from(vectors, generated.distance_metric),
        k=generated.k,
        diversity_metric=generated.diversity_metric,
        constraints=generated.constraints,
    )
    solver = MaxDivSolverBuilder(problem).with_preset(iterations(N_ITERATIONS), preset).with_seed(seed).build()
    return solver.solve(verbosity=Verbosity.SILENT)


def _convert_solution_to_record(solution: MaxDivSolution) -> dict[str, Any]:
    """Extract the deterministic part of a solution (selection + score checkpoints, no wall-clock times)."""
    return {
        "i_selected": [int(i) for i in solution.i_selected],
        "score_checkpoints": [
            {
                "step_index": checkpoint.step_identity.step_index,
                "step": checkpoint.step_identity.step_name,
                "size": checkpoint.score.size,
                "constraints": checkpoint.score.constraints,
                "diversity": checkpoint.score.diversity,
                "div_tie_breakers": list(checkpoint.score.div_tie_breakers),
            }
            for checkpoint in solution.score_checkpoints
        ],
    }


def _case_key(problem_name: str, preset: SolverPreset, seed: int) -> str:
    return f"{problem_name}|{preset.name}|{seed}"


# ==================================================================================================
#  Tests
# ==================================================================================================
@pytest.fixture(scope="module")
def expected_data() -> dict[str, Any]:
    """Load the expected data of the current JIT setting; fail if the jit data's fingerprint differs from this run's."""
    is_numba_jit_enabled = _is_numba_jit_enabled()
    dataset = json.loads(_data_file(is_numba_jit_enabled).read_text())
    runtime_fingerprint = _runtime_fingerprint()
    if is_numba_jit_enabled and dataset["fingerprint"] != runtime_fingerprint:
        pytest.fail(
            f"the jit expected data was generated with {dataset['fingerprint']}, this run has {runtime_fingerprint}. "
            "After a numba upgrade in uv.lock, regenerate it with: "
            "uv run --all-extras --python 3.14 python -m tests._core.solver.test_golden_master. "
            "On another CPU architecture, skip it with: make test JIT_GOLDEN_MASTER_ENV="
        )
    return dataset


@pytest.mark.skipif(
    _is_numba_jit_enabled() and not IS_JIT_GOLDEN_MASTER_REQUESTED,
    reason="with JIT compilation on, the golden master runs only with MAX_DIV_JIT_GOLDEN_MASTER=1",
)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("problem_name", PROBLEMS)
def test_golden_master(problem_name: str, preset: SolverPreset, seed: int, expected_data: dict[str, Any]):
    """A seeded solve reproduces the expected data of the current JIT setting bit for bit."""
    # --- arrange ----------------------
    expected = expected_data["cases"][_case_key(problem_name, preset, seed)]

    # --- act --------------------------
    solution = _solve(problem_name, preset, seed)

    # --- assert -----------------------
    assert _convert_solution_to_record(solution) == expected  # exact equality, incl. float bits (see module docstring)


# ==================================================================================================
#  Regeneration mode
# ==================================================================================================
def regenerate_for_current_jit_setting() -> None:
    """Recompute and overwrite the expected data file of the current JIT setting, for the full matrix."""
    records = {}
    for problem_name in PROBLEMS:
        for preset in PRESETS:
            for seed in SEEDS:
                records[_case_key(problem_name, preset, seed)] = _convert_solution_to_record(
                    _solve(problem_name, preset, seed)
                )
    data_file = _data_file(_is_numba_jit_enabled())
    data = {"fingerprint": _runtime_fingerprint(), "cases": records}
    data_file.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {len(records)} cases to {data_file}")


if __name__ == "__main__":
    if "--current-jit-setting-only" in sys.argv:
        regenerate_for_current_jit_setting()
    else:
        # numba reads NUMBA_DISABLE_JIT at import time, so each JIT setting runs in its own interpreter
        for disable_jit in ("0", "1"):
            env = {**os.environ, "NUMBA_DISABLE_JIT": disable_jit}
            subprocess.run([sys.executable, "-m", __spec__.name, "--current-jit-setting-only"], env=env, check=True)  # noqa: S603 -- fixed args, script mode
