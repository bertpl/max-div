"""Bit-exact golden master over seeded solves.

Permanent determinism guard: solves a matrix of preset x benchmark problem x seed and compares
the full deterministic output (final selection + all score checkpoints) against committed
expected data, with exact equality. Any drift is a regression, unless it comes from an
intentional numeric change, in which case the expected data must be regenerated in the same
change (see below) so the drift is explicit.

What it guards is the *search*: the RNG, the scoring, the swap sequence, the tie-breaks. It
deliberately does not guard distance computation, and is built so it cannot. Each case is
solved from a precomputed distance matrix, so no pair kernel runs during the solve, and that
matrix is built here with plain numpy rather than through the library's own kernels. Distance
computation is guarded separately, by the cross-backend agreement tests.

That separation is what keeps this guard meaningful. A compiler is free to reassociate the
sums inside a distance kernel, and reassociation is target- and toolchain-dependent — so a
bit-exact pin over computed distances would go red on a new numba release or a different CPU
for reasons that are not regressions, and the habit of regenerating it to restore a green
build would leave it guarding nothing.

JIT-compiled and NUMBA_DISABLE_JIT execution produce identical selections but slightly different
score floats (float32 register arithmetic vs numpy's float64 scalar promotion), while each
regime is bit-stable across runs. The expected data is therefore committed per regime, and the
test asserts against the dataset matching the active regime:

- 'nojit' (interpreted) output is environment-independent (verified across platforms and
  Python versions), so it is asserted unconditionally — including on the coverage legs.
- 'jit' output depends on numba's codegen, which varies with Python and numba version (numba
  compiles from Python bytecode; e.g. Python 3.11 produces different float rounding than
  3.12+ with the same numba). The jit dataset therefore records the fingerprint of the
  environment it was generated in, and the test skips when the runtime doesn't match —
  a version-driven codegen change is numba's business, not a regression of this codebase.

To regenerate the expected data (both regimes) after an intentional numeric change:

    uv run --all-extras --python 3.13 python -m tests._core.solver.test_golden_master
"""

import json
import os
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

# =================================================================================================
#  Matrix & expected data
# =================================================================================================
PROBLEMS = ["U1", "U2", "U3", "U4", "C1", "C2", "C3", "C4"]
PRESETS = [SolverPreset.RANDOM, SolverPreset.GUIDED, SolverPreset.SMART, SolverPreset.THOROUGH]
SEEDS = [42, 123]

# small problems (PROBLEM_N) + tiny iteration budget: full matrix must stay
# fast with NUMBA_DISABLE_JIT, where these solves run interpreted
N_ITERATIONS = 30
PROBLEM_N = 100


def _active_regime() -> str:
    return "nojit" if numba_config.DISABLE_JIT else "jit"


def _runtime_fingerprint() -> dict[str, str]:
    """Identify the properties of the runtime that jit-compiled numeric output depends on."""
    import numba

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numba": numba.__version__,
    }


def _data_file(regime: str) -> Path:
    return Path(__file__).parent / f"golden_master_data_{regime}.json"


def _distances_from(vectors: NDArray[np.float32], metric: DistanceMetric) -> NDArray[np.float32]:
    """Build a distance matrix with plain numpy, independent of the library's own kernels.

    Computing it here rather than through the library is what makes this guard indifferent to
    changes in distance kernels: it pins the search given fixed distances. The result is
    quantized and symmetrized so the matrix is bit-identical on every machine — the same
    reasoning that quantizes the vectors below, applied one step later.
    """
    diff = vectors[:, None, :].astype(np.float64) - vectors[None, :, :].astype(np.float64)
    if metric == DistanceMetric.L1_MANHATTAN:
        raw = np.abs(diff).sum(axis=-1)
    elif metric == DistanceMetric.LINF_CHEBYSHEV:
        raw = np.abs(diff).max(axis=-1)
    elif metric == DistanceMetric.L2S_EUCLIDEAN_SQUARED:
        raw = (diff * diff).sum(axis=-1)
    else:  # euclidean, and cosine's pre-normalized rows reduce to it monotonically
        raw = np.sqrt((diff * diff).sum(axis=-1))
    matrix = np.round(raw, 4).astype(np.float32)
    matrix = np.minimum(matrix, matrix.T)  # exact symmetry, structurally rather than by tolerance
    np.fill_diagonal(matrix, np.float32(0.0))
    return np.ascontiguousarray(matrix)


def _solve(problem_name: str, preset: SolverPreset, seed: int) -> MaxDivSolution:
    generated = BenchmarkProblemFactory.construct_problem(
        problem_name, n=PROBLEM_N, diversity_metric=DiversityMetric.APPROX_GEOMEAN_SEPARATION
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
    regime = _active_regime()
    expected_data = json.loads(_data_file(regime).read_text())
    if regime == "jit" and expected_data["fingerprint"] != _runtime_fingerprint():
        pytest.skip(
            f"jit-compiled output is fingerprint-specific; data was generated with {expected_data['fingerprint']}"
        )
    expected = expected_data["cases"][_case_key(problem_name, preset, seed)]

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
    data = {"fingerprint": _runtime_fingerprint(), "cases": records}
    data_file.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {len(records)} cases to {data_file}")


if __name__ == "__main__":
    if "--active-regime-only" in sys.argv:
        regenerate_active_regime()
    else:
        # numba reads NUMBA_DISABLE_JIT at import time, so each regime runs in its own interpreter
        for disable_jit in ("0", "1"):
            env = {**os.environ, "NUMBA_DISABLE_JIT": disable_jit}
            subprocess.run([sys.executable, "-m", __spec__.name, "--active-regime-only"], env=env, check=True)  # noqa: S603 -- fixed args, script mode
