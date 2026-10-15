"""Check that numba callers pass the types that a `lazy_njit` function's declared signature names.

numba copies a function declared with `inline="always"` into each numba function that calls it, and compiles the
copy for its caller's argument types, so it never checks those types against the declared signature. A signature
can then name types that no caller passes, and nothing fails.

This check compiles every numba function of the package with inlining switched off, so that each caller makes a
real call, which numba types against the callee's declared signatures. It records the argument types of every call
to a `lazy_njit` function, and ranks them against the callee's declared signatures with numba's own conversion
rules:

- an exact match, a promotion (`float32` to `float64`) or a safe conversion (`int32` to `int64`, or a writable array
  to a read-only one) passes;
- an unsafe conversion (`int64` to `int32`, `float64` to `float32`), or no conversion at all, is a drift. numba
  accepts an unsafe conversion without notice, so a call that compiles can still be a drift.

An integer literal passes when its value fits the declared integer type.

Each check runs in a fresh interpreter with its own empty numba cache: the cache does not record the inline option,
so the versions compiled here must not mix with the inlined ones that the rest of the suite loads.

The check compiles the whole package on an empty cache, so it runs only when the environment variable
MAX_DIV_SIGNATURE_DRIFT_CHECK is 1, as `make test-signature-drift` and its CI job set it.
"""

import inspect
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import numba
import numpy as np
import pytest
from numba.core import types
from numba.core.dispatcher import Dispatcher
from numba.core.registry import cpu_target
from numba.core.typeconv import Conversion

from max_div._core.jit import LazilyCompiledFunction, lazy_njit
from max_div._core.numba_precompile import numba_precompile

IS_SIGNATURE_DRIFT_CHECK_REQUESTED = os.environ.get("MAX_DIV_SIGNATURE_DRIFT_CHECK") == "1"

pytestmark = [
    pytest.mark.skipif(numba.config.DISABLE_JIT, reason="with the JIT disabled nothing compiles"),
    pytest.mark.skipif(
        not IS_SIGNATURE_DRIFT_CHECK_REQUESTED,
        reason="the signature drift check runs only with MAX_DIV_SIGNATURE_DRIFT_CHECK=1",
    ),
]


# ==================================================================================================
#  Tests
# ==================================================================================================
def test_every_numba_caller_passes_types_that_the_declared_signature_accepts(tmp_path: Path) -> None:
    """No numba function of the package calls a `lazy_njit` function with types that drift from its signature."""
    # --- act --------------------------
    drifts = _find_drifts_in_fresh_interpreter("package", tmp_path)

    # --- assert -----------------------
    assert drifts == []


def test_the_check_reports_a_caller_that_passes_a_drifting_type(tmp_path: Path) -> None:
    """A caller that passes float64 to an inline-only function declared for float32 is reported."""
    # --- act --------------------------
    drifts = _find_drifts_in_fresh_interpreter("self-test", tmp_path)

    # --- assert -----------------------
    assert len(drifts) == 1
    assert "halve" in drifts[0]
    assert "float64" in drifts[0]


# ==================================================================================================
#  Helpers
# ==================================================================================================
def _find_drifts_in_fresh_interpreter(target: str, numba_cache_dir: Path) -> list[str]:
    """Run this module as a script on `target`, with its own numba cache, and return the drifts that it prints."""
    env = {**os.environ, "NUMBA_CACHE_DIR": str(numba_cache_dir)}
    output = subprocess.run(  # noqa: S603 -- fixed args, script mode
        [sys.executable, "-m", __name__, target], env=env, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in output.splitlines() if line]


def _find_drifts(compile_functions: Callable[[], None]) -> list[str]:
    """Compile with inlining off, record the argument types of each call, and describe each call that drifts."""
    call_types_by_callee: dict[Dispatcher, set[tuple[types.Type, ...]]] = {}

    # --- switch inlining off --------------------
    compile_inlined = LazilyCompiledFunction.compile

    def compile_without_inlining(lazily_compiled_function: LazilyCompiledFunction) -> Callable[..., object]:
        if lazily_compiled_function._options.get("inline") == "always":
            lazily_compiled_function._options = {**lazily_compiled_function._options, "inline": "never"}
        return compile_inlined(lazily_compiled_function)

    LazilyCompiledFunction.compile = compile_without_inlining

    # --- record the argument types of each call --
    get_call_template = Dispatcher.get_call_template

    def recording_get_call_template(dispatcher: Dispatcher, args: tuple, kws: dict) -> object:
        # a keyword argument goes to the position of its parameter, as numba binds it
        parameter_names = list(inspect.signature(dispatcher.py_func).parameters)
        call_types = tuple(args) + tuple(kws[name] for name in parameter_names[len(args) :] if name in kws)
        call_types_by_callee.setdefault(dispatcher, set()).add(call_types)
        return get_call_template(dispatcher, args, kws)

    Dispatcher.get_call_template = recording_get_call_template

    compile_functions()

    # --- rank the calls of lazy_njit functions --
    lazy_njit_py_functions = {function.__wrapped__ for function in LazilyCompiledFunction.instances()}
    drifts = []
    for dispatcher, calls in call_types_by_callee.items():
        if dispatcher.py_func in lazy_njit_py_functions:
            declared_signatures = [tuple(signature) for signature in dispatcher.signatures]
            for call_types in calls:
                if not _is_accepted(call_types, declared_signatures):
                    name = f"{dispatcher.py_func.__module__}.{dispatcher.py_func.__qualname__}"
                    drifts.append(f"{name}: called with {call_types}, declared {declared_signatures}")
    return sorted(drifts)


def _is_accepted(call_types: tuple[types.Type, ...], declared_signatures: Iterable[tuple[types.Type, ...]]) -> bool:
    """Return whether some declared signature accepts the call types with at most safe conversions."""
    typing_context = cpu_target.typing_context
    for declared_types in declared_signatures:
        if len(declared_types) == len(call_types):
            conversions = [
                _conversion(typing_context, call_type, declared_type)
                for call_type, declared_type in zip(call_types, declared_types)
            ]
            if all(conversion is not None and conversion <= Conversion.safe for conversion in conversions):
                return True
    return False


def _conversion(typing_context: object, call_type: types.Type, declared_type: types.Type) -> Conversion | None:
    """Return numba's conversion from `call_type` to `declared_type`; an integer literal that fits is exact."""
    if isinstance(call_type, types.IntegerLiteral) and isinstance(declared_type, types.Integer):
        bounds = np.iinfo(str(declared_type))
        if bounds.min <= call_type.literal_value <= bounds.max:
            return Conversion.exact
    return typing_context.can_convert(call_type, declared_type)


def _compile_a_caller_that_drifts() -> None:
    """Compile a caller that passes float64 to an inline-only function whose signature declares float32."""

    @lazy_njit("float32(float32)", inline="always", is_inline_only=True, cache=False)
    def halve(x):
        return x * np.float32(0.5)

    @lazy_njit("float32(float64)", cache=False)
    def call_halve(x):
        return halve(x)

    call_halve.compile()


if __name__ == "__main__":
    compile_functions = numba_precompile if sys.argv[1] == "package" else _compile_a_caller_that_drifts
    for drift in _find_drifts(compile_functions):
        print(drift)
