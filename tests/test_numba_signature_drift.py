"""Check that numba callers pass the argument types of a `lazy_njit` function's declared signature.

numba copies a function declared with `inline="always"` into each numba function that calls it, and compiles the
copy for its caller's argument types, so numba never checks those types against the declared signature. A signature
can then name types that no caller passes, and nothing fails.

This check compiles every numba function of the package with inlining switched off, so that each caller makes a
real call, and numba matches the call's argument types to the callee's declared signatures.

The check records the argument types of every call to a `lazy_njit` function, and compares them with the callee's
declared signatures, using numba's own conversion rules:

- an exact match, a promotion (`float32` to `float64`) or a safe conversion (`int32` to `int64`, or a writable array
  to a read-only one) passes;
- an unsafe conversion (`int64` to `int32`, `float64` to `float32`), or no conversion at all, is a drift. numba
  accepts an unsafe conversion without notice, so a call that compiles can still be a drift.

An integer literal passes when its value fits the declared integer type.

Each test runs in a fresh interpreter with its own empty numba cache: the cache does not record the inline option,
so the functions compiled here without inlining must not mix with the inlined versions that the rest of the suite
loads.

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
    """A caller passing float64 to a function declared as `float32(float32)` with `is_inline_only=True` is reported."""
    # --- act --------------------------
    drifts = _find_drifts_in_fresh_interpreter("self-test", tmp_path)

    # --- assert -----------------------
    assert len(drifts) == 1
    assert "halve" in drifts[0]
    assert "float64" in drifts[0]


# ==================================================================================================
#  Helpers
# ==================================================================================================
def _find_drifts_in_fresh_interpreter(check_target: str, numba_cache_dir: Path) -> list[str]:
    """Run this module as a script on `check_target`, with its own numba cache, and return the drifts that it prints.

    Args:
        check_target: `"package"` checks every numba function of the package; any other value checks the caller
            that `_compile_a_caller_that_drifts` builds.
        numba_cache_dir: An empty directory for the numba cache of the script run.
    """
    env = {**os.environ, "NUMBA_CACHE_DIR": str(numba_cache_dir)}
    output = subprocess.run(  # noqa: S603 -- fixed args, script mode
        [sys.executable, "-m", __name__, check_target], env=env, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in output.splitlines() if line]


def _find_drifts(compile_functions: Callable[[], None]) -> list[str]:
    """Run `compile_functions` with inlining off, and describe each call whose argument types drift.

    This function replaces `LazilyCompiledFunction.__new__` and `Dispatcher.get_call_template` for the rest of the
    process and does not restore them, so call it only in an interpreter that compiles nothing else.
    """
    call_types_by_dispatcher: dict[Dispatcher, set[tuple[types.Type, ...]]] = {}

    # --- switch inlining off --------------------
    # numba reads the inline option while it compiles a caller, so inlining is switched off before anything
    # compiles, both for every function that exists now and for every function that is created later
    for lazily_compiled_function in LazilyCompiledFunction.instances():
        if lazily_compiled_function._options.get("inline") == "always":
            lazily_compiled_function._options = {**lazily_compiled_function._options, "inline": "never"}

    original_new = LazilyCompiledFunction.__new__

    def new_without_inlining(cls: type, *args: object, **kwargs: object) -> LazilyCompiledFunction:
        """Create a `LazilyCompiledFunction` as usual, with inlining switched off."""
        lazily_compiled_function = original_new(cls, *args, **kwargs)
        if lazily_compiled_function._options.get("inline") == "always":
            lazily_compiled_function._options = {**lazily_compiled_function._options, "inline": "never"}
        return lazily_compiled_function

    LazilyCompiledFunction.__new__ = new_without_inlining

    # --- record the argument types of each call -
    original_get_call_template = Dispatcher.get_call_template

    def get_call_template_with_recording(dispatcher: Dispatcher, args: tuple, kws: dict) -> object:
        """Record the argument types of a call to `dispatcher`, then type the call as numba does."""
        # a keyword argument goes to the position of its parameter, as numba binds it
        parameter_names = list(inspect.signature(dispatcher.py_func).parameters)
        call_types = tuple(args) + tuple(kws[name] for name in parameter_names[len(args) :] if name in kws)
        call_types_by_dispatcher.setdefault(dispatcher, set()).add(call_types)
        return original_get_call_template(dispatcher, args, kws)

    Dispatcher.get_call_template = get_call_template_with_recording

    compile_functions()

    # --- compare calls with signatures ----------
    lazy_njit_py_functions = {function.__wrapped__ for function in LazilyCompiledFunction.instances()}
    drifts = []
    for dispatcher, all_call_types in call_types_by_dispatcher.items():
        if dispatcher.py_func in lazy_njit_py_functions:
            declared_signatures = dispatcher.signatures
            for call_types in all_call_types:
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
    """Return numba's conversion from `call_type` to `declared_type`.

    An integer literal whose value fits the declared integer type counts as an exact match.
    """
    if isinstance(call_type, types.IntegerLiteral) and isinstance(declared_type, types.Integer):
        bounds = np.iinfo(str(declared_type))
        if bounds.min <= call_type.literal_value <= bounds.max:
            return Conversion.exact
    return typing_context.can_convert(call_type, declared_type)


def _compile_a_caller_that_drifts() -> None:
    """Compile a caller that passes float64 to a function declared as `float32(float32)` with `is_inline_only=True`."""

    @lazy_njit("float32(float32)", inline="always", is_inline_only=True, cache=False)
    def halve(x):
        """Return half of `x`; the declared signature takes only float32."""
        return x * np.float32(0.5)

    @lazy_njit("float32(float64)", cache=False)
    def call_halve(x):
        """Call `halve` with a float64, which its declared signature does not accept."""
        return halve(x)

    call_halve.compile()


if __name__ == "__main__":
    compile_functions = numba_precompile if sys.argv[1] == "package" else _compile_a_caller_that_drifts
    for drift in _find_drifts(compile_functions):
        print(drift)
