import pickle
from types import MethodType, SimpleNamespace

import numba
import numpy as np
import pytest

from max_div._core.jit.lazily_compiled_function import LazilyCompiledFunction, lazy_njit

_needs_jit = pytest.mark.skipif(numba.config.DISABLE_JIT, reason="needs numba's JIT")


def _double(x: float) -> float:
    """Return twice `x`."""
    return 2.0 * x


def _new_lazy_double() -> LazilyCompiledFunction:
    """Return a new, uncompiled `LazilyCompiledFunction` around `_double` that numba does not cache on disk."""
    return LazilyCompiledFunction(_double, "float64(float64)", {"cache": False})


# The `LazilyCompiledFunction` replaces `_halve` under the same name, as `lazy_njit` would, so that pickling by
# reference finds the `LazilyCompiledFunction` when pickle looks up `_halve` in this module.
def _halve(x: float) -> float:
    """Return half of `x`."""
    return 0.5 * x


_halve = LazilyCompiledFunction(_halve, "float64(float64)", {"cache": False})


# ==================================================================================================
#  Decorator
# ==================================================================================================
@pytest.mark.parametrize("is_jit_disabled", [False, True])
def test_lazy_njit_wraps_the_function_unless_the_jit_is_disabled(
    monkeypatch: pytest.MonkeyPatch, is_jit_disabled: bool
) -> None:
    """With the JIT on, `lazy_njit` returns an uncompiled, recorded `LazilyCompiledFunction`; off, the function."""
    # --- arrange ----------------------
    monkeypatch.setattr(numba.config, "DISABLE_JIT", is_jit_disabled)
    monkeypatch.setattr(LazilyCompiledFunction, "_instances", [])

    # --- act --------------------------
    decorated = lazy_njit("float64(float64)", cache=False)(_double)

    # --- assert -----------------------
    if is_jit_disabled:
        assert decorated is _double
        assert LazilyCompiledFunction.instances() == ()
    else:
        assert isinstance(decorated, LazilyCompiledFunction)
        assert not decorated.is_compiled
        assert LazilyCompiledFunction.instances() == (decorated,)


# ==================================================================================================
#  LazilyCompiledFunction
# ==================================================================================================
def test_a_python_call_compiles_the_function_once() -> None:
    """The first call compiles the function and returns its result; later uses get the same compiled function."""
    # --- arrange ----------------------
    lazy_double = _new_lazy_double()
    assert not lazy_double.is_compiled

    # --- act --------------------------
    result = lazy_double(1.5)

    # --- assert -----------------------
    assert result == 3.0
    assert lazy_double.is_compiled
    assert lazy_double.compile() is lazy_double.compile()


def test_calls_after_compiling_go_straight_to_the_compiled_function() -> None:
    """Once compiled, the partial calls the compiled function itself, not `_compile_and_call`."""
    # --- arrange ----------------------
    lazy_double = _new_lazy_double()

    # --- act --------------------------
    lazy_double(1.5)

    # --- assert -----------------------
    assert lazy_double.func is lazy_double.compile()
    assert lazy_double.args == ()


def test_a_lazily_compiled_function_binds_as_a_method_when_read_from_an_instance() -> None:
    """Stored on a class, a `LazilyCompiledFunction` binds to an instance like a function does."""
    # --- arrange ----------------------
    lazy_double = _new_lazy_double()

    class Holder:
        method = lazy_double

    holder = Holder()

    # --- act --------------------------
    bound = holder.method

    # --- assert -----------------------
    assert Holder.method is lazy_double
    assert isinstance(bound, MethodType)
    assert bound.__self__ is holder
    assert bound.__func__ is lazy_double
    assert not lazy_double.is_compiled


@_needs_jit
def test_compiling_an_njit_caller_compiles_the_function_it_calls() -> None:
    """Compiling an njit function that calls a `LazilyCompiledFunction` compiles it too, through numba's typing hook."""
    # --- arrange ----------------------
    lazy_double = _new_lazy_double()

    def add_one_to_double(x: float) -> float:
        """Return twice `x`, plus 1."""
        return lazy_double(x) + 1.0

    # --- act --------------------------
    result = numba.njit(add_one_to_double)(1.5)

    # --- assert -----------------------
    assert result == 4.0
    assert lazy_double.is_compiled


@_needs_jit
def test_a_call_matching_no_declared_signature_raises_type_error() -> None:
    """A call whose argument types match no declared signature raises `TypeError`, as with `numba.njit`."""

    # --- arrange ----------------------
    def sum_of(values: np.ndarray) -> float:
        """Return the sum of `values`."""
        return values.sum()

    lazy_sum = LazilyCompiledFunction(sum_of, "float64(float64[::1])", {"cache": False})

    # --- act / assert -----------------
    with pytest.raises(TypeError, match="No matching definition"):
        lazy_sum(np.arange(3, dtype=np.int64))


def test_public_attributes_are_forwarded_to_the_compiled_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """A public attribute not found on the `LazilyCompiledFunction` is read from its compiled function."""
    # --- arrange ----------------------
    lazy_double = _new_lazy_double()
    monkeypatch.setattr(lazy_double, "compile", lambda: SimpleNamespace(signatures=["declared"]))

    # --- act / assert -----------------
    assert lazy_double.signatures == ["declared"]


def test_private_names_are_not_forwarded() -> None:
    """Reading a private or dunder name raises `AttributeError` without compiling the function."""
    # --- arrange ----------------------
    lazy_double = _new_lazy_double()

    # --- act / assert -----------------
    with pytest.raises(AttributeError):
        _ = lazy_double._not_an_attribute
    assert not lazy_double.is_compiled


def test_a_lazily_compiled_function_pickles_by_reference() -> None:
    """Unpickling returns the module's own `LazilyCompiledFunction`, and neither step compiles it."""
    # --- arrange ----------------------
    was_compiled = _halve.is_compiled  # tests/test_jit_compilation.py may already have compiled every instance

    # --- act --------------------------
    unpickled = pickle.loads(pickle.dumps(_halve))  # noqa: S301 -- round trip of an object this test created

    # --- assert -----------------------
    assert unpickled is _halve
    assert _halve.is_compiled == was_compiled


def test_repr_names_the_function_and_whether_it_is_compiled() -> None:
    """The repr names the wrapped function and whether it is compiled."""
    # --- arrange ----------------------
    lazy_double = _new_lazy_double()

    # --- act / assert -----------------
    assert repr(lazy_double) == f"<LazilyCompiledFunction {__name__}._double (compiled: False)>"
