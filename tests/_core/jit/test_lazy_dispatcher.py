import pickle
from types import MethodType, SimpleNamespace

import numba
import numpy as np
import pytest

from max_div._core.jit.lazy_dispatcher import LazyDispatcher, lazy_njit

_needs_jit = pytest.mark.skipif(numba.config.DISABLE_JIT, reason="needs numba's JIT")


def _double(x: float) -> float:
    """Return twice `x`."""
    return 2.0 * x


def _new_lazy_double() -> LazyDispatcher:
    """Return a new, unbuilt `LazyDispatcher` around `_double` that numba does not cache on disk."""
    return LazyDispatcher(_double, "float64(float64)", {"cache": False})


# The `LazyDispatcher` replaces `_halve` under the same name, as `lazy_njit` would, so that pickling by
# reference finds the `LazyDispatcher` when pickle looks up `_halve` in this module.
def _halve(x: float) -> float:
    """Return half of `x`."""
    return 0.5 * x


_halve = LazyDispatcher(_halve, "float64(float64)", {"cache": False})


# =================================================================================================
#  Decorator
# =================================================================================================
@pytest.mark.parametrize("is_jit_disabled", [False, True])
def test_lazy_njit_wraps_the_function_unless_the_jit_is_disabled(
    monkeypatch: pytest.MonkeyPatch, is_jit_disabled: bool
) -> None:
    """With the JIT on, `lazy_njit` returns an unbuilt `LazyDispatcher` in `instances()`; with it off, the function."""
    # --- arrange ----------------------
    monkeypatch.setattr(numba.config, "DISABLE_JIT", is_jit_disabled)
    monkeypatch.setattr(LazyDispatcher, "_instances", [])

    # --- act --------------------------
    decorated = lazy_njit("float64(float64)", cache=False)(_double)

    # --- assert -----------------------
    if is_jit_disabled:
        assert decorated is _double
        assert LazyDispatcher.instances() == ()
    else:
        assert isinstance(decorated, LazyDispatcher)
        assert not decorated.is_built
        assert LazyDispatcher.instances() == (decorated,)


# =================================================================================================
#  LazyDispatcher
# =================================================================================================
def test_a_python_call_builds_the_dispatcher_once() -> None:
    """The first call builds the dispatcher and returns its result; later uses get the same dispatcher."""
    # --- arrange ----------------------
    lazy_double = _new_lazy_double()
    assert not lazy_double.is_built

    # --- act --------------------------
    result = lazy_double(1.5)

    # --- assert -----------------------
    assert result == 3.0
    assert lazy_double.is_built
    assert lazy_double.build() is lazy_double.build()


def test_calls_after_the_build_go_straight_to_the_dispatcher() -> None:
    """Once built, the partial calls the dispatcher itself, not `_build_and_call`."""
    # --- arrange ----------------------
    lazy_double = _new_lazy_double()

    # --- act --------------------------
    lazy_double(1.5)

    # --- assert -----------------------
    assert lazy_double.func is lazy_double.build()
    assert lazy_double.args == ()


def test_a_lazy_dispatcher_binds_as_a_method_when_read_from_an_instance() -> None:
    """Stored on a class, a `LazyDispatcher` binds to an instance like a function does."""
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
    assert not lazy_double.is_built


@_needs_jit
def test_an_njit_caller_builds_the_dispatcher_it_calls() -> None:
    """Compiling an njit function that calls a `LazyDispatcher` builds it, through numba's typing hook."""
    # --- arrange ----------------------
    lazy_double = _new_lazy_double()

    def add_one_to_double(x: float) -> float:
        """Return twice `x`, plus 1."""
        return lazy_double(x) + 1.0

    # --- act --------------------------
    result = numba.njit(add_one_to_double)(1.5)

    # --- assert -----------------------
    assert result == 4.0
    assert lazy_double.is_built


@_needs_jit
def test_a_call_matching_no_declared_signature_raises_type_error() -> None:
    """A call whose argument types match no declared signature raises `TypeError`, as with `numba.njit`."""

    # --- arrange ----------------------
    def sum_of(values: np.ndarray) -> float:
        """Return the sum of `values`."""
        return values.sum()

    lazy_sum = LazyDispatcher(sum_of, "float64(float64[::1])", {"cache": False})

    # --- act / assert -----------------
    with pytest.raises(TypeError, match="No matching definition"):
        lazy_sum(np.arange(3, dtype=np.int64))


def test_public_attributes_are_forwarded_to_the_dispatcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """A public attribute not found on the `LazyDispatcher` is read from its built numba dispatcher."""
    # --- arrange ----------------------
    lazy_double = _new_lazy_double()
    monkeypatch.setattr(lazy_double, "build", lambda: SimpleNamespace(signatures=["declared"]))

    # --- act / assert -----------------
    assert lazy_double.signatures == ["declared"]


def test_private_names_are_not_forwarded() -> None:
    """Reading a private or dunder name raises `AttributeError` without building the dispatcher."""
    # --- arrange ----------------------
    lazy_double = _new_lazy_double()

    # --- act / assert -----------------
    with pytest.raises(AttributeError):
        _ = lazy_double._not_an_attribute
    assert not lazy_double.is_built


def test_a_lazy_dispatcher_pickles_by_reference() -> None:
    """Unpickling returns the module's own `LazyDispatcher`, and neither step builds it."""
    # --- arrange ----------------------
    was_built = _halve.is_built  # tests/test_jit_compilation.py may already have built every instance

    # --- act --------------------------
    unpickled = pickle.loads(pickle.dumps(_halve))  # noqa: S301 -- round trip of an object this test created

    # --- assert -----------------------
    assert unpickled is _halve
    assert _halve.is_built == was_built


def test_repr_names_the_function_and_whether_it_is_built() -> None:
    """The repr names the wrapped function and whether it is built."""
    # --- arrange ----------------------
    lazy_double = _new_lazy_double()

    # --- act / assert -----------------
    assert repr(lazy_double) == f"<LazyDispatcher {__name__}._double (built: False)>"
