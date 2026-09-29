"""Compile a numba function with declared signatures on its first use, not when its module is imported.

`numba.njit(signature)` compiles as soon as the decorator runs, so every function declared that way
compiles when its module is imported. On an empty numba cache that makes importing the package take
tens of seconds, for functions that most programs never call. `lazy_njit(signature)` wraps the
function in a `LazyDispatcher`, which calls `numba.njit(signature)` on the function's first use and
delegates to the resulting numba dispatcher from then on. The declared signatures stay the only ones
compiled, so argument checks, casts and results are the same as with `numba.njit(signature)`.

The first use is whichever of these comes first:

- a call from Python;
- the compilation of a numba function that calls it: numba asks for the type of every global that
  the caller references, and the `typeof_impl` hook below builds the dispatcher to answer;
- reading a public attribute of the dispatcher, such as `py_func` or `signatures`.
"""

import functools
import threading
from collections.abc import Callable
from types import FunctionType, MethodType
from typing import Any, Self, TypeVar, cast

import numba
from numba import config
from numba.extending import typeof_impl

_F = TypeVar("_F", bound=FunctionType)


# =================================================================================================
#  LazyDispatcher
# =================================================================================================
class LazyDispatcher(functools.partial):
    """Stand-in for `numba.njit(signature, **options)(py_function)` that is built on first use.

    A `functools.partial`, because its call is implemented in C. Solver code calls compiled functions
    from Python dozens of times per iteration, and forwarding each call through Python code would
    add about 50 ns to every one of them. Until the first use, the partial calls `_build_and_call`;
    the build then points the partial at the numba dispatcher, so later calls reach the dispatcher
    with no Python code in between.
    """

    # Guards storing a built dispatcher, not the build. Holding a lock during the build could
    # deadlock: the build takes numba's compiler lock, and a thread compiling a caller already holds
    # numba's lock when it builds this dispatcher through `typeof_impl`. Two threads may therefore
    # build the same dispatcher at once; the results are equivalent, and the first one stored is
    # the one every later use gets.
    _store_lock = threading.Lock()

    _py_function: FunctionType
    _signature: object
    _options: dict[str, object]
    _dispatcher: Callable[..., Any] | None
    _build_arguments: tuple[tuple[object, ...], dict[str, object]]

    def __new__(cls, py_function: FunctionType, signature: object, options: dict[str, object]) -> Self:
        """Hold `py_function` with the arguments for `numba.njit`, without compiling anything."""
        self = super().__new__(cls, cls._build_and_call)
        functools.update_wrapper(self, py_function)
        self._py_function = py_function
        self._signature = signature
        self._options = options
        self._dispatcher = None
        # The partial's arguments until the first build. Kept in the instance dict so that repointing
        # the partial does not free them while another thread may still be calling through them.
        self._build_arguments = ((self,), {})
        self._point_partial_at(cls._build_and_call, *self._build_arguments)
        return self

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    @property
    def is_built(self) -> bool:
        """Whether the numba dispatcher has been built, i.e. its declared signatures compiled."""
        return self._dispatcher is not None

    def build(self) -> Callable[..., Any]:
        """Return the numba dispatcher, building it first if this is the first use.

        Building compiles the declared signatures, or loads them from numba's cache, and points the
        partial at the dispatcher.
        """
        if self._dispatcher is None:
            # numba's type stub has one overload per signature form; this passes whichever form it got
            dispatcher = numba.njit(self._signature, **self._options)(self._py_function)  # ty: ignore[no-matching-overload]
            with LazyDispatcher._store_lock:
                if self._dispatcher is None:
                    self._dispatcher = dispatcher
                    self._point_partial_at(dispatcher, (), {})
        return self._dispatcher

    # -------------------------------------------------------------------------
    #  Function behavior
    # -------------------------------------------------------------------------
    def __get__(self, instance: object, owner: type | None = None) -> Callable[..., Any]:
        # Bind as a method when stored on a class and read from an instance, like a function or a
        # numba dispatcher does. A partial binds that way only from Python 3.14 on.
        return self if instance is None else MethodType(self, instance)

    def __getattr__(self, name: str) -> object:
        # Called only for names not found on the instance. Private and dunder names are never
        # forwarded: probes such as copy's `__deepcopy__` must not trigger a compile.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.build(), name)

    def __reduce__(self) -> str:
        # Pickle by reference, like a plain function: unpickling looks the name up in its module.
        return self._py_function.__qualname__

    def __repr__(self) -> str:
        name = f"{self._py_function.__module__}.{self._py_function.__qualname__}"
        return f"<LazyDispatcher {name} (built: {self.is_built})>"

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def _point_partial_at(
        self, func: Callable[..., Any], args: tuple[object, ...], keywords: dict[str, object]
    ) -> None:
        """Make the partial call `func(*args, *call_args, **keywords, **call_kwargs)`, keeping the instance dict."""
        # `__setstate__` is how pickle restores a partial, and the only way to change the function of
        # an existing one; typeshed's partial stub does not declare it
        self.__setstate__((func, args, keywords, self.__dict__))  # ty: ignore[call-non-callable]

    @staticmethod
    def _build_and_call(lazy_dispatcher: "LazyDispatcher", *args: object, **kwargs: object) -> object:
        """Build the dispatcher of `lazy_dispatcher` and call it; the partial's function until the first build."""
        return lazy_dispatcher.build()(*args, **kwargs)


# =================================================================================================
#  Decorator
# =================================================================================================
_LAZY_DISPATCHERS: list[LazyDispatcher] = []


def lazy_njit(signature: object, **options: object) -> Callable[[_F], _F]:
    """Decorate a function like `numba.njit(signature, **options)`, but compile it on first use.

    Args:
        signature: The declared signature(s), in any form `numba.njit` accepts. Only these are ever
            compiled, and a call matching none of them raises `TypeError`, as with `numba.njit`.
        **options: Options passed on to `numba.njit` unchanged (`cache`, `fastmath`, `inline`, ...).

    Returns:
        A decorator that returns a `LazyDispatcher`, or the undecorated function when numba's JIT is
        disabled, as `numba.njit` does.
    """

    def decorator(py_function: _F) -> _F:
        if config.DISABLE_JIT:  # ty: ignore[unresolved-attribute] -- set at runtime from NUMBA_DISABLE_JIT
            return py_function
        lazy_dispatcher = LazyDispatcher(py_function, signature, options)
        _LAZY_DISPATCHERS.append(lazy_dispatcher)
        return cast("_F", lazy_dispatcher)

    return decorator


def lazy_dispatchers() -> tuple[LazyDispatcher, ...]:
    """Return every `LazyDispatcher` created by `lazy_njit` so far, in creation order."""
    return tuple(_LAZY_DISPATCHERS)


# =================================================================================================
#  numba typing hook
# =================================================================================================
@typeof_impl.register(LazyDispatcher)
def _typeof_lazy_dispatcher(lazy_dispatcher: LazyDispatcher, context: object) -> numba.types.Type:
    """Type a `LazyDispatcher` global as the numba dispatcher it builds."""
    return typeof_impl(lazy_dispatcher.build(), context)
