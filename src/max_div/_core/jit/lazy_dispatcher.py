"""Compile a numba function with declared signatures on its first use, not when its module is imported.

`numba.njit(signature)` compiles as soon as the decorator runs, so every function declared that way
compiles when its module is imported. On an empty numba cache, this compilation makes importing the
package take tens of seconds, for functions that most programs never call.

`lazy_njit(signature)` wraps the function in a `LazyDispatcher`, which calls `numba.njit(signature)`
on the function's first use and delegates to the resulting numba dispatcher from then on. The
declared signatures stay the only ones compiled, so argument checks, casts and results are the same
as with `numba.njit(signature)`.

The first use is whichever of these comes first:

- a call from Python;
- the compilation of a numba function that calls it: numba asks for the type of every global that
  the caller references, and the `typeof_impl` hook below builds the dispatcher to answer that
  request;
- reading a public attribute of the dispatcher, such as `py_func` or `signatures`.
"""

import functools
import threading
from collections.abc import Callable
from types import FunctionType, MethodType
from typing import Any, ClassVar, Self, TypeVar, cast

import numba
from numba import config
from numba.extending import typeof_impl

_F = TypeVar("_F", bound=FunctionType)


# =================================================================================================
#  LazyDispatcher
# =================================================================================================
class LazyDispatcher(functools.partial):
    """A `LazyDispatcher` stands in for `numba.njit(signature, **options)(py_function)`, built on first use.

    The class subclasses `functools.partial`, because a partial's call is implemented in C. Solver
    code calls compiled functions from Python dozens of times per iteration, and forwarding each call
    through Python code would slow every one of them.

    Until the first use, calling the partial runs `_build_and_call`; `build` then replaces the
    partial's function with the numba dispatcher, so later calls reach the dispatcher with no Python
    code in between.
    """

    # The lock is held only while a built dispatcher is stored, not while it is built. Holding a lock
    # during compilation could deadlock:
    #
    # - thread A holds this lock and waits inside `numba.njit` for numba's compiler lock;
    # - thread B, compiling a numba function that calls this one, holds numba's compiler lock and
    #   waits inside `typeof_impl` for this lock.
    #
    # Therefore 2 threads may build the same dispatcher at once; the results are equivalent, and
    # every later use gets the first one stored.
    _dispatcher_assignment_lock = threading.Lock()

    # Every instance is recorded here, in creation order, so that tests can build all of them.
    _instances: ClassVar[list["LazyDispatcher"]] = []

    _py_function: FunctionType
    _signature: object
    _options: dict[str, object]
    _dispatcher: Callable[..., Any] | None
    _build_and_call_arguments: tuple[tuple[object, ...], dict[str, object]]

    def __new__(cls, py_function: FunctionType, signature: object, options: dict[str, object]) -> Self:
        """Hold `py_function` with the arguments for `numba.njit`, without compiling anything."""
        self = super().__new__(cls, cls._build_and_call)
        functools.update_wrapper(self, py_function)
        self._py_function = py_function
        self._signature = signature
        self._options = options
        self._dispatcher = None
        # `_build_and_call_arguments` holds the partial's arguments until `build` first runs. The instance
        # dict keeps a reference to them, so that when `build` replaces the partial's function and
        # arguments, the old arguments are not freed while another thread is still inside a call that
        # uses them.
        self._build_and_call_arguments = ((self,), {})
        self._point_partial_at(cls._build_and_call, *self._build_and_call_arguments)
        LazyDispatcher._instances.append(self)
        return self

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    @classmethod
    def instances(cls) -> tuple["LazyDispatcher", ...]:
        """Return every `LazyDispatcher` created so far, in creation order."""
        return tuple(LazyDispatcher._instances)

    @property
    def is_built(self) -> bool:
        """Return whether the numba dispatcher is built, i.e. whether its declared signatures are compiled."""
        return self._dispatcher is not None

    def build(self) -> Callable[..., Any]:
        """Return the numba dispatcher, building it first if this is the first use.

        Building compiles the declared signatures, or loads them from numba's cache.
        """
        if self._dispatcher is None:
            # numba's type stub declares one overload per form of signature, and `self._signature` may
            # hold any of those forms, so no single overload matches
            dispatcher = numba.njit(self._signature, **self._options)(self._py_function)  # ty: ignore[no-matching-overload]
            with LazyDispatcher._dispatcher_assignment_lock:
                if self._dispatcher is None:
                    self._dispatcher = dispatcher
                    self._point_partial_at(dispatcher, (), {})
        return self._dispatcher

    # -------------------------------------------------------------------------
    #  Function behavior
    # -------------------------------------------------------------------------
    def __get__(self, instance: object, owner: type | None = None) -> Callable[..., Any]:
        """Bind as a method when stored on a class and read from an instance, as a numba dispatcher does.

        A `functools.partial` binds that way only from Python 3.14 on.
        """
        return self if instance is None else MethodType(self, instance)

    def __getattr__(self, name: str) -> object:
        """Read a public attribute not found on the instance from the numba dispatcher, building it first.

        Private and dunder names are never forwarded, because a lookup such as `copy.deepcopy` checking
        for `__deepcopy__` must not trigger a compile.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.build(), name)

    def __reduce__(self) -> str:
        """Pickle by reference, like a plain function: unpickling looks the name up in its module.

        Pickling therefore works only when the `LazyDispatcher` is bound at module level under the
        function's own name, as `lazy_njit` used as a decorator binds it.
        """
        return self._py_function.__qualname__

    def __repr__(self) -> str:
        """Return a repr naming the wrapped function and whether it is built."""
        name = f"{self._py_function.__module__}.{self._py_function.__qualname__}"
        return f"<LazyDispatcher {name} (built: {self.is_built})>"

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def _point_partial_at(
        self, func: Callable[..., Any], args: tuple[object, ...], keywords: dict[str, object]
    ) -> None:
        """Set the partial's function and arguments, keeping the instance dict.

        Calling the partial as `self(*call_args, **call_kwargs)` then runs
        `func(*args, *call_args, **keywords, **call_kwargs)`.
        """
        # `__setstate__` is how pickle restores a partial, and the only way to change the function of
        # an existing one; typeshed's partial stub does not declare it
        self.__setstate__((func, args, keywords, self.__dict__))  # ty: ignore[call-non-callable]

    @staticmethod
    def _build_and_call(lazy_dispatcher: "LazyDispatcher", *args: object, **kwargs: object) -> object:
        """Build the dispatcher of `lazy_dispatcher` and call it."""
        return lazy_dispatcher.build()(*args, **kwargs)


# =================================================================================================
#  Decorator
# =================================================================================================
def lazy_njit(signature: object, **options: object) -> Callable[[_F], _F]:
    """Decorate a function like `numba.njit(signature, **options)`, but compile it on first use.

    Args:
        signature: The declared signature(s), in any form accepted by `numba.njit`. Only these are ever
            compiled, and a call matching none of them raises `TypeError`, as with `numba.njit`.
        **options: Options passed on to `numba.njit` unchanged (`cache`, `fastmath`, `inline`, ...).

    Returns:
        A decorator that returns a `LazyDispatcher`. When numba's JIT is disabled, the decorator
        returns the function unchanged, as `numba.njit` does.
    """

    def decorator(py_function: _F) -> _F:
        """Wrap `py_function` in a `LazyDispatcher`, or return it unchanged when numba's JIT is disabled."""
        if config.DISABLE_JIT:  # ty: ignore[unresolved-attribute] -- set at runtime from NUMBA_DISABLE_JIT
            return py_function
        else:
            return cast("_F", LazyDispatcher(py_function, signature, options))

    return decorator


# =================================================================================================
#  numba typing hook
# =================================================================================================
@typeof_impl.register(LazyDispatcher)
def _typeof_lazy_dispatcher(lazy_dispatcher: LazyDispatcher, context: object) -> numba.types.Type:
    """Type a `LazyDispatcher` global as its built numba dispatcher."""
    return typeof_impl(lazy_dispatcher.build(), context)
