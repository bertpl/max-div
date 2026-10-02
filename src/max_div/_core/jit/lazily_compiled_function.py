"""Compile a numba function with declared signatures on its first use, not when its module is imported.

`numba.njit(signature)` compiles as soon as the decorator runs, so every function declared that way
compiles when its module is imported. On an empty numba cache, this compilation makes importing the
package take tens of seconds, for functions that most programs never call.

`lazy_njit(signature)` wraps the function in a `LazilyCompiledFunction`, which calls
`numba.njit(signature)` on the function's first use and delegates to the resulting compiled function
from then on. numba calls that compiled function a dispatcher, because it holds one compiled version
per signature and picks one per call. The declared signatures stay the only ones compiled, so
argument checks, casts and results are the same as with `numba.njit(signature)`.

The first use is whichever of these comes first:

- a call from Python;
- the compilation of a numba function that calls it: numba asks for the type of every global that
  the caller references, and the `typeof_impl` hook below compiles the function to answer that
  request;
- reading a public attribute of the compiled function, such as `py_func` or `signatures`.
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


# ==================================================================================================
#  LazilyCompiledFunction
# ==================================================================================================
class LazilyCompiledFunction(functools.partial):
    """A `LazilyCompiledFunction` stands in for `numba.njit(signature, **options)(py_function)`, compiled on first use.

    The class subclasses `functools.partial`, because a partial's call is implemented in C. Solver
    code calls compiled functions from Python dozens of times per iteration, and forwarding each call
    through Python code would slow every one of them.

    Until the first use, calling the partial runs `_compile_and_call`; `compile` then replaces the
    partial's function with the compiled function, so later calls reach the compiled function with no
    Python code in between.
    """

    # The lock is held only while a compiled function is stored, not while it is compiled. Holding a
    # lock during compilation could deadlock:
    #
    # - thread A holds this lock and waits inside `numba.njit` for numba's compiler lock;
    # - thread B, compiling a numba function that calls this one, holds numba's compiler lock and
    #   waits inside `typeof_impl` for this lock.
    #
    # Therefore 2 threads may compile the same function at once; the results are equivalent, and
    # every later use gets the first one stored.
    _compiled_fun_assignment_lock = threading.Lock()

    # Every instance is recorded here, in creation order, so that tests can compile all of them.
    _instances: ClassVar[list["LazilyCompiledFunction"]] = []

    _py_function: FunctionType
    _signature: object
    _options: dict[str, object]
    _compiled_fun: Callable[..., Any] | None
    _compile_and_call_arguments: tuple[tuple[object, ...], dict[str, object]]

    def __new__(cls, py_function: FunctionType, signature: object, options: dict[str, object]) -> Self:
        """Hold `py_function` with the arguments for `numba.njit`, without compiling anything."""
        self = super().__new__(cls, cls._compile_and_call)
        functools.update_wrapper(self, py_function)
        self._py_function = py_function
        self._signature = signature
        self._options = options
        self._compiled_fun = None
        # `_compile_and_call_arguments` holds the partial's arguments until `compile` first runs. The
        # instance dict keeps a reference to them, so that when `compile` replaces the partial's
        # function and arguments, the old arguments are not freed while another thread is still inside
        # a call that uses them.
        self._compile_and_call_arguments = ((self,), {})
        self._point_partial_at(cls._compile_and_call, *self._compile_and_call_arguments)
        LazilyCompiledFunction._instances.append(self)
        return self

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    @classmethod
    def instances(cls) -> tuple["LazilyCompiledFunction", ...]:
        """Return every `LazilyCompiledFunction` created so far, in creation order."""
        return tuple(LazilyCompiledFunction._instances)

    @property
    def is_compiled(self) -> bool:
        """Return whether the declared signatures are compiled."""
        return self._compiled_fun is not None

    def compile(self) -> Callable[..., Any]:
        """Return the compiled function, compiling it first if this is the first use.

        Compiling compiles the declared signatures, or loads them from numba's cache.
        """
        if self._compiled_fun is None:
            # numba's type stub declares one overload per form of signature, and `self._signature` may
            # hold any of those forms, so no single overload matches
            compiled_fun = numba.njit(self._signature, **self._options)(self._py_function)  # ty: ignore[no-matching-overload]
            with LazilyCompiledFunction._compiled_fun_assignment_lock:
                if self._compiled_fun is None:
                    self._compiled_fun = compiled_fun
                    self._point_partial_at(compiled_fun, (), {})
        return self._compiled_fun

    # -------------------------------------------------------------------------
    #  Function behavior
    # -------------------------------------------------------------------------
    def __get__(self, instance: object, owner: type | None = None) -> Callable[..., Any]:
        """Bind as a method when stored on a class and read from an instance, as a compiled function does.

        A `functools.partial` binds that way only from Python 3.14 on.
        """
        return self if instance is None else MethodType(self, instance)

    def __getattr__(self, name: str) -> object:
        """Read a public attribute not found on the instance from the compiled function, compiling it first.

        Private and dunder names are never forwarded, because a lookup such as `copy.deepcopy` checking
        for `__deepcopy__` must not trigger a compile.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.compile(), name)

    def __reduce__(self) -> str:
        """Pickle by reference, like a plain function: unpickling looks the name up in its module.

        Pickling therefore works only when the `LazilyCompiledFunction` is bound at module level under
        the function's own name, as `lazy_njit` used as a decorator binds it.
        """
        return self._py_function.__qualname__

    def __repr__(self) -> str:
        """Return a repr naming the wrapped function and whether it is compiled."""
        name = f"{self._py_function.__module__}.{self._py_function.__qualname__}"
        return f"<LazilyCompiledFunction {name} (compiled: {self.is_compiled})>"

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
    def _compile_and_call(
        lazily_compiled_function: "LazilyCompiledFunction", *args: object, **kwargs: object
    ) -> object:
        """Compile `lazily_compiled_function` and call the compiled function."""
        return lazily_compiled_function.compile()(*args, **kwargs)


# ==================================================================================================
#  Decorator
# ==================================================================================================
def lazy_njit(signature: object, **options: object) -> Callable[[_F], _F]:
    """Decorate a function like `numba.njit(signature, **options)`, but compile it on first use.

    Args:
        signature: The declared signature(s), in any form accepted by `numba.njit`. Only these are ever
            compiled, and a call matching none of them raises `TypeError`, as with `numba.njit`.
        **options: Options passed on to `numba.njit` unchanged (`cache`, `fastmath`, `inline`, ...).

    Returns:
        A decorator that returns a `LazilyCompiledFunction`. When numba's JIT is disabled, the
        decorator returns the function unchanged, as `numba.njit` does.
    """

    def decorator(py_function: _F) -> _F:
        """Wrap `py_function` in a `LazilyCompiledFunction`, or return it unchanged when numba's JIT is disabled."""
        if config.DISABLE_JIT:  # ty: ignore[unresolved-attribute] -- set at runtime from NUMBA_DISABLE_JIT
            return py_function
        else:
            return cast("_F", LazilyCompiledFunction(py_function, signature, options))

    return decorator


# ==================================================================================================
#  numba typing hook
# ==================================================================================================
@typeof_impl.register(LazilyCompiledFunction)
def _typeof_lazily_compiled_function(
    lazily_compiled_function: LazilyCompiledFunction, context: object
) -> numba.types.Type:
    """Type a `LazilyCompiledFunction` global as its compiled function."""
    return typeof_impl(lazily_compiled_function.compile(), context)
