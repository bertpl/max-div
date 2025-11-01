"""
Implements a dummy version of Numba for environments where Numba is not available.
Main goal is that code that uses @numba.njit or numba.tuped.Dict, ... does not break and transparently falls back to standard Python.
"""

from ._helpers import dummy_decorator


# =================================================================================================
#  Dummy typed containers
# =================================================================================================
class NumbaTypedDict(dict):
    @staticmethod
    def empty(*args, **kwargs):
        return dict()


class NumbaTypedList(list):
    @staticmethod
    def empty_list(*args, **kwargs):
        return list()


# =================================================================================================
#  Main dummy package tree
# =================================================================================================
class NumbaTyped:
    # implements subset
    Dict = NumbaTypedDict
    List = NumbaTypedList


class NumbaTypes:
    # implements dummies for all members of numba.types.__all__
    int8 = object()
    int16 = object()
    int32 = object()
    int64 = object()
    uint8 = object()
    uint16 = object()
    uint32 = object()
    uint64 = object()
    intp = object()
    uintp = object()
    intc = object()
    uintc = object()
    ssize_t = object()
    size_t = object()
    boolean = object()
    float32 = object()
    float64 = object()
    complex64 = object()
    complex128 = object()
    bool_ = object()
    byte = object()
    char = object()
    uchar = object()
    short = object()
    ushort = object()
    int_ = object()
    uint = object()
    long_ = object()
    ulong = object()
    longlong = object()
    ulonglong = object()
    double = object()
    void = object()
    none = object()
    b1 = object()
    i1 = object()
    i2 = object()
    i4 = object()
    i8 = object()
    u1 = object()
    u2 = object()
    u4 = object()
    u8 = object()
    f4 = object()
    f8 = object()
    c8 = object()
    c16 = object()
    optional = object()
    ffi_forced_object = object()
    ffi = object()
    deferred_type = object()
    bool = object()


class Numba:
    __version__ = "0.0.0"
    jit = dummy_decorator
    njit = dummy_decorator
    typed = NumbaTyped
    types = NumbaTypes
