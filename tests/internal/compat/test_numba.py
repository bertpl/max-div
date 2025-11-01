from max_div.internal.compat import is_numba_installed, numba
from max_div.internal.compat._numba._dummy_numba import Numba


def test_is_numba_installed():
    # --- arrange -----------------------------------------
    try:
        import numba as real_numba

        is_numba_truly_installed = True
    except ImportError:
        is_numba_truly_installed = False

    # --- act ---------------------------------------------
    is_numba_deemed_installed = is_numba_installed()

    # --- assert ------------------------------------------
    assert is_numba_deemed_installed == is_numba_truly_installed
    if not is_numba_deemed_installed:
        assert not isinstance(numba, Numba), "Dummy numba instance not used when numba is installed."


def test_numba_decorator_always_works():
    # --- arrange -----------------------------------------
    @numba.jit
    def func_jit(x):
        return x + 1

    @numba.njit
    def func_njit(x):
        return x + 2

    @numba.njit(fastmath=True)
    def func_njit2(x):
        return x + 3

    # --- act ---------------------------------------------
    result_jit = func_jit(10)
    result_njit = func_njit(10)
    result_njit2 = func_njit2(10)

    # --- assert ------------------------------------------
    assert result_jit == 11, "numba.jit decorator did not work as expected."
    assert result_njit == 12, "numba.njit decorator did not work as expected."
    assert result_njit2 == 13, "numba.njit(...) decorator did not work as expected."


def test_dummy_numba_typed_dict():
    # --- arrange -----------------------------------------
    @numba.njit
    def try_typed_dict() -> float:
        typed_dict = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.float64,
        )
        typed_dict[1] = 2.71
        typed_dict[1] = 3.14
        return typed_dict[1]

    # --- act ---------------------------------------------
    result = try_typed_dict()

    # --- assert ------------------------------------------
    assert result == 3.14


def test_dummy_numba_typed_list():
    # --- arrange -----------------------------------------
    @numba.njit
    def try_typed_list() -> float:
        typed_list = numba.typed.List.empty_list(numba.types.float64)
        typed_list.append(2.71)
        typed_list.append(3.14)
        return typed_list[1]

    # --- act ---------------------------------------------
    result = try_typed_list()

    # --- assert ------------------------------------------
    assert result == 3.14


def test_dummy_numba_typed_list_conversion():
    # --- arrange -----------------------------------------
    lst_python = [1.1, 2.2, 3.3]

    # --- act ---------------------------------------------
    lst_numba = numba.typed.List(lst_python)

    # --- assert ------------------------------------------
    assert lst_numba[0] == 1.1
    assert lst_numba[1] == 2.2
    assert lst_numba[2] == 3.3
    assert len(lst_numba) == 3
