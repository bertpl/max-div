import numpy as np
import pytest

from max_div._core._utils import delete_sorted, insert_sorted, move_within

CAPACITY_SLACK = 4


def _buffer(values: list[int], slack: int = CAPACITY_SLACK) -> np.ndarray:
    """Return a buffer holding `values` in its prefix, padded with a sentinel."""
    buffer = np.full(len(values) + slack, -1, dtype=np.int32)
    buffer[: len(values)] = values
    return buffer


# =================================================================================================
#  move_within
# =================================================================================================
@pytest.mark.parametrize(
    "dest_offset, src_offset, count, expected",
    [
        (1, 0, 3, [10, 10, 20, 30, -1]),  # shift right, source and destination overlap
        (0, 1, 3, [20, 30, 40, 40, -1]),  # shift left, source and destination overlap
        (0, 0, 4, [10, 20, 30, 40, -1]),  # same range, a no-op
        (0, 0, 0, [10, 20, 30, 40, -1]),  # empty move
    ],
)
def test_move_within(dest_offset: int, src_offset: int, count: int, expected: list[int]):
    # --- arrange ----------------------
    buffer = _buffer([10, 20, 30, 40], slack=1)

    # --- act --------------------------
    move_within(buffer, dest_offset, src_offset, count)

    # --- assert -----------------------
    assert list(buffer) == expected


# =================================================================================================
#  insert_sorted
# =================================================================================================
@pytest.mark.parametrize(
    "values, value, expected",
    [
        ([], 5, [5]),  # into an empty list
        ([10], 5, [5, 10]),  # before the only entry
        ([10], 15, [10, 15]),  # after the only entry
        ([10, 20, 30], 5, [5, 10, 20, 30]),  # at the front
        ([10, 20, 30], 25, [10, 20, 25, 30]),  # in the middle
        ([10, 20, 30], 40, [10, 20, 30, 40]),  # at the end
    ],
)
def test_insert_sorted(values: list[int], value: int, expected: list[int]):
    # --- arrange ----------------------
    buffer = _buffer(values)

    # --- act --------------------------
    insert_sorted(buffer, np.int32(len(values)), np.int32(value))

    # --- assert -----------------------
    assert list(buffer[: len(expected)]) == expected


# =================================================================================================
#  delete_sorted
# =================================================================================================
@pytest.mark.parametrize(
    "values, value, expected",
    [
        ([10], 10, []),  # the only entry
        ([10, 20, 30], 10, [20, 30]),  # at the front
        ([10, 20, 30], 20, [10, 30]),  # in the middle
        ([10, 20, 30], 30, [10, 20]),  # at the end
    ],
)
def test_delete_sorted(values: list[int], value: int, expected: list[int]):
    # --- arrange ----------------------
    buffer = _buffer(values)

    # --- act --------------------------
    delete_sorted(buffer, np.int32(len(values)), np.int32(value))

    # --- assert -----------------------
    assert list(buffer[: len(expected)]) == expected


# =================================================================================================
#  round trip
# =================================================================================================
def test_insert_then_delete_restores_the_list():
    """Inserting a value and deleting it again leaves the list as it was, at every position."""
    # --- arrange ----------------------
    rng = np.random.default_rng(0)
    original = np.sort(rng.choice(1_000, 200, replace=False)).astype(np.int32)

    for value in rng.choice(np.setdiff1d(np.arange(1_000), original), 25, replace=False):
        buffer = _buffer(list(original))

        # --- act ----------------------
        insert_sorted(buffer, np.int32(len(original)), np.int32(value))
        after_insert = buffer[: len(original) + 1].copy()
        delete_sorted(buffer, np.int32(len(original) + 1), np.int32(value))

        # --- assert -------------------
        assert list(after_insert) == sorted([*original.tolist(), int(value)])
        assert list(buffer[: len(original)]) == original.tolist()


def test_the_list_stays_ascending_under_many_mixed_operations():
    """A long run of interleaved inserts and deletes keeps the list ascending and correct."""
    # --- arrange ----------------------
    rng = np.random.default_rng(1)
    capacity = 500
    buffer = np.full(capacity + 1, -1, dtype=np.int32)
    n_live = 0
    reference: set[int] = set()

    # --- act --------------------------
    for _ in range(2_000):
        if n_live and (n_live == capacity or rng.random() < 0.5):
            value = int(rng.choice(sorted(reference)))
            delete_sorted(buffer, np.int32(n_live), np.int32(value))
            reference.remove(value)
            n_live -= 1
        else:
            value = int(rng.integers(0, 10_000))
            if value in reference:
                continue
            insert_sorted(buffer, np.int32(n_live), np.int32(value))
            reference.add(value)
            n_live += 1

    # --- assert -----------------------
    assert list(buffer[:n_live]) == sorted(reference)


def test_delete_sorted_absent_value_past_the_end_is_a_safe_no_op():
    """A value absent and sorting past every live entry leaves the list untouched (see `delete_sorted`)."""
    # --- arrange ----------------------
    buffer = _buffer([10, 20, 30])

    # --- act --------------------------
    delete_sorted(buffer, np.int32(3), np.int32(999))

    # --- assert -----------------------
    assert list(buffer[:3]) == [10, 20, 30]
