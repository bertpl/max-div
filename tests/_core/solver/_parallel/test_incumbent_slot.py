import multiprocessing

import numpy as np
import pytest

from max_div._core.solver._parallel import GroupIncumbentSlot


# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
@pytest.fixture
def slot() -> GroupIncumbentSlot:
    """Return a fresh, never-written slot."""
    return GroupIncumbentSlot(multiprocessing.get_context("spawn"), k=4, score_length=3)


def _selection(*indices: int) -> np.ndarray:
    """Build an int32 selection array from the given indices."""
    return np.array(indices, dtype=np.int32)


# =================================================================================================
#  Tests
# =================================================================================================
def test_first_exchange_publishes(slot: GroupIncumbentSlot):
    """The first visitor always publishes: a never-written slot holds nothing to compare against."""
    # --- arrange -----------------------------------------
    assert not slot.written

    # --- act ---------------------------------------------
    outcome = slot.exchange((1.0, 1.0, 0.5), _selection(0, 1, 2, 3))

    # --- assert ------------------------------------------
    assert outcome is None
    assert slot.written


def test_better_score_publishes(slot: GroupIncumbentSlot):
    """A strictly better score replaces the stored selection."""
    # --- arrange -----------------------------------------
    slot.exchange((1.0, 1.0, 0.5), _selection(0, 1, 2, 3))

    # --- act ---------------------------------------------
    outcome = slot.exchange((1.0, 1.0, 0.7), _selection(4, 5, 6, 7))

    # --- assert ------------------------------------------
    assert outcome is None
    # a third, worse visitor receives the newly stored selection
    np.testing.assert_array_equal(slot.exchange((1.0, 1.0, 0.6), _selection(0, 1, 2, 3)), [4, 5, 6, 7])


def test_worse_score_receives_the_stored_selection(slot: GroupIncumbentSlot):
    """A strictly worse visitor gets the stored selection back and stores nothing."""
    # --- arrange -----------------------------------------
    slot.exchange((1.0, 1.0, 0.5), _selection(3, 1, 0, 2))

    # --- act ---------------------------------------------
    outcome = slot.exchange((1.0, 0.9, 0.8), _selection(4, 5, 6, 7))  # earlier component dominates

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(outcome, [3, 1, 0, 2])  # stored order preserved
    assert outcome.dtype == np.int32
    # the worse visitor stored nothing: a later visitor still receives the original selection
    np.testing.assert_array_equal(slot.exchange((1.0, 0.9, 0.8), _selection(4, 5, 6, 7)), [3, 1, 0, 2])


def test_equal_score_neither_publishes_nor_returns(slot: GroupIncumbentSlot):
    """Equal scores leave the slot untouched: adoption is strictly-better only."""
    # --- arrange -----------------------------------------
    slot.exchange((1.0, 1.0, 0.5), _selection(0, 1, 2, 3))

    # --- act ---------------------------------------------
    outcome = slot.exchange((1.0, 1.0, 0.5), _selection(4, 5, 6, 7))

    # --- assert ------------------------------------------
    assert outcome is None
    # the equal visitor stored nothing: a worse visitor still receives the original selection
    np.testing.assert_array_equal(slot.exchange((1.0, 1.0, 0.4), _selection(4, 5, 6, 7)), [0, 1, 2, 3])


def test_partial_selection_round_trips(slot: GroupIncumbentSlot):
    """A selection smaller than k comes back at its own size, not padded to k."""
    # --- arrange -----------------------------------------
    slot.exchange((0.5, 1.0, 0.5), _selection(2, 3))

    # --- act ---------------------------------------------
    outcome = slot.exchange((0.4, 1.0, 0.5), _selection(1))

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(outcome, [2, 3])


def test_returned_selection_is_an_independent_copy(slot: GroupIncumbentSlot):
    """Mutating a returned selection must not corrupt what the slot stores."""
    # --- arrange -----------------------------------------
    slot.exchange((1.0, 1.0, 0.5), _selection(0, 1, 2, 3))

    # --- act ---------------------------------------------
    outcome = slot.exchange((1.0, 1.0, 0.4), _selection(4, 5, 6, 7))
    outcome[:] = -1

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(slot.exchange((1.0, 1.0, 0.4), _selection(4, 5, 6, 7)), [0, 1, 2, 3])
