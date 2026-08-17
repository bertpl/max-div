import numpy as np
import pytest
from numpy.typing import NDArray

from max_div._core.solver._diversity_contribution import DiversityContributionTracker


# =================================================================================================
#  Stub tracker
# =================================================================================================
class _CallRecordingTracker(DiversityContributionTracker):
    """Minimal concrete tracker that records mutation calls, for testing the base-class defaults."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def contribution_wrt_selection(self, selected: NDArray[np.bool], n_selected: np.int32) -> NDArray[np.float32]:
        return np.array([], dtype=np.float32)

    @property
    def contribution_wrt_dataset(self) -> NDArray[np.float32]:
        return np.array([], dtype=np.float32)

    def contribution_wrt_dataset_for(self, indices: NDArray[np.int32]) -> NDArray[np.float32]:
        return np.array([], dtype=np.float32)  # not exercised by these tests

    def add(self, index: np.int32) -> None:
        self.calls.append(("add", int(index)))

    def remove(self, index: np.int32, new_selection: NDArray[np.int32]) -> None:
        self.calls.append(("remove", int(index), list(new_selection)))

    def reset(self) -> None:
        pass  # not exercised by these tests

    def push_snapshot(self) -> None:
        pass  # not exercised by these tests

    def pop_snapshot(self, restore: bool) -> None:
        pass  # not exercised by these tests

    def copy(self) -> DiversityContributionTracker:
        return self  # not exercised by these tests


# =================================================================================================
#  Tests
# =================================================================================================
def test_tracker_abc_cannot_be_instantiated():
    # --- act / assert -----------------
    with pytest.raises(TypeError):
        DiversityContributionTracker()  # ty: ignore[missing-argument]  # deliberate: abstract class


def test_add_many_delegates_to_add_in_order():
    # --- arrange ----------------------
    tracker = _CallRecordingTracker()
    indices = np.array([4, 1, 7], dtype=np.int32)

    # --- act --------------------------
    tracker.add_many(indices)

    # --- assert -----------------------
    assert tracker.calls == [("add", 4), ("add", 1), ("add", 7)]


def test_remove_many_delegates_to_remove_in_order_with_same_selection():
    # --- arrange ----------------------
    tracker = _CallRecordingTracker()
    indices = np.array([2, 5], dtype=np.int32)
    new_selection = np.array([0, 9], dtype=np.int32)

    # --- act --------------------------
    tracker.remove_many(indices, new_selection)

    # --- assert -----------------------
    assert tracker.calls == [("remove", 2, [0, 9]), ("remove", 5, [0, 9])]
