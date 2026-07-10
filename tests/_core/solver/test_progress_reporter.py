from collections.abc import Callable

import numpy as np
import pytest

from max_div._core.solver._progress_reporting import (
    ProgressReporter,
    SilentProgressReporter,
    TabularProgressReporter,
    TqdmProgressReporter,
)


@pytest.mark.parametrize(
    "factory_method, expected_class",
    [
        (ProgressReporter.silent, SilentProgressReporter),
        (ProgressReporter.tqdm, TqdmProgressReporter),
        (ProgressReporter.tabular, TabularProgressReporter),
    ],
)
def test_progress_reporter_factory_methods(factory_method: Callable, expected_class: type[ProgressReporter]):
    # --- act ---------------------------------------------
    reporter = factory_method()

    # --- assert ------------------------------------------
    assert isinstance(reporter, expected_class)


@pytest.mark.parametrize(
    "selection, n",
    [
        (np.array([1, 2, 100, 516], dtype=np.int32), 20),
        (np.array([1, 2, 100, 516, 700], dtype=np.int32), 30),
        (np.arange(1000, dtype=np.int32), 40),
    ],
)
def test_progress_reporter_selection_hash(selection: np.ndarray, n: int):
    # --- arrange -----------------------------------------
    selection_modified = selection.copy()
    selection_modified[-1] += 1  # modify selection to ensure hash changes

    # --- act ---------------------------------------------
    hash_str_1 = TabularProgressReporter._get_selection_hash(selection, n)
    hash_str_2 = TabularProgressReporter._get_selection_hash(selection_modified, n)

    # --- assert ------------------------------------------
    assert len(hash_str_1) == n
    assert all([char in "0123456789abcdef" for char in hash_str_1])

    assert len(hash_str_2) == n
    assert all([char in "0123456789abcdef" for char in hash_str_2])

    assert hash_str_1 != hash_str_2
    assert hash_str_1[:8] != hash_str_2[:8]  # even first part should be different, if just the last input digit changed
