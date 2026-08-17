import pytest

from max_div._core._utils import stdout_to_file


@pytest.mark.parametrize("enabled", [True, False])
def test_stdout_to_file(tmp_path, enabled):
    # --- arrange ----------------------
    capture_path = tmp_path / "stdout.txt"
    message = "capture me"

    # --- act --------------------------
    with stdout_to_file(enabled=enabled, filename=capture_path):
        print(message)

    # --- assert -----------------------
    file_exists = capture_path.exists()
    assert file_exists is enabled
    if enabled:
        assert capture_path.read_text().strip() == message


def test_stdout_to_file_no_filename():
    # --- act & assert -----------------
    with pytest.raises(ValueError), stdout_to_file(enabled=True, filename=None):
        pass
