import pytest

from max_div._core.metrics._distance._build._common import parallel_build_enabled


@pytest.mark.parametrize(
    "env_value, expected",
    [
        (None, True),
        ("1", True),
        ("0", False),
    ],
)
def test_parallel_build_enabled(monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected: bool):
    """Parallel builds are on by default and disabled only by MAXDIV_PARALLEL_BUILD=0."""
    # --- arrange ----------------------
    if env_value is None:
        monkeypatch.delenv("MAXDIV_PARALLEL_BUILD", raising=False)
    else:
        monkeypatch.setenv("MAXDIV_PARALLEL_BUILD", env_value)

    # --- act / assert -----------------
    assert parallel_build_enabled() is expected
