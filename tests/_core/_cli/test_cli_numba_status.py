from click.testing import CliRunner

from max_div._core._cli import numba_status


# =================================================================================================
#  numba_status
# =================================================================================================
def test_cli_numba_status():
    # --- arrange ----------------------
    runner = CliRunner()

    # --- act --------------------------
    result = runner.invoke(numba_status)

    # --- assert -----------------------
    assert result.exit_code == 0
