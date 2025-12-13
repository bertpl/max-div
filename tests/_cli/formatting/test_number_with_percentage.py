import pytest

from max_div._cli.formatting._content_types import NumberWithUncertainty


def test_number_with_percentage_aggregate_corner_cases():
    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        NumberWithUncertainty.aggregate([], method="mean")

    with pytest.raises(ValueError):
        NumberWithUncertainty.aggregate(
            [NumberWithUncertainty(value_q_25=1.0, value_q_50=2.0, value_q_75=3.0, decimals=2)],
            method="confobulated_mean",
        )
