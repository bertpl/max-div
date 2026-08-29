from max_div._core.metrics._distance._metric import _distance_metric
from tests._core.metrics._distance.conftest import NAMED_METRICS


def test_fixture_covers_every_metric_kind():
    """Every METRIC_KIND_* selector must appear in the fixture, or a new metric escapes the generic tests."""
    # --- arrange ----------------------
    all_kinds = {value for name, value in vars(_distance_metric).items() if name.startswith("METRIC_KIND_")}

    # --- act --------------------------
    covered_kinds = {metric.kind for metric in NAMED_METRICS}

    # --- assert -----------------------
    assert covered_kinds == all_kinds
