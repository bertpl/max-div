import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "max_div.benchmark_problems",
        "max_div.metrics",
        "max_div.problem",
        "max_div.solution",
        "max_div.solver",
    ],
)
def test_api_module_patching(module_name: str) -> None:
    # sys.modules.pop(module_name, None)  # avoid cached import, leading to coverage miss this
    module = importlib.import_module(module_name)
    for name in module.__all__:
        obj = getattr(module, name)
        assert obj.__module__ == module_name, (
            f"{module_name}.{name}.__module__ == {obj.__module__!r}, expected {module_name!r}"
        )
