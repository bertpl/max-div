"""These skip markers key on the same extras check that the runtime guard uses.

A test that needs an optional extra asks the question exactly the way the package asks it, so a test can
never skip in an environment where the feature works, or run in one where it does not.
"""

import pytest

from max_div._core.extras import is_extra_installed, missing_extra_message


def needs_extra(extra: str) -> pytest.MarkDecorator:
    """Skip unless the extra is installed."""
    return pytest.mark.skipif(not is_extra_installed(extra), reason=missing_extra_message(extra))


def without_extra(extra: str) -> pytest.MarkDecorator:
    """Skip unless the extra is absent, for tests of what the package does without it."""
    return pytest.mark.skipif(is_extra_installed(extra), reason=f"the {extra!r} extra is installed")


def skip_module_unless_extra(extra: str) -> None:
    """Skip the calling test module unless the extra is installed.

    For a module that imports the guarded package at module level: a marker acts only after the
    imports have run, so the skip has to come before them.
    """
    if not is_extra_installed(extra):
        pytest.skip(missing_extra_message(extra), allow_module_level=True)
