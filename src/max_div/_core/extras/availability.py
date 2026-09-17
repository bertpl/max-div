"""This module answers whether an optional extra of this package is installed, from the package's own metadata.

`pyproject.toml` is the single source of what extras exist and what each one installs, and this module
reads that back from the installed distribution instead of restating any of it:

- `Provides-Extra` lists the extras that exist
- the ``extra == '<name>'`` markers on `Requires-Dist` list the distributions each one installs

So adding a package to an extra needs no change here, and renaming an extra cannot leave a guard that
tells users to install an extra name that no longer exists.

What is checked is whether those distributions are installed, not how they got there: a plain install
of this package with the distributions present through any other route reads as available. Versions
are not checked, on purpose: this answers "is it here", and checking versions would repeat the
version-constraint check the installer already did at install time.
"""

import re
from functools import cache
from importlib.metadata import PackageNotFoundError, distribution, metadata

_DISTRIBUTION_NAME = "max-div"

# `matplotlib>=3.9; extra == 'plot'` -> name `matplotlib`, extra `plot`. Version specifiers and other
# environment markers are dropped: only the distribution name and the extra it belongs to matter here.
_REQUIREMENT_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_EXTRA_MARKER_PATTERN = re.compile(r"""extra\s*==\s*['"]([^'"]+)['"]""")


class MissingExtraError(ImportError):
    """A part of the package was reached without the extra that makes it work."""


# ==================================================================================================
#  Main API
# ==================================================================================================
def is_extra_installed(extra: str) -> bool:
    """Return whether every distribution the extra `extra` lists is installed.

    Raises:
        ValueError: If the package declares no extra by that name.
    """
    return all(_is_installed(name) for name in _required_distributions(extra))


def missing_extra_message(extra: str) -> str:
    """Return the install instruction shown when `extra` is needed and absent."""
    return f'This feature requires the "{extra}" extra: pip install "{_DISTRIBUTION_NAME}[{extra}]"'


def require_extra(extra: str) -> None:
    """Raise `MissingExtraError` naming the install instruction unless the extra `extra` is installed.

    A precondition, not an error caught after the fact: a caller runs the guarded imports only once
    the extra is known to be there, so a genuine error inside them is reported as that error, not as
    a missing extra.

    Raises:
        MissingExtraError: If a distribution the extra lists is not installed.
        ValueError: If the package declares no extra by that name.
    """
    if not is_extra_installed(extra):
        raise MissingExtraError(missing_extra_message(extra))


@cache
def declared_extras() -> frozenset[str]:
    """Return every extra this package declares."""
    return frozenset(metadata(_DISTRIBUTION_NAME).get_all("Provides-Extra") or ())


# ==================================================================================================
#  Helpers
# ==================================================================================================
@cache
def _required_distributions(extra: str) -> frozenset[str]:
    """Return the distributions the extra `extra` installs, read from this package's own metadata."""
    if extra not in declared_extras():
        raise ValueError(f"{_DISTRIBUTION_NAME} declares no extra {extra!r}; declared: {sorted(declared_extras())}")
    requirements = metadata(_DISTRIBUTION_NAME).get_all("Requires-Dist") or []
    return frozenset(
        name.group(1)
        for requirement in requirements
        if (marker := _EXTRA_MARKER_PATTERN.search(requirement)) and marker.group(1) == extra
        if (name := _REQUIREMENT_NAME_PATTERN.match(requirement))
    )


@cache
def _is_installed(distribution_name: str) -> bool:
    """Return whether a distribution is present in the running environment."""
    try:
        distribution(distribution_name)
    except PackageNotFoundError:
        return False
    else:
        return True
