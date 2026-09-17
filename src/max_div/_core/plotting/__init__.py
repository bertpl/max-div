"""Figures drawn from solver results, behind the optional `plot` extra.

Matplotlib is imported at module level only inside this package, and this `__init__` is the one place
that turns a missing matplotlib into an error naming the extra: every module below runs after it, so no
other module needs a guard of its own. Nothing outside the package imports matplotlib, and nothing
outside it imports this package at module level, so `import max_div` never pulls matplotlib in.

Layout: `helpers` holds what any figure can use and nothing that belongs to one figure; each figure
gets a sibling subpackage of its own.
"""

try:
    import matplotlib  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Plotting needs matplotlib, which is an optional dependency: install it with `pip install 'max-div[plot]'`"
    ) from e
