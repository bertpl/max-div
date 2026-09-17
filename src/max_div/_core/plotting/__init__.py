"""This package implements the plotting features, behind the optional `plot` extra.

Matplotlib is imported at module level only inside this package, and this `__init__` is the one place
that checks for the extra: every submodule of this package runs after this `__init__`, so no other
module needs a guard of its own.

Nothing outside this package imports matplotlib, and no other module imports this package at module
level, so `import max_div` never pulls matplotlib in.
"""

from max_div._core.extras import require_extra

require_extra("plot")

import matplotlib  # noqa: E402
