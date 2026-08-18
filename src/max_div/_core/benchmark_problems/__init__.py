"""This package implements a limited set of benchmark problems that serve multiple purposes.

Purposes:
- help guide development of the tool
- compare different solver (initialization/optimization) strategies, to guide the user
- help create documentation
- help provide performance comparisons with other available tools, by providing reference problems.
"""

from ._factory import BenchmarkProblemFactory
from ._problems import IMPORT_ME_FOR_BENCHMARK_PROBLEM_DISCOVERY
from ._registry import MIN_N, BenchmarkProblem
