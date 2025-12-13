"""
Package containing the actual definitions of benchmark problems.
"""

# importing this will trigger import of all defined benchmark problems, also triggering execution
# of their decorators and hence their registration in the benchmark problem registry
IMPORT_ME_FOR_BENCHMARK_PROBLEM_DISCOVERY = object()

# import actual benchmark problems to register them
from ._problem_a1 import BenchmarkProblem_A1
from ._problem_a2 import BenchmarkProblem_A2
from ._problem_a3 import BenchmarkProblem_A3
from ._problem_a4 import BenchmarkProblem_A4
from ._problem_a5 import BenchmarkProblem_A5
