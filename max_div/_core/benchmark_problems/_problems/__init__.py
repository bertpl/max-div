"""
Package containing the actual definitions of benchmark problems.
"""

# importing this will trigger import of all defined benchmark problems, also triggering execution
# of their decorators and hence their registration in the benchmark problem registry
IMPORT_ME_FOR_BENCHMARK_PROBLEM_DISCOVERY = object()

# import actual benchmark problems to register them
from ._problem_c1 import BenchmarkProblem_C1
from ._problem_c2 import BenchmarkProblem_C2
from ._problem_c3 import BenchmarkProblem_C3
from ._problem_c4 import BenchmarkProblem_C4
from ._problem_u1 import BenchmarkProblem_U1
from ._problem_u2 import BenchmarkProblem_U2
from ._problem_u3 import BenchmarkProblem_U3
from ._problem_u4 import BenchmarkProblem_U4
