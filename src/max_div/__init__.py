from max_div.benchmark_problems import *  # noqa: F403
from max_div.benchmark_problems import __all__ as _all_benchmark_problems
from max_div.metrics import *  # noqa: F403
from max_div.metrics import __all__ as _all_metrics
from max_div.problem import *  # noqa: F403
from max_div.problem import __all__ as _all_problem
from max_div.solution import *  # noqa: F403
from max_div.solution import __all__ as _all_solution
from max_div.solver import *  # noqa: F403
from max_div.solver import __all__ as _all_solver

__all__ = sorted(  # noqa: PLE0605 — sorted() returns a list; only the literal form is statically recognized
    [
        *_all_benchmark_problems,
        *_all_metrics,
        *_all_problem,
        *_all_solution,
        *_all_solver,
    ]
)

del _all_benchmark_problems, _all_metrics, _all_problem, _all_solution, _all_solver
