from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property

from tqdm import tqdm

from max_div._cli.formatting import (
    BoldLabels,
    FastestBenchmark,
    HighestNumberWithUncertainty,
    NumberWithUncertainty,
    extend_table_with_aggregate_row,
    format_table_as_markdown,
    format_table_for_console,
)
from max_div.benchmarks import BenchmarkProblemFactory
from max_div.internal.benchmarking import BenchmarkResult
from max_div.internal.utils import stdout_to_file
from max_div.solver import DiversityMetric, MaxDivProblem, MaxDivSolver


# =================================================================================================
#  Benchmark Executor
# =================================================================================================
class SolverBenchmarkExecutor:
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, scope: SolverBenchmarkScope):
        self._scope = scope

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def execute(self, markdown: bool, file: bool = False):
        # --- run benchmarks ------------------------------
        with self._scope as scope:
            for size, strat_name, seed in scope.params():
                # --- construct solver ---
                solver = scope.construct_solver(size, strat_name, seed)

                # --- run solver  ---
                solution = solver.solve(verbosity=0)

                # --- get results ---
                t_elapsed_sec = list(solution.step_durations.values())[-1].t_elapsed_sec
                diversity_score = solution.score.diversity
                constraint_score = solution.score.constraints

                # --- register results ---
                scope.register_result(
                    size=size,
                    strat_name=strat_name,
                    t_elapsed_sec=t_elapsed_sec,
                    diversity_score=diversity_score,
                    constraint_score=constraint_score,
                )

        # --- show results --------------------------------
        scope.show_results_tables(markdown, file)


# =================================================================================================
#  Benchmark Scope
# =================================================================================================
class SolverBenchmarkScope:
    """
    Base class for Scope of benchmarks to run for a solver benchmark, limited to a specific test problem.

    A scope spans all (size, seed, strat_name)-tuples for one test problem.

    The SolverBenchmarkExecutor can use this info to construct a pre-configured Solver for said problem with given size,
    such that it can be benchmarked.  Such class will typically focus on testing...
      - initialization strategies
      - optimization strategies
      - specific solver presets.
    """

    # -------------------------------------------------------------------------
    #  Constructor / Configuration
    # -------------------------------------------------------------------------
    def __init__(self, solver_constructor: BenchmarkSolverConstructor, speed: float, leave_pbar: bool):
        # arguments influencing scope
        self._solver_constructor = solver_constructor
        self._constraints = solver_constructor.has_constraints
        self._speed = speed
        self._leave_pbar = leave_pbar  # leave progress bar after completion

        # data structures to keep track of results
        self._t_elapsed: dict[tuple[int, str], list[float]] = defaultdict(list)
        self._diversity_scores: dict[tuple[int, str], list[float]] = defaultdict(list)
        self._constraint_scores: dict[tuple[int, str], list[float]] = defaultdict(list)

        # context mgr state
        self._context_active: bool = False
        self._pbar: tqdm | None = None

    @property
    def benchmark_type(self) -> str:
        return self._solver_constructor.benchmark_type

    @property
    def problem_name(self) -> str:
        return self._solver_constructor.problem_name

    # -------------------------------------------------------------------------
    #  Context Manager
    # -------------------------------------------------------------------------
    def __enter__(self) -> SolverBenchmarkScope:
        self._pbar = tqdm(
            total=len(self.params()),
            desc=f"Problem {self.problem_name} - {self.benchmark_type.capitalize()}".ljust(40),
            leave=self._leave_pbar,
        )
        self._context_active = True

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._context_active = False
        if not self._leave_pbar:
            self._pbar.close()
        self._pbar = None

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def params(self) -> list[tuple[int, str, int]]:
        """Returns list of (size, strat_name, seed)-tuples to benchmark."""

        # --- calibrate -------------------------
        n_seeds_min = 3  # we don't execute benchmarks if n_seeds < n_seeds_min
        n_seeds_max = 16  # we never do more than n_seeds_max
        full_size_range = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]  # size range for speed=0.0
        nominal_max_size = max(full_size_range)  # max_size for speed=0.0

        # --- speed-dependent settings ----------
        speed = self._speed
        n_seeds_min = round(n_seeds_min ** (1 - speed))  # reduce to 1 at speed=1.0
        n_seeds_max = round(n_seeds_max - speed * (n_seeds_max - n_seeds_min))  # reduce to n_seeds_min at speed=1.0
        # speed = 0.0  -->  size_seeds_limit = n_seeds_min * nominal_max_size
        # speed = 1.0  -->  size_seeds_limit = n_seeds_min * 1
        size_seeds_limit = round(n_seeds_min * (nominal_max_size ** (1 - speed)))

        # --- generate list ---------------------
        lst = []
        for size in full_size_range:
            # determine n_seeds such that size * n_seeds <= size_seeds_limit
            #                         and        n_seeds <= n_seeds_max
            n_seeds = min(round(size_seeds_limit / size), n_seeds_max)

            # only generate benchmarks if n_seeds >= n_seeds_min; otherwise size is too big for this speed setting
            if n_seeds >= n_seeds_min:
                for seed in range(42, 42 + n_seeds):
                    for strat_name in self._solver_constructor.strategy_names():
                        lst.append((size, strat_name, seed))

        return lst

    def construct_solver(self, size: int, strat_name: str, seed: int) -> MaxDivSolver:
        """Constructs and returns a Solver for given (size, strat_name, seed)-tuple."""
        return self._solver_constructor.construct_solver(size, strat_name, seed)

    def register_result(
        self,
        size: int,
        strat_name: str,
        t_elapsed_sec: float,
        diversity_score: float,
        constraint_score: float,
    ):
        """Register benchmark results for given (size, strat_name, seed)-tuple."""

        # --- register results ---
        self._t_elapsed[size, strat_name].append(t_elapsed_sec)
        self._diversity_scores[size, strat_name].append(diversity_score)
        self._constraint_scores[size, strat_name].append(constraint_score)

        # --- update progress bar ---
        if self._pbar:
            self._pbar.n += 1
            self._pbar.refresh()

    def show_results_tables(self, markdown: bool, file: bool):
        benchmark_type = self.benchmark_type.lower()
        problem_name = self._solver_constructor.problem_name

        # redirect stdout to file if requested
        with stdout_to_file(enabled=file, filename=f"benchmark_{benchmark_type}_{problem_name}.md"):
            # --- show tested strategies table ----------------
            self._solver_constructor.show_strategies_table(markdown)

            # --- aggregate data ------------------------------
            t_elapsed_agg = {
                (size, strat_name): BenchmarkResult.from_list(result_lst)
                for (size, strat_name), result_lst in self._t_elapsed.items()
            }
            diversity_scores_agg = {
                (size, strat_name): NumberWithUncertainty.from_list(result_lst)
                for (size, strat_name), result_lst in self._diversity_scores.items()
            }
            constraint_scores_agg = {
                (size, strat_name): NumberWithUncertainty.from_list(result_lst)
                for (size, strat_name), result_lst in self._constraint_scores.items()
            }

            # --- prepare table data --------------------------

            # --- prep ---
            strat_names = self._solver_constructor.strategy_names()
            size_range = sorted({size for size, _, _ in self.params()})
            scope = [
                (t_elapsed_agg, "Time Duration", "geomean"),
                (diversity_scores_agg, "Diversity Score", "geomean"),
            ]  # (data, title, agg_type)-tuples
            if self._constraints:
                scope.append((constraint_scores_agg, "Constraint Score", "mean"))

            # --- show all tables ---
            for data, title, agg_type in scope:
                # header
                if markdown:
                    headers = ["`d`", "`n`", "`k`", "`m`"] + [f"`{s}`" for s in strat_names]
                else:
                    headers = ["d", "n", "k", "m"] + strat_names

                # table data
                table_data = []
                for size in size_range:
                    problem = self._solver_constructor.construct_problem(size)
                    table_data.append(
                        [
                            str(problem.d),
                            str(problem.n),
                            str(problem.k),
                            str(problem.m),
                        ]
                        + [data[size, strat_name] for strat_name in strat_names]
                    )

                # add aggregates
                table_data = extend_table_with_aggregate_row(table_data, agg=agg_type)

                # --- show title ---
                if markdown:
                    print(f"### {title}")
                else:
                    print(f"{title}:")

                # --- show table ---
                if markdown:
                    display_data = format_table_as_markdown(
                        headers,
                        table_data,
                        highlighters=[
                            BoldLabels(),
                            FastestBenchmark(),
                            HighestNumberWithUncertainty(),
                        ],
                    )
                else:
                    display_data = format_table_for_console(headers, table_data)

                print()
                for line in display_data:
                    print(line)
                print()


# =================================================================================================
#  BenchmarkSolverConstructor
# =================================================================================================
class BenchmarkSolverConstructor(ABC):
    """
    Base class for constructing Solvers for given benchmark scope and (size, strat_name, seed)-tuple.

    The SolverBenchmarkExecutor can use this info to construct a pre-configured Solver for said problem with given size,
    such that it can be benchmarked.  Such class will typically focus on testing...
      - initialization strategies
      - optimization strategies
      - specific solver presets.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, benchmark_type: str, problem_name: str, diversity_metric: DiversityMetric):
        self._benchmark_type = benchmark_type
        self._problem_name = problem_name
        self._diversity_metric = diversity_metric

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    @property
    def problem_name(self) -> str:
        return self._problem_name

    @property
    def has_constraints(self) -> bool:
        """Determine if problems with 'problem_name' have constraints, assuming this property is size-independent."""
        d, n, k, m, n_con_indices = self.get_problem_dimensions(size=10)
        return m > 0

    @property
    def benchmark_type(self) -> str:
        return self._benchmark_type

    def construct_problem(self, size: int) -> MaxDivProblem:
        return BenchmarkProblemFactory.construct_problem(
            name=self._problem_name,
            size=size,
            diversity_metric=self._diversity_metric,
        )

    def get_problem_dimensions(self, size: int) -> tuple[int, int, int, int, int]:
        """Get problem dimensions as (d, n, k, m, n_con_indices)-tuple for the benchmark problem with given size."""
        return BenchmarkProblemFactory.get_problem_dimensions(self._problem_name, size=size)

    # -------------------------------------------------------------------------
    #  API - ABSTRACT
    # -------------------------------------------------------------------------
    @abstractmethod
    def construct_solver(self, size: int, strat_name: str, seed: int) -> MaxDivSolver:
        """Constructs and returns a Solver for given (size, strat_name, seed)-tuple."""
        raise NotImplementedError()

    @abstractmethod
    def strategy_names(self) -> list[str]:
        """Returns list of strategy names that can be constructed by this class."""
        raise NotImplementedError()

    @abstractmethod
    def show_strategies_table(self, markdown: bool):
        """Displays a table summarizing the strategies that can be constructed by this class."""
        raise NotImplementedError()
