from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Protocol

from tqdm import tqdm

from max_div._core._cli.benchmarks._helpers.solver_sizing import K_VALUES, determine_problem_size_for_k
from max_div._core._cli.benchmarks._helpers.speed_scaling import SpeedParam
from max_div._core._markdown import (
    Report,
    ReportElement,
    Table,
    TableAggregationType,
    TableTimeElapsed,
    TableValueWithUncertainty,
    h3,
)
from max_div._core._utils import stdout_to_file
from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.solver import Verbosity

if TYPE_CHECKING:
    from types import TracebackType

    from max_div._core._markdown import TableElement
    from max_div._core.metrics import DiversityMetric
    from max_div._core.problem import VectorMaxDivProblem
    from max_div._core.solver import MaxDivSolver


# =================================================================================================
#  StrategyPreset protocol
# =================================================================================================
class StrategyPreset(Protocol):
    """A StrategyPreset exposes the metadata both benchmark preset enums (init and optim) share."""

    def class_name(self) -> str: ...

    def class_kwargs(self) -> dict[str, Any]: ...

    def is_constraint_aware(self) -> bool: ...

    def preset_note(self) -> str: ...


# =================================================================================================
#  Benchmark Executor
# =================================================================================================
class SolverBenchmarkExecutor:
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, scope: SolverBenchmarkScope) -> None:
        self._scope = scope

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def execute(self, markdown: bool, file: bool = False) -> None:
        # --- run benchmarks ---------------------
        with self._scope as scope:
            for n, strat_name, seed in scope.params():
                # --- construct solver -----------
                solver = scope.construct_solver(n, strat_name, seed)

                # --- run solver -----------------
                solution = solver.solve(verbosity=Verbosity.SILENT)

                # --- get results ----------------
                t_elapsed_sec = list(solution.step_durations.values())[-1].t_elapsed_sec
                diversity_score = solution.score.diversity
                constraint_score = solution.score.constraints

                # --- register results -----------
                scope.register_result(
                    n=n,
                    strat_name=strat_name,
                    t_elapsed_sec=t_elapsed_sec,
                    diversity_score=diversity_score,
                    constraint_score=constraint_score,
                )

        # --- show results -----------------------
        scope.show_results_tables(markdown, file)


# =================================================================================================
#  Benchmark Scope
# =================================================================================================
class SolverBenchmarkScope:
    """The scope of benchmarks to run for one test problem.

    A scope spans all (n, seed, strat_name)-tuples for that problem; the SolverBenchmarkExecutor
    iterates them and registers the results here.
    """

    # -------------------------------------------------------------------------
    #  Constructor / Configuration
    # -------------------------------------------------------------------------
    def __init__(self, solver_constructor: BenchmarkSolverConstructor, speed: float, leave_pbar: bool) -> None:
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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._context_active = False
        if not self._leave_pbar:
            # `_pbar` is always set in __enter__, but ty cannot see the link
            self._pbar.close()  # ty: ignore[unresolved-attribute]
        self._pbar = None

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def params(self) -> list[tuple[int, str, int]]:
        """Return the list of (n, strat_name, seed)-tuples to benchmark."""
        # --- calibrate --------------------------
        # at speed=0.0 the size axis maps each k in K_VALUES to this problem's n
        full_n_range = sorted({determine_problem_size_for_k(self.problem_name, k) for k in K_VALUES})

        # --- speed-dependent settings -----------
        # the seed budget below is computed on n/100 ("weight"), the scale the min/max/limit
        # formulas here are tuned for
        speed = self._speed
        max_weight = max(full_n_range) / 100.0
        n_seeds_min = SpeedParam(slow=3, fast=1).at(speed)  # skip the benchmark when n_seeds falls below this
        n_seeds_max = SpeedParam(slow=16, fast=1, scale="linear").at(speed)  # the seed count never exceeds this
        # The per-problem seed budget allows n_seeds_min curves at the largest size and more at smaller sizes.
        weight_seeds_limit = round(n_seeds_min * SpeedParam(slow=max_weight, fast=1.0).at(speed))

        # --- generate list ----------------------
        lst = []
        for n in full_n_range:
            # determine n_seeds such that (n/100) * n_seeds <= weight_seeds_limit
            #                         and           n_seeds <= n_seeds_max
            n_seeds = min(round(weight_seeds_limit / (n / 100.0)), n_seeds_max)

            # only generate benchmarks if n_seeds >= n_seeds_min; otherwise n is too big for this speed setting
            if n_seeds >= n_seeds_min:
                for seed in range(42, 42 + n_seeds):
                    for strat_name in self._solver_constructor.strategy_names():
                        lst.append((n, strat_name, seed))

        return lst

    def construct_solver(self, n: int, strat_name: str, seed: int) -> MaxDivSolver:
        """Construct and return a Solver for the given (n, strat_name, seed)-tuple."""
        return self._solver_constructor.construct_solver(n, strat_name, seed)

    def register_result(
        self,
        n: int,
        strat_name: str,
        t_elapsed_sec: float,
        diversity_score: float,
        constraint_score: float,
    ) -> None:
        """Register benchmark results for given (n, strat_name, seed)-tuple."""
        # --- register results -------------------
        self._t_elapsed[n, strat_name].append(t_elapsed_sec)
        self._diversity_scores[n, strat_name].append(diversity_score)
        self._constraint_scores[n, strat_name].append(constraint_score)

        # --- update progress bar ----------------
        if self._pbar:
            self._pbar.n += 1
            self._pbar.refresh()

    def show_results_tables(self, markdown: bool, file: bool) -> None:
        benchmark_type = self.benchmark_type.lower()
        problem_name = self._solver_constructor.problem_name

        # redirect stdout to file if requested
        with stdout_to_file(enabled=file, filename=f"benchmark_{benchmark_type}_{problem_name}.md"):
            # --- initialize report --------------
            report = Report()
            report += self._solver_constructor.build_strategies_table()

            # --- aggregate data -----------------
            t_elapsed_agg = {
                (n, strat_name): TableTimeElapsed.from_values(result_lst)
                for (n, strat_name), result_lst in self._t_elapsed.items()
            }
            diversity_scores_agg = {
                (n, strat_name): TableValueWithUncertainty.from_values(result_lst)
                for (n, strat_name), result_lst in self._diversity_scores.items()
            }
            constraint_scores_agg = {
                (n, strat_name): TableValueWithUncertainty.from_values(result_lst)
                for (n, strat_name), result_lst in self._constraint_scores.items()
            }

            # --- prepare table data -------------

            # --- prep ---------------------------
            strat_names = self._solver_constructor.strategy_names()
            n_range = sorted({n for n, _, _ in self.params()})
            scope: list[tuple[dict, str, TableAggregationType]] = [
                (t_elapsed_agg, "Time Duration", TableAggregationType.GEOMEAN),
                (diversity_scores_agg, "Diversity Score", TableAggregationType.GEOMEAN),
            ]  # (data, title, agg_type)-tuples
            if self._constraints:
                scope.append((constraint_scores_agg, "Constraint Score", TableAggregationType.MEAN))

            # --- show all tables ----------------
            headers = ["`d`", "`n`", "`k`", "`m`"] + [f"`{s}`" for s in strat_names]
            for data, title, agg_type in scope:
                # create table
                table = Table(headers)
                for n in n_range:
                    problem = self._solver_constructor.construct_problem(n)
                    table.add_row(
                        [
                            str(problem.d),
                            str(problem.n),
                            str(problem.k),
                            str(problem.m),
                        ]
                        + [data[n, strat_name] for strat_name in strat_names]
                    )

                # finalize table & add to report
                table.add_aggregate_row(agg_type)
                table.highlight_results(TableTimeElapsed, clr_highest=Table.RED)
                table.highlight_results(TableTimeElapsed, clr_lowest=Table.GREEN)
                table.highlight_results(TableValueWithUncertainty, clr_lowest=Table.RED)
                table.highlight_results(TableValueWithUncertainty, clr_highest=Table.GREEN)

                report += [h3(title), table]

            # show final report
            report.print(markdown=markdown)


# =================================================================================================
#  BenchmarkSolverConstructor
# =================================================================================================
class BenchmarkSolverConstructor(ABC):
    """Base class for constructing pre-configured Solvers, one strategy family per subclass.

    A subclass owns one test problem and one set of named strategies, and builds the solver a
    SolverBenchmarkScope asks for.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, benchmark_type: str, problem_name: str, diversity_metric: DiversityMetric) -> None:
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
        _d, _n, _k, m, _n_con_indices = self.get_problem_dimensions(n=1000)
        return m > 0

    @property
    def benchmark_type(self) -> str:
        return self._benchmark_type

    def construct_problem(self, n: int) -> VectorMaxDivProblem:
        return BenchmarkProblemFactory.construct_problem(
            name=self._problem_name,
            n=n,
            diversity_metric=self._diversity_metric,
        )

    def get_problem_dimensions(self, n: int) -> tuple[int, int, int, int, int]:
        """Get problem dimensions as (d, n, k, m, n_con_indices)-tuple for the benchmark problem with given size n."""
        return BenchmarkProblemFactory.get_problem_dimensions(self._problem_name, n=n)

    def _strategies_table(self, intro: str, presets: dict[str, StrategyPreset]) -> list[ReportElement | str]:
        """Build the intro line + table describing the tested strategies, shared by the subclasses.

        The Note column appears only when at least one preset carries a preset note (its exact
        correspondence to a shipped solver preset); the Constraint-aware column only on
        constrained problems.
        """
        has_notes = any(preset.preset_note() for preset in presets.values())
        headers = ["`name`", "`class`", "`params`"]
        if self.has_constraints:
            headers.append("Constraint-aware")
        if has_notes:
            headers.append("Note")

        table = Table(headers)
        for name, preset in presets.items():
            row: list[str | TableElement] = [
                f"`{name}`",
                preset.class_name(),
                "\n".join(f"{k}={v!s}" for k, v in preset.class_kwargs().items()),
            ]
            if self.has_constraints:
                row.append(str(preset.is_constraint_aware()))
            if has_notes:
                row.append(preset.preset_note())
            table.add_row(row)

        return [intro, table]

    # -------------------------------------------------------------------------
    #  API - ABSTRACT
    # -------------------------------------------------------------------------
    @abstractmethod
    def construct_solver(self, n: int, strat_name: str, seed: int) -> MaxDivSolver:
        """Construct and return a Solver for the given (n, strat_name, seed)-tuple."""
        raise NotImplementedError

    @abstractmethod
    def strategy_names(self) -> list[str]:
        """Returns list of strategy names that can be constructed by this class."""
        raise NotImplementedError

    @abstractmethod
    def build_strategies_table(self) -> list[ReportElement | str]:
        """Builds a Table object summarizing the strategies that can be constructed by this class."""
        raise NotImplementedError
