from collections import defaultdict
from itertools import product

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
from max_div.internal.benchmarking import BenchmarkResult
from max_div.internal.utils import stdout_to_file
from max_div.solver import DiversityMetric, MaxDivSolverBuilder
from max_div.solver._duration import iterations
from max_div.solver._solver_step import OptimizationStep
from max_div.solver._strategies import InitializationStrategy

from ._helpers import (
    OptimStrategyInfo,
    construct_problem_instance,
    get_optimization_strategies,
    get_size_range,
    problem_has_constraints,
)


# =================================================================================================
#  Benchmark function
# =================================================================================================
def benchmark_optimization_strategies(problem_name: str, markdown: bool, file: bool = False, speed: float = 0.0):
    """
    Benchmark initialization strategies on a given benchmark problem across different sizes.

    The setup is relatively straightforward:
      - for each problem size we...
        - initialize the solver with InitDummy (=very poor initialization)
        - we let the optimization strategy run for 1000 iterations, independent of problem size, for a fixed set of seeds
        - we record time taken, final diversity score, and final constraint score

    :param problem_name: Name of the benchmark problem
    :param markdown: If True, outputs the results as a Markdown table, otherwise plain text without markup.
    :param file: If True, redirects output to a file instead of console.
    :param speed: Speed factor to adjust the benchmark duration (0.0 = full, 1.0 = fastest).
    """

    # --- prep --------------------------------------------
    diversity_metric = DiversityMetric.geomean_separation()
    size_range = get_size_range(speed)
    n_seeds = int(10 - speed * 9)  # from 1 (speed=1.0) to 10 (speed=0.0)
    n_iterations = int(1000 - speed * 999)  # from 1 (speed=1.0) to 1000 (speed=0.0)
    has_constraints = problem_has_constraints(problem_name, [min(size_range), max(size_range)])
    optim_strategies: list[OptimStrategyInfo] = get_optimization_strategies(has_constraints)

    # --- benchmark across sizes --------------------------
    # Initialize data structures for benchmark results
    times: dict[int, dict[str, BenchmarkResult]] = defaultdict(dict)
    diversity_scores: dict[int, dict[str, NumberWithUncertainty]] = defaultdict(dict)
    constraint_scores: dict[int, dict[str, NumberWithUncertainty]] = defaultdict(dict)

    size_exp = 1.0  # exponent with which size influences time spent
    pbar = tqdm(
        desc=f"problem {problem_name} - Optimization strategies".ljust(40),
        total=sum([round(s**size_exp) for s in size_range]) * len(optim_strategies) * n_seeds,
        leave=file,
    )

    for size in size_range:
        # Create problem instance
        problem = construct_problem_instance(problem_name, size, diversity_metric)

        # go over all optimization strategies
        for strat_info in optim_strategies:
            # initialize lists for this (size, strategy)
            times_lst = []
            diversity_scores_lst = []
            constraint_scores_lst = []

            # Repeat n_seeds times with different seed
            for seed in range(1, n_seeds + 1):
                # Create solver with explicit initialization strategy
                solver = (
                    MaxDivSolverBuilder(problem)
                    .set_initialization_strategy(InitializationStrategy.dummy())
                    .add_solver_step(
                        OptimizationStep(
                            optim_strategy=strat_info.factory(),
                            duration=iterations(n_iterations),
                        )
                    )
                    .with_seed(seed)
                    .build()
                )

                # Execute solver
                solution = solver.solve()

                # Track elapsed time and diversity score
                times_lst.append(
                    list(solution.step_durations.values())[-1].t_elapsed_sec
                )  # last step is optimization we're testing
                diversity_scores_lst.append(solution.score.diversity)
                constraint_scores_lst.append(solution.score.constraints)

                # Update progress bar
                pbar.n += round(size**size_exp)
                pbar.refresh()

            # Register results for this (size, strategy)
            times[size][strat_info.name] = BenchmarkResult.from_list(times_lst)
            diversity_scores[size][strat_info.name] = NumberWithUncertainty.from_list(diversity_scores_lst)
            constraint_scores[size][strat_info.name] = NumberWithUncertainty.from_list(constraint_scores_lst)

    # --- show results ------------------------------------
    with stdout_to_file(enabled=file, filename=f"benchmark_optimization_{problem_name}.md"):
        # show tested optimization strategies
        show_strategies_table(markdown, optim_strategies, has_constraints, n_iterations)

        # prepare scope of what we need to show
        strategy_names = [strat_info.name for strat_info in optim_strategies]
        scope = [
            (times, "Time Duration", "geomean"),
            (diversity_scores, "Diversity Score", "geomean"),
        ]  # (data, title, agg_type)-tuples
        if has_constraints:
            scope.append((constraint_scores, "Constraint Score", "mean"))

        # show all relevant data
        for data, title, agg_type in scope:
            # --- create table data ---
            if markdown:
                headers = ["`d`", "`n`", "`k`", "`m`"] + [f"`{s}`" for s in strategy_names]
            else:
                headers = ["d", "n", "k", "m"] + strategy_names

            table_data = []
            for size in size_range:
                problem = construct_problem_instance(problem_name, size, diversity_metric)
                table_data.append(
                    [
                        str(problem.d),
                        str(problem.n),
                        str(problem.k),
                        str(problem.m),
                    ]
                    + [data[size][strat_name] for strat_name in strategy_names]
                )

            # --- add aggregates ---
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
#  Helpers
# =================================================================================================
def show_strategies_table(
    markdown: bool,
    optim_strategies: list[OptimStrategyInfo],
    problem_has_constraints: bool,
    n_iterations: int,
) -> None:
    # --- prepare table data ------------------------------
    if markdown:
        headers = ["`name`", "`class`", "`params`"]
    else:
        headers = ["name", "class", "params"]

    if problem_has_constraints:
        headers.append("Constraint-aware")

    table_data = []
    for strat_info in optim_strategies:
        table_data.append(
            [
                f"`{strat_info.name}`" if markdown else strat_info.name,
                strat_info.class_name,
                strat_info.class_kwargs,
            ]
        )
        if problem_has_constraints:
            table_data[-1].append(strat_info.uses_constraints)

    # --- show table ---
    if markdown:
        display_data = format_table_as_markdown(headers, table_data)
    else:
        display_data = format_table_for_console(headers, table_data)

    print(f"Tested Optimization strategies ({n_iterations} iterations):")
    print()
    for line in display_data:
        print(line)
    print()
