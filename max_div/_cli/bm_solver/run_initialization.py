import numpy as np
from tqdm import tqdm

from max_div._cli.formatting import (
    BoldLabels,
    format_as_markdown,
    format_for_console,
)
from max_div.benchmarks import BenchmarkProblemFactory
from max_div.internal.benchmarking import BenchmarkResult
from max_div.solver import DiversityMetric, MaxDivSolverBuilder

from ...internal.formatting import ljust_str_list
from ._helpers import construct_problem_instance, get_initialization_strategies, problem_has_constraints


def benchmark_initialization_strategies(problem_name: str, markdown: bool):
    """
    Benchmark initialization strategies on a given benchmark problem across different sizes.

    :param problem_name: Name of the benchmark problem
    :param markdown: If True, outputs the results as a Markdown table, otherwise plain text without markup.
    """
    print(f"Benchmarking initialization strategies on problem: {problem_name}")
    print()

    # --- prep --------------------------------------------
    diversity_metric = DiversityMetric.geomean_separation()
    size_range = list(range(1, 21))
    n_seeds = 32
    has_constraints = problem_has_constraints(problem_name, [min(size_range), max(size_range)])
    init_strategies = get_initialization_strategies(has_constraints)

    print("Testing Initialization strategies:")
    strat_names_ljust = ljust_str_list([f"`{strat_name}`" for strat_name, _, _ in init_strategies])
    for strat_name_ljust, (_, desc, _) in zip(strat_names_ljust, init_strategies):
        print(f" - {strat_name_ljust}: {desc}")
    print()

    # --- benchmark across sizes --------------------------
    # Initialize data structures for benchmark results
    times: dict[int, dict[str, BenchmarkResult]] = dict()
    diversity_scores: dict[int, dict[str, str]] = dict()

    for size in tqdm(size_range, leave=False):
        # initialize data structures for this size
        times[size] = dict()
        diversity_scores[size] = dict()

        # Create problem instance
        problem = construct_problem_instance(problem_name, size, diversity_metric)

        # go over all initialization strategies
        for strat_name, _, strategy_factory_method in init_strategies:
            # Repeat n_seeds times with different seed
            times_lst = []
            diversity_scores_lst = []
            for seed in range(1, n_seeds + 1):
                # Create solver with explicit initialization strategy
                solver = (
                    MaxDivSolverBuilder(problem)
                    .set_initialization_strategy(strategy_factory_method())
                    .with_seed(seed)
                    .build()
                )

                # Execute solver
                solution = solver.solve()

                # Track elapsed time and diversity score
                times_lst.append(
                    list(solution.step_durations.values())[-1].t_elapsed_sec
                )  # last step is initialization
                diversity_scores_lst.append(solution.score.diversity)

            # Register results for this (size, strategy)
            times[size][strat_name] = BenchmarkResult(
                t_sec_q_25=float(np.quantile(times_lst, 0.25)),
                t_sec_q_50=float(np.quantile(times_lst, 0.50)),
                t_sec_q_75=float(np.quantile(times_lst, 0.75)),
            )
            diversity_scores[size][strat_name] = f"{float(np.median(diversity_scores_lst)):.3f}"

    # --- show results ------------------------------------
    strategy_names = [strat_name for strat_name, _, _ in init_strategies]
    for data, title in [
        (times, "Time Duration"),
        (diversity_scores, "Diversity Score"),
    ]:
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

        # --- show title ---
        if markdown:
            print(f"### {title}")
        else:
            print(f"{title}:")

        # --- show table ---
        if markdown:
            display_data = format_as_markdown(headers, table_data, highlighters=[BoldLabels()])
        else:
            display_data = format_for_console(headers, table_data)

        print()
        for line in display_data:
            print(line)
        print()
