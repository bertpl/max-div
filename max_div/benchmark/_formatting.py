from max_div.internal.benchmarking import BenchmarkResult
from max_div.internal.formatting import md_bold, md_colored, md_table


def format_as_markdown(headers: list[str], data: list[list[str | BenchmarkResult]]) -> list[str]:
    """
    Format benchmark data as a Markdown table.

    Converts BenchmarkResult objects to strings using t_sec_with_uncertainty_str.
    The fastest BenchmarkResult in each row is highlighted in bold and green.

    :param headers: List of column headers
    :param data: 2D list where each row contains strings and/or BenchmarkResult objects
    :return: List of strings representing the Markdown table lines
    """
    # Convert data to string format and identify the fastest results
    converted_data: list[list[str]] = [headers]

    for row in data:
        # Find all BenchmarkResult objects in this row and their indices
        benchmark_results: list[tuple[int, BenchmarkResult]] = []
        for i, cell in enumerate(row):
            if isinstance(cell, BenchmarkResult):
                benchmark_results.append((i, cell))

        # Find the fastest BenchmarkResult (minimum median time)
        fastest_idx = None
        if benchmark_results:
            fastest_idx = min(benchmark_results, key=lambda x: x[1].t_sec_q_50)[0]

        # Convert row to strings, highlighting the fastest
        converted_row: list[str] = []
        for i, cell in enumerate(row):
            if isinstance(cell, BenchmarkResult):
                text = cell.t_sec_with_uncertainty_str
                if i == fastest_idx:
                    text = md_colored(md_bold(text), "#00aa00")
                converted_row.append(text)
            else:
                converted_row.append(str(cell))

        converted_data.append(converted_row)

    return md_table(converted_data)


def format_for_console(headers: list[str], data: list[list[str | BenchmarkResult]]) -> list[str]:
    """Similar to `format_as_markdown`, but without extensive formatting, to keep it readable with rendering."""
    table_data = [headers]
    for row in data:
        converted_row: list[str] = []
        for cell in row:
            if isinstance(cell, BenchmarkResult):
                converted_row.append(cell.t_sec_with_uncertainty_str)
            else:
                converted_row.append(str(cell))
        table_data.append(converted_row)
    return md_table(table_data)
