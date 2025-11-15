from typing import Literal

from max_div.internal.benchmarking import BenchmarkResult
from max_div.internal.formatting import md_bold, md_colored, md_table


def extend_table_with_aggregate_row(
    data: list[list[str | BenchmarkResult]], agg: Literal["mean", "geomean", "sum"]
) -> list[list[str | BenchmarkResult]]:
    """
    Extend an extra row to the provided data that contains aggregate statistics of the provided data:
     - for each column that has at least 1 row containing a BenchmarkResult object, compute an aggregate BenchmarkResult object
     - all other columns are kept empty

    The last column not containing any BenchmarkResult objects that comes before the first column containing
      BenchmarkResult objects is used as label for the aggregate row, based on the 'agg' argument, capitilized.

    BenchmarkResults are aggregated by aggregation the q25, q50, and q75 times separately.
    """
    n_cols = len(data[0])

    # Identify which columns contain BenchmarkResult objects
    has_benchmark_result = [False] * n_cols
    for row in data:
        for col_idx, cell in enumerate(row):
            if isinstance(cell, BenchmarkResult):
                has_benchmark_result[col_idx] = True

    # Find the first column with BenchmarkResult
    first_benchmark_col = None
    for col_idx, has_result in enumerate(has_benchmark_result):
        if has_result:
            first_benchmark_col = col_idx
            break

    # Find the last non-BenchmarkResult column before the first BenchmarkResult column
    label_col = None
    for col_idx in range(first_benchmark_col - 1, -1, -1):
        if not has_benchmark_result[col_idx]:
            label_col = col_idx
            break

    # Create the aggregate row
    agg_row: list[str | BenchmarkResult] = [""] * n_cols

    # Set the label if we found a label column
    if label_col is not None:
        agg_row[label_col] = agg.capitalize() + ":"

    # Compute aggregates for each column with BenchmarkResult objects
    for col_idx in range(n_cols):
        # Collect all BenchmarkResult values from this column
        results = [row[col_idx] for row in data if isinstance(row[col_idx], BenchmarkResult)]
        if results:  # Only compute if we have values
            agg_row[col_idx] = BenchmarkResult.aggregate(results, method=agg)

    # Return data with the aggregate row appended
    return data + [agg_row]


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
        # Find the fastest BenchmarkResult (minimum median time)
        t_q50_min = min([value.t_sec_q_50 for value in row if isinstance(value, BenchmarkResult)])

        # Convert row to strings, highlighting the results with t_q25 <= t_q50_min
        converted_row: list[str] = []
        for i, value in enumerate(row):
            if isinstance(value, BenchmarkResult):
                text = value.t_sec_with_uncertainty_str
                if value.t_sec_q_25 <= t_q50_min:
                    text = md_colored(md_bold(text), "#00aa00")
                converted_row.append(text)
            else:
                value = str(value)
                if value.endswith(":"):
                    value = md_bold(value)  # this very likely represents a label (e.g. "Mean:")
                converted_row.append(value)

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
