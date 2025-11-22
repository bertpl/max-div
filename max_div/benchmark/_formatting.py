from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from max_div.internal.benchmarking import BenchmarkResult
from max_div.internal.formatting import md_bold, md_colored, md_table


# =================================================================================================
#  Helper classes / types
# =================================================================================================
@dataclass
class Percentage:
    frac: float  # fraction between 0.0 and 1.0
    decimals: int = 1  # number of decimals to display

    def __str__(self):
        return f"{(self.frac * 100):.{self.decimals}f}%"


CellContent = str | BenchmarkResult | Percentage


# =================================================================================================
#  Aggregation
# =================================================================================================
def extend_table_with_aggregate_row(
    data: list[list[CellContent]],
    agg: Literal["mean", "geomean", "sum"],
    include_benchmark_result: bool = True,
    include_percentage: bool = True,
) -> list[list[CellContent]]:
    """
    This function adds aggregate statistics for BenchmarkResult | Percentage (=Aggregatable) columns to the data table.

    Extend an extra row to the provided data that contains aggregate statistics of the provided data:
     - for each column that has at least 1 row containing a Aggregatable object, compute an aggregate
     - all other columns are kept empty

    The last column not containing any Aggregatable objects that comes before the first column containing
      Aggregatable objects is used as label for the aggregate row, based on the 'agg' argument, capitalized.

    BenchmarkResults are aggregated by aggregation the q25, q50, and q75 times separately.
    Percentage objects are aggregated with decimals equal to max of what we observed in that col.
    """
    n_cols = len(data[0])

    Aggregatable = BenchmarkResult | Percentage

    # Identify which columns contain Aggregatable objects
    has_aggregatable = [False] * n_cols
    for row in data:
        for col_idx, cell in enumerate(row):
            if isinstance(cell, Aggregatable):
                has_aggregatable[col_idx] = True

    # Find the first column with Aggregatable objects
    first_aggregatable_col = None
    for col_idx, has_result in enumerate(has_aggregatable):
        if has_result:
            first_aggregatable_col = col_idx
            break

    # Find the last non-Aggregatable column before the first Aggregatable column
    label_col = None
    for col_idx in range(first_aggregatable_col - 1, -1, -1):
        if not has_aggregatable[col_idx]:
            label_col = col_idx
            break

    # Create the aggregate row
    agg_row: list[CellContent] = [""] * n_cols

    # Set the label if we found a label column
    if label_col is not None:
        agg_row[label_col] = agg.capitalize() + ":"

    # Compute aggregates for each column with BenchmarkResult objects
    for col_idx in range(n_cols):
        if include_benchmark_result:
            # Collect all BenchmarkResult values from this column
            results = [row[col_idx] for row in data if isinstance(row[col_idx], BenchmarkResult)]
            if results:  # Only compute if we have values
                agg_row[col_idx] = BenchmarkResult.aggregate(results, method=agg)

        if include_percentage:
            # Collect all Percentage values from this column
            percentages = [row[col_idx] for row in data if isinstance(row[col_idx], Percentage)]
            if percentages:  # Only compute if we have values
                # Compute average fraction and max decimals
                avg_frac = sum(p.frac for p in percentages) / len(percentages)
                max_decimals = max(p.decimals for p in percentages)
                agg_row[col_idx] = Percentage(frac=avg_frac, decimals=max_decimals + 1)

    # Return data with the aggregate row appended
    return data + [agg_row]


# =================================================================================================
#  Markdown highlighters
# =================================================================================================
class HighLighter(ABC):
    @abstractmethod
    def process_row(self, row: list[CellContent]) -> list[CellContent]:
        raise NotImplementedError()


class FastestBenchmark(HighLighter):
    def __init__(self, bold: bool = True, color: str = "#00aa00"):
        self.bold = bold
        self.color = color

    def process_row(self, row: list[CellContent]) -> list[CellContent]:
        if any(isinstance(value, BenchmarkResult) for value in row):
            # Find the fastest BenchmarkResult (minimum median time)
            t_q50_min = min([value.t_sec_q_50 for value in row if isinstance(value, BenchmarkResult)])

            # Convert row to strings, highlighting the results with t_q25 <= t_q50_min
            converted_row: list[CellContent] = []
            for i, value in enumerate(row):
                if isinstance(value, BenchmarkResult):
                    text = str(value)
                    if value.t_sec_q_25 <= t_q50_min:
                        if self.bold:
                            text = md_bold(text)
                        text = md_colored(text, self.color)
                    converted_row.append(text)
                else:
                    converted_row.append(value)
            return converted_row
        else:
            return row


class HighestPercentage(HighLighter):
    def __init__(self, bold: bool = True, color: str = "#00aa00"):
        self.bold = bold
        self.color = color

    def process_row(self, row: list[CellContent]) -> list[CellContent]:
        if any(isinstance(value, Percentage) for value in row):
            # Find the highest Percentage (maximum frac)
            max_perc = max([value for value in row if isinstance(value, Percentage)], key=lambda x: x.frac)

            # Convert row to strings, highlighting the results with frac == max_frac
            converted_row: list[CellContent] = []
            for i, value in enumerate(row):
                if isinstance(value, Percentage):
                    text = str(value)
                    if text == str(max_perc):  # make green if its str-representation is equal
                        if self.bold:
                            text = md_bold(text)
                        text = md_colored(text, self.color)
                    converted_row.append(text)
                else:
                    converted_row.append(value)
            return converted_row
        else:
            return row


class BoldLabels(HighLighter):
    def process_row(self, row: list[CellContent]) -> list[CellContent]:
        converted_row: list[CellContent] = []
        for value in row:
            if isinstance(value, str) and value.endswith(":"):
                converted_row.append(md_bold(value))
            else:
                converted_row.append(value)
        return converted_row


# =================================================================================================
#  Formatting
# =================================================================================================
def format_as_markdown(
    headers: list[str], data: list[list[CellContent]], highlighters: list[HighLighter] | None = None
) -> list[str]:
    """
    Format benchmark data as a Markdown table.

    Converts BenchmarkResult objects to strings using t_sec_with_uncertainty_str.
    The fastest BenchmarkResult in each row is highlighted in bold and green.

    :param headers: List of column headers
    :param data: 2D list where each row contains strings and/or BenchmarkResult objects
    :param highlighters: Optional list of HighLighter objects to apply to each row
    :return: List of strings representing the Markdown table lines
    """
    # Convert data to string format and identify the fastest results
    converted_data: list[list[str]] = [headers]

    for row in data:
        # highlight if requested
        for highlighter in highlighters or []:
            row = highlighter.process_row(row)

        # convert to str
        row = [str(cell) for cell in row]

        # append to converted data
        converted_data.append(row)

    return md_table(converted_data)


def format_for_console(headers: list[str], data: list[list[CellContent]]) -> list[str]:
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
