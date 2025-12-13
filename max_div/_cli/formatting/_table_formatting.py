from max_div.internal.formatting import md_table

from ._content_types import CellContent
from ._table_highlighters import HighLighter


# =================================================================================================
#  Table Formatting
# =================================================================================================
def format_table_as_markdown(
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


def format_table_for_console(headers: list[str], data: list[list[CellContent]]) -> list[str]:
    """Similar to `format_as_markdown`, but without extensive formatting, to keep it readable with rendering."""
    table_data = [headers]
    for row in data:
        converted_row = [str(cell) for cell in row]
        table_data.append(converted_row)
    return md_table(table_data)
