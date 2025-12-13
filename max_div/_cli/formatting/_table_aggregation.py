from __future__ import annotations

from typing import Literal

from ._content_types import BenchmarkResult, CellContent, NumberWithUncertainty, Percentage


# =================================================================================================
#  Aggregation
# =================================================================================================
def extend_table_with_aggregate_row(
    data: list[list[CellContent]],
    agg: Literal["mean", "geomean", "sum"],
    include_benchmark_result: bool = True,
    include_percentage: bool = True,
    include_number_with_uncertainty: bool = True,
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

    Aggregatable = BenchmarkResult | Percentage | NumberWithUncertainty

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

        if include_number_with_uncertainty:
            # Collect all NumberWithUncertainty values from this column
            numbers = [row[col_idx] for row in data if isinstance(row[col_idx], NumberWithUncertainty)]
            if numbers:  # Only compute if we have values
                agg_row[col_idx] = NumberWithUncertainty.aggregate(numbers, method=agg)

    # Return data with the aggregate row appended
    return data + [agg_row]
