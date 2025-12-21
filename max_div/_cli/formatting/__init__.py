from ._content_types import BenchmarkResult, CellContent, NumberWithUncertainty, Percentage
from ._table_aggregation import extend_table_with_aggregate_row
from ._table_formatting import format_table_as_markdown, format_table_for_console
from ._table_highlighters import (
    BoldLabels,
    FastestBenchmark,
    HighestNumberWithUncertainty,
    HighestPercentage,
    HighLighter,
    LowestPercentage,
)
