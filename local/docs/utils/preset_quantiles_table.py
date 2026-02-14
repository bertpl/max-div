from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from local.docs.utils.quantile_curves import QuantileCurves
from max_div.internal.formatting import format_long_time_duration
from max_div.internal.markdown import Report, Table, TableElement, TableValueRange, h4
from max_div.solver import SolverPreset


@dataclass
class _QuantileCurvesEntry:
    """Internal data class to store a QuantileCurves object with its axis meanings."""

    quantile_curves: QuantileCurves
    preset: SolverPreset
    x_axis: Literal["iteration", "elapsed_sec"]
    y_axis: Literal["constraint_score", "diversity_score"]


class PresetQuantilesTable:
    """
    A class to generate Markdown tables with quantile data for solver presets.

    Collects QuantileCurves for different presets and axis configurations,
    then generates a Markdown report with tables showing q10/q50/q90 values
    at selected x-values (iterations or elapsed time).
    """

    # Row x-values for iteration-based tables (1000, 2000, 5000, then multiply by 10)
    ITERATION_BASE_VALUES: list[int] = [1000, 2000, 5000]

    # Row x-values for elapsed time-based tables (in seconds)
    # Max: 6 hours, increments <= 15 minutes
    ELAPSED_TIME_VALUES: list[int] = [
        10,
        15,
        30,
        45,
        60,
        2 * 60,
        5 * 60,
        10 * 60,
    ] + [n * 15 * 60 for n in range(1, 25)]  # 15 min to 6 hours in 15-minute increments

    def __init__(self, problem_name: str, size: int, target_md_folder: Path):
        """
        Initialize a PresetQuantilesTable.

        :param problem_name: Name of the benchmark problem.
        :param size: Size of the problem.
        :param target_md_folder: Path to the folder where markdown files should be saved.
        """
        self._problem_name = problem_name
        self._size = size
        self._target_md_folder = target_md_folder
        self._entries: list[_QuantileCurvesEntry] = []

    def add_quantile_curves(
        self,
        quantile_curves: QuantileCurves,
        preset: SolverPreset,
        x_axis: Literal["iteration", "elapsed_sec"],
        y_axis: Literal["constraint_score", "diversity_score"],
    ) -> None:
        """
        Add a QuantileCurves object with its axis meanings.

        :param quantile_curves: The QuantileCurves object to add.
        :param preset: The solver preset this curve belongs to.
        :param x_axis: The meaning of the x-axis ("iteration" or "elapsed_sec").
        :param y_axis: The meaning of the y-axis ("constraint_score" or "diversity_score").
        """
        self._entries.append(
            _QuantileCurvesEntry(
                quantile_curves=quantile_curves,
                preset=preset,
                x_axis=x_axis,
                y_axis=y_axis,
            )
        )

    def generate_tables(self) -> None:
        """
        Generate Markdown tables and output to a .md file in the repo root.

        Creates a Markdown report with tables for each (y_axis, x_axis) combination,
        showing q10/q50/q90 quantile values at selected x-values for each preset.
        """
        if not self._entries:
            return

        all_presets = SolverPreset.all_sorted()

        # Group entries by (y_axis, x_axis)
        # Sort to ensure iteration tables come before elapsed_sec tables
        axis_combinations = sorted(
            set((e.y_axis, e.x_axis) for e in self._entries),
            key=lambda t: (t[0], 0 if t[1] == "iteration" else 1),
        )

        # Create report
        report = Report()

        for y_axis, x_axis in axis_combinations:
            # Filter entries for this axis combination
            entries_for_axes = [e for e in self._entries if e.x_axis == x_axis and e.y_axis == y_axis]

            # Build a dict: preset -> QuantileCurves
            preset_quantile_curves: dict[SolverPreset, QuantileCurves] = {
                e.preset: e.quantile_curves for e in entries_for_axes
            }

            if not preset_quantile_curves:
                continue

            # Determine x/y axis labels and title
            if x_axis == "iteration":
                x_title = "Total Iterations"
            else:
                x_title = "Total Time"

            if y_axis == "constraint_score":
                y_title = "Constraint Score"
            else:
                y_title = "Diversity Score"

            # Determine x values and row labels
            all_x_max = [qc.x_max for qc in preset_quantile_curves.values()]
            max_x = max(all_x_max)

            row_x_values, row_labels = self._get_row_x_values_and_labels(x_axis, max_x)

            # Create table with headers = presets
            table_headers = [x_title] + [f"`{preset.value.upper()}`" for preset in all_presets]
            table = Table(table_headers)

            # Fill table rows
            for row_label, x_val in zip(row_labels, row_x_values):
                row: list[str | TableElement] = [row_label]
                for preset in all_presets:
                    if preset in preset_quantile_curves:
                        qc = preset_quantile_curves[preset]
                        # Only add data if x_val is within the valid range of the quantile curves
                        if qc.x_min <= x_val <= qc.x_max:
                            q10, q50, q90 = qc.get_quantiles_at_x(x_val)
                            row.append(TableValueRange(q10, q50, q90, max_decimals=9, diff_decimals=3))
                        else:
                            row.append("-")
                    else:
                        row.append("-")
                table.add_row(row)

            # Highlight best and worst results
            # Note: red is applied first, then green, so green takes precedence when all values are equal
            # Don't highlight rows with only a single value (highlight_single_values=False)
            table.highlight_results(
                element_type=TableValueRange,
                clr_lowest=Table.RED,
                make_bold=True,
                highlight_single_values=False,
            )
            table.highlight_results(
                element_type=TableValueRange,
                clr_highest=Table.GREEN,
                make_bold=True,
                highlight_single_values=False,
            )

            # Add table to report
            report += h4(f"{y_title} vs {x_title}")
            report += "(estimated q10...q90 ranges)"
            report += table

        # Save Markdown report to target markdown folder
        report_file = self._target_md_folder / f"preset_result_quantiles_{self._problem_name}_{self._size}.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report.render(markdown=True)))

    @classmethod
    def _get_row_x_values_and_labels(
        cls,
        x_axis: Literal["iteration", "elapsed_sec"],
        max_x: float,
    ) -> tuple[list[float], list[str]]:
        """
        Determine x values and corresponding row labels for the table.

        :param x_axis: The x-axis type ("iteration" or "elapsed_sec").
        :param max_x: Maximum x value in the data.
        :return: Tuple of (x_values, labels).
        """
        if x_axis == "iteration":
            # iterations: 1000, 2000, 5000, 10_000, 20_000, 50_000, ...
            row_x_values: list[float] = []
            multiplier = 1
            while True:
                for base in cls.ITERATION_BASE_VALUES:
                    val = base * multiplier
                    if val <= max_x:
                        row_x_values.append(float(val))
                    else:
                        break
                else:
                    multiplier *= 10
                    continue
                break
            row_labels = [f"{int(v):_}" for v in row_x_values]
        else:
            # elapsed time (seconds)
            # Use set to automatically deduplicate, then sort
            row_x_values_set = {float(t) for t in cls.ELAPSED_TIME_VALUES if t <= max_x}
            # Add rounded-down max minutes (will be deduplicated if already present)
            max_minutes = int(max_x // 60)
            if max_minutes > 0:
                row_x_values_set.add(float(max_minutes * 60))
            row_x_values = sorted(row_x_values_set)
            row_labels = [format_long_time_duration(t, n_chars=5).strip() for t in row_x_values]

        return row_x_values, row_labels
