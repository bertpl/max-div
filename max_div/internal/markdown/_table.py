from collections import defaultdict
from itertools import chain

from ._report_element import ReportElement
from ._table_aggregation import TableAggregationType
from ._table_element import TableElement, TableText
from ._text_layout import TextLayout


class Table(ReportElement):
    GREEN = "#00aa00"
    RED = "#dd0000"

    # -------------------------------------------------------------------------
    #  Construction / Configuration
    # -------------------------------------------------------------------------
    def __init__(self, headers: list[str]):
        self.headers: list[TableText] = [TableText(h) for h in headers]
        self.rows: list[list[TableElement]] = []
        self._text_layout: dict[tuple[int, int], TextLayout] = defaultdict(TextLayout)  # (row_idx, col_idx) -> layout

    def n_cols(self) -> int:
        """# of table columns"""
        return len(self.headers)

    def n_rows(self) -> int:
        """# of table rows, excluding header"""
        return len(self.rows)

    # -------------------------------------------------------------------------
    #  Build
    # -------------------------------------------------------------------------
    def add_row(self, row: list[str | TableElement]):
        if len(row) < self.n_cols():
            row += [""] * (self.n_cols() - len(row))
        if len(row) > self.n_cols():
            row = row[: self.n_cols()]
        row = [TableText(str(cell)) if not isinstance(cell, TableElement) else cell for cell in row]
        self.rows.append(row)

    def add_aggregate_row(
        self,
        agg_type: TableAggregationType,
        restrict_to_types: list[type[TableElement]] | None = None,
    ):
        # Identify which columns contain Aggregatable objects
        has_aggregatable = [False] * self.n_cols()
        for row in self.rows:
            for col_idx, cell in enumerate(row):
                if cell.supports_aggregation:
                    has_aggregatable[col_idx] = True

        # Find the first column with Aggregatable objects
        first_aggregatable_col: int | None = None
        for col_idx, flag in enumerate(has_aggregatable):
            if flag:
                first_aggregatable_col = col_idx
                break

        # Find the right_most non-Aggregatable column before the first Aggregatable column
        label_col: int | None = None
        for col_idx in range(first_aggregatable_col - 1, -1, -1):
            if not has_aggregatable[col_idx]:
                label_col = col_idx
                break

        # Create the aggregate row
        agg_row: list[TableElement] = [TableText("")] * self.n_cols()

        # Set the label if we found a label column
        if label_col is not None:
            agg_row[label_col] = agg_type.value.capitalize() + ":"
            self.layout(self.n_rows(), label_col).bold = True

        # Now finally aggregate column by column and insert result in agg_row
        for col_idx in range(self.n_cols()):
            if has_aggregatable[col_idx]:
                # get all types of aggregatable elements in this column
                aggregatable_types = {
                    type(self.rows[row_idx][col_idx])
                    for row_idx in range(self.n_rows())
                    if self.rows[row_idx][col_idx].supports_aggregation
                }
                if restrict_to_types:
                    aggregatable_types = aggregatable_types.intersection(restrict_to_types)

                # compute aggregation, if possible
                if len(aggregatable_types) == 1:
                    element_type = aggregatable_types.pop()
                    elements_for_aggregation = [
                        self.rows[row_idx][col_idx]
                        for row_idx in range(self.n_rows())
                        if isinstance(self.rows[row_idx][col_idx], element_type)
                    ]
                    agg_row[col_idx] = element_type.aggregate(elements_for_aggregation, agg_type)

        # add aggregation row
        self.add_row(agg_row)

    # -------------------------------------------------------------------------
    #  Modify
    # -------------------------------------------------------------------------
    def layout(self, i_row: int, i_col: int) -> TextLayout:
        """
        Get the TextLayout object for the specified cell, allowing to modify its layout properties.
        """
        return self._text_layout[i_row, i_col]

    def highlight_results(
        self,
        element_type: type[TableElement],
        clr_lowest: str | None = None,
        clr_highest: str | None = None,
        make_bold: bool = True,
        make_italic: bool = False,
    ):
        """
        For each ROW, highlights the lowest and/or highest values in the table for the specified element type.

        EXAMPLE:

            table.highlight_results(TablePercentage, clr_lowest=table.RED, clr_highest=table.GREEN, make_bold=True)

                --> Highlights lowest percentages in red and highest percentages in green, making both bold.
        """

        for row_idx in range(self.n_rows()):
            # check if we have any element of the specified type in this row
            if not any(isinstance(cell, element_type) for cell in self.rows[row_idx]):
                continue

            # highlight lowest
            if clr_lowest is not None:
                min_value = min([cell for cell in self.rows[row_idx] if isinstance(cell, element_type)])
                for col_idx, cell in enumerate(self.rows[row_idx]):
                    if isinstance(cell, element_type) and cell.is_equalish(min_value):
                        layout = self.layout(row_idx, col_idx)
                        layout.bold |= make_bold
                        layout.italic |= make_italic
                        layout.color = clr_lowest

            # highlight highest
            if clr_highest is not None:
                max_value = max([cell for cell in self.rows[row_idx] if isinstance(cell, element_type)])
                for col_idx, cell in enumerate(self.rows[row_idx]):
                    if isinstance(cell, element_type) and cell.is_equalish(max_value):
                        layout = self.layout(row_idx, col_idx)
                        layout.bold |= make_bold
                        layout.italic |= make_italic
                        layout.color = clr_highest

    # -------------------------------------------------------------------------
    #  Render
    # -------------------------------------------------------------------------
    def render(self, markdown: bool) -> list[str]:
        """
        Convert the table into a list of lines that can be shown in the terminal or written to a Markdown file.

        The following steps are taken:
            1. Convert all cell elements to a single- or multi-line string representation
            2. If markdown==True, apply layout of each cell to the str contents
            3. Split rows than span multiple lines, removing 1 level of our nested list
            4. Convert final cell contents into table lines with proper padding and header separator.
        """

        # --- 1. Render all cells -------------------------
        # header indices  --> s =     headers[i_col][i_cell_line]
        # rows indices    --> s = rows[i_row][i_col][i_cell_line]
        headers: list[list[str]] = self._render_single_elements_of_single_row(markdown, self.headers)
        rows: list[list[list[str]]] = [self._render_single_elements_of_single_row(markdown, row) for row in self.rows]

        # --- 2. Apply layout to rows, if Markdown --------
        if markdown:
            rows_with_layout = [
                [
                    [self.layout(row_idx, col_idx).apply(cell_line) for cell_line in cell]
                    for col_idx, cell in enumerate(row)
                ]
                for row_idx, row in enumerate(rows)
            ]
        else:
            rows_with_layout = rows

        # --- 3. Split rows spanning multiple lines -------
        final_headers = self._split_row_spanning_multiple_lines(headers)
        final_rows = list(chain(*[self._split_row_spanning_multiple_lines(row) for row in rows_with_layout]))

        # --- 4. Convert to final table lines -------------
        return self._render_cell_contents_to_table_lines(final_headers, final_rows)

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    @staticmethod
    def _render_single_elements_of_single_row(markdown: bool, row: list[TableElement]) -> list[list[str]]:
        """
        Renders elements of a single row, possibly spanning multiple lines (if markdown==False)
        In the return list[list[str]], the outer list spans all columns, while each inner list represents multiple
        lines of a single cell.

        Example for a 3-column row:

        [
            ["col 0 - single line"],
            ["col 1 - line 0", "col 1 - line 1", "col 1 - line 2"],
            ["col 2 - single line"],
        ]
        """
        if markdown:
            # in this case, there will be no row duplication; .to_mark_down() always returns a single string
            return [[el.to_mark_down()] for el in row]
        else:
            # in this case, there might be row duplication; .to_plain_text() can return multiple lines for 1 element
            return [el.to_plain_text() for el in row]

    @staticmethod
    def _split_row_spanning_multiple_lines(row: list[list[str]]) -> list[list[str]]:
        """
        Takes a single row (list[list[str]]) representing cell contents that may span multiple lines,
        and splits it into multiple rows (list[list[str]]), such that each row only spans a single line.

        'row' indices:    --> s = row[i_col][i_cell_line]
        'result' indices: --> s = result[i_row][i_col]

        Example for a 3-column row:

            row =       [
                            ["col 0 - single line"],
                            ["col 1 - line 0", "col 1 - line 1", "col 1 - line 2"],
                            ["col 2 - single line"],
                        ]

            result =    [
                            ["col 0 - single line", "col 1 - line 0", "col 2 - single line"],
                            ["",                    "col 1 - line 1", ""                   ],
                            ["",                    "col 1 - line 2", ""                   ],
                        ]
        """
        n_cols = len(row)
        n_rows = max([len(cell_lines) for cell_lines in row])
        return [
            [row[i_col][i_row] if len(row[i_col]) > i_row else "" for i_col in range(n_cols)] for i_row in range(n_rows)
        ]

    @staticmethod
    def _render_cell_contents_to_table_lines(headers: list[list[str]], rows: list[list[str]]) -> list[str]:
        """
        Takes headers & rows (both list[list[str]]) representing cell contents of the entire table and
        renders it into a Markdown table...
          - with a horizontal separator after the header row(s)
          - ensure that all columns are properly aligned by padding with spaces

        `headers` & `rows` nested lists are such that the outer list spans rows, while inner lists represents 1 line.
        """

        # --- init -------------------------------
        n_header_rows = len(headers)
        contents = headers + rows

        # --- determine column widths -----------
        n_cols = len(contents[0])
        col_widths = [0] * n_cols
        for row in contents:
            for col_idx, cell in enumerate(row):
                col_widths[col_idx] = max(col_widths[col_idx], len(cell))

        # --- insert header separator -----------
        contents = contents[:n_header_rows] + [["-" * cw for cw in col_widths]] + contents[n_header_rows:]

        # --- left justify all cells ------------
        contents = [[el.ljust(col_widths[col_idx]) for col_idx, el in enumerate(row)] for row in contents]

        # --- build table lines -----------------
        return [""] + ["| " + " | ".join(row) + " |" for row in contents] + [""]
