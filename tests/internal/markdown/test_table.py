from max_div.internal.markdown import Table, TableAggregationType, TablePercentage, TableTimeElapsed


def test_table():
    # --- arrange -----------------------------------------
    table = Table(headers=["Name", "Age", "City"])
    table.add_row(["Alice", "30", "New York\n(US)"])
    table.add_row(["Bob", "25", "Los Angeles\n(US)"])
    table.add_row(["Charlie", "31"])  # incomplete row
    table.add_row(["David", "26", "Los Angeles\n(US)", "blah"])  # too many columns, last one ignored
    table.layout(0, 2).bold = True  # make "New York (US)" bold
    table.layout(1, 0).italic = True  # make "Bob" italic

    # --- act ---------------------------------------------
    md_lines = table.render(markdown=True)
    txt_lines = table.render(markdown=False)

    # --- assert ------------------------------------------
    assert table.n_rows() == 4
    assert table.n_cols() == 3

    assert md_lines == [
        "",
        "| Name    | Age | City                     |",
        "| ------- | --- | ------------------------ |",
        "| Alice   | 30  | **New York**<br>**(US)** |",
        "| *Bob*   | 25  | Los Angeles<br>(US)      |",
        "| Charlie | 31  |                          |",
        "| David   | 26  | Los Angeles<br>(US)      |",
        "",
    ]
    assert txt_lines == [
        "",
        "| Name    | Age | City        |",
        "| ------- | --- | ----------- |",
        "| Alice   | 30  | New York    |",
        "|         |     | (US)        |",
        "| Bob     | 25  | Los Angeles |",
        "|         |     | (US)        |",
        "| Charlie | 31  |             |",
        "| David   | 26  | Los Angeles |",
        "|         |     | (US)        |",
        "",
    ]


def test_table_add_aggregate_row():
    # --- arrange -----------------------------------------
    table = Table(headers=["Name", "Score (%)", "Time"])
    table.add_row(["Alice", TablePercentage(0.95, 1), TableTimeElapsed(950, 1000, 1050)])
    table.add_row(["Bob", TablePercentage(0.751, 1), TableTimeElapsed(450, 500, 550)])

    # --- act ---------------------------------------------
    table.add_aggregate_row(agg_type=TableAggregationType.MEAN, restrict_to_types=[TablePercentage])
    md_lines = table.render(markdown=True)

    # --- assert ------------------------------------------
    assert table.n_cols() == 3  # 3 columns as specified during construction
    assert table.n_rows() == 3  # 2 data rows + 1 aggregate row

    assert md_lines == [
        "",
        "| Name      | Score (%) | Time              |",
        "| --------- | --------- | ----------------- |",
        "| Alice     | 95.0%     | 1000 sec ± 5.0%   |",
        "| Bob       | 75.1%     | 500.0 sec ± 10.0% |",
        "| **Mean:** | 85.05%    |                   |",
        "",
    ]


def test_table_highlight_results():
    # --- arrange -----------------------------------------
    table = Table(headers=["Period", "Ellie", "Tony", "Ziva"])
    table.add_row(["(Effort)", "Strong", "Mediocre", "Strong"])
    table.add_row(["Period 1", TablePercentage(0.95), TablePercentage(0.70), TablePercentage(0.95001)])
    table.add_row(["Period 2", TablePercentage(0.95), TablePercentage(0.70), TablePercentage(0.94)])
    table.add_row(["Period 3", TablePercentage(0.90), TablePercentage(0.91), TablePercentage(0.90)])
    table.add_row(["Period 4", "-", "-", TablePercentage(0.93)])
    table.add_aggregate_row(agg_type=TableAggregationType.MEAN, restrict_to_types=[TablePercentage])

    # --- act ---------------------------------------------
    table.highlight_results(
        element_type=TablePercentage,
        clr_lowest=Table.RED,
        clr_highest=Table.GREEN,
        highlight_single_values=False,
    )
    md_lines = table.render(markdown=True)

    # --- assert ------------------------------------------
    assert md_lines == [
        "",
        "| Period    | Ellie                                         | Tony                                          | Ziva                                         |",
        "| --------- | --------------------------------------------- | --------------------------------------------- | -------------------------------------------- |",
        "| (Effort)  | Strong                                        | Mediocre                                      | Strong                                       |",
        '| Period 1  | <span style="color:#00aa00">**95.0%**</span>  | <span style="color:#dd0000">**70.0%**</span>  | <span style="color:#00aa00">**95.0%**</span> |',
        '| Period 2  | <span style="color:#00aa00">**95.0%**</span>  | <span style="color:#dd0000">**70.0%**</span>  | 94.0%                                        |',
        '| Period 3  | <span style="color:#dd0000">**90.0%**</span>  | <span style="color:#00aa00">**91.0%**</span>  | <span style="color:#dd0000">**90.0%**</span> |',
        "| Period 4  | -                                             | -                                             | 93.0%                                        |",
        '| **Mean:** | <span style="color:#00aa00">**93.33%**</span> | <span style="color:#dd0000">**77.00%**</span> | 93.00%                                       |',
        "",
    ]
