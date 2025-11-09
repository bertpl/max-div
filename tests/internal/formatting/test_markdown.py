from max_div.internal.formatting import md_bold, md_colored, md_italic, md_multiline, md_table


def test_md_table():
    # --- arrange -----------------------------------------
    data = [
        ["Header 1", "Header 2", "Header 3", ""],
        ["Row 1 Col 1", "Row 1 Col 2", "", ""],
        ["Row 2 Col 1", "", "Row 2 Col 3", ""],
    ]

    expected = [
        "| Header 1    | Header 2    | Header 3    |   |",
        "| ----------- | ----------- | ----------- | - |",
        "| Row 1 Col 1 | Row 1 Col 2 |             |   |",
        "| Row 2 Col 1 |             | Row 2 Col 3 |   |",
    ]

    # --- act ---------------------------------------------
    result = md_table(data)

    # --- assert ------------------------------------------
    assert result == expected


def test_md_multiline():
    # --- arrange -----------------------------------------
    lines = ["Line 1", "Line 2", "Line 3"]
    expected = "Line 1<br>Line 2<br>Line 3"

    # --- act ---------------------------------------------
    result = md_multiline(lines)

    # --- assert ------------------------------------------
    assert result == expected


def test_md_bold():
    # --- arrange -----------------------------------------
    text = "Bold Text"
    expected = "**Bold Text**"

    # --- act ---------------------------------------------
    result = md_bold(text)

    # --- assert ------------------------------------------
    assert result == expected


def test_md_italic():
    # --- arrange -----------------------------------------
    text = "Italic Text"
    expected = "*Italic Text*"

    # --- act ---------------------------------------------
    result = md_italic(text)

    # --- assert ------------------------------------------
    assert result == expected


def test_md_colored():
    # --- arrange -----------------------------------------
    text = "Colored Text"
    hex_color = "#ff5733"
    expected = '<span style="color:#ff5733">Colored Text</span>'

    # --- act ---------------------------------------------
    result = md_colored(text, hex_color)

    # --- assert ------------------------------------------
    assert result == expected
