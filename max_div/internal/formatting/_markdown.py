def md_table(data: list[list[str]]) -> list[str]:
    """Creates a Markdown table from a 2D list of strings, where the first list represents the table headers."""

    # Calculate column widths based on all rows
    num_cols = len(data[0])
    col_widths = [max(1, max(len(row[i]) for row in data)) for i in range(num_cols)]  # max width of content and >=1

    result = []

    # Add header row
    header = "| " + " | ".join(data[0][i].ljust(col_widths[i]) for i in range(num_cols)) + " |"
    result.append(header)

    # Add separator row
    separator = "| " + " | ".join("-" * col_widths[i] for i in range(num_cols)) + " |"
    result.append(separator)

    # Add data rows
    for row in data[1:]:
        row_str = "| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(num_cols)) + " |"
        result.append(row_str)

    return result


def md_multiline(lines: list[str]) -> str:
    """Puts multiple Markdown lines into a single line by means of html line breaks."""
    return "<br>".join(lines)


def md_bold(text: str) -> str:
    """Makes the given text bold in Markdown."""
    return f"**{text}**"


def md_italic(text: str) -> str:
    """Makes the given text italic in Markdown."""
    return f"*{text}*"


def md_colored(text: str, hex_color: str) -> str:
    """Colors the given text in Markdown using HTML span tags (hex_color="#rrggbb" or "#rgb")."""
    return f'<span style="color:{hex_color}">{text}</span>'
