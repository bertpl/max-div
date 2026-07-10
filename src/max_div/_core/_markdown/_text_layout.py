from dataclasses import dataclass


@dataclass
class TextLayout:
    color: str | None = None  # when specified, should be a hex color string like "#rrggbb"
    bold: bool = False
    italic: bool = False

    def apply(self, text: str) -> str:
        """Applies the layout to the given text in Markdown format."""
        if not text:
            return text  # empty string needs no layout
        elif "<br>" in text:
            # multi-line text
            return "<br>".join(self.apply(line) for line in text.split("<br>"))
        else:
            # single-line text
            result = text
            if self.bold and self.italic:
                result = f"***{result}***"
            elif self.bold:
                result = f"**{result}**"
            elif self.italic:
                result = f"*{result}*"
            if self.color is not None:
                result = f'<span style="color:{self.color}">{result}</span>'
            return result
