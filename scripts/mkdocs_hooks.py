"""Build hooks for the documentation site.

Two rewrites run on every rendered page:

- **README image paths.** The docs home page includes README.md, and the README addresses its
  images with repo-root-relative paths (`docs/images/...`) because that is what GitHub resolves.
  The other two surfaces already correct for this in their own way: PyPI's long description is
  built by hatch-fancy-pypi-readme, which rewrites those attributes to tag-pinned absolute URLs
  (see the substitutions in pyproject.toml). MkDocs serves the `docs/` directory *as* the site
  root, so without a rewrite the same paths land one directory too deep and the images 404. The
  hook strips the `docs/` prefix and re-anchors what remains against the including page.

- **Figure display width.** Every raster figure in the docs is rendered by Matplotlib at the
  `savefig.dpi` of the shared style sheet, from a size in inches chosen per figure. Left to the
  browser, each image fills the content column whatever its design size, so an 8-inch chart shows
  its 10 pt labels half again as large as a 12-inch one. The hook sets each image's `width` to its
  size in inches at CSS resolution (96 px per inch), so a figure appears at the size it was
  designed at and a wide one is capped by the column (`max-width: 100%` in the style sheet).
  Images that already carry a `width` or `style` attribute are hand-sized and left alone, as is
  the home page, whose README images are styled for GitHub.

Both run on the rendered page rather than on its markdown: the README is pulled in by a snippet,
which is expanded during markdown conversion, so at `on_page_markdown` time the home page is still
a one-line include directive and none of the README's paths exist yet.
"""

import posixpath
import re
import struct
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parent.parent
STYLE_SHEET = REPO_ROOT / "local" / "docs" / "figures" / "docs.mplstyle"
CSS_PIXELS_PER_INCH = 96

# `src` and `srcset` both carry paths in the README's <picture> element; a plain `src`-only
# rule would silently leave the dark variant broken.
DOCS_RELATIVE_ATTR = re.compile(r'\b(src|srcset)="docs/([^"]+)"')

IMG_TAG = re.compile(r"<img\b[^>]*>")
SRC_ATTR = re.compile(r'\bsrc="([^"]+)"')
HAND_SIZED_ATTR = re.compile(r'\b(width|style)="')
CLASS_ATTR = re.compile(r'\bclass="([^"]*)"')
RASTER_SUFFIXES = (".webp", ".png")

# The class the style sheet uses to center a sized figure and cap it at the column width.
FIGURE_CLASS = "figure"

# max-div has a capability record like every compared tool, because both the hero table and the
# comparison page need its row — but the third-party solver profiles deliberately do not
# profile the package those tools are being compared against. Omitting it from the nav is not
# enough: MkDocs builds every page under the docs tree regardless, so it would still get a URL
# and land in site search. Dropping the file here is what actually keeps it unpublished.
UNPUBLISHED_RECORDS = ("solvers/max-div.md",)


def on_files(files, config):
    """Drop the records that exist as data but are not published as pages."""
    return files.__class__([f for f in files if f.src_uri not in UNPUBLISHED_RECORDS])


# =================================================================================================
#  README image paths
# =================================================================================================
def _relative_prefix(page_url: str) -> str:
    """Return the `../` chain that walks from a page back up to the site root.

    Deliberately not MkDocs' own `get_relative_url`: importing mkdocs here would make the test
    suite depend on the documentation extra, which the test matrix does not install. Counting
    separators is equivalent for the site-root-relative targets this module produces, and holds
    whether or not the site is built with directory URLs.
    """
    return "../" * page_url.count("/")


def _re_anchor_readme_paths(html: str, page_url: str) -> str:
    """Re-anchor repo-root-relative image paths onto the rendered page."""
    prefix = _relative_prefix(page_url)

    def rewrite(match: re.Match) -> str:
        attribute, path = match.group(1), match.group(2)
        return f'{attribute}="{prefix}{path}"'

    return DOCS_RELATIVE_ATTR.sub(rewrite, html)


# =================================================================================================
#  Figure display width
# =================================================================================================
def figure_dpi() -> float:
    """Return `savefig.dpi` from the docs style sheet, the resolution every raster figure is rendered at."""
    match = re.search(r"^savefig\.dpi\s*:\s*([\d.]+)", STYLE_SHEET.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError(f"{STYLE_SHEET} sets no savefig.dpi")
    return float(match.group(1))


def image_pixel_width(path: Path) -> int | None:
    """Return the pixel width of a webp or png file from its header, or None for any other content.

    Parsed by hand: neither the docs nor the test dependency groups install an image library, and
    the two container formats put the width within the first few dozen bytes.
    """
    header = path.read_bytes()[:30]
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">I", header[16:20])[0]
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        chunk = header[12:16]
        if chunk == b"VP8 ":  # lossy: 14-bit width after the 3-byte frame tag and 3-byte start code
            return struct.unpack("<H", header[26:28])[0] & 0x3FFF
        if chunk == b"VP8L":  # lossless: 14-bit width right after the one-byte signature
            return (struct.unpack("<I", header[21:25])[0] & 0x3FFF) + 1
        if chunk == b"VP8X":  # extended: 24-bit canvas width, stored minus one
            return int.from_bytes(header[24:27], "little") + 1
    return None


def _figure_path(docs_dir: Path, page_url: str, src: str) -> Path:
    """Return the docs-tree file an image `src` names.

    By the time the hook runs, MkDocs has already rewritten every image link to be relative to the
    page's URL (a directory under directory URLs, a file otherwise), so the source file is found
    by walking that URL-relative path from the docs directory, not from the page's markdown file.
    """
    page_directory = page_url if page_url.endswith("/") else posixpath.dirname(page_url) + "/"
    return docs_dir / posixpath.normpath(posixpath.join(page_directory, unquote(src)))


def _sized_img_tag(tag: str, docs_dir: Path, page_url: str, dpi: float) -> str:
    """Return the tag with a `width` at its design size and the figure class.

    A tag that is hand-sized, remote, vector, or missing on disk is returned unchanged.
    """
    if HAND_SIZED_ATTR.search(tag):
        return tag
    src = SRC_ATTR.search(tag)
    if src is None or "://" in src.group(1):
        return tag
    path = _figure_path(docs_dir, page_url, src.group(1))
    if path.suffix.lower() not in RASTER_SUFFIXES or not path.is_file():
        return tag
    pixels = image_pixel_width(path)
    if pixels is None:
        return tag
    css_width = round(pixels * CSS_PIXELS_PER_INCH / dpi)
    if CLASS_ATTR.search(tag):
        tag = CLASS_ATTR.sub(lambda m: f'class="{m.group(1)} {FIGURE_CLASS}"', tag, count=1)
        added = f' width="{css_width}"'
    else:
        added = f' class="{FIGURE_CLASS}" width="{css_width}"'
    # Python-Markdown emits self-closing `<img ... />`; the new attributes go before that ending.
    ending = " />" if tag.endswith("/>") else ">"
    return tag[: -len(ending)].rstrip() + added + ending


def _size_figures(html: str, docs_dir: Path, page_url: str, dpi: float) -> str:
    """Give every raster figure on the page its design width."""
    return IMG_TAG.sub(lambda m: _sized_img_tag(m.group(0), docs_dir, page_url, dpi), html)


# =================================================================================================
#  Hook entry point
# =================================================================================================
def on_page_content(html: str, *, page, config, files) -> str:
    """Re-anchor the README's image paths, and size the raster figures of every page but the home page."""
    html = _re_anchor_readme_paths(html, page.url)
    if page.url == "":
        return html
    return _size_figures(html, Path(config["docs_dir"]), page.url, figure_dpi())
