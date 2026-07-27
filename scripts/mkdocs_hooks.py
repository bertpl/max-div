"""Build hooks for the documentation site.

The docs home page includes README.md, and the README addresses its images with
repo-root-relative paths (`docs/images/...`) because that is what GitHub resolves. The other
two surfaces already correct for this in their own way: PyPI's long description is built by
hatch-fancy-pypi-readme, which rewrites those attributes to tag-pinned absolute URLs (see the
substitutions in pyproject.toml). MkDocs serves the `docs/` directory *as* the site root, so
without a rewrite the same paths land one directory too deep and the images 404.

This hook is that missing layer. It strips the `docs/` prefix and re-anchors what remains
against the including page, so the result is correct wherever the include happens rather than
only on a page that sits at the site root.

It deliberately runs on the rendered page rather than on its markdown: the README is pulled in
by a snippet, which is expanded during markdown conversion. At `on_page_markdown` time the home
page is still a one-line include directive and none of the README's paths exist yet.
"""

import re

from mkdocs.utils import get_relative_url

# `src` and `srcset` both carry paths in the README's <picture> element; a plain `src`-only
# rule would silently leave the dark variant broken.
DOCS_RELATIVE_ATTR = re.compile(r'\b(src|srcset)="docs/([^"]+)"')


def on_page_content(html: str, *, page, config, files) -> str:
    """Re-anchor repo-root-relative image paths onto the rendered page."""

    def rewrite(match: re.Match) -> str:
        attribute, path = match.group(1), match.group(2)
        return f'{attribute}="{get_relative_url(path, page.url)}"'

    return DOCS_RELATIVE_ATTR.sub(rewrite, html)
