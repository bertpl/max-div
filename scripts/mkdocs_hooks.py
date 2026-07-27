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

# `src` and `srcset` both carry paths in the README's <picture> element; a plain `src`-only
# rule would silently leave the dark variant broken.
DOCS_RELATIVE_ATTR = re.compile(r'\b(src|srcset)="docs/([^"]+)"')

# max-div has a capability record like every compared tool, because both the hero table and the
# comparison page need its row — but the third-party solver reference deliberately does not
# profile the package those tools are being compared against. Omitting it from the nav is not
# enough: MkDocs builds every page under the docs tree regardless, so it would still get a URL
# and land in site search. Dropping the file here is what actually keeps it unpublished.
UNPUBLISHED_RECORDS = ("solvers/max-div.md",)


def on_files(files, config):
    """Drop the records that exist as data but are not published as pages."""
    return files.__class__([f for f in files if f.src_uri not in UNPUBLISHED_RECORDS])


def _relative_prefix(page_url: str) -> str:
    """Return the `../` chain that walks from a page back up to the site root.

    Deliberately not MkDocs' own `get_relative_url`: importing mkdocs here would make the test
    suite depend on the documentation extra, which the test matrix does not install. Counting
    separators is equivalent for the site-root-relative targets this module produces, and holds
    whether or not the site is built with directory URLs.
    """
    return "../" * page_url.count("/")


def on_page_content(html: str, *, page, config, files) -> str:
    """Re-anchor repo-root-relative image paths onto the rendered page."""
    prefix = _relative_prefix(page.url)

    def rewrite(match: re.Match) -> str:
        attribute, path = match.group(1), match.group(2)
        return f'{attribute}="{prefix}{path}"'

    return DOCS_RELATIVE_ATTR.sub(rewrite, html)
