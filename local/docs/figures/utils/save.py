from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure


def save_fig(fig: Figure, filepath: Path):
    """Save a figure at the docs style sheet's `savefig.dpi`, so its pixel size follows its size in inches.

    The documentation build sizes every raster image from that one dpi (see `scripts/mkdocs_hooks.py`).
    """
    # apply styles not settable in Matplotlib style files
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg:
            leg.get_title().set_fontweight("semibold")
            leg._legend_box.align = "left"

    # build save kwargs based on format
    save_kwargs = dict(
        bbox_inches="tight",
        dpi=plt.rcParams["savefig.dpi"],
    )

    if filepath.suffix.lower() == ".webp":
        save_kwargs |= dict(
            format="webp",
            pil_kwargs=dict(
                lossless=True,  # lossless, no quality loss
                quality=100,  # highest compression effort
                method=1,  # higher methods are slower but not significantly smaller
            ),
        )

    # save figure
    fig.savefig(filepath, **save_kwargs)
