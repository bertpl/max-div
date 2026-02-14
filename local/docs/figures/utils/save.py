from pathlib import Path

from matplotlib.pyplot import Figure


def save_fig(fig: Figure, filepath: Path, tgt_pixels: int | None = None):

    # apply styles not settable in Matplotlib style files
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg:
            leg.get_title().set_fontweight("semibold")
            leg._legend_box.align = "left"

    # determine file format and default pixel target
    file_suffix = filepath.suffix.lower()
    if tgt_pixels is None:
        if file_suffix == ".webp":
            tgt_pixels = 40_000_000
        elif file_suffix == ".png":
            tgt_pixels = 10_000_000
        else:
            tgt_pixels = 10_000_000  # default fallback

    # determine appropriate DPI for target # of pixels
    w, h = fig.get_size_inches()
    dpi = round((tgt_pixels / (w * h)) ** 0.5)

    # build save kwargs based on format
    save_kwargs = dict(
        bbox_inches="tight",
        dpi=dpi,
    )

    if file_suffix == ".webp":
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
