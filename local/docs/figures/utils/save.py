from pathlib import Path

from matplotlib.pyplot import Figure


def save_fig(fig: Figure, filepath: Path, tgt_pixels: int = 10_000_000):

    # apply styles not settable in Matplotlib style files
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg:
            leg.get_title().set_fontweight("semibold")
            leg._legend_box.align = "left"

    # determine appropriate DPI for target # of pixels
    w, h = fig.get_size_inches()
    dpi = round((tgt_pixels / (w * h)) ** 0.5)

    # save
    fig.savefig(filepath, bbox_inches="tight", dpi=dpi)
