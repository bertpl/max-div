from pathlib import Path

from matplotlib.pyplot import Figure


def save_fig(fig: Figure, filepath: Path, tgt_pixels: int = 10_000_000):
    w, h = fig.get_size_inches()
    dpi = round((tgt_pixels / (w * h)) ** 0.5)

    fig.savefig(filepath, bbox_inches="tight", dpi=dpi)
