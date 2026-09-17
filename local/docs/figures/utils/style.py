import matplotlib.pyplot as plt

from max_div._core.plotting.helpers import figure_style_path


def set_docs_style():
    plt.style.use(figure_style_path())
