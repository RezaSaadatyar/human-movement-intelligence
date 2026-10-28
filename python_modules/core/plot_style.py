# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Apply consistent Matplotlib styles and save reusable research figures."""

from pathlib import Path

import matplotlib as mpl


# =============================== 1. Plot Style ===============================
def plot_style(
    font_family="DejaVu Sans",
    title_size=12.5,
    title_weight="normal",
    label_size=11.5,
    label_weight="normal",
    tick_size=11,
    legend_size=10,
    figure_dpi=100,
    figure_size=None,
):
    """Configure plotting defaults for subsequently created figures.

    No-argument calls retain the existing research style. Pass figure_size
    explicitly to change the default size of newly created figures; an
    individual plt.subplots(figsize=...) takes precedence over this setting.
    """
    settings = {
        "font.family": font_family,
        "axes.titlesize": title_size,
        "axes.titleweight": title_weight,
        "axes.labelsize": label_size,
        "axes.labelweight": label_weight,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        "figure.dpi": figure_dpi,
    }
    if figure_size is not None:
        settings["figure.figsize"] = figure_size
    mpl.rcParams.update(settings)


# ============================== 2. Figure Export ==============================
def save_figure(figure, destination, dpi=300):
    """Save an existing Matplotlib figure without changing its plot content.

    Args:
        figure (matplotlib.figure.Figure): Completed figure to save.
        destination (str | pathlib.Path): Full destination filename.
        dpi (int): PNG or other raster export resolution (default 300).

    Returns:
        pathlib.Path: Path to the saved figure.

    This function does not automatically commit files to GitHub.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    return destination
