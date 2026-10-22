# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Apply a consistent, readable Matplotlib style to research figures."""

import matplotlib as mpl


# =============================== 1. Plot Style ===============================
def plot_style():
    """Set the standard scientific plotting defaults for this process.

    Updates Matplotlib's global rcParams for subsequent plots.
    Existing figures are not automatically restyled.
    """
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12.5,
            "axes.titleweight": "normal",
            "axes.labelsize": 11.5,
            "axes.labelweight": "normal",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 100,
        }
    )
