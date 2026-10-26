"""Shared chart styling used by round_plots.py and plot_selected.py."""

LINE_COLOR = "#2a78d6"      # categorical slot 1 (blue)
LINE_COLOR_2 = "#eb6834"    # categorical slot 2 (orange)
STATUS_GOOD = "#0ca30c"     # status palette: good
STATUS_CRITICAL = "#d03b3b"  # status palette: critical
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
MUTED_TEXT = "#898781"
PRIMARY_TEXT = "#0b0b0b"


def style_axis(ax):
    ax.set_facecolor("#fcfcfb")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED_TEXT, labelsize=8)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
