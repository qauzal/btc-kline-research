"""One consistent chart style for every notebook."""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# sequential: one hue, light -> dark
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "figure.figsize": (9, 3.6),
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "text.color": INK,
        "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False, "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, AQUA, YELLOW]),
        "font.family": "sans-serif",
    })
