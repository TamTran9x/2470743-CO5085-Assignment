"""Shared matplotlib style: one palette for every figure in the report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# Categorical slots (fixed order): CNN always slot 1, Transformer always slot 2
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
FAMILY_COLOR = {"CNN": BLUE, "Transformer": ORANGE}
ARCH_COLOR = {"resnet50": BLUE, "deit_s": ORANGE}
ARCH_LABEL = {"resnet50": "ResNet-50 (CNN)", "deit_s": "DeiT-S (ViT)"}
MODE_LABEL = {"scratch": "From scratch", "linear": "Freeze backbone", "partial": "Freeze một phần",
              "full": "Full fine-tune"}
MODE_ORDER = ["scratch", "linear", "partial", "full"]

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_red_blue", ["#c0392b", "#e34948", "#f3b2b1", "#f0efec", "#9ec5f4", "#2a78d6", "#184f95"])


def apply():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
