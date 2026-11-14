"""Shared matplotlib style: one palette for every figure in the report (same as A1)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PHOTO_DPI = 130  # figures made of photographs are saved at a lower resolution to keep the PDF small
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# Categorical slots (fixed order): CNN always blue, Transformer always orange
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6"
FAMILY_COLOR = {"CNN": BLUE, "Transformer": ORANGE}
RUN_ORDER = ["deeplabv3plus_r50", "resnet50_mlp", "segformer_b0", "segformer_b1", "segformer_b2",
             "segformer_b0_scratch"]
RUN_LABEL = {
    "deeplabv3plus_r50": "DeepLabV3+ R50",
    "resnet50_mlp": "ResNet-50 + MLP",
    "segformer_b0": "SegFormer-B0",
    "segformer_b1": "SegFormer-B1",
    "segformer_b2": "SegFormer-B2",
    "segformer_b0_scratch": "SegFormer-B0 (scratch)",
}
# CNN runs in blues, Transformer runs in oranges (darker = larger), scratch run in grey
RUN_COLOR = {
    "deeplabv3plus_r50": BLUE,
    "resnet50_mlp": "#86b6ef",
    "segformer_b0": "#f5b08e",
    "segformer_b1": "#f08a5c",
    "segformer_b2": ORANGE,
    "segformer_b0_scratch": MUTED,
}
RUN_STYLE = {"segformer_b0_scratch": "--"}
MAIN_CNN, MAIN_VIT = "deeplabv3plus_r50", "segformer_b2"

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_orange", ["#184f95", "#2a78d6", "#9ec5f4", "#f0efec", "#f5b08e", "#eb6834", "#a8401a"])
HEAT = LinearSegmentedColormap.from_list(
    "heat", ["#0b0b0b", "#0d366b", "#256abf", "#3987e5", "#86b6ef", "#fde9a8", "#ffffff"])


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
