"""matplotlib 统一样式：中文字体和配色。"""
import matplotlib.pyplot as plt

# 分类配色按固定顺序使用（已通过色盲区分度校验）
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "D"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"


def setup_matplotlib():
    """设置中文字体和统一的图表外观。

    注意：标签里的 $ 会被 matplotlib 当成数学公式符号，需要写成 \\$。
    """
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def smd_plot(smd, threshold=0.1, title=None):
    """平衡性图：每行一个特征，横轴是 SMD（标准化均值差），虚线是 ±threshold。

    smd：pandas Series，索引为特征名。
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(smd) + 1.4))
    y = np.arange(len(smd))[::-1]
    ax.scatter(smd.to_numpy(), y, s=42, color=SERIES[0], edgecolor=SURFACE, linewidth=1, zorder=3)
    for x in (-threshold, threshold):
        ax.axvline(x, color=MUTED, linestyle="--", linewidth=1, zorder=1)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1, zorder=2)
    lim = max(threshold * 1.6, float(np.nanmax(np.abs(smd.to_numpy()))) * 1.2)
    ax.set_xlim(-lim, lim)
    ax.set_yticks(y, [str(s).replace("$", r"\$") for s in smd.index])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"标准化均值差 SMD（实验组 − 对照组）；虚线为 ±{threshold}")
    if title:
        ax.set_title(title, loc="left", color=INK)
    return fig, ax
