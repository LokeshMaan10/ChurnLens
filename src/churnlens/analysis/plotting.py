"""Matplotlib styling and chart helpers for report figures.

Colours follow the dataviz reference palette: categorical slots in fixed order
(slot 1 blue, slot 2 orange, ...), a single-hue blue ramp for magnitude, text in
ink tokens rather than series colours, recessive grid and axes. Static PNGs have
no hover layer, so key values are direct-labelled and every figure has a
CSV/markdown table twin in reports/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Categorical slots, fixed order (validated adjacent-pair order).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
          "#e34948"]
KEPT, CHURNED = SERIES[0], SERIES[1]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)


def apply_style() -> None:
    """Global rcParams: light surface, recessive axes, thin marks."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY, "text.color": TEXT_PRIMARY, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": TEXT_PRIMARY,
        "lines.linewidth": 2, "legend.frameon": False, "axes.axisbelow": True,
    })


def finish(fig: plt.Figure, title: str, provenance: str, path: Path, dpi: int) -> Path:
    """Add title + provenance footer (mode/sample/snapshot), save, close."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY)
    # Below the axes area (negative y); bbox_inches="tight" grows the canvas to include it,
    # so the footer never collides with x-axis labels.
    fig.text(0.01, -0.01, provenance, ha="left", va="top", fontsize=7.5, color=TEXT_SECONDARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def rate_bars(ax: plt.Axes, labels: list[str], rates: list[float], los: list[float],
              his: list[float], ns: list[int], overall: float, title: str) -> None:
    """Horizontal churn-rate bars with 95% CI whiskers, n in the label, overall reference."""
    y = list(range(len(labels)))[::-1]
    ax.barh(y, [r * 100 for r in rates], height=0.6, color=SERIES[0], edgecolor=SURFACE,
            linewidth=2)
    ax.errorbar([r * 100 for r in rates], y,
                xerr=[[(r - lo) * 100 for r, lo in zip(rates, los, strict=True)],
                      [(hi - r) * 100 for r, hi in zip(rates, his, strict=True)]],
                fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1, capsize=2)
    ax.axvline(overall * 100, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.set_yticks(y, [f"{lab}  (n={n:,})" for lab, n in zip(labels, ns, strict=True)])
    for yi, hi, r in zip(y, his, rates, strict=True):
        ax.text(hi * 100 + 0.6, yi, f"{r * 100:.1f}%", va="center", fontsize=7.5,
                color=TEXT_PRIMARY)
    ax.set_xlabel("churn rate (%)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
