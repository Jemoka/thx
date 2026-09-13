"""CPU plotting helpers with minimalist Seaborn styling following jemoka.com/design."""

from collections.abc import Callable
from typing import Any

import matplotlib as mpl
from cycler import cycler
from matplotlib.figure import Figure
from matplotlib.container import BarContainer
import numpy as np
import seaborn as sns

BLUE = "#37A5BE"
GREEN = "#26A671"
ORANGE = "#F27200"
RED = "#FF3900"
PALETTE = [BLUE, ORANGE, GREEN, RED]

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 13,
    "axes.labelsize": 14.5,
    "axes.labelpad": 12,
    "axes.titlesize": 15,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "xtick.major.pad": 9,
    "ytick.major.pad": 6,
    "legend.fontsize": 11.5,
    "legend.title_fontsize": 11.5,
    "legend.frameon": False,
    "text.color": "#2F3338",
    "axes.labelcolor": "#2F3338",
    "axes.edgecolor": "#929292",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.prop_cycle": cycler(color=PALETTE),
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "lines.linewidth": 1.55,
    "patch.linewidth": 0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 150,
    "figure.constrained_layout.w_pad": 0.08,
    "figure.constrained_layout.h_pad": 0.08,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _plot(
    draw: Callable[..., Any],
    data: Any,
    *,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (5.35, 3.6),
    **kwargs: Any,
) -> Figure:
    with mpl.rc_context({**sns.axes_style("white"), **STYLE}):
        figure = Figure(figsize=figsize, layout="constrained")
        ax = figure.subplots()
        draw(data=data, ax=ax, **kwargs)
        ax.set_title(title, loc="left", pad=14)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        return figure


def line(data: Any = None, **kwargs: Any) -> Figure:
    """Draw observations as lines; no implicit averaging or confidence band."""
    return _plot(sns.lineplot, data, **{"estimator": None, "errorbar": None, **kwargs})


def scatter(data: Any = None, **kwargs: Any) -> Figure:
    """Draw points without outlines; accepts Seaborn x/y/hue arguments."""
    return _plot(
        sns.scatterplot, data, **{"s": 24, "linewidth": 0, "alpha": 0.85, **kwargs}
    )


def heatmap(data: Any, **kwargs: Any) -> Figure:
    """Draw a continuous matrix without cell borders; override cmap as needed."""
    figure = _plot(
        sns.heatmap,
        data,
        **{
            "cmap": sns.light_palette(BLUE, as_cmap=True),
            "linewidths": 0,
            "figsize": (5.35, 4.2),
            "xticklabels": max(1, int(np.ceil(np.shape(data)[1] / 6))),
            "yticklabels": max(1, int(np.ceil(np.shape(data)[0] / 6))),
            **kwargs,
        },
    )
    figure.axes[0].tick_params(axis="both", labelrotation=0)
    return figure


def bar(
    data: Any = None, *, labels: bool = True, fmt: str = ".3g", **kwargs: Any
) -> Figure:
    """Draw bars with direct value labels, headroom, and no automatic error bars."""
    figure = _plot(
        sns.barplot,
        data,
        **{
            "errorbar": None,
            "saturation": 1,
            "edgecolor": "none",
            "width": 0.62,
            **kwargs,
        },
    )
    ax = figure.axes[0]
    for container in ax.containers:
        if isinstance(container, BarContainer):
            if labels:
                ax.bar_label(
                    container,
                    fmt="{:" + fmt + "}",
                    padding=5,
                    fontsize=11.8,
                    fontweight=600,
                    fontfamily="DejaVu Sans",
                    color="#2F3338",
                )
            ax.margins(**{"y" if container.orientation == "vertical" else "x": 0.18})
    return figure
