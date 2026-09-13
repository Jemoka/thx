"""Independent Plotly trace and axis projections for terminal-styled charts."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

from theseus.cli.interface.plot_math import format_timestamp_tick, ticks


def trace_options(
    points: Sequence[tuple[float, float]],
    name: str,
    color: str,
    log_x: bool,
    log_y: bool,
    *,
    markers: bool = False,
    opacity: float = 1,
) -> dict[str, object]:
    """Serialize finite transformed points into a Plotly line or marker trace."""
    return {
        "type": "scatter",
        "mode": "markers" if markers or len(points) == 1 else "lines",
        "name": name,
        "x": [10**x if log_x else x for x, _ in points],
        "y": [10**y if log_y else y for _, y in points],
        "line": {"color": color, "width": 1.4},
        "opacity": opacity,
        "marker": {"color": color, "size": 6 if markers else 4},
        "hoverinfo": "skip",
    }


def axis_options(
    key: str | None,
    logarithmic: bool,
    bounds: tuple[float, float] | None,
    preview: bool,
) -> dict[str, object]:
    """Preserve compact magnitude and local timestamp ticks on either axis."""
    options: dict[str, object] = {
        "type": "log" if logarithmic else "linear",
        "showgrid": False,
        "zeroline": False,
        "showline": not preview and bounds is not None,
        "linecolor": "#d5d5d5",
        "showticklabels": not preview and bounds is not None,
        "tickfont": {"color": "#929292"},
        "fixedrange": True,
    }
    if bounds is None:
        return options
    low, high = bounds
    if low == high:
        padding = 0.5 if logarithmic else max(abs(low) * 0.01, 0.5)
        low, high = low - padding, high + padding
    formatter = (
        partial(
            format_timestamp_tick,
            span=10**high - 10**low if logarithmic else high - low,
        )
        if key == "timestamp"
        else None
    )
    marks = ticks(low, high, logarithmic, 5, formatter)
    options.update(
        range=[low, high],
        tickmode="array",
        tickvals=[
            10 ** (low + (high - low) * p) if logarithmic else low + (high - low) * p
            for p, _ in marks
        ],
        ticktext=[text for _, text in marks],
    )
    return options
