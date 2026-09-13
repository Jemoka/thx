"""Scalar transformations and formatting shared by browser plots."""

from bisect import bisect_left
from collections.abc import Callable
from datetime import datetime
from math import isfinite, log10, sqrt


_TWEMA_VIEWPORT_SCALE = 100.0


def transform_points(
    points: tuple[tuple[float, float], ...], log_x: bool, log_y: bool
) -> list[tuple[float, float]]:
    """Transform finite points into the selected linear or log domains."""
    transformed: list[tuple[float, float]] = []
    for x, y in points:
        if (log_x and x <= 0) or (log_y and y <= 0):
            continue
        x = log10(x) if log_x else x
        y = log10(y) if log_y else y
        if isfinite(x) and isfinite(y):
            transformed.append((x, y))
    return transformed


def interpolate(points: list[tuple[float, float]], x: float) -> float | None:
    """Interpolate a connected series at ``x``, clamping to its endpoints."""
    if not points:
        return None
    index = bisect_left([point[0] for point in points], x)
    if index == 0:
        return points[0][1]
    if index == len(points):
        return points[-1][1]
    left, right = points[index - 1 : index + 1]
    span = right[0] - left[0]
    position = 0.0 if span == 0 else (x - left[0]) / span
    return left[1] + position * (right[1] - left[1])


def twema(
    points: tuple[tuple[float, float], ...], smoothing: float
) -> tuple[tuple[float, float], ...]:
    """Apply debiased time-weighted EMA using normalized x spacing."""
    if smoothing == 0 or len(points) < 2:
        return points
    weight = min(sqrt(smoothing), 0.999)
    range_x = points[-1][0] - points[0][0]
    last_y = 0.0
    debias = 0.0
    previous_x = points[0][0]
    result: list[tuple[float, float]] = []
    for x, y in points:
        change_x = 0.0 if range_x == 0 else (x - previous_x) / range_x
        adjusted = weight ** (change_x * _TWEMA_VIEWPORT_SCALE)
        last_y = last_y * adjusted + y
        debias = debias * adjusted + 1.0
        result.append((x, last_y / debias))
        previous_x = x
    return tuple(result)


def format_tick(value: float) -> str:
    """Format an axis value with compact uppercase magnitude suffixes."""
    for scale, suffix in (
        (1e12, "T"),
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "K"),
    ):
        if abs(value) >= scale:
            return f"{value / scale:.4g}{suffix}"
    return f"{value:.4g}"


def format_timestamp(value: float) -> str:
    """Format a stored timestamp in local time for detailed display."""
    return datetime.fromtimestamp(value).isoformat(sep=" ", timespec="milliseconds")


def format_timestamp_tick(value: float, span: float) -> str:
    """Format a compact local-time label appropriate to an axis span."""
    timestamp = datetime.fromtimestamp(value)
    if span < 10:
        return timestamp.strftime("%H:%M:%S.%f")[:-3]
    if span < 60:
        return timestamp.strftime("%H:%M:%S")
    if span < 86_400:
        return timestamp.strftime("%H:%M")
    if span < 15_552_000:
        return timestamp.strftime("%b %d")
    return timestamp.strftime("%Y-%m-%d")


def format_field(key: str, value: float) -> str:
    """Format a plotted field according to its semantic type."""
    return format_timestamp(value) if key == "timestamp" else format_tick(value)


def ticks(
    low: float,
    high: float,
    logarithmic: bool,
    count: int,
    formatter: Callable[[float], str] | None = None,
) -> tuple[tuple[float, str], ...]:
    """Return well-spaced normalized tick positions and original-unit labels."""
    format_value = formatter or format_tick
    positions = [index / (count - 1) for index in range(count)]
    if logarithmic:
        for exponent in range(int(low) + 1, int(high) + 1):
            position = (exponent - low) / max(high - low, 1e-12)
            nearest = min(
                range(count), key=lambda index: abs(positions[index] - position)
            )
            positions[nearest] = position
        positions.sort()
    return tuple(
        (
            position,
            format_value(
                10 ** (low + (high - low) * position)
                if logarithmic
                else low + (high - low) * position
            ),
        )
        for position in positions
    )
