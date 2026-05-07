from __future__ import annotations

import re


HIGH_CONTRAST_PALETTE = [
    "#0047ff",
    "#00a651",
    "#e60012",
    "#7a00cc",
    "#00b7c7",
    "#ff00a8",
    "#704214",
    "#111111",
    "#b3de00",
    "#005a32",
    "#ff1493",
    "#4d4dff",
    "#ff7f00",
    "#6a3d9a",
    "#00e5ff",
    "#b15928",
    "#008080",
    "#ff4500",
    "#3b8f00",
    "#c000ff",
    "#0080ff",
    "#ff0050",
    "#6f6f00",
    "#00a0a0",
    "#a50021",
    "#2f4b7c",
    "#9c755f",
    "#bcbd22",
    "#17becf",
    "#8c564b",
]


def _label_sort_key(label: str) -> tuple[int, str]:
    match = re.fullmatch(r"ON(\d+)", str(label))
    if match:
        return int(match.group(1)), str(label)
    return 10**9, str(label)


def assign_high_contrast_colors(labels: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Assign deterministic high-contrast colors to niche labels."""

    ordered = sorted({str(label) for label in labels}, key=_label_sort_key)
    colors: dict[str, str] = {}
    for pos, label in enumerate(ordered):
        colors[label] = HIGH_CONTRAST_PALETTE[pos % len(HIGH_CONTRAST_PALETTE)]
    if "ON12" in colors:
        colors["ON12"] = "#ff7f00"
    return colors


__all__ = ["HIGH_CONTRAST_PALETTE", "assign_high_contrast_colors"]
