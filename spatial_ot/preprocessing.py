"""Backward-compatible facade for legacy preprocessing utilities."""

from __future__ import annotations

from .legacy.preprocessing import *  # noqa: F401,F403
from .legacy.preprocessing import (
    _extract_cell_types as _extract_cell_types,
    _resolve_raw_counts as _resolve_raw_counts,
    _spatial_grid_subset as _spatial_grid_subset,
)

__all__ = [name for name in globals() if not name.startswith("__")]
