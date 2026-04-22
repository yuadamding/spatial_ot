from __future__ import annotations

"""Backward-compatible facade for legacy preprocessing utilities."""

from .legacy.preprocessing import *  # noqa: F401,F403
from .legacy.preprocessing import _extract_cell_types, _resolve_raw_counts, _spatial_grid_subset

__all__ = [name for name in globals() if not name.startswith("__")]
