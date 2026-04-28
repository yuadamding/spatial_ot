"""Backward-compatible facade for the active multilevel OT namespace."""

from __future__ import annotations

from .multilevel import *  # noqa: F401,F403
from .multilevel import __all__ as _multilevel_all

__all__ = list(_multilevel_all)
