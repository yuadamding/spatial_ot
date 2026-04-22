from __future__ import annotations

"""Backward-compatible facade for legacy OT niche utilities."""

from .legacy.ot import *  # noqa: F401,F403
from .legacy.ot import _build_shell_ground

__all__ = [name for name in globals() if not name.startswith("__")]
