"""Backward-compatible facade for the legacy communication module."""

from __future__ import annotations

from .legacy.communication import CommunicationResult, _masked_sinkhorn, fit_communication_flows

__all__ = [
    "CommunicationResult",
    "_masked_sinkhorn",
    "fit_communication_flows",
]
