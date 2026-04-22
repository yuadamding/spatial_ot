from __future__ import annotations

"""Backward-compatible facade for the legacy communication module."""

from .legacy.communication import CommunicationResult, _masked_sinkhorn, fit_communication_flows

__all__ = [
    "CommunicationResult",
    "_masked_sinkhorn",
    "fit_communication_flows",
]
