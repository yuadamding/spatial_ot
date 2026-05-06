from __future__ import annotations

import torch


def weighted_pairwise_sqdist(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean cost for batched point clouds."""

    return torch.cdist(left, right, p=2.0).pow(2)


def _log_weights(weights: torch.Tensor) -> torch.Tensor:
    neg = torch.full_like(weights, -1.0e9)
    return torch.where(weights > 0.0, torch.log(weights.clamp_min(1.0e-12)), neg)


def sinkhorn_balanced_distance(
    cost: torch.Tensor,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    epsilon: float = 0.05,
    n_iters: int = 30,
) -> torch.Tensor:
    """Balanced entropic OT cost for batches of weighted point clouds.

    Parameters
    ----------
    cost
        Tensor with shape ``(batch, n_source, n_target)``.
    source_weights
        Nonnegative source masses with shape ``(batch, n_source)``.
    target_weights
        Nonnegative target masses with shape ``(batch, n_target)``.
    """

    if cost.ndim != 3:
        raise ValueError("cost must have shape (batch, n_source, n_target).")
    eps = max(float(epsilon), 1e-4)
    a = source_weights.to(dtype=cost.dtype, device=cost.device)
    b = target_weights.to(dtype=cost.dtype, device=cost.device)
    a = a / a.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    b = b / b.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    log_a = _log_weights(a)
    log_b = _log_weights(b)
    log_k = -cost / eps
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(max(int(n_iters), 1)):
        log_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(1), dim=2)
        log_v = log_b - torch.logsumexp(log_k + log_u.unsqueeze(2), dim=1)
    log_plan = log_k + log_u.unsqueeze(2) + log_v.unsqueeze(1)
    plan = torch.exp(log_plan)
    return torch.sum(plan * cost, dim=(1, 2))


def sinkhorn_divergence(
    cost_xy: torch.Tensor,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    cost_xx: torch.Tensor,
    cost_yy: torch.Tensor,
    epsilon: float = 0.05,
    n_iters: int = 30,
) -> torch.Tensor:
    """Debiased Sinkhorn divergence for weighted point clouds.

    The transport cost from :func:`sinkhorn_balanced_distance` is entropically
    biased and can be nonzero for identical multi-support measures. This helper
    subtracts the self-costs and is intended for diagnostics/distillation
    targets, not as a replacement for the current prototype training loss.
    """

    if cost_xx.ndim != 3 or cost_yy.ndim != 3:
        raise ValueError("cost_xx and cost_yy must have shape (batch, n_support, n_support).")
    if cost_xx.shape[0] != cost_xy.shape[0] or cost_yy.shape[0] != cost_xy.shape[0]:
        raise ValueError("self-cost batches must match cost_xy.")
    if cost_xx.shape[1:] != (cost_xy.shape[1], cost_xy.shape[1]):
        raise ValueError("cost_xx must match the source support size.")
    if cost_yy.shape[1:] != (cost_xy.shape[2], cost_xy.shape[2]):
        raise ValueError("cost_yy must match the target support size.")
    ot_xy = sinkhorn_balanced_distance(
        cost_xy,
        source_weights,
        target_weights,
        epsilon=epsilon,
        n_iters=n_iters,
    )
    ot_xx = sinkhorn_balanced_distance(
        cost_xx,
        source_weights,
        source_weights,
        epsilon=epsilon,
        n_iters=n_iters,
    )
    ot_yy = sinkhorn_balanced_distance(
        cost_yy,
        target_weights,
        target_weights,
        epsilon=epsilon,
        n_iters=n_iters,
    )
    return ot_xy - 0.5 * ot_xx - 0.5 * ot_yy


__all__ = [
    "sinkhorn_balanced_distance",
    "sinkhorn_divergence",
    "weighted_pairwise_sqdist",
]
