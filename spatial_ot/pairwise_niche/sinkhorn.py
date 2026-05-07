from __future__ import annotations

import torch


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    w = weights.clamp_min(0.0)
    total = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return w / total


def batched_sinkhorn_cost(
    cost: torch.Tensor,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    epsilon: float = 0.05,
    n_iters: int = 50,
) -> torch.Tensor:
    """Return transport costs under entropy-regularized Sinkhorn plans."""

    if cost.ndim != 3:
        raise ValueError("cost must have shape (batch, n_source, n_target).")
    eps = max(float(epsilon), 1e-6)
    a = _normalize_weights(source_weights.to(device=cost.device, dtype=cost.dtype))
    b = _normalize_weights(target_weights.to(device=cost.device, dtype=cost.dtype))
    log_a = torch.log(a.clamp_min(1e-12))
    log_b = torch.log(b.clamp_min(1e-12))
    log_k = -cost / eps
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(max(int(n_iters), 1)):
        log_u = log_a - torch.logsumexp(log_k + log_v[:, None, :], dim=2)
        log_v = log_b - torch.logsumexp(log_k.transpose(1, 2) + log_u[:, None, :], dim=2)
    plan = torch.exp(log_u[:, :, None] + log_k + log_v[:, None, :])
    return torch.sum(plan * cost, dim=(1, 2))


def _pairwise_cost(tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> torch.Tensor:
    diff = tokens_a[:, None, :, None, :] - tokens_b[None, :, None, :, :]
    return diff.pow(2).sum(dim=-1)


def sinkhorn_ot_block(
    tokens_a: torch.Tensor,
    weights_a: torch.Tensor,
    tokens_b: torch.Tensor,
    weights_b: torch.Tensor,
    *,
    epsilon: float = 0.05,
    n_iters: int = 50,
) -> torch.Tensor:
    """Compute an A x B block of balanced Sinkhorn OT costs."""

    if tokens_a.ndim != 3 or tokens_b.ndim != 3:
        raise ValueError("tokens_a and tokens_b must have shape (batch, support, dim).")
    a_count, support_a, _ = tokens_a.shape
    b_count, support_b, _ = tokens_b.shape
    cost = _pairwise_cost(tokens_a, tokens_b).reshape(a_count * b_count, support_a, support_b)
    a = weights_a[:, None, :].expand(a_count, b_count, support_a).reshape(
        a_count * b_count, support_a
    )
    b = weights_b[None, :, :].expand(a_count, b_count, support_b).reshape(
        a_count * b_count, support_b
    )
    out = batched_sinkhorn_cost(cost, a, b, epsilon=epsilon, n_iters=n_iters)
    return out.reshape(a_count, b_count)
