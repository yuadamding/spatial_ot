from __future__ import annotations

import torch


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    w = weights.clamp_min(0.0)
    total = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return w / total


def _log_normalized_weights(weights: torch.Tensor) -> torch.Tensor:
    """Return log normalized weights while preserving exact zero mass."""

    normalized = _normalize_weights(weights)
    return torch.where(
        normalized > 0,
        torch.log(normalized.clamp_min(1e-45)),
        torch.full_like(normalized, -torch.inf),
    )


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
    log_a = _log_normalized_weights(source_weights.to(device=cost.device, dtype=cost.dtype))
    log_b = _log_normalized_weights(target_weights.to(device=cost.device, dtype=cost.dtype))
    log_k = -cost / eps
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(max(int(n_iters), 1)):
        log_u = log_a - torch.logsumexp(log_k + log_v[:, None, :], dim=2)
        log_v = log_b - torch.logsumexp(log_k.transpose(1, 2) + log_u[:, None, :], dim=2)
    plan = torch.exp(log_u[:, :, None] + log_k + log_v[:, None, :])
    return torch.sum(plan * cost, dim=(1, 2))


def pairwise_sqdist_block(tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> torch.Tensor:
    """Return squared distances with shape A x B x L_a x L_b.

    This uses a Gram expansion instead of materializing the much larger
    A x B x L_a x L_b x D broadcasted difference tensor.
    """

    if tokens_a.ndim != 3 or tokens_b.ndim != 3:
        raise ValueError("tokens_a and tokens_b must have shape (batch, support, dim).")
    if tokens_a.shape[2] != tokens_b.shape[2]:
        raise ValueError("tokens_a and tokens_b must have the same feature dimension.")
    a2 = torch.sum(tokens_a * tokens_a, dim=-1)[:, None, :, None]
    b2 = torch.sum(tokens_b * tokens_b, dim=-1)[None, :, None, :]
    dot = torch.einsum("ald,bmd->ablm", tokens_a, tokens_b)
    return (a2 + b2 - 2.0 * dot).clamp_min_(0.0)


def pairwise_sqdist_self(tokens: torch.Tensor) -> torch.Tensor:
    """Return within-measure squared distances with shape B x L x L."""

    if tokens.ndim != 3:
        raise ValueError("tokens must have shape (batch, support, dim).")
    x2 = torch.sum(tokens * tokens, dim=-1)
    dot = torch.bmm(tokens, tokens.transpose(1, 2))
    return (x2[:, :, None] + x2[:, None, :] - 2.0 * dot).clamp_min_(0.0)


def sinkhorn_self_cost_batch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    *,
    epsilon: float = 0.05,
    n_iters: int = 50,
) -> torch.Tensor:
    """Compute entropy-plan self transport costs for a batch of measures."""

    cost = pairwise_sqdist_self(tokens)
    return batched_sinkhorn_cost(cost, weights, weights, epsilon=epsilon, n_iters=n_iters)


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
    cost = pairwise_sqdist_block(tokens_a, tokens_b).reshape(
        a_count * b_count,
        support_a,
        support_b,
    )
    a = weights_a[:, None, :].expand(a_count, b_count, support_a).reshape(
        a_count * b_count, support_a
    )
    b = weights_b[None, :, :].expand(a_count, b_count, support_b).reshape(
        a_count * b_count, support_b
    )
    out = batched_sinkhorn_cost(cost, a, b, epsilon=epsilon, n_iters=n_iters)
    return out.reshape(a_count, b_count)
