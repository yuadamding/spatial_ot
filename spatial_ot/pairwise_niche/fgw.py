from __future__ import annotations

import torch


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    w = weights.clamp_min(0.0)
    total = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return w / total


def _sinkhorn_plan(
    cost: torch.Tensor,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    epsilon: float,
    n_iters: int,
) -> torch.Tensor:
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
    return torch.exp(log_u[:, :, None] + log_k + log_v[:, None, :])


def _pairwise_feature_cost(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    diff = features_a[:, None, :, None, :] - features_b[None, :, None, :, :]
    return diff.pow(2).sum(dim=-1)


def _structure_cost(
    structure_a: torch.Tensor,
    structure_b: torch.Tensor,
    plan: torch.Tensor,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    a = _normalize_weights(source_weights.to(device=plan.device, dtype=plan.dtype))
    b = _normalize_weights(target_weights.to(device=plan.device, dtype=plan.dtype))
    left = torch.bmm(structure_a.pow(2), a[:, :, None]).squeeze(-1)
    right = torch.bmm(structure_b.pow(2), b[:, :, None]).squeeze(-1)
    cross = torch.bmm(torch.bmm(structure_a, plan), structure_b.transpose(1, 2))
    return (left[:, :, None] + right[:, None, :] - 2.0 * cross).clamp_min(0.0)


def batched_fused_gromov_wasserstein_cost(
    feature_cost: torch.Tensor,
    structure_a: torch.Tensor,
    structure_b: torch.Tensor,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    alpha: float = 0.5,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 30,
    fgw_iters: int = 5,
) -> torch.Tensor:
    """Batched entropic fused Gromov-Wasserstein cost for fixed-size local graphs.

    The feature term compares node attributes; the structure term compares
    intra-neighborhood graph distance matrices through the transport plan.
    """

    if feature_cost.ndim != 3:
        raise ValueError("feature_cost must have shape (batch, n_source, n_target).")
    if structure_a.ndim != 3 or structure_b.ndim != 3:
        raise ValueError("structure matrices must have shape (batch, support, support).")
    if feature_cost.shape[0] != structure_a.shape[0] or feature_cost.shape[0] != structure_b.shape[0]:
        raise ValueError("feature and structure batches must align.")
    blend = float(min(max(alpha, 0.0), 1.0))
    a = _normalize_weights(source_weights.to(device=feature_cost.device, dtype=feature_cost.dtype))
    b = _normalize_weights(target_weights.to(device=feature_cost.device, dtype=feature_cost.dtype))
    plan = a[:, :, None] * b[:, None, :]
    fused_cost = feature_cost
    for _ in range(max(int(fgw_iters), 1)):
        struct_cost = _structure_cost(structure_a, structure_b, plan, a, b)
        fused_cost = (1.0 - blend) * feature_cost + blend * struct_cost
        plan = _sinkhorn_plan(
            fused_cost,
            a,
            b,
            epsilon=float(epsilon),
            n_iters=int(sinkhorn_iters),
        )
    struct_cost = _structure_cost(structure_a, structure_b, plan, a, b)
    feature_value = torch.sum(plan * feature_cost, dim=(1, 2))
    structure_value = torch.sum(plan * struct_cost, dim=(1, 2))
    return (1.0 - blend) * feature_value + blend * structure_value


def fused_gromov_wasserstein_block(
    features_a: torch.Tensor,
    structures_a: torch.Tensor,
    weights_a: torch.Tensor,
    features_b: torch.Tensor,
    structures_b: torch.Tensor,
    weights_b: torch.Tensor,
    *,
    alpha: float = 0.5,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 30,
    fgw_iters: int = 5,
) -> torch.Tensor:
    """Compute an A x B block of graph-topology FGW distances."""

    if features_a.ndim != 3 or features_b.ndim != 3:
        raise ValueError("features_a and features_b must have shape (batch, support, dim).")
    a_count, support_a, _ = features_a.shape
    b_count, support_b, _ = features_b.shape
    feature_cost = _pairwise_feature_cost(features_a, features_b).reshape(
        a_count * b_count,
        support_a,
        support_b,
    )
    struct_a = structures_a[:, None, :, :].expand(
        a_count,
        b_count,
        support_a,
        support_a,
    ).reshape(a_count * b_count, support_a, support_a)
    struct_b = structures_b[None, :, :, :].expand(
        a_count,
        b_count,
        support_b,
        support_b,
    ).reshape(a_count * b_count, support_b, support_b)
    a = weights_a[:, None, :].expand(a_count, b_count, support_a).reshape(
        a_count * b_count,
        support_a,
    )
    b = weights_b[None, :, :].expand(a_count, b_count, support_b).reshape(
        a_count * b_count,
        support_b,
    )
    out = batched_fused_gromov_wasserstein_cost(
        feature_cost,
        struct_a,
        struct_b,
        a,
        b,
        alpha=float(alpha),
        epsilon=float(epsilon),
        sinkhorn_iters=int(sinkhorn_iters),
        fgw_iters=int(fgw_iters),
    )
    return out.reshape(a_count, b_count)


__all__ = [
    "batched_fused_gromov_wasserstein_cost",
    "fused_gromov_wasserstein_block",
]
