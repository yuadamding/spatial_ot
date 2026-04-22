from __future__ import annotations

import torch
import torch.nn.functional as F


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("off_diagonal expects a square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def variance_loss(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return torch.relu(1.0 - std).mean()


def decorrelation_loss(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(z.shape[0], 1)
    return off_diagonal(cov).pow(2).mean()


def cross_correlation_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    if z_a.shape[0] < 2 or z_b.shape[0] < 2:
        return z_a.new_tensor(0.0)
    za = z_a - z_a.mean(dim=0, keepdim=True)
    zb = z_b - z_b.mean(dim=0, keepdim=True)
    za = za / torch.sqrt(za.var(dim=0, unbiased=False, keepdim=True) + 1e-4)
    zb = zb / torch.sqrt(zb.var(dim=0, unbiased=False, keepdim=True) + 1e-4)
    cross = (za.T @ zb) / max(z_a.shape[0], 1)
    return cross.pow(2).mean()


def edge_contrastive_loss(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    temperature: float = 0.2,
    max_edges: int = 4096,
) -> torch.Tensor:
    if z.shape[0] < 2 or edge_index.numel() == 0:
        return z.new_tensor(0.0)
    src, dst = edge_index
    if src.numel() > max_edges:
        keep = torch.randperm(src.numel(), device=z.device)[:max_edges]
        src = src[keep]
        dst = dst[keep]
    z_norm = F.normalize(z, dim=-1)
    pos = torch.sum(z_norm[src] * z_norm[dst], dim=-1) / max(float(temperature), 1e-6)
    neg_dst = torch.randint(0, z.shape[0], size=(src.numel(),), device=z.device)
    neg = torch.sum(z_norm[src] * z_norm[neg_dst], dim=-1) / max(float(temperature), 1e-6)
    return F.softplus(neg - pos).mean()
