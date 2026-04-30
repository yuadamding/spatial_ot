from __future__ import annotations

import torch


def _as_tensor(x, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def _log(x: torch.Tensor) -> torch.Tensor:
    # Use a float32-representable lower bound; 1e-300 underflows to 0 in fp32.
    return torch.log(torch.clamp(x, min=1e-30))


def sinkhorn_balanced_log_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    eps: float,
    *,
    num_iter: int = 600,
    tol: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, bool, float]:
    """Log-domain balanced Sinkhorn in torch, supporting an optional leading batch dim.

    Returns transport plan, total cost <T, C>, converged flag, and final error.

    Uses float32-appropriate tolerance and coarser convergence checks to keep the
    GPU saturated without excessive host<->device sync.
    """
    eps = max(float(eps), 1e-6)
    log_a = _log(a)
    log_b = _log(b)
    log_k = -cost / eps
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    converged = False
    last_err = float("inf")

    for it in range(int(num_iter)):
        g_new = log_b - torch.logsumexp(log_k + f.unsqueeze(-1), dim=-2)
        f_new = log_a - torch.logsumexp(log_k + g_new.unsqueeze(-2), dim=-1)
        if (it & 127) == 127:
            err = (f_new - f).abs().amax().detach()
            err_val = float(err.item())
            last_err = err_val
            f = f_new
            g = g_new
            if err_val < tol:
                converged = True
                break
        else:
            f = f_new
            g = g_new
    log_t = f.unsqueeze(-1) + log_k + g.unsqueeze(-2)
    t = torch.exp(log_t)
    transport_cost = (t * cost).sum(dim=(-1, -2))
    return t, transport_cost, converged, last_err


def sinkhorn_semirelaxed_unbalanced_log_torch(
    a: torch.Tensor,
    beta: torch.Tensor,
    cost: torch.Tensor,
    eps: float,
    rho: float,
    *,
    num_iter: int = 600,
    tol: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, bool, float]:
    """Semi-relaxed unbalanced Sinkhorn with KL regularization in torch.

    Solves approximately:
        min_{T>=0, T 1 = a} <T, C> + eps * KL(T || a b^T) + rho * KL(T^T 1 || beta)

    which corresponds to POT's ``sinkhorn_unbalanced`` with ``reg_m=(inf, rho)`` and
    ``reg_type="kl"``.

    Inputs may be shape (n,) / (m,) / (n, m), or they may all share a leading batch
    dimension (B, n), (B, m), (B, n, m).

    Returns transport plan T, scalar total cost (<T,C> + regularizers), converged,
    last max update magnitude.
    """
    eps = max(float(eps), 1e-6)
    rho = max(float(rho), 1e-6)
    eta = rho / (rho + eps)
    log_a = _log(a)
    log_b = _log(beta)
    log_k = -cost / eps
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    last_err = float("inf")
    converged = False

    for it in range(int(num_iter)):
        # hard row marginal update: f = log a - lse_j(log_k + g)
        f_new = log_a - torch.logsumexp(log_k + g.unsqueeze(-2), dim=-1)
        # relaxed column update: g = eta * (log b - lse_i(log_k + f))
        g_new = eta * (log_b - torch.logsumexp(log_k + f_new.unsqueeze(-1), dim=-2))
        if (it & 127) == 127:
            err = (f_new - f).abs().amax().detach()
            err_val = float(err.item())
            last_err = err_val
            f = f_new
            g = g_new
            if err_val < tol:
                converged = True
                break
        else:
            f = f_new
            g = g_new

    log_t = f.unsqueeze(-1) + log_k + g.unsqueeze(-2)
    t = torch.exp(log_t)

    # Transport cost term.
    transport_cost = (t * cost).sum(dim=(-1, -2))

    # eps * KL(T || a b^T) = eps * sum T (log T - log a - log b) - eps * (sum T - sum a b)
    # Using the generalized KL convention used by POT.
    t_log_t = (t * log_t).sum(dim=(-1, -2))
    t_log_ab = (t * log_a.unsqueeze(-1)).sum(dim=(-1, -2)) + (
        t * log_b.unsqueeze(-2)
    ).sum(dim=(-1, -2))
    mass_t = t.sum(dim=(-1, -2))
    mass_ab = a.sum(dim=-1) * beta.sum(dim=-1)
    kl_t = t_log_t - t_log_ab - mass_t + mass_ab

    # rho * KL(col_mass || beta).
    col_mass = t.sum(dim=-2)
    log_col = _log(col_mass)
    kl_col = (
        (col_mass * (log_col - log_b)).sum(dim=-1)
        - col_mass.sum(dim=-1)
        + beta.sum(dim=-1)
    )

    objective = transport_cost + eps * kl_t + rho * kl_col
    return t, objective, converged, last_err


__all__ = [
    "sinkhorn_balanced_log_torch",
    "sinkhorn_semirelaxed_unbalanced_log_torch",
]
