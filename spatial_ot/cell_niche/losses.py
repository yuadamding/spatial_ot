from __future__ import annotations

import torch
import torch.nn.functional as F


def _variance_loss(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.shape[0] < 2:
        return embedding.new_tensor(0.0)
    std = torch.sqrt(torch.var(embedding, dim=0, unbiased=False) + 1.0e-4)
    return torch.mean(F.relu(0.5 - std))


def _decorrelation_loss(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.shape[0] < 2 or embedding.shape[1] < 2:
        return embedding.new_tensor(0.0)
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(int(embedding.shape[0]) - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return torch.mean(off_diag.pow(2))


def deepshe_loss(
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, torch.Tensor],
    *,
    context_reconstruction_weight: float = 1.0,
    ot_prototype_weight: float = 0.5,
    prototype_balance_weight: float = 0.05,
    variance_weight: float = 0.02,
    decorrelation_weight: float = 0.005,
    ot_temperature: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    embedding = outputs["embedding"]
    if not isinstance(embedding, torch.Tensor):
        raise ValueError("outputs must contain an embedding tensor.")
    total = embedding.new_tensor(0.0)
    parts: dict[str, torch.Tensor] = {}

    if "descriptor_targets" in batch and float(context_reconstruction_weight) > 0.0:
        decoded = outputs["decoded_descriptor"]
        if isinstance(decoded, torch.Tensor):
            context = F.mse_loss(decoded, batch["descriptor_targets"])
            total = total + float(context_reconstruction_weight) * context
            parts["context_reconstruction"] = context

    proto_dist = outputs.get("prototype_distances")
    proto_post = outputs.get("prototype_posterior")
    if isinstance(proto_dist, torch.Tensor) and float(ot_prototype_weight) > 0.0:
        tau = max(float(ot_temperature), 1.0e-6)
        proto_loss = torch.sum(torch.softmax(-proto_dist / tau, dim=1) * proto_dist, dim=1).mean()
        total = total + float(ot_prototype_weight) * proto_loss
        parts["ot_prototype"] = proto_loss
    if isinstance(proto_post, torch.Tensor) and float(prototype_balance_weight) > 0.0:
        mean_post = proto_post.mean(dim=0).clamp_min(1.0e-8)
        uniform = torch.full_like(mean_post, 1.0 / max(int(mean_post.numel()), 1))
        balance = torch.sum(mean_post * (torch.log(mean_post) - torch.log(uniform)))
        total = total + float(prototype_balance_weight) * balance
        parts["prototype_balance"] = balance

    if float(variance_weight) > 0.0:
        variance = _variance_loss(embedding)
        total = total + float(variance_weight) * variance
        parts["variance"] = variance
    if float(decorrelation_weight) > 0.0:
        decorrelation = _decorrelation_loss(embedding)
        total = total + float(decorrelation_weight) * decorrelation
        parts["decorrelation"] = decorrelation

    metrics = {"loss": float(total.detach().cpu())}
    metrics.update({key: float(value.detach().cpu()) for key, value in parts.items()})
    return total, metrics


__all__ = ["deepshe_loss"]
