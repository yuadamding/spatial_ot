from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .ot import sinkhorn_balanced_distance, weighted_pairwise_sqdist


class TokenMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.GELU(),
            nn.LayerNorm(int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleDeepSetEncoder(nn.Module):
    def __init__(
        self,
        *,
        z_dim: int,
        token_input_dim: int,
        n_radii: int,
        token_dim: int = 128,
        hidden_dim: int = 256,
        embedding_dim: int = 64,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.n_radii = int(n_radii)
        self.token_dim = int(token_dim)
        self.use_attention = bool(use_attention)
        self.token_mlps = nn.ModuleList(
            [
                TokenMLP(int(token_input_dim), int(hidden_dim), int(token_dim))
                for _ in range(int(n_radii))
            ]
        )
        self.attn_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(token_dim), max(int(hidden_dim) // 2, 8)),
                    nn.GELU(),
                    nn.Linear(max(int(hidden_dim) // 2, 8), 1),
                )
                for _ in range(int(n_radii))
            ]
        )
        self.self_mlp = nn.Sequential(
            nn.Linear(int(z_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(token_dim)),
            nn.GELU(),
            nn.LayerNorm(int(token_dim)),
        )
        self.fuse = nn.Sequential(
            nn.Linear(int(token_dim) * (int(n_radii) + 1), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(embedding_dim)),
        )

    def forward(
        self,
        *,
        z_self: torch.Tensor,
        token_inputs: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
        is_isolated: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale_contexts: list[torch.Tensor] = []
        token_embeddings: list[torch.Tensor] = []
        for radius_idx in range(self.n_radii):
            x = token_inputs[:, radius_idx]
            w = weights[:, radius_idx]
            m = mask[:, radius_idx]
            isolated = (
                is_isolated[:, radius_idx]
                if is_isolated is not None
                else torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
            )
            e = self.token_mlps[radius_idx](x)
            if torch.any(isolated):
                e = torch.where(isolated[:, None, None], torch.zeros_like(e), e)
            if self.use_attention:
                logits = self.attn_mlps[radius_idx](e).squeeze(-1)
            else:
                logits = torch.zeros_like(w)
            logits = logits + torch.log(w.clamp_min(1.0e-8))
            logits = logits.masked_fill(~m, -1.0e9)
            attn = torch.softmax(logits, dim=1)
            context = torch.sum(attn.unsqueeze(-1) * e, dim=1)
            scale_contexts.append(context)
            token_embeddings.append(e)
        self_context = self.self_mlp(z_self)
        h = self.fuse(torch.cat([self_context] + scale_contexts, dim=1))
        return F.normalize(h, dim=1), torch.stack(token_embeddings, dim=1)


class OTPrototypeHead(nn.Module):
    """Learned local-measure prototype layer with balanced Sinkhorn distances."""

    def __init__(
        self,
        *,
        n_radii: int,
        token_dim: int,
        n_prototypes: int = 20,
        support_size: int = 32,
        epsilon: float = 0.05,
        temperature: float = 0.25,
        sinkhorn_iters: int = 30,
    ) -> None:
        super().__init__()
        self.n_radii = int(n_radii)
        self.n_prototypes = int(n_prototypes)
        self.support_size = int(support_size)
        self.epsilon = float(epsilon)
        self.temperature = float(temperature)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.prototype_embeddings = nn.Parameter(
            torch.randn(
                int(n_radii),
                int(n_prototypes),
                int(support_size),
                int(token_dim),
            )
            * 0.02
        )
        self.prototype_mass_logits = nn.Parameter(
            torch.zeros(int(n_radii), int(n_prototypes), int(support_size))
        )

    def forward(
        self,
        *,
        token_embeddings: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(token_embeddings.shape[0])
        source_weights = weights * mask.to(dtype=weights.dtype)
        source_weights = source_weights / source_weights.sum(dim=2, keepdim=True).clamp_min(
            1.0e-12
        )
        distances = []
        for proto_idx in range(self.n_prototypes):
            total = torch.zeros(
                batch_size,
                dtype=token_embeddings.dtype,
                device=token_embeddings.device,
            )
            for radius_idx in range(self.n_radii):
                source = token_embeddings[:, radius_idx]
                proto = self.prototype_embeddings[radius_idx, proto_idx]
                target = torch.softmax(
                    self.prototype_mass_logits[radius_idx, proto_idx], dim=0
                )
                cost = weighted_pairwise_sqdist(source, proto.unsqueeze(0).expand(batch_size, -1, -1))
                total = total + sinkhorn_balanced_distance(
                    cost,
                    source_weights[:, radius_idx],
                    target.unsqueeze(0).expand(batch_size, -1),
                    epsilon=self.epsilon,
                    n_iters=self.sinkhorn_iters,
                )
            distances.append(total / max(self.n_radii, 1))
        dist = torch.stack(distances, dim=1)
        posterior = torch.softmax(-dist / max(float(self.temperature), 1.0e-6), dim=1)
        return dist, posterior


class OTDeepSHEModel(nn.Module):
    def __init__(
        self,
        *,
        z_dim: int,
        token_input_dim: int,
        n_radii: int,
        descriptor_dim: int,
        token_dim: int = 128,
        hidden_dim: int = 256,
        embedding_dim: int = 64,
        use_attention: bool = True,
        use_ot_prototypes: bool = False,
        n_ot_prototypes: int = 20,
        prototype_support_size: int = 32,
        ot_epsilon: float = 0.05,
        ot_temperature: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = MultiScaleDeepSetEncoder(
            z_dim=int(z_dim),
            token_input_dim=int(token_input_dim),
            n_radii=int(n_radii),
            token_dim=int(token_dim),
            hidden_dim=int(hidden_dim),
            embedding_dim=int(embedding_dim),
            use_attention=bool(use_attention),
        )
        self.context_decoder = nn.Sequential(
            nn.Linear(int(embedding_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(descriptor_dim)),
        )
        self.ot_head = (
            OTPrototypeHead(
                n_radii=int(n_radii),
                token_dim=int(token_dim),
                n_prototypes=int(n_ot_prototypes),
                support_size=int(prototype_support_size),
                epsilon=float(ot_epsilon),
                temperature=float(ot_temperature),
            )
            if bool(use_ot_prototypes)
            else None
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
        h, token_embeddings = self.encoder(
            z_self=batch["z_self"],
            token_inputs=batch["tokens"],
            weights=batch["weights"],
            mask=batch["mask"],
            is_isolated=batch.get("is_isolated"),
        )
        decoded = self.context_decoder(h)
        if self.ot_head is not None:
            distances, posterior = self.ot_head(
                token_embeddings=token_embeddings,
                weights=batch["weights"],
                mask=batch["mask"],
            )
        else:
            distances, posterior = None, None
        return {
            "embedding": h,
            "decoded_descriptor": decoded,
            "token_embeddings": token_embeddings,
            "prototype_distances": distances,
            "prototype_posterior": posterior,
        }


__all__ = [
    "MultiScaleDeepSetEncoder",
    "OTDeepSHEModel",
    "OTPrototypeHead",
    "TokenMLP",
]
