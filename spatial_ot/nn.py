from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))


def nb_nll(counts: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    theta = theta.unsqueeze(0)
    log_theta_mu = torch.log(theta + mu + 1e-8)
    result = (
        torch.lgamma(theta + counts)
        - torch.lgamma(theta)
        - torch.lgamma(counts + 1.0)
        + theta * (torch.log(theta + 1e-8) - log_theta_mu)
        + counts * (torch.log(mu + 1e-8) - log_theta_mu)
    )
    return -result.mean()


def aggregate_mean(features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros_like(features)
    src, dst = edge_index
    out = torch.zeros_like(features)
    deg = torch.zeros((features.size(0), 1), device=features.device, dtype=features.dtype)
    out.index_add_(0, dst, features[src])
    deg.index_add_(0, dst, torch.ones((dst.numel(), 1), device=features.device, dtype=features.dtype))
    deg = torch.clamp(deg, min=1.0)
    return out / deg


def cross_covariance_penalty(z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    zc = z - z.mean(dim=0, keepdim=True)
    sc = s - s.mean(dim=0, keepdim=True)
    cov = zc.T @ sc / max(z.size(0) - 1, 1)
    return torch.mean(cov**2)


def sample_negative_edges(n_nodes: int, positive_edge_index: torch.Tensor, ratio: float, generator: torch.Generator | None = None) -> torch.Tensor:
    n_positive = positive_edge_index.size(1)
    n_negative = max(1, int(math.ceil(n_positive * ratio)))
    src = torch.randint(0, n_nodes, (n_negative,), generator=generator)
    dst = torch.randint(0, n_nodes, (n_negative,), generator=generator)
    return torch.stack([src, dst], dim=0).to(positive_edge_index.device)


def edge_bce_loss(embedding: torch.Tensor, positive_edges: torch.Tensor, negative_edges: torch.Tensor) -> torch.Tensor:
    def score(edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return torch.sum(embedding[src] * embedding[dst], dim=1)

    pos_score = score(positive_edges)
    neg_score = score(negative_edges)
    logits = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat(
        [
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score),
        ],
        dim=0,
    )
    return F.binary_cross_entropy_with_logits(logits, labels)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        return self.mu(hidden), self.logvar(hidden)


class NBDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.logits = MLP(latent_dim, hidden_dim, output_dim, dropout=dropout)
        self.theta = nn.Parameter(torch.zeros(output_dim))

    def forward(self, latent: torch.Tensor, library: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(latent)
        mu = torch.softmax(logits, dim=1) * library
        theta = F.softplus(self.theta) + 1e-4
        return mu, theta


class MaskedProgramDecoder(nn.Module):
    def __init__(
        self,
        z_dim: int,
        prior_dim: int,
        novo_dim: int,
        n_genes: int,
        mask: np.ndarray | None = None,
    ):
        super().__init__()
        self.prior_dim = prior_dim
        self.novo_dim = novo_dim
        self.z_proj = nn.Linear(z_dim, n_genes)
        self.bias = nn.Parameter(torch.zeros(n_genes))
        self.theta = nn.Parameter(torch.zeros(n_genes))
        if prior_dim > 0:
            self.prior_weight = nn.Parameter(torch.randn(prior_dim, n_genes) * 0.01)
            if mask is None:
                mask = np.ones((prior_dim, n_genes), dtype=np.float32)
            self.register_buffer("prior_mask", torch.as_tensor(mask, dtype=torch.float32))
        else:
            self.prior_weight = None
            self.register_buffer("prior_mask", torch.zeros((0, n_genes), dtype=torch.float32))
        if novo_dim > 0:
            self.novo_weight = nn.Parameter(torch.randn(novo_dim, n_genes) * 0.01)
        else:
            self.novo_weight = None

    def forward(self, z: torch.Tensor, s_prior: torch.Tensor, s_novo: torch.Tensor, library: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.z_proj(z) + self.bias
        if self.prior_dim > 0:
            logits = logits + s_prior @ (self.prior_weight * self.prior_mask)
        if self.novo_dim > 0:
            logits = logits + s_novo @ self.novo_weight
        mu = torch.softmax(logits, dim=1) * library
        theta = F.softplus(self.theta) + 1e-4
        return mu, theta


class TeacherContextModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, teacher_dim: int, teacher_logit_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = MLP(input_dim * 2, hidden_dim, teacher_dim, dropout=dropout)
        self.logit_head = nn.Linear(teacher_dim, teacher_logit_dim)
        self.self_decoder = NBDecoder(teacher_dim, hidden_dim, input_dim, dropout=dropout)
        self.nb_decoder = NBDecoder(teacher_dim, hidden_dim, input_dim, dropout=dropout)

    def forward(
        self,
        x_log: torch.Tensor,
        neighbor_log: torch.Tensor,
        library: torch.Tensor,
        neighbor_library: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(torch.cat([x_log, neighbor_log], dim=1))
        mu_self, theta_self = self.self_decoder(hidden, library)
        mu_nb, theta_nb = self.nb_decoder(hidden, neighbor_library)
        return {
            "u": hidden,
            "q": self.logit_head(hidden),
            "mu_self": mu_self,
            "theta_self": theta_self,
            "mu_nb": mu_nb,
            "theta_nb": theta_nb,
        }


class MultiViewFusion(nn.Module):
    def __init__(self, view_dims: list[int], hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoders = nn.ModuleList([MLP(dim, hidden_dim, hidden_dim, dropout=dropout) for dim in view_dims])
        self.attn = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in view_dims])

    def forward(self, views: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = [encoder(view) for encoder, view in zip(self.encoders, views)]
        logits = torch.cat([head(h) for head, h in zip(self.attn, hidden)], dim=1)
        weights = torch.softmax(logits, dim=1)
        fused = 0.0
        for idx, h in enumerate(hidden):
            fused = fused + h * weights[:, idx : idx + 1]
        return fused, weights


class StudentSpatialModel(nn.Module):
    def __init__(
        self,
        n_genes: int,
        aux_dim: int,
        batch_dim: int,
        teacher_dim: int,
        z_dim: int,
        prior_dim: int,
        novo_dim: int,
        hidden_dim: int,
        teacher_logit_dim: int,
        n_cell_types: int,
        self_mask: np.ndarray | None,
        neighborhood_mask: np.ndarray | None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.prior_dim = prior_dim
        self.novo_dim = novo_dim
        self.s_dim = prior_dim + novo_dim
        intrinsic_in = n_genes + aux_dim + batch_dim
        self.intrinsic_encoder = GaussianEncoder(intrinsic_in, hidden_dim, z_dim, dropout=dropout)
        self.intrinsic_decoder = NBDecoder(z_dim, hidden_dim, n_genes, dropout=dropout)
        view_dims = [z_dim, z_dim, n_cell_types, max(prior_dim, 1), teacher_dim, aux_dim]
        self.view_fusion = MultiViewFusion(view_dims, hidden_dim, dropout=dropout)
        self.context_encoder = GaussianEncoder(hidden_dim, hidden_dim, self.s_dim, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(z_dim + max(prior_dim, 1), self.s_dim), nn.Sigmoid())
        self.teacher_proj = nn.Linear(self.s_dim, teacher_dim)
        self.teacher_logits = nn.Linear(self.s_dim, teacher_logit_dim)
        self.self_program_decoder = MaskedProgramDecoder(z_dim, prior_dim, novo_dim, n_genes, mask=self_mask)
        self.nb_program_decoder = MaskedProgramDecoder(z_dim, prior_dim, novo_dim, n_genes, mask=neighborhood_mask)
        self.func_head = nn.Linear(z_dim, n_cell_types)

    def encode_intrinsic(self, x_log: torch.Tensor, aux: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.intrinsic_encoder(torch.cat([x_log, aux, batch], dim=1))
        z = reparameterize(mu, logvar)
        return z, mu, logvar

    def decode_intrinsic(self, z: torch.Tensor, library: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.intrinsic_decoder(z, library)

    def encode_context(
        self,
        z: torch.Tensor,
        short_view: torch.Tensor,
        mid_view: torch.Tensor,
        composition_view: torch.Tensor,
        marker_view: torch.Tensor,
        teacher_view: torch.Tensor,
        aux_view: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if marker_view.size(1) == 0:
            marker_view = torch.zeros((z.size(0), 1), device=z.device, dtype=z.dtype)
        fused, weights = self.view_fusion([short_view, mid_view, composition_view, marker_view, teacher_view, aux_view])
        mu, logvar = self.context_encoder(fused)
        base_s = reparameterize(mu, logvar)
        gate = self.gate(torch.cat([z, marker_view], dim=1))
        s = gate * base_s
        return s, mu, logvar, weights

    def split_programs(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_prior = s[:, : self.prior_dim] if self.prior_dim > 0 else s.new_zeros((s.size(0), 0))
        s_novo = s[:, self.prior_dim :] if self.novo_dim > 0 else s.new_zeros((s.size(0), 0))
        return s_prior, s_novo

    def decode_programs(self, z: torch.Tensor, s: torch.Tensor, library: torch.Tensor, neighbor_library: torch.Tensor) -> dict[str, torch.Tensor]:
        s_prior, s_novo = self.split_programs(s)
        mu_self, theta_self = self.self_program_decoder(z, s_prior, s_novo, library)
        mu_nb, theta_nb = self.nb_program_decoder(z, s_prior, s_novo, neighbor_library)
        return {
            "mu_self": mu_self,
            "theta_self": theta_self,
            "mu_nb": mu_nb,
            "theta_nb": theta_nb,
        }

    def teacher_targets(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.teacher_proj(s), self.teacher_logits(s)

    def embedding(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return torch.cat([z, s], dim=1)
