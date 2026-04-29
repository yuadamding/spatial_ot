from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..config import DeepFeatureConfig
from .graph import MeanGraphLayer, build_multiscale_graphs
from .validation import context_radii


def _decode_counts_from_factors(
    latent: torch.Tensor,
    *,
    factor_layer: nn.Module,
    gene_factors: torch.Tensor,
    gene_bias: torch.Tensor,
    log_theta: torch.Tensor,
    gene_index: torch.Tensor,
    library_log: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    low_rank = factor_layer(latent)
    factors = gene_factors.index_select(1, gene_index)
    bias = gene_bias.index_select(0, gene_index)
    theta = torch.nn.functional.softplus(log_theta.index_select(0, gene_index)) + 1e-4
    logits = low_rank @ factors + bias.unsqueeze(0) + library_log.unsqueeze(1)
    mu = torch.exp(logits).clamp_min(1e-6)
    return mu, theta.unsqueeze(0).expand_as(mu)


class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim: int, context_dim: int, config: DeepFeatureConfig, *, count_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = int(config.hidden_dim)
        layers = []
        in_dim = int(input_dim)
        for _ in range(max(int(config.layers), 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.encoder_body = nn.Sequential(*layers)
        self.intrinsic_head = nn.Linear(in_dim, int(config.latent_dim))
        self.context_head = nn.Linear(in_dim, int(config.latent_dim))
        self.fusion = nn.Sequential(
            nn.Linear(int(config.latent_dim) * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(config.latent_dim)),
        )

        decoder_layers = []
        dec_in = int(config.latent_dim)
        for _ in range(max(int(config.layers), 1)):
            decoder_layers.append(nn.Linear(dec_in, hidden_dim))
            decoder_layers.append(nn.ReLU())
            dec_in = hidden_dim
        decoder_layers.append(nn.Linear(dec_in, int(input_dim)))
        self.decoder = nn.Sequential(*decoder_layers)

        context_layers = []
        ctx_in = int(config.latent_dim)
        for _ in range(max(int(config.layers) - 1, 0)):
            context_layers.append(nn.Linear(ctx_in, hidden_dim))
            context_layers.append(nn.ReLU())
            ctx_in = hidden_dim
        context_layers.append(nn.Linear(ctx_in, int(context_dim)))
        self.context_predictor = nn.Sequential(*context_layers)
        self.count_factor: nn.Linear | None = None
        self.count_gene_factors: nn.Parameter | None = None
        self.count_gene_bias: nn.Parameter | None = None
        self.count_log_theta: nn.Parameter | None = None
        if count_dim is not None and int(count_dim) > 0:
            rank = int(config.count_decoder_rank)
            self.count_factor = nn.Linear(int(config.latent_dim), rank)
            self.count_gene_factors = nn.Parameter(torch.empty(rank, int(count_dim)))
            self.count_gene_bias = nn.Parameter(torch.zeros(int(count_dim)))
            self.count_log_theta = nn.Parameter(torch.zeros(int(count_dim)))
            nn.init.normal_(self.count_gene_factors, std=0.02)

    def _encode_all(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        intrinsic = self.intrinsic_head(h)
        context = self.context_head(h)
        joint = self.fusion(torch.cat([intrinsic, context], dim=-1))
        return intrinsic, context, joint

    def encode(self, x: torch.Tensor, *, output_embedding: str = "joint") -> torch.Tensor:
        intrinsic, context, joint = self._encode_all(x)
        if output_embedding == "intrinsic":
            return intrinsic
        if output_embedding == "context":
            return context
        return joint

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        intrinsic, context, joint = self._encode_all(x)
        return {
            "intrinsic": intrinsic,
            "context": context,
            "joint": joint,
            "recon": self.decoder(intrinsic),
            "context_pred": self.context_predictor(context),
        }

    def decode_counts(
        self,
        intrinsic: torch.Tensor,
        *,
        gene_index: torch.Tensor,
        library_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.count_factor is None
            or self.count_gene_factors is None
            or self.count_gene_bias is None
            or self.count_log_theta is None
        ):
            raise RuntimeError("Count reconstruction was requested, but the count decoder was not initialized.")
        return _decode_counts_from_factors(
            intrinsic,
            factor_layer=self.count_factor,
            gene_factors=self.count_gene_factors,
            gene_bias=self.count_gene_bias,
            log_theta=self.count_log_theta,
            gene_index=gene_index,
            library_log=library_log,
        )


class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim: int, context_dim: int, config: DeepFeatureConfig, *, count_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = int(config.hidden_dim)
        layers = []
        in_dim = int(input_dim)
        for _ in range(max(int(config.layers), 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        self.encoder_body = nn.Sequential(*layers)
        self.encoder_head = nn.Linear(in_dim, int(config.latent_dim))
        self.short_layers = nn.ModuleList(
            [MeanGraphLayer(int(config.latent_dim), hidden_dim) for _ in range(max(int(config.graph_layers), 1))]
        )
        self.mid_layers = nn.ModuleList(
            [MeanGraphLayer(int(config.latent_dim), hidden_dim) for _ in range(max(int(config.graph_layers), 1))]
        )
        self.fusion = nn.Sequential(
            nn.Linear(int(config.latent_dim) * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(config.latent_dim)),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(config.latent_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(input_dim)),
        )
        self.context_head = nn.Sequential(
            nn.Linear(int(config.latent_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(context_dim)),
        )
        self.count_factor: nn.Linear | None = None
        self.count_gene_factors: nn.Parameter | None = None
        self.count_gene_bias: nn.Parameter | None = None
        self.count_log_theta: nn.Parameter | None = None
        if count_dim is not None and int(count_dim) > 0:
            rank = int(config.count_decoder_rank)
            self.count_factor = nn.Linear(int(config.latent_dim), rank)
            self.count_gene_factors = nn.Parameter(torch.empty(rank, int(count_dim)))
            self.count_gene_bias = nn.Parameter(torch.zeros(int(count_dim)))
            self.count_log_theta = nn.Parameter(torch.zeros(int(count_dim)))
            nn.init.normal_(self.count_gene_factors, std=0.02)

    def _encode_all(
        self,
        x: torch.Tensor,
        edge_index_short: torch.Tensor,
        edge_index_mid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intrinsic = self.encoder_head(self.encoder_body(x))
        short = intrinsic
        for layer in self.short_layers:
            short = short + layer(short, edge_index_short)
        context = short
        for layer in self.mid_layers:
            context = context + layer(context, edge_index_mid)
        joint = self.fusion(torch.cat([intrinsic, short, context], dim=-1))
        return intrinsic, context, joint

    def encode(
        self,
        x: torch.Tensor,
        *,
        edge_index_short: torch.Tensor,
        edge_index_mid: torch.Tensor,
        output_embedding: str = "joint",
    ) -> torch.Tensor:
        intrinsic, context, joint = self._encode_all(x, edge_index_short=edge_index_short, edge_index_mid=edge_index_mid)
        if output_embedding == "intrinsic":
            return intrinsic
        if output_embedding == "context":
            return context
        return joint

    def forward(
        self,
        x: torch.Tensor,
        *,
        edge_index_short: torch.Tensor,
        edge_index_mid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        intrinsic, context, joint = self._encode_all(x, edge_index_short=edge_index_short, edge_index_mid=edge_index_mid)
        return {
            "intrinsic": intrinsic,
            "context": context,
            "joint": joint,
            "recon": self.decoder(intrinsic),
            "context_pred": self.context_head(context),
        }

    def decode_counts(
        self,
        intrinsic: torch.Tensor,
        *,
        gene_index: torch.Tensor,
        library_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.count_factor is None
            or self.count_gene_factors is None
            or self.count_gene_bias is None
            or self.count_log_theta is None
        ):
            raise RuntimeError("Count reconstruction was requested, but the count decoder was not initialized.")
        return _decode_counts_from_factors(
            intrinsic,
            factor_layer=self.count_factor,
            gene_factors=self.count_gene_factors,
            gene_bias=self.count_gene_bias,
            log_theta=self.count_log_theta,
            gene_index=gene_index,
            library_log=library_log,
        )


def make_model(input_dim: int, context_dim: int, config: DeepFeatureConfig, *, count_dim: int | None = None) -> nn.Module:
    if config.method == "graph_autoencoder":
        return GraphAutoencoder(input_dim=input_dim, context_dim=context_dim, config=config, count_dim=count_dim)
    return MLPAutoencoder(input_dim=input_dim, context_dim=context_dim, config=config, count_dim=count_dim)


def tensor_graphs(
    coords_um: np.ndarray,
    *,
    config: DeepFeatureConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    short_radius, mid_radius = context_radii(config)
    short_graph, mid_graph = build_multiscale_graphs(
        coords_um,
        neighbor_k=config.neighbor_k,
        base_radius_um=config.radius_um,
        short_radius_um=short_radius,
        mid_radius_um=mid_radius,
        max_neighbors=config.graph_max_neighbors,
    )
    return (
        torch.as_tensor(short_graph, dtype=torch.long, device=device),
        torch.as_tensor(mid_graph, dtype=torch.long, device=device),
    )
