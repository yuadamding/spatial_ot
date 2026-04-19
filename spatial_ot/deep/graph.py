from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn


def build_neighbor_graph(
    coords_um: np.ndarray,
    *,
    neighbor_k: int,
    radius_um: float | None,
) -> np.ndarray:
    coords = np.asarray(coords_um, dtype=np.float32)
    n = coords.shape[0]
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64)

    if radius_um is not None:
        nn_model = NearestNeighbors(radius=float(radius_um), metric="euclidean")
        nn_model.fit(coords)
        neighborhoods = nn_model.radius_neighbors(coords, return_distance=False)
    else:
        n_neighbors = min(max(int(neighbor_k) + 1, 2), n)
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn_model.fit(coords)
        neighborhoods = nn_model.kneighbors(coords, return_distance=False)

    src_index: list[int] = []
    dst_index: list[int] = []
    for center_idx, neighbors in enumerate(neighborhoods):
        neigh = np.asarray(neighbors, dtype=np.int64)
        neigh = neigh[neigh != center_idx]
        if neigh.size == 0:
            continue
        src_index.extend(neigh.tolist())
        dst_index.extend([center_idx] * int(neigh.size))
    if not src_index:
        return np.zeros((2, 0), dtype=np.int64)
    return np.vstack(
        [
            np.asarray(src_index, dtype=np.int64),
            np.asarray(dst_index, dtype=np.int64),
        ]
    )


def build_multiscale_graphs(
    coords_um: np.ndarray,
    *,
    neighbor_k: int,
    base_radius_um: float | None,
    short_radius_um: float | None,
    mid_radius_um: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    short_radius = short_radius_um if short_radius_um is not None else base_radius_um
    mid_radius = mid_radius_um
    if mid_radius is None and short_radius is not None:
        mid_radius = float(short_radius) * 2.0
    short_graph = build_neighbor_graph(
        coords_um,
        neighbor_k=neighbor_k,
        radius_um=short_radius,
    )
    mid_graph = build_neighbor_graph(
        coords_um,
        neighbor_k=max(int(neighbor_k) * 2, int(neighbor_k) + 2),
        radius_um=mid_radius,
    )
    return short_graph, mid_graph


def aggregate_neighbor_mean_torch(
    features: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    if edge_index.numel() == 0 or features.shape[0] == 0:
        return features
    src, dst = edge_index
    agg = torch.zeros_like(features)
    deg = torch.zeros((features.shape[0], 1), dtype=features.dtype, device=features.device)
    agg.index_add_(0, dst, features[src])
    deg.index_add_(0, dst, torch.ones((src.numel(), 1), dtype=features.dtype, device=features.device))
    agg = agg / deg.clamp_min(1.0)
    isolated = deg.squeeze(1) == 0
    if torch.any(isolated):
        agg[isolated] = features[isolated]
    return agg


def build_context_distribution_targets(
    coords_um: np.ndarray,
    features_std: np.ndarray,
    *,
    neighbor_k: int,
    base_radius_um: float | None,
    short_radius_um: float | None,
    mid_radius_um: float | None,
    device: torch.device | None = None,
) -> np.ndarray:
    feats = np.asarray(features_std, dtype=np.float32)
    if feats.shape[0] == 0:
        return np.zeros((0, feats.shape[1] * 2), dtype=np.float32)
    short_edges, mid_edges = build_multiscale_graphs(
        coords_um,
        neighbor_k=neighbor_k,
        base_radius_um=base_radius_um,
        short_radius_um=short_radius_um,
        mid_radius_um=mid_radius_um,
    )
    target_device = device or torch.device("cpu")
    feats_t = torch.as_tensor(feats, dtype=torch.float32, device=target_device)
    short_t = torch.as_tensor(short_edges, dtype=torch.long, device=target_device)
    mid_t = torch.as_tensor(mid_edges, dtype=torch.long, device=target_device)
    short_mean = aggregate_neighbor_mean_torch(feats_t, short_t)
    mid_mean = aggregate_neighbor_mean_torch(feats_t, mid_t)
    targets = torch.cat([short_mean, mid_mean], dim=1)
    return targets.detach().cpu().numpy().astype(np.float32, copy=False)


class MeanGraphLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        agg = aggregate_neighbor_mean_torch(h, edge_index)
        return self.net(torch.cat([h, agg], dim=-1))
