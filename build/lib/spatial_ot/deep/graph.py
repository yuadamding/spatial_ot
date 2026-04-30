from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn

_NEIGHBOR_AGGREGATION_MAX_GATHER_BYTES = 128 * 1024 * 1024


def build_neighbor_graph(
    coords_um: np.ndarray,
    *,
    neighbor_k: int,
    radius_um: float | None,
    max_neighbors: int | None = None,
) -> np.ndarray:
    coords = np.asarray(coords_um, dtype=np.float32)
    n = coords.shape[0]
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64)

    if radius_um is not None:
        nn_model = NearestNeighbors(radius=float(radius_um), metric="euclidean")
        nn_model.fit(coords)
        distances, neighborhoods = nn_model.radius_neighbors(
            coords, return_distance=True
        )
    else:
        n_neighbors = min(max(int(neighbor_k) + 1, 2), n)
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn_model.fit(coords)
        distances, neighborhoods = nn_model.kneighbors(coords, return_distance=True)

    src_index: list[int] = []
    dst_index: list[int] = []
    for center_idx, (neighbors, neighbor_dist) in enumerate(
        zip(neighborhoods, distances, strict=False)
    ):
        neigh = np.asarray(neighbors, dtype=np.int64)
        dist = np.asarray(neighbor_dist, dtype=np.float32)
        keep = neigh != center_idx
        neigh = neigh[keep]
        dist = dist[keep]
        if max_neighbors is not None and neigh.size > int(max_neighbors):
            order = np.argsort(dist, kind="stable")[: int(max_neighbors)]
            neigh = neigh[order]
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
    max_neighbors: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    short_radius = short_radius_um if short_radius_um is not None else base_radius_um
    mid_radius = mid_radius_um
    if mid_radius is None and short_radius is not None:
        mid_radius = float(short_radius) * 2.0
    short_graph = build_neighbor_graph(
        coords_um,
        neighbor_k=neighbor_k,
        radius_um=short_radius,
        max_neighbors=max_neighbors,
    )
    mid_graph = build_neighbor_graph(
        coords_um,
        neighbor_k=max(int(neighbor_k) * 2, int(neighbor_k) + 2),
        radius_um=mid_radius,
        max_neighbors=max_neighbors,
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
    deg = torch.zeros(
        (features.shape[0], 1), dtype=features.dtype, device=features.device
    )
    bytes_per_row = max(
        int(features.shape[1]) * max(int(features.element_size()), 1), 1
    )
    chunk_size = max(int(_NEIGHBOR_AGGREGATION_MAX_GATHER_BYTES // bytes_per_row), 1)
    for start in range(0, int(src.numel()), chunk_size):
        stop = min(start + chunk_size, int(src.numel()))
        src_chunk = src[start:stop]
        dst_chunk = dst[start:stop]
        agg.index_add_(0, dst_chunk, features.index_select(0, src_chunk))
        deg.index_add_(
            0,
            dst_chunk,
            torch.ones(
                (dst_chunk.numel(), 1), dtype=features.dtype, device=features.device
            ),
        )
    agg = agg / deg.clamp_min(1.0)
    isolated = deg.squeeze(1) == 0
    if torch.any(isolated):
        agg[isolated] = features[isolated]
    return agg


def aggregate_neighbor_degree_torch(
    n_nodes: int,
    edge_index: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if edge_index.numel() == 0 or n_nodes <= 0:
        return torch.zeros((n_nodes, 1), dtype=dtype, device=device)
    dst = edge_index[1]
    degree = torch.zeros((n_nodes, 1), dtype=dtype, device=device)
    degree.index_add_(
        0,
        dst,
        torch.ones((dst.numel(), 1), dtype=dtype, device=device),
    )
    return degree


def build_context_distribution_targets(
    coords_um: np.ndarray,
    features_std: np.ndarray,
    *,
    neighbor_k: int,
    base_radius_um: float | None,
    short_radius_um: float | None,
    mid_radius_um: float | None,
    max_neighbors: int | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    feats = np.asarray(features_std, dtype=np.float32)
    if feats.shape[0] == 0:
        return np.zeros((0, feats.shape[1] * 4 + 2), dtype=np.float32)
    short_edges, mid_edges = build_multiscale_graphs(
        coords_um,
        neighbor_k=neighbor_k,
        base_radius_um=base_radius_um,
        short_radius_um=short_radius_um,
        mid_radius_um=mid_radius_um,
        max_neighbors=max_neighbors,
    )
    target_device = device or torch.device("cpu")
    feats_t = torch.as_tensor(feats, dtype=torch.float32, device=target_device)
    target_chunks: list[np.ndarray] = []
    for edges in (short_edges, mid_edges):
        edge_t = torch.as_tensor(edges, dtype=torch.long, device=target_device)
        mean_t = aggregate_neighbor_mean_torch(feats_t, edge_t)
        mean_sq_t = aggregate_neighbor_mean_torch(feats_t.pow(2), edge_t)
        var_t = (mean_sq_t - mean_t.pow(2)).clamp_min(0.0)
        degree_t = aggregate_neighbor_degree_torch(
            feats_t.shape[0],
            edge_t,
            dtype=feats_t.dtype,
            device=target_device,
        )
        density_t = torch.log1p(degree_t)
        target_chunks.extend(
            [
                mean_t.detach().cpu().numpy().astype(np.float32, copy=False),
                var_t.detach().cpu().numpy().astype(np.float32, copy=False),
                density_t.detach().cpu().numpy().astype(np.float32, copy=False),
            ]
        )
        del edge_t
        del mean_t
        del mean_sq_t
        del var_t
        del degree_t
        del density_t
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(target_chunks, axis=1).astype(np.float32, copy=False)


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
