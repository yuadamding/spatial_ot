from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .graph import NeighborhoodGraph


@dataclass(frozen=True)
class CellNicheBatchSpec:
    n_radii: int
    max_neighbors: int
    token_input_dim: int
    graph_keys: tuple[str, ...]


class CellNicheDataset(Dataset):
    """Mini-batch view that materializes padded cell-centered neighborhood tokens."""

    def __init__(
        self,
        *,
        features: np.ndarray,
        posteriors: np.ndarray,
        coords_um: np.ndarray,
        graphs: dict[str, NeighborhoodGraph],
        descriptor_targets: np.ndarray | None = None,
        radial_shells: int = 3,
        max_neighbors_per_graph: int = 64,
    ) -> None:
        z = np.asarray(features, dtype=np.float32)
        q = np.asarray(posteriors, dtype=np.float32)
        coords = np.asarray(coords_um, dtype=np.float32)
        if z.ndim != 2 or q.ndim != 2 or z.shape[0] != q.shape[0]:
            raise ValueError("features and posteriors must be 2D with matching rows.")
        if coords.shape != (z.shape[0], 2):
            raise ValueError("coords_um must have shape (n_cells, 2).")
        if not graphs:
            raise ValueError("at least one graph is required.")
        if descriptor_targets is not None:
            targets = np.asarray(descriptor_targets, dtype=np.float32)
            if targets.ndim != 2 or targets.shape[0] != z.shape[0]:
                raise ValueError("descriptor_targets must be 2D with one row per cell.")
        else:
            targets = None
        self.features = z
        self.posteriors = q
        self.coords_um = coords
        self.graphs = dict(graphs)
        self.graph_keys = tuple(self.graphs)
        self.descriptor_targets = targets
        self.radial_shells = max(int(radial_shells), 1)
        self.max_neighbors_per_graph = max(int(max_neighbors_per_graph), 1)
        self.token_input_dim = int(z.shape[1] + q.shape[1] + 2 + 1 + self.radial_shells)

    @property
    def spec(self) -> CellNicheBatchSpec:
        return CellNicheBatchSpec(
            n_radii=len(self.graph_keys),
            max_neighbors=int(self.max_neighbors_per_graph),
            token_input_dim=int(self.token_input_dim),
            graph_keys=self.graph_keys,
        )

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> int:
        return int(index)

    def _row_neighbors(self, graph: NeighborhoodGraph, row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        conn = graph.connectivities.tocsr()
        dist = graph.distances.tocsr()
        start, stop = int(conn.indptr[row]), int(conn.indptr[row + 1])
        if start == stop:
            return (
                np.asarray([row], dtype=np.int64),
                np.asarray([0.0], dtype=np.float32),
                np.asarray([1.0], dtype=np.float32),
            )
        cols = conn.indices[start:stop].astype(np.int64, copy=False)
        weights = np.asarray(conn.data[start:stop], dtype=np.float32)
        distances = np.asarray(dist.data[start:stop], dtype=np.float32)
        if cols.size > self.max_neighbors_per_graph:
            order = np.argsort(-weights, kind="stable")[: self.max_neighbors_per_graph]
            cols = cols[order]
            weights = weights[order]
            distances = distances[order]
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            weights = np.full(weights.shape, 1.0 / max(int(weights.size), 1), dtype=np.float32)
        else:
            weights = (weights / weight_sum).astype(np.float32, copy=False)
        return cols, distances.astype(np.float32, copy=False), weights

    def _radius_denominator(self, graph: NeighborhoodGraph, distances: np.ndarray) -> float:
        if graph.radius_um is not None and float(graph.radius_um) > 0.0:
            return float(graph.radius_um)
        positive = distances[np.isfinite(distances) & (distances > 0.0)]
        return float(np.max(positive)) if positive.size else 1.0

    def collate_fn(self, anchor_ids: list[int] | tuple[int, ...] | np.ndarray) -> dict[str, torch.Tensor]:
        anchors = np.asarray(anchor_ids, dtype=np.int64).reshape(-1)
        batch_size = int(anchors.shape[0])
        n_graphs = len(self.graph_keys)
        max_neighbors = int(self.max_neighbors_per_graph)
        token_dim = int(self.token_input_dim)
        tokens = np.zeros((batch_size, n_graphs, max_neighbors, token_dim), dtype=np.float32)
        weights = np.zeros((batch_size, n_graphs, max_neighbors), dtype=np.float32)
        mask = np.zeros((batch_size, n_graphs, max_neighbors), dtype=bool)

        z_dim = int(self.features.shape[1])
        q_dim = int(self.posteriors.shape[1])
        shell_start = z_dim + q_dim + 3
        for batch_row, anchor in enumerate(anchors):
            anchor_xy = self.coords_um[int(anchor)]
            for graph_pos, graph_key in enumerate(self.graph_keys):
                graph = self.graphs[graph_key]
                cols, distances, row_weights = self._row_neighbors(graph, int(anchor))
                keep = min(int(cols.size), max_neighbors)
                cols = cols[:keep]
                distances = distances[:keep]
                row_weights = row_weights[:keep]
                denom = max(self._radius_denominator(graph, distances), 1e-8)
                rel = (self.coords_um[cols] - anchor_xy[None, :]) / denom
                relative_distance = distances / denom
                shell_idx = np.floor(
                    np.clip(relative_distance, 0.0, 0.999999) * self.radial_shells
                ).astype(np.int64)
                tokens[batch_row, graph_pos, :keep, :z_dim] = self.features[cols]
                tokens[batch_row, graph_pos, :keep, z_dim : z_dim + q_dim] = self.posteriors[
                    cols
                ]
                tokens[
                    batch_row,
                    graph_pos,
                    :keep,
                    z_dim + q_dim : z_dim + q_dim + 2,
                ] = rel.astype(np.float32)
                tokens[batch_row, graph_pos, :keep, z_dim + q_dim + 2] = relative_distance
                tokens[
                    batch_row,
                    graph_pos,
                    np.arange(keep),
                    shell_start + shell_idx[:keep],
                ] = 1.0
                weights[batch_row, graph_pos, :keep] = row_weights
                mask[batch_row, graph_pos, :keep] = True

        out: dict[str, torch.Tensor] = {
            "anchor_ids": torch.as_tensor(anchors, dtype=torch.long),
            "z_self": torch.as_tensor(self.features[anchors], dtype=torch.float32),
            "tokens": torch.as_tensor(tokens, dtype=torch.float32),
            "weights": torch.as_tensor(weights, dtype=torch.float32),
            "mask": torch.as_tensor(mask, dtype=torch.bool),
        }
        if self.descriptor_targets is not None:
            out["descriptor_targets"] = torch.as_tensor(
                self.descriptor_targets[anchors], dtype=torch.float32
            )
        return out


__all__ = ["CellNicheBatchSpec", "CellNicheDataset"]
