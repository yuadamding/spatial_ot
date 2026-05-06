from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class NeighborhoodGraph:
    key: str
    connectivities: sparse.csr_matrix
    distances: sparse.csr_matrix
    mode: str
    radius_um: float | None = None
    k: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def _as_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must be a 2D array with shape (n_cells, 2).")
    if arr.shape[0] == 0:
        raise ValueError("coords must contain at least one cell.")
    return arr


def _as_sample_ids(sample_ids: np.ndarray | None, n_cells: int) -> np.ndarray:
    if sample_ids is None:
        return np.full(int(n_cells), "sample_0", dtype=object)
    values = np.asarray(sample_ids, dtype=object).reshape(-1)
    if values.shape[0] != int(n_cells):
        raise ValueError("sample_ids must have one value per cell.")
    return values.astype(str)


def _format_number(value: float | int) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _kernel_weights(distances: np.ndarray, *, kernel: str, scale: float) -> np.ndarray:
    dist = np.asarray(distances, dtype=np.float32)
    requested = str(kernel or "gaussian").strip().lower()
    if requested in {"binary", "uniform", "none"}:
        return np.ones_like(dist, dtype=np.float32)
    if requested in {"inverse", "inverse_distance"}:
        return (1.0 / np.maximum(dist, 1e-6)).astype(np.float32)
    if requested in {"gaussian", "rbf"}:
        bandwidth = max(float(scale), 1e-6)
        return np.exp(-0.5 * (dist / bandwidth) ** 2).astype(np.float32)
    raise ValueError("kernel must be gaussian, binary, or inverse.")


def _apply_density_correction(
    connectivities: sparse.csr_matrix, *, density_correction: float
) -> sparse.csr_matrix:
    alpha = float(np.clip(float(density_correction), 0.0, 1.0))
    if alpha <= 0.0 or connectivities.nnz == 0:
        return connectivities.tocsr()
    coo = connectivities.tocoo(copy=True)
    degree = np.asarray(connectivities.sum(axis=1)).reshape(-1).astype(np.float64)
    neighbor_degree = np.maximum(degree[coo.col], 1e-8)
    coo.data = (np.asarray(coo.data, dtype=np.float64) / (neighbor_degree**alpha)).astype(
        np.float32
    )
    return coo.tocsr()


def _graph_metadata(
    matrix: sparse.csr_matrix,
    *,
    sample_ids: np.ndarray,
    mode: str,
    kernel: str,
    max_neighbors: int,
    density_correction: float,
    radius_um: float | None = None,
    k: int | None = None,
) -> dict[str, object]:
    degree = np.diff(matrix.indptr).astype(np.int32)
    sample_values, sample_counts = np.unique(sample_ids.astype(str), return_counts=True)
    return {
        "mode": str(mode),
        "radius_um": None if radius_um is None else float(radius_um),
        "k": None if k is None else int(k),
        "kernel": str(kernel),
        "max_neighbors": int(max_neighbors),
        "density_correction": float(density_correction),
        "n_cells": int(matrix.shape[0]),
        "n_edges": int(matrix.nnz),
        "mean_degree": float(np.mean(degree)) if degree.size else 0.0,
        "max_degree": int(np.max(degree)) if degree.size else 0,
        "isolated_fraction": float(np.mean(degree == 0)) if degree.size else 0.0,
        "sample_counts": {
            str(sample): int(count)
            for sample, count in zip(sample_values, sample_counts, strict=False)
        },
        "cross_sample_edges_allowed": False,
    }


def build_radius_graphs(
    coords: np.ndarray,
    sample_ids: np.ndarray | None,
    radii_um: tuple[float, ...] | list[float] | np.ndarray,
    *,
    max_neighbors: int = 256,
    kernel: str = "gaussian",
    density_correction: float = 0.0,
    include_self: bool = False,
) -> dict[str, NeighborhoodGraph]:
    """Build sample-isolated radius graphs for cell-centered contexts."""

    xy = _as_coords(coords)
    samples = _as_sample_ids(sample_ids, xy.shape[0])
    radii = tuple(float(r) for r in radii_um if float(r) > 0.0)
    if not radii:
        raise ValueError("At least one positive radius is required.")
    max_neighbors = max(int(max_neighbors), 1)

    graphs: dict[str, NeighborhoodGraph] = {}
    sample_order = list(dict.fromkeys(samples.tolist()))
    for radius in radii:
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        distances: list[np.ndarray] = []
        for sample in sample_order:
            global_idx = np.flatnonzero(samples == sample)
            if global_idx.size == 0:
                continue
            local = xy[global_idx]
            model = NearestNeighbors(radius=float(radius), algorithm="auto")
            model.fit(local)
            dist_list, ind_list = model.radius_neighbors(
                local, return_distance=True, sort_results=True
            )
            for row_local, (dist, ind) in enumerate(zip(dist_list, ind_list, strict=True)):
                dist = np.asarray(dist, dtype=np.float32)
                ind = np.asarray(ind, dtype=np.int64)
                finite = np.isfinite(dist)
                if not include_self:
                    finite &= ind != int(row_local)
                dist = dist[finite]
                ind = ind[finite]
                if dist.size == 0:
                    continue
                if dist.size > max_neighbors:
                    dist = dist[:max_neighbors]
                    ind = ind[:max_neighbors]
                rows.append(
                    np.full(ind.shape[0], int(global_idx[row_local]), dtype=np.int64)
                )
                cols.append(global_idx[ind].astype(np.int64, copy=False))
                distances.append(dist.astype(np.float32, copy=False))
                weights.append(
                    _kernel_weights(dist, kernel=kernel, scale=max(radius / 2.0, 1e-6))
                )
        n_cells = int(xy.shape[0])
        if rows:
            row_arr = np.concatenate(rows)
            col_arr = np.concatenate(cols)
            weight_arr = np.concatenate(weights).astype(np.float32, copy=False)
            dist_arr = np.concatenate(distances).astype(np.float32, copy=False)
        else:
            row_arr = col_arr = np.zeros(0, dtype=np.int64)
            weight_arr = dist_arr = np.zeros(0, dtype=np.float32)
        connectivities = sparse.csr_matrix(
            (weight_arr, (row_arr, col_arr)), shape=(n_cells, n_cells)
        )
        connectivities = _apply_density_correction(
            connectivities, density_correction=density_correction
        )
        distances_matrix = sparse.csr_matrix(
            (dist_arr, (row_arr, col_arr)), shape=(n_cells, n_cells)
        )
        key = f"r{_format_number(radius)}"
        graphs[key] = NeighborhoodGraph(
            key=key,
            connectivities=connectivities,
            distances=distances_matrix,
            mode="radius",
            radius_um=float(radius),
            metadata=_graph_metadata(
                connectivities,
                sample_ids=samples,
                mode="radius",
                kernel=kernel,
                max_neighbors=max_neighbors,
                density_correction=density_correction,
                radius_um=float(radius),
            ),
        )
    return graphs


def build_knn_graphs(
    coords: np.ndarray,
    sample_ids: np.ndarray | None,
    k_values: tuple[int, ...] | list[int] | np.ndarray,
    *,
    max_neighbors: int = 256,
    kernel: str = "gaussian",
    density_correction: float = 0.0,
    include_self: bool = False,
) -> dict[str, NeighborhoodGraph]:
    """Build sample-isolated kNN graphs for density-normalized context checks."""

    xy = _as_coords(coords)
    samples = _as_sample_ids(sample_ids, xy.shape[0])
    requested_k = tuple(int(k) for k in k_values if int(k) > 0)
    if not requested_k:
        raise ValueError("At least one positive k value is required.")
    max_neighbors = max(int(max_neighbors), 1)

    graphs: dict[str, NeighborhoodGraph] = {}
    sample_order = list(dict.fromkeys(samples.tolist()))
    for k_value in requested_k:
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        distances: list[np.ndarray] = []
        for sample in sample_order:
            global_idx = np.flatnonzero(samples == sample)
            if global_idx.size < 2 and not include_self:
                continue
            local = xy[global_idx]
            n_neighbors = min(
                int(k_value) + (0 if include_self else 1),
                int(global_idx.size),
                max_neighbors + (0 if include_self else 1),
            )
            if n_neighbors <= 0:
                continue
            model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
            model.fit(local)
            dist_matrix, ind_matrix = model.kneighbors(local, return_distance=True)
            positive = dist_matrix[np.isfinite(dist_matrix) & (dist_matrix > 0.0)]
            scale = float(np.median(positive)) if positive.size else 1.0
            for row_local in range(int(local.shape[0])):
                dist = np.asarray(dist_matrix[row_local], dtype=np.float32)
                ind = np.asarray(ind_matrix[row_local], dtype=np.int64)
                finite = np.isfinite(dist)
                if not include_self:
                    finite &= ind != int(row_local)
                dist = dist[finite]
                ind = ind[finite]
                if dist.size == 0:
                    continue
                keep = min(dist.size, int(k_value), max_neighbors)
                dist = dist[:keep]
                ind = ind[:keep]
                rows.append(
                    np.full(ind.shape[0], int(global_idx[row_local]), dtype=np.int64)
                )
                cols.append(global_idx[ind].astype(np.int64, copy=False))
                distances.append(dist.astype(np.float32, copy=False))
                weights.append(_kernel_weights(dist, kernel=kernel, scale=scale))
        n_cells = int(xy.shape[0])
        if rows:
            row_arr = np.concatenate(rows)
            col_arr = np.concatenate(cols)
            weight_arr = np.concatenate(weights).astype(np.float32, copy=False)
            dist_arr = np.concatenate(distances).astype(np.float32, copy=False)
        else:
            row_arr = col_arr = np.zeros(0, dtype=np.int64)
            weight_arr = dist_arr = np.zeros(0, dtype=np.float32)
        connectivities = sparse.csr_matrix(
            (weight_arr, (row_arr, col_arr)), shape=(n_cells, n_cells)
        )
        connectivities = _apply_density_correction(
            connectivities, density_correction=density_correction
        )
        distances_matrix = sparse.csr_matrix(
            (dist_arr, (row_arr, col_arr)), shape=(n_cells, n_cells)
        )
        key = f"k{int(k_value)}"
        graphs[key] = NeighborhoodGraph(
            key=key,
            connectivities=connectivities,
            distances=distances_matrix,
            mode="knn",
            k=int(k_value),
            metadata=_graph_metadata(
                connectivities,
                sample_ids=samples,
                mode="knn",
                kernel=kernel,
                max_neighbors=max_neighbors,
                density_correction=density_correction,
                k=int(k_value),
            ),
        )
    return graphs


__all__ = ["NeighborhoodGraph", "build_knn_graphs", "build_radius_graphs"]
