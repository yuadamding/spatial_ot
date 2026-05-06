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
    full_neighbor_counts: np.ndarray | None = None
    retained_neighbor_counts: np.ndarray | None = None
    full_neighbor_radii_um: np.ndarray | None = None
    retained_neighbor_radii_um: np.ndarray | None = None
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


def _canonical_kernel(kernel: str) -> str:
    requested = str(kernel or "gaussian").strip().lower()
    aliases = {
        "binary": "uniform",
        "none": "uniform",
        "inverse": "inverse_distance",
        "inverse_distance": "inverse_distance",
        "rbf": "gaussian",
        "gaussian": "gaussian",
        "uniform": "uniform",
    }
    try:
        return aliases[requested]
    except KeyError as exc:
        raise ValueError(
            "kernel must be gaussian, uniform, or inverse_distance "
            "(aliases: rbf, binary, none, inverse)."
        ) from exc


def _kernel_weights(distances: np.ndarray, *, kernel: str, scale: float) -> np.ndarray:
    dist = np.asarray(distances, dtype=np.float32)
    requested = _canonical_kernel(kernel)
    if requested == "uniform":
        return np.ones_like(dist, dtype=np.float32)
    if requested == "inverse_distance":
        return (1.0 / np.maximum(dist, 1e-6)).astype(np.float32)
    if requested == "gaussian":
        bandwidth = max(float(scale), 1e-6)
        return np.exp(-0.5 * (dist / bandwidth) ** 2).astype(np.float32)
    raise AssertionError("unreachable kernel branch")


def _stratified_distance_cap(
    *,
    indices: np.ndarray,
    distances: np.ndarray,
    weights: np.ndarray,
    max_neighbors: int,
    radius_um: float,
    n_shells: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cap dense radius neighborhoods while retaining outer radial shells."""

    if indices.size <= int(max_neighbors):
        return indices, distances, weights
    shells = max(int(n_shells), 1)
    denom = max(float(radius_um), 1e-8)
    shell_idx = np.floor(np.clip(distances / denom, 0.0, 0.999999) * shells).astype(
        np.int32
    )
    present = [int(shell) for shell in range(shells) if np.any(shell_idx == shell)]
    if not present:
        order = np.argsort(distances, kind="stable")[: int(max_neighbors)]
        return indices[order], distances[order], weights[order]
    selected: list[int] = []
    base_quota = int(max_neighbors) // len(present)
    remainder = int(max_neighbors) - base_quota * len(present)
    for pos, shell in enumerate(present):
        quota = base_quota + (1 if pos < remainder else 0)
        if quota <= 0:
            continue
        candidates = np.flatnonzero(shell_idx == shell)
        order = candidates[np.argsort(distances[candidates], kind="stable")]
        selected.extend(int(idx) for idx in order[:quota])
    if len(selected) < int(max_neighbors):
        remaining = np.setdiff1d(np.arange(indices.size), np.asarray(selected), assume_unique=False)
        top_up = remaining[np.argsort(distances[remaining], kind="stable")]
        selected.extend(int(idx) for idx in top_up[: int(max_neighbors) - len(selected)])
    chosen = np.asarray(selected[: int(max_neighbors)], dtype=np.int64)
    chosen = chosen[np.argsort(distances[chosen], kind="stable")]
    return indices[chosen], distances[chosen], weights[chosen]


def _apply_density_correction(
    connectivities: sparse.csr_matrix,
    *,
    density_correction: float,
    degree_reference: np.ndarray | None = None,
) -> sparse.csr_matrix:
    alpha = float(np.clip(float(density_correction), 0.0, 1.0))
    if alpha <= 0.0 or connectivities.nnz == 0:
        return connectivities.tocsr()
    coo = connectivities.tocoo(copy=True)
    if degree_reference is None:
        degree = np.asarray(connectivities.sum(axis=1)).reshape(-1).astype(np.float64)
    else:
        degree = np.asarray(degree_reference, dtype=np.float64).reshape(-1)
        if degree.shape[0] != connectivities.shape[0]:
            raise ValueError("degree_reference must have one value per graph row.")
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
    full_neighbor_counts: np.ndarray | None = None,
    retained_neighbor_counts: np.ndarray | None = None,
    radius_um: float | None = None,
    k: int | None = None,
    shell_coordinate_system: str | None = None,
) -> dict[str, object]:
    degree = np.diff(matrix.indptr).astype(np.int32)
    full_counts = (
        np.asarray(full_neighbor_counts, dtype=np.int32).reshape(-1)
        if full_neighbor_counts is not None
        else degree
    )
    retained_counts = (
        np.asarray(retained_neighbor_counts, dtype=np.int32).reshape(-1)
        if retained_neighbor_counts is not None
        else degree
    )
    retention_fraction = np.ones(full_counts.shape[0], dtype=np.float32)
    positive = full_counts > 0
    retention_fraction[positive] = retained_counts[positive] / full_counts[positive]
    sample_values, sample_counts = np.unique(sample_ids.astype(str), return_counts=True)
    return {
        "mode": str(mode),
        "radius_um": None if radius_um is None else float(radius_um),
        "k": None if k is None else int(k),
        "kernel": _canonical_kernel(kernel),
        "max_neighbors": int(max_neighbors),
        "density_correction": float(density_correction),
        "density_correction_degree_source": "full_neighbor_counts"
        if full_neighbor_counts is not None
        else "retained_weighted_degree",
        "n_cells": int(matrix.shape[0]),
        "n_edges": int(matrix.nnz),
        "mean_degree": float(np.mean(retained_counts)) if retained_counts.size else 0.0,
        "max_degree": int(np.max(retained_counts)) if retained_counts.size else 0,
        "mean_full_degree": float(np.mean(full_counts)) if full_counts.size else 0.0,
        "max_full_degree": int(np.max(full_counts)) if full_counts.size else 0,
        "mean_retained_degree": float(np.mean(retained_counts))
        if retained_counts.size
        else 0.0,
        "max_retained_degree": int(np.max(retained_counts))
        if retained_counts.size
        else 0,
        "mean_neighbor_retention_fraction": float(np.mean(retention_fraction))
        if retention_fraction.size
        else 1.0,
        "isolated_fraction": float(np.mean(full_counts == 0)) if full_counts.size else 0.0,
        "shell_coordinate_system": str(shell_coordinate_system or "unknown"),
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
        full_counts = np.zeros(int(xy.shape[0]), dtype=np.int32)
        retained_counts = np.zeros(int(xy.shape[0]), dtype=np.int32)
        full_radii = np.full(int(xy.shape[0]), float(radius), dtype=np.float32)
        retained_radii = np.full(int(xy.shape[0]), float(radius), dtype=np.float32)
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
                row_full_count = int(dist.size)
                full_counts[int(global_idx[row_local])] = row_full_count
                if dist.size == 0:
                    continue
                row_weights = _kernel_weights(
                    dist, kernel=kernel, scale=max(radius / 2.0, 1e-6)
                )
                if dist.size > max_neighbors:
                    ind, dist, row_weights = _stratified_distance_cap(
                        indices=ind,
                        distances=dist,
                        weights=row_weights,
                        max_neighbors=max_neighbors,
                        radius_um=float(radius),
                    )
                retained_counts[int(global_idx[row_local])] = int(dist.size)
                rows.append(
                    np.full(ind.shape[0], int(global_idx[row_local]), dtype=np.int64)
                )
                cols.append(global_idx[ind].astype(np.int64, copy=False))
                distances.append(dist.astype(np.float32, copy=False))
                weights.append(row_weights.astype(np.float32, copy=False))
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
            connectivities,
            density_correction=density_correction,
            degree_reference=full_counts,
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
            full_neighbor_counts=full_counts,
            retained_neighbor_counts=retained_counts,
            full_neighbor_radii_um=full_radii,
            retained_neighbor_radii_um=retained_radii,
            metadata=_graph_metadata(
                connectivities,
                sample_ids=samples,
                mode="radius",
                kernel=kernel,
                max_neighbors=max_neighbors,
                density_correction=density_correction,
                full_neighbor_counts=full_counts,
                retained_neighbor_counts=retained_counts,
                radius_um=float(radius),
                shell_coordinate_system="physical_radius_fraction",
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
        full_counts = np.zeros(int(xy.shape[0]), dtype=np.int32)
        retained_counts = np.zeros(int(xy.shape[0]), dtype=np.int32)
        full_radii = np.zeros(int(xy.shape[0]), dtype=np.float32)
        retained_radii = np.zeros(int(xy.shape[0]), dtype=np.float32)
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
                row_full_count = int(min(dist.size, int(k_value)))
                full_counts[int(global_idx[row_local])] = row_full_count
                if row_full_count > 0:
                    full_radii[int(global_idx[row_local])] = float(
                        np.max(dist[:row_full_count])
                    )
                if dist.size == 0:
                    continue
                keep = min(dist.size, int(k_value), max_neighbors)
                dist = dist[:keep]
                ind = ind[:keep]
                retained_counts[int(global_idx[row_local])] = int(keep)
                if keep > 0:
                    retained_radii[int(global_idx[row_local])] = float(np.max(dist))
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
            connectivities,
            density_correction=density_correction,
            degree_reference=full_counts,
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
            full_neighbor_counts=full_counts,
            retained_neighbor_counts=retained_counts,
            full_neighbor_radii_um=full_radii,
            retained_neighbor_radii_um=retained_radii,
            metadata=_graph_metadata(
                connectivities,
                sample_ids=samples,
                mode="knn",
                kernel=kernel,
                max_neighbors=max_neighbors,
                density_correction=density_correction,
                full_neighbor_counts=full_counts,
                retained_neighbor_counts=retained_counts,
                k=int(k_value),
                shell_coordinate_system="row_max_distance_fraction",
            ),
        )
    return graphs


__all__ = ["NeighborhoodGraph", "build_knn_graphs", "build_radius_graphs"]
