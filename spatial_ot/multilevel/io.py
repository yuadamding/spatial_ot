from __future__ import annotations

from dataclasses import asdict
import json
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import subprocess
import warnings

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
import torch

from ..config import DeepFeatureConfig, MultilevelExperimentConfig
from ..deep.features import SpatialOTFeatureEncoder, fit_deep_features, save_deep_feature_history
from ..feature_source import resolve_h5ad_features
from .core import _resolve_compute_device, fit_multilevel_ot
from .geometry import (
    _shape_descriptor_frame,
    _shape_leakage_balanced_accuracy,
    _shape_leakage_permutation_baseline,
    _shape_leakage_spatial_block_accuracy,
)
from .types import MultilevelOTResult, RegionGeometry

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _compute_subregion_embedding(weights: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    try:
        import umap.umap_ as umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(30, max(3, weights.shape[0] - 1)),
            min_dist=0.2,
            metric="euclidean",
            random_state=seed,
            transform_seed=seed,
        )
        return reducer.fit_transform(weights).astype(np.float32), "UMAP"
    except Exception:
        pca = PCA(n_components=2, random_state=seed)
        return pca.fit_transform(weights).astype(np.float32), "PCA"


def _package_version() -> str:
    try:
        return version("spatial-ot")
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if pyproject.exists():
            payload = tomllib.loads(pyproject.read_text())
            return str(payload.get("project", {}).get("version", "unknown"))
        return "unknown"


def _git_sha() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return completed.stdout.strip() or None
    return None


def _cluster_palette(n_clusters: int) -> np.ndarray:
    cmap_name = "tab20" if n_clusters <= 20 else "gist_ncar"
    cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(n_clusters)
    rgba = np.asarray([cmap(i) for i in range(n_clusters)], dtype=np.float32)
    return np.clip(np.rint(rgba[:, :3] * 255.0), 0, 255).astype(np.uint8)


def _runtime_memory_snapshot(device: torch.device) -> dict[str, float | int | bool | str]:
    snapshot: dict[str, float | int | bool | str] = {
        "device": str(device),
        "cuda": bool(device.type == "cuda" and torch.cuda.is_available()),
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        return snapshot
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        free_bytes, total_bytes = 0, 0
    snapshot.update(
        {
            "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "memory_free_bytes": int(free_bytes),
            "memory_total_bytes": int(total_bytes),
        }
    )
    return snapshot


def _latent_source_label(feature_source: dict, deep_summary: dict) -> str:
    if bool(deep_summary.get("enabled")):
        output_embedding = deep_summary.get("output_embedding")
        return f"deep_{output_embedding}" if output_embedding is not None else "deep_unspecified"

    feature_key = str(feature_source.get("feature_key", ""))
    input_mode = str(feature_source.get("input_mode", "obsm"))
    preprocessing = str(feature_source.get("preprocessing", ""))
    if feature_key.startswith("X_spatial_ot_x_svd_"):
        return "prepared_svd"
    if input_mode == "X" and "truncated_svd" in preprocessing:
        return "runtime_svd"
    if feature_key == "X":
        return "raw_x"
    return f"obsm:{feature_key}"


def _extract_count_target(adata: ad.AnnData, *, count_layer: str | None):
    if count_layer is None:
        return None, None
    layer_key = str(count_layer)
    if layer_key in {"X", "counts"}:
        if adata.X is None:
            raise ValueError("deep.count_layer requested the primary count matrix, but adata.X is missing.")
        return adata.X, "X"
    if layer_key not in adata.layers:
        raise KeyError(f"deep.count_layer '{layer_key}' was not found in adata.layers.")
    return adata.layers[layer_key], layer_key


def _method_stack_summary(
    *,
    feature_source: dict,
    deep_summary: dict,
    feature_obsm_key: str,
) -> dict[str, object]:
    return {
        "method_family": "multilevel_ot",
        "active_path": "multilevel-ot",
        "core_model": "shape_normalized_multilevel_semi_relaxed_ot",
        "deep_feature_adapter": (
            str(deep_summary.get("method", "none"))
            if bool(deep_summary.get("enabled"))
            else "none"
        ),
        "latent_used_for_ot": _latent_source_label(feature_source, deep_summary),
        "ot_feature_obsm_key": str(feature_obsm_key),
        "feature_input_mode": str(feature_source.get("input_mode", "obsm")),
        "feature_space_kind": str(feature_source.get("feature_space_kind", "unknown")),
        "legacy_teacher_student_used": False,
        "communication_source": "none",
        "cell_projection_mode": "approximate_assigned_subregion",
    }


def _assigned_transport_cost_decomposition(result: MultilevelOTResult) -> dict[str, float]:
    geometry = np.asarray(result.subregion_assigned_geometry_transport_costs, dtype=np.float64)
    feature = np.asarray(result.subregion_assigned_feature_transport_costs, dtype=np.float64)
    transform = np.asarray(result.subregion_assigned_transform_penalties, dtype=np.float64)
    overlap = np.asarray(result.subregion_assigned_overlap_consistency_penalties, dtype=np.float64)
    transport_plus_transform = geometry + feature + transform
    assigned_transport_objective = np.asarray(
        result.subregion_cluster_transport_costs[
            np.arange(result.subregion_cluster_labels.shape[0], dtype=np.int64),
            result.subregion_cluster_labels.astype(np.int64),
        ],
        dtype=np.float64,
    )
    assigned_total_objective = np.asarray(
        result.subregion_cluster_costs[
            np.arange(result.subregion_cluster_labels.shape[0], dtype=np.int64),
            result.subregion_cluster_labels.astype(np.int64),
        ],
        dtype=np.float64,
    )
    transport_sum = float(np.sum(transport_plus_transform))
    transport_objective_sum = float(np.sum(assigned_transport_objective))
    total_objective_sum = float(np.sum(assigned_total_objective))
    transport_denom = max(transport_sum, 1e-12)
    transport_objective_denom = max(transport_objective_sum, 1e-12)
    total_objective_denom = max(total_objective_sum, 1e-12)
    return {
        "mean_geometry_transport_cost": float(np.mean(geometry)) if geometry.size else 0.0,
        "mean_feature_transport_cost": float(np.mean(feature)) if feature.size else 0.0,
        "mean_transform_penalty": float(np.mean(transform)) if transform.size else 0.0,
        "mean_overlap_consistency_penalty": float(np.mean(overlap)) if overlap.size else 0.0,
        "mean_transport_plus_transform_cost": float(np.mean(transport_plus_transform)) if transport_plus_transform.size else 0.0,
        "mean_regularized_objective": float(np.mean(assigned_transport_objective)) if assigned_transport_objective.size else 0.0,
        "mean_transport_assignment_objective": float(np.mean(assigned_transport_objective)) if assigned_transport_objective.size else 0.0,
        "mean_total_assignment_cost": float(np.mean(assigned_total_objective)) if assigned_total_objective.size else 0.0,
        "mean_ot_regularization_gap": float(np.mean(assigned_transport_objective - transport_plus_transform)) if assigned_transport_objective.size else 0.0,
        "geometry_transport_fraction": float(np.sum(geometry) / transport_denom),
        "feature_transport_fraction": float(np.sum(feature) / transport_denom),
        "transform_penalty_fraction": float(np.sum(transform) / transport_denom),
        "overlap_consistency_fraction_of_total": float(np.sum(overlap) / total_objective_denom),
        "ot_regularization_gap_fraction_of_transport_objective": float(
            np.sum(assigned_transport_objective - transport_plus_transform) / transport_objective_denom
        ),
    }


def _zscore_columns(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return arr.astype(np.float32, copy=False)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((arr - mean) / std).astype(np.float32, copy=False)


def _native_subregion_embedding(result: MultilevelOTResult) -> np.ndarray:
    summaries = np.asarray(result.subregion_measure_summaries, dtype=np.float32)
    if summaries.ndim == 2 and summaries.shape[0] == result.subregion_cluster_labels.shape[0] and summaries.shape[1] > 0:
        return _zscore_columns(summaries)
    weights = np.asarray(result.subregion_atom_weights, dtype=np.float32)
    return _zscore_columns(weights)


def _subregion_embedding_compactness(result: MultilevelOTResult) -> dict[str, object]:
    embedding = _native_subregion_embedding(result)
    labels = np.asarray(result.subregion_cluster_labels, dtype=np.int32)
    if embedding.shape[0] == 0:
        return {
            "native_embedding_dim": 0,
            "silhouette_native": None,
            "davies_bouldin_native": None,
            "calinski_harabasz_native": None,
            "within_cluster_centroid_distance_mean": None,
            "within_cluster_centroid_distance_median": None,
            "between_cluster_centroid_distance_min": None,
            "between_cluster_centroid_distance_mean": None,
            "compactness_ratio": None,
            "within_cluster_assigned_transport_cost_mean": None,
            "within_cluster_assigned_transport_cost_median": None,
            "within_cluster_assigned_transport_cost_iqr": None,
            "within_cluster_assigned_total_cost_mean": None,
            "within_cluster_assigned_total_cost_median": None,
            "within_cluster_assigned_cost_mean": None,
            "within_cluster_assigned_cost_median": None,
            "within_cluster_assigned_cost_iqr": None,
            "within_cluster_assigned_cost_by_cluster": {},
        }

    unique_labels = np.unique(labels)
    silhouette_native = None
    davies_bouldin_native = None
    calinski_harabasz_native = None
    if unique_labels.size >= 2 and embedding.shape[0] > unique_labels.size:
        try:
            silhouette_native = float(silhouette_score(embedding, labels, metric="euclidean"))
        except Exception:
            silhouette_native = None
        try:
            davies_bouldin_native = float(davies_bouldin_score(embedding, labels))
        except Exception:
            davies_bouldin_native = None
        try:
            calinski_harabasz_native = float(calinski_harabasz_score(embedding, labels))
        except Exception:
            calinski_harabasz_native = None

    within_distances: list[np.ndarray] = []
    cluster_centroids: list[np.ndarray] = []
    assigned_transport_cost = np.asarray(
        result.subregion_cluster_transport_costs[
            np.arange(result.subregion_cluster_labels.shape[0], dtype=np.int64),
            result.subregion_cluster_labels.astype(np.int64),
        ],
        dtype=np.float64,
    )
    assigned_total_cost = np.asarray(
        result.subregion_cluster_costs[
            np.arange(result.subregion_cluster_labels.shape[0], dtype=np.int64),
            result.subregion_cluster_labels.astype(np.int64),
        ],
        dtype=np.float64,
    )
    cost_by_cluster: dict[str, dict[str, float]] = {}
    for cid in unique_labels.tolist():
        mask = labels == int(cid)
        cluster_embedding = embedding[mask]
        centroid = cluster_embedding.mean(axis=0)
        cluster_centroids.append(np.asarray(centroid, dtype=np.float64))
        dist = np.linalg.norm(cluster_embedding - centroid[None, :], axis=1)
        within_distances.append(np.asarray(dist, dtype=np.float64))
        cluster_transport_cost = assigned_transport_cost[mask]
        cost_by_cluster[f"C{int(cid)}"] = {
            "transport_mean": float(np.mean(cluster_transport_cost)),
            "transport_median": float(np.median(cluster_transport_cost)),
            "transport_iqr": float(np.quantile(cluster_transport_cost, 0.75) - np.quantile(cluster_transport_cost, 0.25)),
        }

    within_concat = np.concatenate(within_distances) if within_distances else np.zeros(0, dtype=np.float64)
    if len(cluster_centroids) >= 2:
        centroid_matrix = np.vstack(cluster_centroids).astype(np.float64)
        centroid_dist = np.sqrt(np.sum((centroid_matrix[:, None, :] - centroid_matrix[None, :, :]) ** 2, axis=2))
        tri = centroid_dist[np.triu_indices(centroid_dist.shape[0], k=1)]
        between_centroid_distance_min = float(np.min(tri)) if tri.size else None
        between_centroid_distance_mean = float(np.mean(tri)) if tri.size else None
    else:
        between_centroid_distance_min = None
        between_centroid_distance_mean = None
    compactness_ratio = (
        float(np.mean(within_concat) / max(float(between_centroid_distance_min), 1e-12))
        if within_concat.size and between_centroid_distance_min is not None and np.isfinite(between_centroid_distance_min)
        else None
    )
    return {
        "native_embedding_dim": int(embedding.shape[1]),
        "silhouette_native": silhouette_native,
        "davies_bouldin_native": davies_bouldin_native,
        "calinski_harabasz_native": calinski_harabasz_native,
        "within_cluster_centroid_distance_mean": float(np.mean(within_concat)) if within_concat.size else None,
        "within_cluster_centroid_distance_median": float(np.median(within_concat)) if within_concat.size else None,
        "between_cluster_centroid_distance_min": between_centroid_distance_min,
        "between_cluster_centroid_distance_mean": between_centroid_distance_mean,
        "compactness_ratio": compactness_ratio,
        "within_cluster_assigned_transport_cost_mean": float(np.mean(assigned_transport_cost)) if assigned_transport_cost.size else None,
        "within_cluster_assigned_transport_cost_median": float(np.median(assigned_transport_cost)) if assigned_transport_cost.size else None,
        "within_cluster_assigned_transport_cost_iqr": (
            float(np.quantile(assigned_transport_cost, 0.75) - np.quantile(assigned_transport_cost, 0.25))
            if assigned_transport_cost.size
            else None
        ),
        "within_cluster_assigned_total_cost_mean": float(np.mean(assigned_total_cost)) if assigned_total_cost.size else None,
        "within_cluster_assigned_total_cost_median": float(np.median(assigned_total_cost)) if assigned_total_cost.size else None,
        "within_cluster_assigned_cost_mean": float(np.mean(assigned_total_cost)) if assigned_total_cost.size else None,
        "within_cluster_assigned_cost_median": float(np.median(assigned_total_cost)) if assigned_total_cost.size else None,
        "within_cluster_assigned_cost_iqr": (
            float(np.quantile(assigned_total_cost, 0.75) - np.quantile(assigned_total_cost, 0.25))
            if assigned_total_cost.size
            else None
        ),
        "within_cluster_assigned_cost_by_cluster": cost_by_cluster,
    }


def _deterministic_subsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, num=max_points, dtype=np.int64)


def _cell_adjacency_same_label_fraction(
    coords_um: np.ndarray,
    labels: np.ndarray,
    *,
    max_cells: int = 50000,
    neighbors: int = 6,
) -> float | None:
    coords = np.asarray(coords_um, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int32)
    if coords.shape[0] < 2:
        return None
    keep = _deterministic_subsample_indices(coords.shape[0], max_cells)
    coords = coords[keep]
    labels_arr = labels_arr[keep]
    n_neighbors = min(int(neighbors) + 1, coords.shape[0])
    if n_neighbors < 2:
        return None
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(coords)
    neigh = nn.kneighbors(coords, return_distance=False)
    neigh = neigh[:, 1:]
    same = labels_arr[:, None] == labels_arr[neigh]
    return float(np.mean(same.astype(np.float32))) if same.size else None


def _subregion_graph_metrics(
    *,
    n_cells: int,
    result: MultilevelOTResult,
    radius_um: float,
    stride_um: float,
    coords_um: np.ndarray,
) -> dict[str, object]:
    n_subregions = len(result.subregion_members)
    labels = np.asarray(result.subregion_cluster_labels, dtype=np.int32)
    probs = np.asarray(result.subregion_cluster_probs, dtype=np.float64)
    if n_subregions < 2:
        return {
            "subregion_graph_edge_count": 0,
            "overlap_edge_count": 0,
            "proximity_only_edge_count": 0,
            "same_label_edge_fraction": None,
            "boundary_edge_fraction": None,
            "boundary_entropy_mean": None,
            "boundary_entropy_p95": None,
            "overlap_probability_l2_mean": None,
            "high_overlap_edge_count": 0,
            "high_overlap_same_label_fraction": None,
            "isolated_subregion_fraction": None,
            "cluster_connected_components": {},
            "cell_adjacency_same_label_fraction": _cell_adjacency_same_label_fraction(coords_um, result.cell_cluster_labels),
        }

    member_sizes = np.asarray([len(members) for members in result.subregion_members], dtype=np.int64)
    rows = np.concatenate([np.asarray(members, dtype=np.int64) for members in result.subregion_members], dtype=np.int64)
    cols = np.concatenate(
        [np.full(len(members), idx, dtype=np.int64) for idx, members in enumerate(result.subregion_members)],
        dtype=np.int64,
    )
    incidence = sparse.csr_matrix(
        (np.ones(rows.shape[0], dtype=np.int8), (rows, cols)),
        shape=(int(n_cells), int(n_subregions)),
        dtype=np.int8,
    )
    overlap = (incidence.T @ incidence).tocoo()
    edge_info: dict[tuple[int, int], dict[str, float | int | None]] = {}
    for i, j, intersection in zip(overlap.row.tolist(), overlap.col.tolist(), overlap.data.tolist(), strict=False):
        if int(i) >= int(j) or int(intersection) <= 0:
            continue
        union = int(member_sizes[int(i)] + member_sizes[int(j)] - int(intersection))
        edge_info[(int(i), int(j))] = {
            "intersection": int(intersection),
            "jaccard": float(int(intersection) / max(union, 1)),
            "distance": None,
        }

    centers = np.asarray(result.subregion_centers_um, dtype=np.float32)
    proximity_radius = max(float(stride_um) * 1.5, 1e-6)
    if centers.shape[0] >= 2:
        nn = NearestNeighbors(radius=proximity_radius, metric="euclidean")
        nn.fit(centers)
        distances, neighbors = nn.radius_neighbors(centers, return_distance=True)
        for src, (nbrs, dists) in enumerate(zip(neighbors, distances, strict=False)):
            for dst, dist in zip(np.asarray(nbrs, dtype=np.int64).tolist(), np.asarray(dists, dtype=np.float32).tolist(), strict=False):
                if int(dst) <= int(src):
                    continue
                key = (int(src), int(dst))
                entry = edge_info.setdefault(key, {"intersection": 0, "jaccard": 0.0, "distance": None})
                if entry["distance"] is None or float(dist) < float(entry["distance"]):
                    entry["distance"] = float(dist)
    if not edge_info and centers.shape[0] >= 2:
        nn = NearestNeighbors(n_neighbors=min(3, centers.shape[0]), metric="euclidean")
        nn.fit(centers)
        distances, neighbors = nn.kneighbors(centers, return_distance=True)
        for src, (nbrs, dists) in enumerate(zip(neighbors, distances, strict=False)):
            for dst, dist in zip(np.asarray(nbrs, dtype=np.int64)[1:].tolist(), np.asarray(dists, dtype=np.float32)[1:].tolist(), strict=False):
                key = tuple(sorted((int(src), int(dst))))
                entry = edge_info.setdefault(key, {"intersection": 0, "jaccard": 0.0, "distance": None})
                if entry["distance"] is None or float(dist) < float(entry["distance"]):
                    entry["distance"] = float(dist)

    if not edge_info:
        return {
            "subregion_graph_edge_count": 0,
            "overlap_edge_count": 0,
            "proximity_only_edge_count": 0,
            "same_label_edge_fraction": None,
            "boundary_edge_fraction": None,
            "boundary_entropy_mean": None,
            "boundary_entropy_p95": None,
            "overlap_probability_l2_mean": None,
            "high_overlap_edge_count": 0,
            "high_overlap_same_label_fraction": None,
            "isolated_subregion_fraction": 1.0,
            "cluster_connected_components": {f"C{int(cid)}": int(np.sum(labels == int(cid))) for cid in np.unique(labels).tolist()},
            "cell_adjacency_same_label_fraction": _cell_adjacency_same_label_fraction(coords_um, result.cell_cluster_labels),
        }

    edge_pairs = np.asarray(list(edge_info.keys()), dtype=np.int64)
    edge_i = edge_pairs[:, 0]
    edge_j = edge_pairs[:, 1]
    overlap_jaccard = np.asarray([float(edge_info[key]["jaccard"]) for key in edge_info], dtype=np.float64)
    distances = np.asarray(
        [
            float(edge_info[key]["distance"]) if edge_info[key]["distance"] is not None else np.nan
            for key in edge_info
        ],
        dtype=np.float64,
    )
    overlap_mask = overlap_jaccard > 0.0
    proximity_weight = np.exp(-np.nan_to_num(distances, nan=proximity_radius) / max(proximity_radius, 1e-6))
    edge_weight = np.where(overlap_mask, np.maximum(overlap_jaccard, proximity_weight), proximity_weight)
    edge_weight = np.clip(edge_weight.astype(np.float64), 1e-8, None)
    same_label = labels[edge_i] == labels[edge_j]
    boundary_mask = ~same_label

    mean_probs = 0.5 * (probs[edge_i] + probs[edge_j])
    edge_entropy = -np.sum(mean_probs * np.log(np.clip(mean_probs, 1e-8, None)), axis=1)
    overlap_prob_l2 = np.linalg.norm(probs[edge_i] - probs[edge_j], axis=1)
    high_overlap_mask = overlap_jaccard >= 0.5

    degree = np.bincount(
        np.concatenate([edge_i, edge_j]),
        weights=np.concatenate([edge_weight, edge_weight]),
        minlength=n_subregions,
    )
    adjacency = sparse.csr_matrix(
        (
            np.concatenate([edge_weight, edge_weight]),
            (
                np.concatenate([edge_i, edge_j]),
                np.concatenate([edge_j, edge_i]),
            ),
        ),
        shape=(n_subregions, n_subregions),
        dtype=np.float32,
    )
    component_counts: dict[str, int] = {}
    for cid in np.unique(labels).tolist():
        nodes = np.flatnonzero(labels == int(cid))
        if nodes.size == 0:
            continue
        if nodes.size == 1:
            component_counts[f"C{int(cid)}"] = 1
            continue
        subgraph = adjacency[nodes][:, nodes]
        n_components, _ = connected_components(subgraph, directed=False, return_labels=True)
        component_counts[f"C{int(cid)}"] = int(n_components)

    return {
        "subregion_graph_edge_count": int(edge_pairs.shape[0]),
        "overlap_edge_count": int(np.sum(overlap_mask)),
        "proximity_only_edge_count": int(np.sum(~overlap_mask)),
        "same_label_edge_fraction": float(np.sum(edge_weight[same_label]) / np.sum(edge_weight)),
        "boundary_edge_fraction": float(np.sum(edge_weight[boundary_mask]) / np.sum(edge_weight)),
        "boundary_entropy_mean": float(np.mean(edge_entropy[boundary_mask])) if np.any(boundary_mask) else None,
        "boundary_entropy_p95": float(np.quantile(edge_entropy[boundary_mask], 0.95)) if np.any(boundary_mask) else None,
        "overlap_probability_l2_mean": (
            float(np.sum(overlap_jaccard[overlap_mask] * overlap_prob_l2[overlap_mask]) / np.sum(overlap_jaccard[overlap_mask]))
            if np.any(overlap_mask)
            else None
        ),
        "high_overlap_edge_count": int(np.sum(high_overlap_mask)),
        "high_overlap_same_label_fraction": (
            float(np.mean(same_label[high_overlap_mask].astype(np.float32)))
            if np.any(high_overlap_mask)
            else None
        ),
        "isolated_subregion_fraction": float(np.mean((degree <= 0).astype(np.float32))),
        "cluster_connected_components": component_counts,
        "cell_adjacency_same_label_fraction": _cell_adjacency_same_label_fraction(coords_um, result.cell_cluster_labels),
    }


def _cost_reliability_metrics(result: MultilevelOTResult) -> dict[str, object]:
    effective_eps = np.asarray(result.subregion_candidate_effective_eps_matrix, dtype=np.float64)
    used_fallback = np.asarray(result.subregion_candidate_used_ot_fallback_matrix, dtype=bool)
    sorted_costs = np.sort(np.asarray(result.subregion_cluster_costs, dtype=np.float64), axis=1)
    margins = (
        sorted_costs[:, 1] - sorted_costs[:, 0]
        if sorted_costs.shape[1] >= 2
        else np.full(sorted_costs.shape[0], np.nan, dtype=np.float64)
    )
    finite_margins = margins[np.isfinite(margins)]
    mixed_eps = np.mean((np.max(effective_eps, axis=1) - np.min(effective_eps, axis=1)) > 1e-8) if effective_eps.size else 0.0
    mixed_fallback = np.mean(np.any(used_fallback != used_fallback[:, :1], axis=1)) if used_fallback.size else 0.0
    return {
        "effective_eps_matrix_available": True,
        "fallback_fraction_all_costs": float(np.mean(used_fallback.astype(np.float32))) if used_fallback.size else 0.0,
        "fallback_fraction_assigned": float(np.mean(result.subregion_assigned_used_ot_fallback.astype(np.float32))),
        "mixed_candidate_effective_eps_fraction": float(mixed_eps),
        "mixed_candidate_fallback_fraction": float(mixed_fallback),
        "effective_eps_min": float(np.min(effective_eps)) if effective_eps.size else None,
        "effective_eps_max": float(np.max(effective_eps)) if effective_eps.size else None,
        "assignment_margin_mean": float(np.mean(finite_margins)) if finite_margins.size else None,
        "assignment_margin_median": float(np.median(finite_margins)) if finite_margins.size else None,
        "assignment_margin_p10": float(np.quantile(finite_margins, 0.10)) if finite_margins.size else None,
        "fallback_fraction_by_cluster": {
            f"C{int(cid)}": float(np.mean(used_fallback[:, int(cid)].astype(np.float32)))
            for cid in range(used_fallback.shape[1])
        }
        if used_fallback.ndim == 2
        else {},
    }


def _transform_diagnostics(result: MultilevelOTResult) -> dict[str, float | None]:
    rotation = np.asarray(result.subregion_assigned_transform_rotation_deg, dtype=np.float64)
    reflection = np.asarray(result.subregion_assigned_transform_reflection, dtype=bool)
    scale = np.asarray(result.subregion_assigned_transform_scale, dtype=np.float64)
    translation_norm = np.asarray(result.subregion_assigned_transform_translation_norm, dtype=np.float64)
    if rotation.size == 0:
        return {
            "mean_abs_rotation_deg": None,
            "p95_abs_rotation_deg": None,
            "reflection_fraction": None,
            "scale_mean": None,
            "scale_deviation_mean": None,
            "scale_deviation_p95": None,
            "translation_norm_mean": None,
            "translation_norm_p95": None,
        }
    abs_rotation = np.abs(rotation)
    scale_dev = np.abs(scale - 1.0)
    return {
        "mean_abs_rotation_deg": float(np.mean(abs_rotation)),
        "p95_abs_rotation_deg": float(np.quantile(abs_rotation, 0.95)),
        "reflection_fraction": float(np.mean(reflection.astype(np.float32))),
        "scale_mean": float(np.mean(scale)),
        "scale_deviation_mean": float(np.mean(scale_dev)),
        "scale_deviation_p95": float(np.quantile(scale_dev, 0.95)),
        "translation_norm_mean": float(np.mean(translation_norm)),
        "translation_norm_p95": float(np.quantile(translation_norm, 0.95)),
    }


def _qc_warning(code: str, severity: str, message: str, *, value: float | int | str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "code": str(code),
        "severity": str(severity),
        "message": str(message),
    }
    if value is not None:
        payload["value"] = value
    return payload


def _build_qc_warnings(
    *,
    feature_embedding_warning: str | None,
    fallback_fraction: float,
    assigned_ot_fallback_fraction: float,
    assigned_effective_eps_values: list[float],
    requested_ot_eps: float,
    coverage_fraction: float,
    mean_assignment_margin: float | None,
    assigned_transport_cost_decomposition: dict[str, float],
    cost_reliability: dict[str, object],
    transform_diagnostics: dict[str, float | None],
    forced_label_fraction: float,
    deep_summary: dict,
) -> list[dict[str, object]]:
    warnings_out: list[dict[str, object]] = [
        _qc_warning(
            "cell_projection_is_approximate_assigned_subregion",
            "info",
            "Cell-level labels are approximate projections from assigned subregions rather than direct cell-level OT posteriors.",
        )
    ]
    if feature_embedding_warning == "umap_exploratory":
        warnings_out.append(
            _qc_warning(
                "umap_feature_space_exploratory",
                "warning",
                "UMAP was used as the OT feature space, so the run should be treated as exploratory.",
            )
        )
    elif feature_embedding_warning == "visualization_embedding_like":
        warnings_out.append(
            _qc_warning(
                "visualization_like_feature_space",
                "warning",
                "A visualization-like embedding was used as the OT feature space, so the run should be treated as exploratory.",
            )
        )
    if fallback_fraction > 0.0:
        warnings_out.append(
            _qc_warning(
                "observed_hull_geometry_fallback_active",
                "warning",
                "Observed-coordinate convex-hull fallback was used for at least one subregion, so boundary-shape invariance is not fully supported.",
                value=float(fallback_fraction),
            )
        )
    if assigned_ot_fallback_fraction > 0.0:
        warnings_out.append(
            _qc_warning(
                "assigned_ot_regularization_fallback_active",
                "warning",
                "At least one assigned OT solve needed a larger effective epsilon than requested.",
                value=float(assigned_ot_fallback_fraction),
            )
        )
    if assigned_effective_eps_values:
        max_effective_eps = float(max(assigned_effective_eps_values))
        if max_effective_eps > float(requested_ot_eps) * 1.5:
            warnings_out.append(
                _qc_warning(
                    "effective_eps_exceeds_requested",
                    "warning",
                    "The effective OT epsilon increased substantially above the requested value during fallback.",
                    value=max_effective_eps,
                )
            )
    if coverage_fraction < 1.0:
        warnings_out.append(
            _qc_warning(
                "incomplete_cell_subregion_coverage",
                "warning",
                "Some cells were not covered by any retained subregion.",
                value=float(coverage_fraction),
            )
        )
    if mean_assignment_margin is not None and np.isfinite(mean_assignment_margin) and mean_assignment_margin < 0.05:
        warnings_out.append(
            _qc_warning(
                "low_mean_assignment_margin",
                "warning",
                "Mean subregion assignment margin is low, so niche assignments may be unstable or weakly separated.",
                value=float(mean_assignment_margin),
            )
        )
    if float(assigned_transport_cost_decomposition.get("geometry_transport_fraction", 0.0)) >= 0.75:
        warnings_out.append(
            _qc_warning(
                "geometry_dominates_assigned_cost",
                "warning",
                "Most of the assigned transport cost comes from geometry rather than feature matching.",
                value=float(assigned_transport_cost_decomposition["geometry_transport_fraction"]),
            )
        )
    if forced_label_fraction > 0.0:
        warnings_out.append(
            _qc_warning(
                "forced_nonempty_cluster_assignment",
                "warning",
                "At least one subregion label was forced to keep every requested cluster nonempty.",
                value=float(forced_label_fraction),
            )
        )
    mixed_candidate_fallback_fraction = float(cost_reliability.get("mixed_candidate_fallback_fraction", 0.0) or 0.0)
    if mixed_candidate_fallback_fraction > 0.0:
        warnings_out.append(
            _qc_warning(
                "candidate_costs_use_mixed_fallback",
                "warning",
                "Some subregions compared candidate clusters under different OT fallback states or effective eps values.",
                value=mixed_candidate_fallback_fraction,
            )
        )
    reflection_fraction = transform_diagnostics.get("reflection_fraction")
    if reflection_fraction is not None and float(reflection_fraction) > 0.0:
        warnings_out.append(
            _qc_warning(
                "reflection_used_in_assigned_transforms",
                "info",
                "At least one assigned subregion-to-cluster alignment used a reflected transform.",
                value=float(reflection_fraction),
            )
        )
    scale_deviation_p95 = transform_diagnostics.get("scale_deviation_p95")
    if scale_deviation_p95 is not None and float(scale_deviation_p95) > 0.25:
        warnings_out.append(
            _qc_warning(
                "large_transform_scale_drift",
                "warning",
                "Assigned transforms show substantial scale drift from 1.0 for at least part of the run.",
                value=float(scale_deviation_p95),
            )
        )
    if bool(deep_summary.get("enabled")) and deep_summary.get("output_embedding") == "joint":
        warnings_out.append(
            _qc_warning(
                "joint_embedding_used_for_ot",
                "info",
                "The OT feature view used the deep joint embedding under explicit opt-in.",
            )
        )
    return warnings_out


def _probability_diagnostics(probs: np.ndarray, *, prefix: str) -> dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return {
            f"{prefix}_assignment_entropy_mean": 0.0,
            f"{prefix}_assignment_entropy_p95": 0.0,
            f"{prefix}_assignment_confidence_mean": 0.0,
            f"{prefix}_assignment_confidence_p05": 0.0,
        }
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, None)), axis=1)
    confidence = np.max(probs, axis=1)
    return {
        f"{prefix}_assignment_entropy_mean": float(np.mean(entropy)),
        f"{prefix}_assignment_entropy_p95": float(np.quantile(entropy, 0.95)),
        f"{prefix}_assignment_confidence_mean": float(np.mean(confidence)),
        f"{prefix}_assignment_confidence_p05": float(np.quantile(confidence, 0.05)),
    }


def _cell_subregion_coverage(n_cells: int, subregion_members: list[np.ndarray]) -> dict[str, float | int]:
    if n_cells <= 0:
        return {
            "covered_cell_count": 0,
            "uncovered_cell_count": 0,
            "cell_subregion_coverage_fraction": 0.0,
        }
    covered = np.zeros(n_cells, dtype=bool)
    for members in subregion_members:
        covered[np.asarray(members, dtype=np.int64)] = True
    covered_count = int(covered.sum())
    return {
        "covered_cell_count": covered_count,
        "uncovered_cell_count": int(n_cells - covered_count),
        "cell_subregion_coverage_fraction": float(covered_count / max(n_cells, 1)),
    }


def _marker_size(n_points: int, *, low: float = 0.5, high: float = 8.0) -> float:
    if n_points <= 1000:
        return high
    if n_points >= 250000:
        return low
    scale = (np.log10(n_points) - np.log10(1000)) / (np.log10(250000) - np.log10(1000))
    scale = float(np.clip(scale, 0.0, 1.0))
    return high - (high - low) * scale


def _safe_filename_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "sample"


def _resolve_sample_plot_coordinate_keys(
    obs: pd.DataFrame,
    *,
    requested_x_key: str | None,
    requested_y_key: str | None,
    metadata_x_key: str | None,
    metadata_y_key: str | None,
) -> tuple[str, str]:
    candidate_pairs: list[tuple[str | None, str | None]] = [
        (requested_x_key, requested_y_key),
        ("original_cell_x", "original_cell_y"),
        ("cell_x", "cell_y"),
        (metadata_x_key, metadata_y_key),
    ]
    for x_key, y_key in candidate_pairs:
        if x_key is None or y_key is None:
            continue
        if x_key in obs.columns and y_key in obs.columns:
            return str(x_key), str(y_key)
    raise KeyError(
        "Could not find a usable spatial coordinate pair for sample niche plots. "
        "Pass --plot-spatial-x-key and --plot-spatial-y-key explicitly."
    )


def _plot_sample_cluster_maps(
    cells_h5ad: str | Path,
    output_dir: str | Path | None = None,
    *,
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    cluster_obs_key: str = "mlot_cluster_int",
    cluster_label_obs_key: str = "mlot_cluster_id",
    cluster_hex_obs_key: str = "mlot_cluster_hex",
    plot_spatial_x_key: str | None = None,
    plot_spatial_y_key: str | None = None,
    default_sample_id: str = "all_cells",
    spatial_scale: float | None = None,
    title_prefix: str,
    output_filename_suffix: str,
    manifest_filename: str,
) -> dict[str, object]:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    cells_h5ad = Path(cells_h5ad)
    output_dir = Path(output_dir) if output_dir is not None else cells_h5ad.parent / "sample_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(cells_h5ad, backed="r")
    try:
        obs = adata.obs.copy()
        metadata = dict(adata.uns["multilevel_ot"]) if "multilevel_ot" in adata.uns else {}
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()

    if cluster_obs_key not in obs.columns and cluster_label_obs_key not in obs.columns:
        raise KeyError(
            f"Expected either cluster obs key '{cluster_obs_key}' or '{cluster_label_obs_key}' in {cells_h5ad}."
        )

    metadata_x_key = metadata.get("spatial_x_key")
    metadata_y_key = metadata.get("spatial_y_key")
    resolved_x_key, resolved_y_key = _resolve_sample_plot_coordinate_keys(
        obs,
        requested_x_key=plot_spatial_x_key,
        requested_y_key=plot_spatial_y_key,
        metadata_x_key=str(metadata_x_key) if metadata_x_key is not None else None,
        metadata_y_key=str(metadata_y_key) if metadata_y_key is not None else None,
    )
    resolved_scale = float(spatial_scale) if spatial_scale is not None else float(metadata.get("spatial_scale", 1.0))

    if sample_obs_key in obs.columns:
        sample_ids = [str(value) for value in pd.unique(obs[sample_obs_key].astype(str))]
        sample_values = obs[sample_obs_key].astype(str).to_numpy()
    else:
        sample_ids = [str(default_sample_id)]
        sample_values = np.full(obs.shape[0], str(default_sample_id), dtype=object)

    if cluster_label_obs_key in obs.columns:
        cluster_names = obs[cluster_label_obs_key].astype(str).to_numpy()
    elif cluster_obs_key in obs.columns:
        cluster_names = np.asarray([f"C{int(value)}" for value in np.asarray(obs[cluster_obs_key], dtype=np.int32)], dtype=object)
    else:
        raise KeyError("No cluster label information was available for sample plotting.")

    if cluster_obs_key in obs.columns:
        cluster_ids = np.asarray(obs[cluster_obs_key], dtype=np.int32)
    else:
        category = pd.Categorical(cluster_names)
        cluster_ids = category.codes.astype(np.int32)
        cluster_names = np.asarray([str(category.categories[idx]) for idx in cluster_ids], dtype=object)

    if cluster_hex_obs_key in obs.columns:
        cluster_hex = obs[cluster_hex_obs_key].astype(str).to_numpy()
    else:
        palette = _cluster_palette(int(cluster_ids.max()) + 1)
        cluster_hex = np.asarray(
            [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette[cluster_ids].tolist()],
            dtype=object,
        )

    coords_um = np.stack(
        [
            np.asarray(obs[resolved_x_key], dtype=np.float32) * resolved_scale,
            np.asarray(obs[resolved_y_key], dtype=np.float32) * resolved_scale,
        ],
        axis=1,
    )

    cluster_display: dict[int, tuple[str, str]] = {}
    for idx, cluster_id in enumerate(cluster_ids.tolist()):
        cluster_display.setdefault(int(cluster_id), (str(cluster_names[idx]), str(cluster_hex[idx])))

    plots: list[dict[str, object]] = []
    for sample_id in sample_ids:
        sample_mask = sample_values == sample_id
        sample_count = int(np.sum(sample_mask))
        if sample_count == 0:
            continue

        fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
        point_size = _marker_size(sample_count)
        sample_cluster_ids = cluster_ids[sample_mask]
        sample_coords = coords_um[sample_mask]
        for cluster_id in np.unique(sample_cluster_ids):
            cluster_mask = sample_cluster_ids == cluster_id
            label_name, color_hex = cluster_display[int(cluster_id)]
            ax.scatter(
                sample_coords[cluster_mask, 0],
                sample_coords[cluster_mask, 1],
                s=point_size,
                color=color_hex,
                linewidths=0,
                alpha=0.85,
                rasterized=sample_count > 20000,
                label=f"{label_name} ({int(np.sum(cluster_mask))})",
            )

        source_name: str | list[str] | None = None
        if source_file_obs_key in obs.columns:
            sources = [str(value) for value in pd.unique(obs.loc[sample_mask, source_file_obs_key].astype(str))]
            if len(sources) == 1:
                source_name = sources[0]
            elif sources:
                source_name = sources

        title = f"{title_prefix}: {sample_id}"
        if isinstance(source_name, str):
            title = f"{title}\n{source_name}"
        ax.set_title(title)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)

        output_png = output_dir / f"{_safe_filename_component(sample_id)}{output_filename_suffix}"
        fig.savefig(output_png, dpi=250, bbox_inches="tight")
        plt.close(fig)

        plots.append(
            {
                "sample_id": str(sample_id),
                "source_h5ad": source_name,
                "n_cells": sample_count,
                "output_png": str(output_png),
            }
        )

    manifest: dict[str, object] = {
        "cells_h5ad": str(cells_h5ad),
        "output_dir": str(output_dir),
        "n_samples": int(len(plots)),
        "sample_obs_key": str(sample_obs_key),
        "source_file_obs_key": str(source_file_obs_key),
        "cluster_obs_key": str(cluster_obs_key),
        "cluster_label_obs_key": str(cluster_label_obs_key),
        "cluster_hex_obs_key": str(cluster_hex_obs_key),
        "plot_spatial_x_key": str(resolved_x_key),
        "plot_spatial_y_key": str(resolved_y_key),
        "spatial_scale": float(resolved_scale),
        "title_prefix": str(title_prefix),
        "output_filename_suffix": str(output_filename_suffix),
        "plots": plots,
    }
    manifest_path = output_dir / manifest_filename
    manifest["manifest_json"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def plot_sample_niche_maps(
    cells_h5ad: str | Path,
    output_dir: str | Path | None = None,
    *,
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    cluster_obs_key: str = "mlot_cluster_int",
    cluster_label_obs_key: str = "mlot_cluster_id",
    cluster_hex_obs_key: str = "mlot_cluster_hex",
    plot_spatial_x_key: str | None = None,
    plot_spatial_y_key: str | None = None,
    default_sample_id: str = "all_cells",
    spatial_scale: float | None = None,
) -> dict[str, object]:
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(cells_h5ad).parent / "sample_niche_plots"
    return _plot_sample_cluster_maps(
        cells_h5ad=cells_h5ad,
        output_dir=resolved_output_dir,
        sample_obs_key=sample_obs_key,
        source_file_obs_key=source_file_obs_key,
        cluster_obs_key=cluster_obs_key,
        cluster_label_obs_key=cluster_label_obs_key,
        cluster_hex_obs_key=cluster_hex_obs_key,
        plot_spatial_x_key=plot_spatial_x_key,
        plot_spatial_y_key=plot_spatial_y_key,
        default_sample_id=default_sample_id,
        spatial_scale=spatial_scale,
        title_prefix="Spatial niche map",
        output_filename_suffix="_spatial_niche_map.png",
        manifest_filename="sample_niche_plots_manifest.json",
    )


def plot_sample_spatial_maps(
    cells_h5ad: str | Path,
    output_dir: str | Path | None = None,
    *,
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    cluster_obs_key: str = "mlot_cluster_int",
    cluster_label_obs_key: str = "mlot_cluster_id",
    cluster_hex_obs_key: str = "mlot_cluster_hex",
    plot_spatial_x_key: str | None = None,
    plot_spatial_y_key: str | None = None,
    default_sample_id: str = "all_cells",
    spatial_scale: float | None = None,
) -> dict[str, object]:
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(cells_h5ad).parent / "sample_spatial_maps"
    return _plot_sample_cluster_maps(
        cells_h5ad=cells_h5ad,
        output_dir=resolved_output_dir,
        sample_obs_key=sample_obs_key,
        source_file_obs_key=source_file_obs_key,
        cluster_obs_key=cluster_obs_key,
        cluster_label_obs_key=cluster_label_obs_key,
        cluster_hex_obs_key=cluster_hex_obs_key,
        plot_spatial_x_key=plot_spatial_x_key,
        plot_spatial_y_key=plot_spatial_y_key,
        default_sample_id=default_sample_id,
        spatial_scale=spatial_scale,
        title_prefix="Shape-normalized multilevel OT cell labels",
        output_filename_suffix="_multilevel_ot_spatial_map.png",
        manifest_filename="sample_spatial_maps_manifest.json",
    )


def plot_sample_niche_maps_from_run_dir(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    cluster_obs_key: str = "mlot_cluster_int",
    cluster_label_obs_key: str = "mlot_cluster_id",
    cluster_hex_obs_key: str = "mlot_cluster_hex",
    plot_spatial_x_key: str | None = None,
    plot_spatial_y_key: str | None = None,
    default_sample_id: str = "all_cells",
    spatial_scale: float | None = None,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    cells_h5ad = run_dir / "cells_multilevel_ot.h5ad"
    if not cells_h5ad.exists():
        raise FileNotFoundError(f"Expected multilevel OT cell output under {cells_h5ad}.")
    resolved_output_dir = Path(output_dir) if output_dir is not None else run_dir / "sample_niche_plots"
    return plot_sample_niche_maps(
        cells_h5ad=cells_h5ad,
        output_dir=resolved_output_dir,
        sample_obs_key=sample_obs_key,
        source_file_obs_key=source_file_obs_key,
        cluster_obs_key=cluster_obs_key,
        cluster_label_obs_key=cluster_label_obs_key,
        cluster_hex_obs_key=cluster_hex_obs_key,
        plot_spatial_x_key=plot_spatial_x_key,
        plot_spatial_y_key=plot_spatial_y_key,
        default_sample_id=default_sample_id,
        spatial_scale=spatial_scale,
    )


def plot_sample_spatial_maps_from_run_dir(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    cluster_obs_key: str = "mlot_cluster_int",
    cluster_label_obs_key: str = "mlot_cluster_id",
    cluster_hex_obs_key: str = "mlot_cluster_hex",
    plot_spatial_x_key: str | None = None,
    plot_spatial_y_key: str | None = None,
    default_sample_id: str = "all_cells",
    spatial_scale: float | None = None,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    cells_h5ad = run_dir / "cells_multilevel_ot.h5ad"
    if not cells_h5ad.exists():
        raise FileNotFoundError(f"Expected multilevel OT cell output under {cells_h5ad}.")
    resolved_output_dir = Path(output_dir) if output_dir is not None else run_dir / "sample_spatial_maps"
    return plot_sample_spatial_maps(
        cells_h5ad=cells_h5ad,
        output_dir=resolved_output_dir,
        sample_obs_key=sample_obs_key,
        source_file_obs_key=source_file_obs_key,
        cluster_obs_key=cluster_obs_key,
        cluster_label_obs_key=cluster_label_obs_key,
        cluster_hex_obs_key=cluster_hex_obs_key,
        plot_spatial_x_key=plot_spatial_x_key,
        plot_spatial_y_key=plot_spatial_y_key,
        default_sample_id=default_sample_id,
        spatial_scale=spatial_scale,
    )


def _save_multilevel_outputs(
    adata: ad.AnnData,
    result: MultilevelOTResult,
    output_dir: Path,
    feature_obsm_key: str,
    spatial_x_key: str,
    spatial_y_key: str,
    spatial_scale: float,
    radius_um: float,
    stride_um: float,
    embedding_2d: np.ndarray,
    embedding_name: str,
    shape_df: pd.DataFrame,
    summary: dict,
    deep_embedding: np.ndarray | None = None,
    deep_obsm_key: str | None = None,
    extra_outputs: dict[str, str] | None = None,
) -> dict[str, str]:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = output_dir / "cells_multilevel_ot.h5ad"
    subregions_path = output_dir / "subregions_multilevel_ot.parquet"
    supports_path = output_dir / "cluster_supports_multilevel_ot.npz"
    candidate_diag_path = output_dir / "multilevel_ot_candidate_cost_diagnostics.npz"
    map_path = output_dir / "multilevel_ot_spatial_map.png"
    emb_path = output_dir / "multilevel_ot_subregion_embedding.png"
    atom_path = output_dir / "multilevel_ot_atom_layouts.png"
    summary_path = output_dir / "summary.json"
    outputs = {
        "h5ad": str(h5ad_path),
        "subregions": str(subregions_path),
        "supports": str(supports_path),
        "candidate_cost_diagnostics": str(candidate_diag_path),
        "spatial_map": str(map_path),
        "subregion_embedding": str(emb_path),
        "atom_layouts": str(atom_path),
        "summary": str(summary_path),
    }
    if extra_outputs:
        outputs.update(extra_outputs)
    summary["outputs"] = outputs

    palette = _cluster_palette(result.cluster_supports.shape[0])
    label_names = [f"C{int(x)}" for x in result.cell_cluster_labels]
    label_hex = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette[result.cell_cluster_labels].tolist()]

    cells_out = adata.copy()
    cells_out.obs["mlot_cluster_id"] = pd.Categorical(label_names)
    cells_out.obs["mlot_cluster_int"] = result.cell_cluster_labels.astype(np.int32)
    cells_out.obs["mlot_cluster_hex"] = label_hex
    cells_out.obsm["mlot_cluster_probs"] = result.cell_cluster_probs.astype(np.float32)
    cells_out.obsm["mlot_cell_cluster_scores"] = result.cell_cluster_probs.astype(np.float32)
    cells_out.obsm["mlot_feature_cluster_probs"] = result.cell_feature_cluster_probs.astype(np.float32)
    cells_out.obsm["mlot_context_cluster_probs"] = result.cell_context_cluster_probs.astype(np.float32)
    if deep_embedding is not None and deep_obsm_key:
        cells_out.obsm[deep_obsm_key] = np.asarray(deep_embedding, dtype=np.float32)
    cells_out.uns["multilevel_ot"] = {
        "feature_obsm_key": feature_obsm_key,
        "feature_input_mode": summary.get("feature_input_mode"),
        "feature_source": summary.get("feature_source"),
        "spatial_x_key": spatial_x_key,
        "spatial_y_key": spatial_y_key,
        "spatial_scale": float(spatial_scale),
        "radius_um": float(radius_um),
        "stride_um": float(stride_um),
        "basic_niche_size_um": float(result.basic_niche_size_um) if result.basic_niche_size_um is not None else None,
        "cell_projection_mode": "approximate_assigned_subregion",
        "deep_obsm_key": deep_obsm_key,
        "summary_json": json.dumps(summary),
    }
    cells_out.write_h5ad(h5ad_path, compression="gzip")

    subregion_rows = []
    sorted_costs = np.sort(result.subregion_cluster_costs, axis=1)
    subregion_margin = (
        sorted_costs[:, 1] - sorted_costs[:, 0]
        if sorted_costs.shape[1] >= 2
        else np.full(sorted_costs.shape[0], np.nan, dtype=np.float32)
    )
    for idx, members in enumerate(result.subregion_members):
        row = {
            "subregion_id": int(idx),
            "center_x_um": float(result.subregion_centers_um[idx, 0]),
            "center_y_um": float(result.subregion_centers_um[idx, 1]),
            "n_cells": int(len(members)),
            "basic_niche_count": int(len(result.subregion_basic_niche_ids[idx])),
            "geometry_point_count": int(result.subregion_geometry_point_counts[idx]),
            "geometry_source": result.subregion_geometry_sources[idx],
            "geometry_used_fallback": bool(result.subregion_geometry_used_fallback[idx]),
            "forced_label": bool(result.subregion_forced_label_mask[idx]),
            "argmin_cluster_int": int(result.subregion_argmin_labels[idx]),
            "assigned_effective_eps": float(result.subregion_assigned_effective_eps[idx]),
            "assigned_ot_used_fallback": bool(result.subregion_assigned_used_ot_fallback[idx]),
            "candidate_effective_eps_min": float(np.min(result.subregion_candidate_effective_eps_matrix[idx])),
            "candidate_effective_eps_max": float(np.max(result.subregion_candidate_effective_eps_matrix[idx])),
            "candidate_ot_used_fallback_any": bool(np.any(result.subregion_candidate_used_ot_fallback_matrix[idx])),
            "normalizer_radius_p95": float(result.subregion_normalizer_radius_p95[idx]) if np.isfinite(result.subregion_normalizer_radius_p95[idx]) else np.nan,
            "normalizer_radius_max": float(result.subregion_normalizer_radius_max[idx]) if np.isfinite(result.subregion_normalizer_radius_max[idx]) else np.nan,
            "normalizer_interpolation_residual": float(result.subregion_normalizer_interpolation_residual[idx]) if np.isfinite(result.subregion_normalizer_interpolation_residual[idx]) else np.nan,
            "cluster_id": f"C{int(result.subregion_cluster_labels[idx])}",
            "cluster_int": int(result.subregion_cluster_labels[idx]),
            "objective": float(result.subregion_cluster_costs[idx, result.subregion_cluster_labels[idx]]),
            "transport_objective": float(result.subregion_cluster_transport_costs[idx, result.subregion_cluster_labels[idx]]),
            "overlap_consistency_penalty": float(
                result.subregion_cluster_overlap_penalties[idx, result.subregion_cluster_labels[idx]]
            ),
            "assigned_geometry_transport_cost": float(result.subregion_assigned_geometry_transport_costs[idx]),
            "assigned_feature_transport_cost": float(result.subregion_assigned_feature_transport_costs[idx]),
            "assigned_transform_penalty": float(result.subregion_assigned_transform_penalties[idx]),
            "assigned_overlap_consistency_penalty": float(result.subregion_assigned_overlap_consistency_penalties[idx]),
            "assigned_transform_rotation_deg": float(result.subregion_assigned_transform_rotation_deg[idx]),
            "assigned_transform_reflection": bool(result.subregion_assigned_transform_reflection[idx]),
            "assigned_transform_scale": float(result.subregion_assigned_transform_scale[idx]),
            "assigned_transform_translation_norm": float(result.subregion_assigned_transform_translation_norm[idx]),
            "assigned_reconstructed_transport_cost": float(
                result.subregion_assigned_geometry_transport_costs[idx]
                + result.subregion_assigned_feature_transport_costs[idx]
                + result.subregion_assigned_transform_penalties[idx]
            ),
            "assignment_margin": float(subregion_margin[idx]) if np.isfinite(subregion_margin[idx]) else np.nan,
        }
        for j, prob in enumerate(result.subregion_cluster_probs[idx]):
            row[f"cluster_prob_{j:02d}"] = float(prob)
        for j, weight in enumerate(result.subregion_atom_weights[idx]):
            row[f"atom_weight_{j:02d}"] = float(weight)
        row["embed1"] = float(embedding_2d[idx, 0])
        row["embed2"] = float(embedding_2d[idx, 1])
        subregion_rows.append(row)
    subregions_df = pd.DataFrame(subregion_rows)
    if not shape_df.empty:
        subregions_df = subregions_df.merge(shape_df, on="subregion_id", how="left")
    subregions_df.to_parquet(subregions_path, index=False)

    np.savez_compressed(
        supports_path,
        cluster_supports=result.cluster_supports.astype(np.float32),
        cluster_atom_coords=result.cluster_atom_coords.astype(np.float32),
        cluster_atom_features=result.cluster_atom_features.astype(np.float32),
        cluster_prototype_weights=result.cluster_prototype_weights.astype(np.float32),
        subregion_atom_weights=result.subregion_atom_weights.astype(np.float32),
    )
    np.savez_compressed(
        candidate_diag_path,
        subregion_cluster_costs=result.subregion_cluster_costs.astype(np.float32),
        subregion_cluster_transport_costs=result.subregion_cluster_transport_costs.astype(np.float32),
        subregion_cluster_overlap_penalties=result.subregion_cluster_overlap_penalties.astype(np.float32),
        subregion_measure_summaries=result.subregion_measure_summaries.astype(np.float32),
        candidate_effective_eps_matrix=result.subregion_candidate_effective_eps_matrix.astype(np.float32),
        candidate_used_ot_fallback_matrix=result.subregion_candidate_used_ot_fallback_matrix.astype(bool),
    )

    coords = np.stack(
        [
            np.asarray(adata.obs[spatial_x_key], dtype=np.float32) * spatial_scale,
            np.asarray(adata.obs[spatial_y_key], dtype=np.float32) * spatial_scale,
        ],
        axis=1,
    )
    point_size = 4.0 if coords.shape[0] > 100000 else 8.0
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    for cid in range(result.cluster_supports.shape[0]):
        mask = result.cell_cluster_labels == cid
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            color=palette[cid] / 255.0,
            linewidths=0,
            alpha=0.85,
            rasterized=coords.shape[0] > 20000,
            label=f"C{cid} ({int(mask.sum())})",
        )
    ax.set_title("Shape-normalized multilevel OT cell labels")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.savefig(map_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    for cid in range(result.cluster_supports.shape[0]):
        mask = result.subregion_cluster_labels == cid
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            s=16,
            color=palette[cid] / 255.0,
            linewidths=0,
            alpha=0.9,
            label=f"C{cid} ({int(mask.sum())})",
        )
    ax.set_title(f"Subregion {embedding_name} from learned mixture weights")
    ax.set_xlabel(f"{embedding_name} 1")
    ax.set_ylabel(f"{embedding_name} 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.savefig(emb_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        nrows=result.cluster_supports.shape[0],
        ncols=1,
        figsize=(6.5, max(3.0, 2.4 * result.cluster_supports.shape[0])),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    feat_norm = np.linalg.norm(result.cluster_atom_features, axis=2)
    for cid, ax in enumerate(axes):
        sizes = 200.0 * np.clip(result.cluster_prototype_weights[cid], 0.05, None)
        sc = ax.scatter(
            result.cluster_atom_coords[cid, :, 0],
            result.cluster_atom_coords[cid, :, 1],
            s=sizes,
            c=feat_norm[cid],
            cmap="viridis",
            linewidths=0.5,
            edgecolors="black",
        )
        for atom_idx in range(result.cluster_atom_coords.shape[1]):
            ax.text(
                result.cluster_atom_coords[cid, atom_idx, 0],
                result.cluster_atom_coords[cid, atom_idx, 1],
                str(atom_idx),
                fontsize=7,
                ha="center",
                va="center",
                color="white",
            )
        ax.set_title(f"Cluster C{cid} canonical atom layout")
        ax.set_xlabel("canonical x")
        ax.set_ylabel("canonical y")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="feature norm")
    fig.savefig(atom_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    sample_spatial_manifest = plot_sample_spatial_maps(
        cells_h5ad=h5ad_path,
        output_dir=output_dir / "sample_spatial_maps",
    )
    outputs["sample_spatial_maps_dir"] = str(output_dir / "sample_spatial_maps")
    outputs["sample_spatial_maps_manifest"] = str(sample_spatial_manifest["manifest_json"])

    summary_path.write_text(json.dumps(summary, indent=2))
    return outputs


def run_multilevel_ot_on_h5ad(
    input_h5ad: str | Path,
    output_dir: str | Path,
    feature_obsm_key: str,
    spatial_x_key: str,
    spatial_y_key: str,
    spatial_scale: float,
    *,
    region_obs_key: str | None = None,
    allow_umap_as_feature: bool = False,
    n_clusters: int,
    atoms_per_cluster: int,
    radius_um: float,
    stride_um: float,
    min_cells: int,
    max_subregions: int,
    lambda_x: float,
    lambda_y: float,
    geometry_eps: float,
    ot_eps: float,
    rho: float,
    geometry_samples: int,
    compressed_support_size: int,
    align_iters: int,
    allow_reflection: bool = False,
    allow_scale: bool = False,
    min_scale: float = 0.75,
    max_scale: float = 1.33,
    scale_penalty: float = 0.05,
    shift_penalty: float = 0.05,
    n_init: int = 5,
    allow_convex_hull_fallback: bool = False,
    max_iter: int = 10,
    tol: float = 1e-4,
    overlap_consistency_weight: float = 0.0,
    overlap_jaccard_min: float = 0.15,
    overlap_contrast_scale: float = 1.0,
    basic_niche_size_um: float | None = 200.0,
    seed: int = 1337,
    compute_device: str = "auto",
    deep_config: DeepFeatureConfig | None = None,
) -> dict:
    input_h5ad = Path(input_h5ad)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(input_h5ad)
    if spatial_x_key not in adata.obs or spatial_y_key not in adata.obs:
        raise KeyError(f"Spatial keys '{spatial_x_key}' and/or '{spatial_y_key}' not found in obs.")
    deep_config = deep_config or DeepFeatureConfig()
    features, feature_source = resolve_h5ad_features(
        adata,
        feature_obsm_key=feature_obsm_key,
        allow_umap_as_feature=allow_umap_as_feature,
    )
    feature_embedding_warning = feature_source.get("feature_embedding_warning")
    coords_um = np.stack(
        [
            np.asarray(adata.obs[spatial_x_key], dtype=np.float32) * spatial_scale,
            np.asarray(adata.obs[spatial_y_key], dtype=np.float32) * spatial_scale,
        ],
        axis=1,
    )
    feature_obsm_key_used = str(feature_source.get("feature_key", feature_obsm_key))
    resolved_compute_device = _resolve_compute_device(compute_device)
    if resolved_compute_device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(resolved_compute_device)
        except Exception:
            pass
    deep_embedding: np.ndarray | None = None
    deep_outputs: dict[str, str] = {}
    deep_summary = {
        "enabled": False,
        "method": "none",
    }
    if deep_config.method != "none":
        active_deep_config = deep_config
        batch = None
        count_layer_used = None
        if deep_config.batch_key is not None:
            if deep_config.batch_key not in adata.obs:
                raise KeyError(f"Deep-feature batch key '{deep_config.batch_key}' not found in obs.")
            batch = np.asarray(adata.obs[deep_config.batch_key].astype(str))
        if deep_config.pretrained_model is not None:
            encoder = SpatialOTFeatureEncoder.load(deep_config.pretrained_model)
            active_deep_config = encoder.config
            allow_joint_ot_embedding = bool(active_deep_config.allow_joint_ot_embedding or deep_config.allow_joint_ot_embedding)
            if active_deep_config.output_embedding == "joint" and not allow_joint_ot_embedding:
                raise ValueError(
                    "Using deep.output_embedding='joint' as the OT feature view requires explicit opt-in. "
                    "Set deep.allow_joint_ot_embedding=true or pass --deep-allow-joint-ot-embedding."
                )
            deep_embedding = encoder.transform(features=features, coords_um=coords_um)
            history = list(encoder.history)
            model_path = str(Path(deep_config.pretrained_model))
            validation_report = dict(getattr(encoder, "validation_report", {}))
            feature_schema = dict(getattr(encoder, "feature_schema", {}))
            latent_diagnostics = dict(getattr(encoder, "latent_diagnostics", {}))
            if active_deep_config.count_layer is not None:
                count_layer_used = str(active_deep_config.count_layer)
        else:
            allow_joint_ot_embedding = bool(deep_config.allow_joint_ot_embedding)
            if deep_config.output_embedding == "joint" and not allow_joint_ot_embedding:
                raise ValueError(
                    "Using deep.output_embedding='joint' as the OT feature view requires explicit opt-in. "
                    "Set deep.allow_joint_ot_embedding=true or pass --deep-allow-joint-ot-embedding."
                )
            model_path = str(output_dir / "deep_feature_model.pt") if deep_config.save_model else None
            count_matrix, count_layer_used = _extract_count_target(adata, count_layer=deep_config.count_layer)
            deep_result = fit_deep_features(
                features=features,
                coords_um=coords_um,
                config=deep_config,
                batch=batch,
                count_matrix=count_matrix,
                seed=seed,
                save_path=model_path,
            )
            deep_embedding = deep_result.embedding.astype(np.float32)
            history = list(deep_result.history)
            validation_report = dict(deep_result.validation_report)
            feature_schema = dict(deep_result.feature_schema)
            latent_diagnostics = dict(deep_result.latent_diagnostics)
        features = np.asarray(deep_embedding, dtype=np.float32)
        feature_obsm_key_used = active_deep_config.output_obsm_key
        adata.obsm[feature_obsm_key_used] = features.astype(np.float32)
        history_path = output_dir / "deep_feature_history.csv"
        save_deep_feature_history(history, history_path)
        config_path = output_dir / "deep_feature_config.json"
        config_path.write_text(json.dumps(asdict(active_deep_config), indent=2))
        deep_outputs["deep_feature_history"] = str(history_path)
        deep_outputs["deep_feature_config"] = str(config_path)
        if model_path is not None:
            deep_outputs["deep_feature_model"] = str(model_path)
            meta_path = Path(model_path).with_suffix(Path(model_path).suffix + ".meta.json")
            scaler_path = Path(model_path).with_suffix(Path(model_path).suffix + ".scaler.npz")
            if meta_path.exists():
                deep_outputs["deep_feature_model_meta"] = str(meta_path)
            if scaler_path.exists():
                deep_outputs["deep_feature_scaler"] = str(scaler_path)
        final_train_loss = history[-1].get("train_loss") if history else None
        final_val_loss = history[-1].get("val_loss") if history and "val_loss" in history[-1] else None
        count_reconstruction_summary: str | dict[str, object]
        if active_deep_config.count_layer is None:
            count_reconstruction_summary = "disabled"
        else:
            count_reconstruction_summary = {
                "enabled": True,
                "target_layer": str(count_layer_used or active_deep_config.count_layer),
                "decoder_rank": int(active_deep_config.count_decoder_rank),
                "gene_chunk_size": int(active_deep_config.count_chunk_size),
                "loss_weight": float(active_deep_config.count_loss_weight),
            }
        deep_summary = {
            "enabled": True,
            "method": active_deep_config.method,
            "input_feature_obsm_key": feature_obsm_key,
            "output_feature_obsm_key": feature_obsm_key_used,
            "latent_dim": int(features.shape[1]),
            "epochs": int(active_deep_config.epochs),
            "batch_key": active_deep_config.batch_key,
            "neighbor_k": int(active_deep_config.neighbor_k),
            "radius_um": float(active_deep_config.radius_um) if active_deep_config.radius_um is not None else None,
            "short_radius_um": float(active_deep_config.short_radius_um) if active_deep_config.short_radius_um is not None else None,
            "mid_radius_um": float(active_deep_config.mid_radius_um) if active_deep_config.mid_radius_um is not None else None,
            "graph_layers": int(active_deep_config.graph_layers),
            "graph_aggr": active_deep_config.graph_aggr,
            "graph_max_neighbors": int(active_deep_config.graph_max_neighbors),
            "full_batch_max_cells": int(active_deep_config.full_batch_max_cells),
            "validation": active_deep_config.validation,
            "validation_context_mode": active_deep_config.validation_context_mode,
            "allow_joint_ot_embedding": bool(allow_joint_ot_embedding),
            "uses_absolute_coordinate_features": False,
            "uses_spatial_graph": bool(active_deep_config.method == "graph_autoencoder"),
            "output_embedding": active_deep_config.output_embedding,
            "ot_feature_view_warning": (
                "joint_embedding_explicit_opt_in"
                if active_deep_config.output_embedding == "joint"
                else None
            ),
            "final_train_loss": float(final_train_loss) if final_train_loss is not None else None,
            "final_val_loss": float(final_val_loss) if final_val_loss is not None else None,
            "model_path": model_path,
            "batch_correction": "disabled",
            "count_reconstruction": count_reconstruction_summary,
            "pretrained_model_loaded": bool(deep_config.pretrained_model is not None),
            "validation_used_for_early_stopping": bool(deep_config.validation != "none"),
            "runtime_memory": latent_diagnostics.get("runtime_memory"),
            "feature_schema": feature_schema,
            "validation_report": validation_report,
            "latent_diagnostics": latent_diagnostics,
        }
    region_geometries = None
    subregion_members = None
    subregion_centers_um = None
    build_grid_subregions = True
    if region_obs_key is not None:
        if region_obs_key not in adata.obs:
            raise KeyError(f"Region obs key '{region_obs_key}' not found in obs.")
        grouped = pd.Series(np.arange(adata.n_obs), index=adata.obs[region_obs_key].astype(str))
        subregion_members = [group.to_numpy(dtype=np.int32) for _, group in grouped.groupby(level=0)]
        subregion_centers_um = np.vstack([coords_um[members].mean(axis=0) for members in subregion_members]).astype(np.float32)
        region_geometries = [
            RegionGeometry(region_id=str(region_id), members=np.asarray(members, dtype=np.int32))
            for region_id, members in grouped.groupby(level=0)
        ]
        build_grid_subregions = False

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords_um,
        subregion_members=subregion_members,
        subregion_centers_um=subregion_centers_um,
        n_clusters=n_clusters,
        atoms_per_cluster=atoms_per_cluster,
        radius_um=radius_um,
        stride_um=stride_um,
        basic_niche_size_um=basic_niche_size_um,
        min_cells=min_cells,
        max_subregions=max_subregions,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        geometry_eps=geometry_eps,
        ot_eps=ot_eps,
        rho=rho,
        geometry_samples=geometry_samples,
        compressed_support_size=compressed_support_size,
        align_iters=align_iters,
        allow_reflection=allow_reflection,
        allow_scale=allow_scale,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_penalty=scale_penalty,
        shift_penalty=shift_penalty,
        n_init=n_init,
        region_geometries=region_geometries,
        build_grid_subregions=build_grid_subregions,
        allow_convex_hull_fallback=allow_convex_hull_fallback,
        max_iter=max_iter,
        tol=tol,
        overlap_consistency_weight=overlap_consistency_weight,
        overlap_jaccard_min=overlap_jaccard_min,
        overlap_contrast_scale=overlap_contrast_scale,
        seed=seed,
        compute_device=str(resolved_compute_device),
    )
    fallback_fraction = float(np.mean(result.subregion_geometry_used_fallback.astype(np.float32)))
    if fallback_fraction > 0:
        warnings.warn(
            f"{int(result.subregion_geometry_used_fallback.sum())}/{len(result.subregion_members)} subregions used observed-coordinate convex-hull geometry fallback. Treat this run as exploratory rather than boundary-shape-invariant.",
            RuntimeWarning,
            stacklevel=2,
        )
    shape_df = _shape_descriptor_frame(result.subregion_members, coords_um, region_geometries=region_geometries)
    shape_leakage = _shape_leakage_balanced_accuracy(shape_df, result.subregion_cluster_labels, seed=seed)
    shape_leakage_block = _shape_leakage_spatial_block_accuracy(
        shape_df=shape_df,
        labels=result.subregion_cluster_labels,
        centers_um=result.subregion_centers_um,
        seed=seed,
    )
    shape_leakage_perm = _shape_leakage_permutation_baseline(shape_df, result.subregion_cluster_labels, seed=seed)
    embedding_2d, embedding_name = _compute_subregion_embedding(result.subregion_atom_weights, seed=seed)
    silhouette = None
    n_unique_labels = np.unique(result.subregion_cluster_labels).size
    if 1 < n_unique_labels < result.subregion_atom_weights.shape[0]:
        silhouette = float(
            silhouette_score(
                result.subregion_atom_weights,
                result.subregion_cluster_labels,
                metric="euclidean",
            )
        )
    sorted_costs = np.sort(result.subregion_cluster_costs, axis=1)
    margin = None
    if sorted_costs.shape[1] >= 2:
        margin = float(np.mean(sorted_costs[:, 1] - sorted_costs[:, 0]))
    coverage_summary = _cell_subregion_coverage(int(adata.n_obs), result.subregion_members)
    cell_prob_summary = _probability_diagnostics(result.cell_cluster_probs, prefix="cell")
    subregion_prob_summary = _probability_diagnostics(result.subregion_cluster_probs, prefix="subregion")
    compactness_summary = _subregion_embedding_compactness(result)
    boundary_summary = _subregion_graph_metrics(
        n_cells=int(adata.n_obs),
        result=result,
        radius_um=radius_um,
        stride_um=stride_um,
        coords_um=coords_um,
    )
    cost_reliability = _cost_reliability_metrics(result)
    transform_summary = _transform_diagnostics(result)
    cost_scale_summary = {
        "coordinate_scale": float(result.cost_scale_x),
        "feature_scale": float(result.cost_scale_y),
        "feature_to_coordinate_scale_ratio": float(result.cost_scale_y / max(result.cost_scale_x, 1e-8)),
        "effective_feature_to_geometry_weight_ratio": (
            float((lambda_y / max(result.cost_scale_y, 1e-8)) / (lambda_x / max(result.cost_scale_x, 1e-8)))
            if float(lambda_x) > 0
            else None
        ),
    }
    assigned_transport_cost_summary = _assigned_transport_cost_decomposition(result)
    runtime_memory = _runtime_memory_snapshot(resolved_compute_device)
    assigned_effective_eps_values = [float(x) for x in np.unique(np.round(result.subregion_assigned_effective_eps.astype(np.float64), 8))]
    method_stack = _method_stack_summary(
        feature_source=feature_source,
        deep_summary=deep_summary,
        feature_obsm_key=feature_obsm_key_used,
    )
    qc_warnings = _build_qc_warnings(
        feature_embedding_warning=feature_embedding_warning,
        fallback_fraction=float(fallback_fraction),
        assigned_ot_fallback_fraction=float(np.mean(result.subregion_assigned_used_ot_fallback.astype(np.float32))),
        assigned_effective_eps_values=assigned_effective_eps_values,
        requested_ot_eps=float(ot_eps),
        coverage_fraction=float(coverage_summary["cell_subregion_coverage_fraction"]),
        mean_assignment_margin=margin,
        assigned_transport_cost_decomposition=assigned_transport_cost_summary,
        cost_reliability=cost_reliability,
        transform_diagnostics=transform_summary,
        forced_label_fraction=float(result.subregion_forced_label_mask.sum() / max(len(result.subregion_members), 1)),
        deep_summary=deep_summary,
    )
    summary = {
        "summary_schema_version": "1",
        "spatial_ot_version": _package_version(),
        "git_sha": _git_sha(),
        "method_family": "multilevel_ot",
        "active_path": "multilevel-ot",
        "latent_source": _latent_source_label(feature_source, deep_summary),
        "communication_source": "none",
        "input_h5ad": str(input_h5ad),
        "output_dir": str(output_dir),
        "feature_obsm_key": feature_obsm_key_used,
        "feature_obsm_key_requested": feature_obsm_key,
        "feature_input_mode": str(feature_source.get("input_mode", "obsm")),
        "feature_source": dict(feature_source),
        "feature_embedding_warning": feature_embedding_warning,
        "allow_umap_as_feature": bool(allow_umap_as_feature),
        "spatial_x_key": spatial_x_key,
        "spatial_y_key": spatial_y_key,
        "spatial_scale": float(spatial_scale),
        "region_obs_key": region_obs_key,
        "n_cells": int(adata.n_obs),
        "feature_dim": int(features.shape[1]),
        "n_subregions": int(len(result.subregion_members)),
        "n_clusters": int(n_clusters),
        "atoms_per_cluster": int(atoms_per_cluster),
        "radius_um": float(radius_um),
        "stride_um": float(stride_um),
        "basic_niche_size_um": float(result.basic_niche_size_um) if result.basic_niche_size_um is not None else None,
        "basic_niche_radius_um": (
            0.5 * float(result.basic_niche_size_um) * float(np.sqrt(2.0))
        ) if result.basic_niche_size_um is not None else None,
        "n_basic_niches": int(result.basic_niche_centers_um.shape[0]),
        "mean_basic_niches_per_subregion": (
            float(np.mean([len(niche_ids) for niche_ids in result.subregion_basic_niche_ids]))
            if result.subregion_basic_niche_ids
            else 0.0
        ),
        "min_cells": int(min_cells),
        "max_subregions": int(max_subregions),
        "lambda_x": float(lambda_x),
        "lambda_y": float(lambda_y),
        "geometry_eps": float(geometry_eps),
        "ot_eps": float(ot_eps),
        "rho": float(rho),
        "geometry_samples": int(geometry_samples),
        "compressed_support_size": int(compressed_support_size),
        "align_iters": int(align_iters),
        "allow_reflection": bool(allow_reflection),
        "allow_scale": bool(allow_scale),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "scale_penalty": float(scale_penalty),
        "shift_penalty": float(shift_penalty),
        "n_init": int(n_init),
        "allow_convex_hull_fallback": bool(allow_convex_hull_fallback),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "overlap_consistency_weight": float(overlap_consistency_weight),
        "overlap_jaccard_min": float(overlap_jaccard_min),
        "overlap_contrast_scale": float(overlap_contrast_scale),
        "seed": int(seed),
        "compute_device_requested": str(compute_device),
        "compute_device_used": str(resolved_compute_device),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_list_env": os.environ.get("SPATIAL_OT_CUDA_DEVICE_LIST"),
        "parallel_restarts_env": os.environ.get("SPATIAL_OT_PARALLEL_RESTARTS"),
        "cuda_target_vram_gb_env": os.environ.get("SPATIAL_OT_CUDA_TARGET_VRAM_GB"),
        "torch_num_threads_env": os.environ.get("SPATIAL_OT_TORCH_NUM_THREADS"),
        "torch_num_interop_threads_env": os.environ.get("SPATIAL_OT_TORCH_NUM_INTEROP_THREADS"),
        "cost_scale_x": float(result.cost_scale_x),
        "cost_scale_y": float(result.cost_scale_y),
        "cost_scale_diagnostics": cost_scale_summary,
        "method_stack": method_stack,
        "requested_ot_eps": float(ot_eps),
        "assigned_ot_fallback_fraction": float(np.mean(result.subregion_assigned_used_ot_fallback.astype(np.float32))),
        "assigned_effective_eps_values": assigned_effective_eps_values,
        "assigned_transport_cost_decomposition": assigned_transport_cost_summary,
        "subregion_embedding_compactness": compactness_summary,
        "boundary_separation": boundary_summary,
        "cost_reliability": cost_reliability,
        "transform_diagnostics": transform_summary,
        "selected_restart": int(result.selected_restart),
        "restart_summaries": result.restart_summaries,
        "subregion_cluster_counts": {f"C{int(k)}": int(v) for k, v in pd.Series(result.subregion_cluster_labels).value_counts().sort_index().items()},
        "cell_cluster_counts": {f"C{int(k)}": int(v) for k, v in pd.Series(result.cell_cluster_labels).value_counts().sort_index().items()},
        "objective_history": result.objective_history,
        "subregion_embedding_method": embedding_name,
        "subregion_weight_silhouette": silhouette,
        "mean_assignment_margin": margin,
        "shape_leakage_balanced_accuracy": shape_leakage,
        "shape_leakage_spatial_block_accuracy": shape_leakage_block,
        "shape_leakage_permutation": shape_leakage_perm,
        "shape_leakage_diagnostics": {
            "balanced_accuracy": shape_leakage,
            "spatial_block_accuracy": shape_leakage_block,
            "permutation": shape_leakage_perm,
        },
        **coverage_summary,
        **cell_prob_summary,
        **subregion_prob_summary,
        "geometry_point_count_range": [
            int(result.subregion_geometry_point_counts.min()),
            int(result.subregion_geometry_point_counts.max()),
        ],
        "geometry_fallback_fraction": fallback_fraction,
        "convex_hull_fallback_fraction": fallback_fraction,
        "degenerate_geometry_subregion_count": int(np.sum(result.subregion_geometry_point_counts < 3)),
        "degenerate_geometry_subregion_fraction": float(np.mean((result.subregion_geometry_point_counts < 3).astype(np.float32))),
        "geometry_source_counts": {
            key: int(value)
            for key, value in pd.Series(result.subregion_geometry_sources).value_counts().sort_index().items()
        },
        "shape_descriptor_source_counts": {
            key: int(value)
            for key, value in shape_df["shape_descriptor_source"].value_counts().sort_index().items()
        }
        if "shape_descriptor_source" in shape_df.columns
        else {},
        "forced_label_count": int(result.subregion_forced_label_mask.sum()),
        "forced_label_fraction": float(result.subregion_forced_label_mask.sum() / max(len(result.subregion_members), 1)),
        "normalizer_radius_p95_mean": float(np.nanmean(result.subregion_normalizer_radius_p95)),
        "normalizer_radius_max_mean": float(np.nanmean(result.subregion_normalizer_radius_max)),
        "normalizer_interpolation_residual_mean": float(np.nanmean(result.subregion_normalizer_interpolation_residual)),
        "normalizer_diagnostics": {
            "radius_p95_mean": float(np.nanmean(result.subregion_normalizer_radius_p95)),
            "radius_max_mean": float(np.nanmean(result.subregion_normalizer_radius_max)),
            "interpolation_residual_mean": float(np.nanmean(result.subregion_normalizer_interpolation_residual)),
        },
        "boundary_invariance_claim": (
            "supported_with_explicit_geometry"
            if float(np.mean(result.subregion_geometry_used_fallback.astype(np.float32))) == 0.0
            else "not_supported_observed_hull_fallback"
        ),
        "qc_warnings": qc_warnings,
        "qc_warning_count": int(len(qc_warnings)),
        "qc_has_warnings": bool(any(item.get("severity") == "warning" for item in qc_warnings)),
        "runtime_memory": runtime_memory,
        "method_notes": {
            "core": "shape-normalized cluster-specific semi-relaxed Wasserstein dictionary clustering",
            "geometry_normalization": "uniform geometry samples from each subregion are OT-mapped into a shared unit-disk reference domain before clustering; degenerate 1-2 point subregions fall back to centered-and-scaled local coordinates without OT interpolation",
            "geometry_proxy": "when explicit masks are unavailable and convex hull fallback is allowed, geometry samples are drawn from the convex hull of local cell coordinates",
            "basic_niches": "when basic_niche_size_um is set, grid-built subregions are unions of fixed-size basic niches rather than direct raw-cell radius windows",
            "local_measure": "compressed empirical measures over canonical coordinates and standardized cell-level features",
            "local_matching": "semi-relaxed unbalanced Sinkhorn with fixed source marginal and relaxed target marginal",
            "overlap_consistency": "overlapping subregions can be regularized toward compatible cluster assignments using Jaccard overlap gated by subregion-summary contrast",
            "residual_alignment": "weighted similarity transform is optimized during subregion-to-cluster matching",
            "support_sharing": "subregions assigned to the same cluster reuse the same shared atom dictionary but keep subregion-specific mixture weights",
            "cell_boundary_projection": "cell-level scores are an approximate projection from canonical-coordinate plus feature fit to assigned cluster atoms, modulated by overlapping-subregion cluster evidence; they are not an exact posterior under the OT model",
        },
        "deep_features": deep_summary,
    }
    _save_multilevel_outputs(
        adata=adata,
        result=result,
        output_dir=output_dir,
        feature_obsm_key=feature_obsm_key_used,
        spatial_x_key=spatial_x_key,
        spatial_y_key=spatial_y_key,
        spatial_scale=spatial_scale,
        radius_um=radius_um,
        stride_um=stride_um,
        embedding_2d=embedding_2d,
        embedding_name=embedding_name,
        shape_df=shape_df,
        summary=summary,
        deep_embedding=deep_embedding,
        deep_obsm_key=feature_obsm_key_used if deep_embedding is not None else None,
        extra_outputs=deep_outputs,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_multilevel_ot_with_config(config: MultilevelExperimentConfig) -> dict:
    return run_multilevel_ot_on_h5ad(
        input_h5ad=config.paths.input_h5ad,
        output_dir=config.paths.output_dir,
        feature_obsm_key=config.paths.feature_obsm_key,
        spatial_x_key=config.paths.spatial_x_key,
        spatial_y_key=config.paths.spatial_y_key,
        spatial_scale=config.paths.spatial_scale,
        region_obs_key=config.paths.region_obs_key,
        allow_umap_as_feature=config.paths.allow_umap_as_feature,
        n_clusters=config.ot.n_clusters,
        atoms_per_cluster=config.ot.atoms_per_cluster,
        radius_um=config.ot.radius_um,
        stride_um=config.ot.stride_um,
        basic_niche_size_um=config.ot.basic_niche_size_um,
        min_cells=config.ot.min_cells,
        max_subregions=config.ot.max_subregions,
        lambda_x=config.ot.lambda_x,
        lambda_y=config.ot.lambda_y,
        geometry_eps=config.ot.geometry_eps,
        ot_eps=config.ot.ot_eps,
        rho=config.ot.rho,
        geometry_samples=config.ot.geometry_samples,
        compressed_support_size=config.ot.compressed_support_size,
        align_iters=config.ot.align_iters,
        allow_reflection=config.ot.allow_reflection,
        allow_scale=config.ot.allow_scale,
        min_scale=config.ot.min_scale,
        max_scale=config.ot.max_scale,
        scale_penalty=config.ot.scale_penalty,
        shift_penalty=config.ot.shift_penalty,
        n_init=config.ot.n_init,
        allow_convex_hull_fallback=config.ot.allow_convex_hull_fallback,
        max_iter=config.ot.max_iter,
        tol=config.ot.tol,
        overlap_consistency_weight=config.ot.overlap_consistency_weight,
        overlap_jaccard_min=config.ot.overlap_jaccard_min,
        overlap_contrast_scale=config.ot.overlap_contrast_scale,
        seed=config.ot.seed,
        compute_device=config.ot.compute_device,
        deep_config=config.deep,
    )
