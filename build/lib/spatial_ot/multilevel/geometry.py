from __future__ import annotations

import warnings
import os
import sys
import time

from matplotlib.path import Path as MplPath
import numpy as np
import ot
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
import torch

from .runtime import env_float as _env_float, env_int as _env_int
from .types import RegionGeometry, ShapeNormalizer, ShapeNormalizerDiagnostics


_GEOMETRY_PROGRESS_START = time.perf_counter()


def _geometry_progress(message: str) -> None:
    raw = os.environ.get("SPATIAL_OT_PROGRESS", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        elapsed = time.perf_counter() - _GEOMETRY_PROGRESS_START
        print(f"[spatial_ot geometry {elapsed:8.1f}s] {message}", file=sys.stderr, flush=True)


def _standardize_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((x - mean) / std).astype(np.float32)


def _normalize_hist(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    total = float(x.sum())
    if total <= 1e-12:
        return np.full_like(x, 1.0 / max(x.size, 1), dtype=np.float64)
    return x / total


def _sinkhorn_max_iter() -> int:
    return _env_int("SPATIAL_OT_SINKHORN_MAX_ITER", 600)


def _sinkhorn_tol() -> float:
    return _env_float("SPATIAL_OT_SINKHORN_TOL", 1e-5)


def _cpu_sinkhorn_max_iter() -> int:
    return _env_int("SPATIAL_OT_CPU_SINKHORN_MAX_ITER", 3000)


def _cpu_sinkhorn_tol() -> float:
    return _env_float("SPATIAL_OT_CPU_SINKHORN_TOL", 1e-8)


def _softmax_over_negative_costs(costs: np.ndarray, temperature: float) -> np.ndarray:
    scaled = -np.asarray(costs, dtype=np.float32) / max(float(temperature), 1e-5)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    return probs.astype(np.float32)


def _center_from_members(coords_um: np.ndarray, members: np.ndarray) -> np.ndarray:
    return np.asarray(coords_um, dtype=np.float32)[np.asarray(members, dtype=np.int32)].mean(axis=0).astype(np.float32)


def _point_cloud_area_um2(points_um: np.ndarray) -> float:
    points = np.asarray(points_um, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return 0.0
    unique = np.unique(points, axis=0)
    if unique.shape[0] < 3:
        return 0.0
    try:
        return float(max(ConvexHull(unique).volume, 0.0))
    except Exception:
        x = unique[:, 0]
        y = unique[:, 1]
        return float(max((float(x.max()) - float(x.min())) * (float(y.max()) - float(y.min())), 0.0))


def _member_area_um2(coords_um: np.ndarray, members: np.ndarray) -> float:
    member_arr = np.asarray(members, dtype=np.int32)
    if member_arr.size < 3:
        return 0.0
    return _point_cloud_area_um2(np.asarray(coords_um, dtype=np.float32)[member_arr])


def _member_bbox_area_um2(coords_um: np.ndarray, members: np.ndarray) -> float:
    member_arr = np.asarray(members, dtype=np.int32)
    if member_arr.size < 2:
        return 0.0
    local = np.asarray(coords_um, dtype=np.float32)[member_arr]
    span = np.ptp(local, axis=0)
    return float(max(float(span[0] * span[1]), 0.0))


def _target_data_driven_region_count(
    coords_um: np.ndarray,
    *,
    target_scale_um: float,
    min_cells: int,
    max_subregions: int,
    max_area_um2: float | None = None,
) -> int:
    coords = np.asarray(coords_um, dtype=np.float32)
    n_cells = int(coords.shape[0])
    if n_cells == 0:
        raise RuntimeError("No cells were provided, so no subregions can be created.")
    min_cells = max(int(min_cells), 1)
    max_by_min = max(1, n_cells // min_cells)
    target_scale = max(float(target_scale_um), 1e-6)
    span = np.maximum(np.ptp(coords, axis=0), target_scale)
    spatial_estimate = int(np.ceil(float(span[0] * span[1]) / max(target_scale * target_scale, 1e-6)))
    if max_area_um2 is not None and float(max_area_um2) > 0.0:
        max_area = max(float(max_area_um2), 1e-6)
        observed_area = _point_cloud_area_um2(coords)
        if observed_area > 0:
            spatial_estimate = max(spatial_estimate, int(np.ceil(observed_area / max_area)))
    size_estimate = int(np.ceil(n_cells / max(min_cells * 4, 64)))
    data_estimate = max(spatial_estimate, size_estimate)
    cap = int(max_subregions) if int(max_subregions) > 0 else n_cells
    return max(1, min(data_estimate, cap, max_by_min, n_cells))


def _subregion_seed_kmeans_n_init() -> int:
    return max(1, _env_int("SPATIAL_OT_SUBREGION_KMEANS_N_INIT", 1))


def _subregion_seed_kmeans_max_iter() -> int:
    return max(5, _env_int("SPATIAL_OT_SUBREGION_KMEANS_MAX_ITER", 25))


def _subregion_seed_kmeans_batch_multiplier() -> int:
    return max(1, _env_int("SPATIAL_OT_SUBREGION_KMEANS_BATCH_MULTIPLIER", 2))


def _subregion_boundary_refinement_iters() -> int:
    return max(0, _env_int("SPATIAL_OT_SUBREGION_BOUNDARY_REFINEMENT_ITERS", 2))


def _subregion_boundary_refinement_knn() -> int:
    return max(2, _env_int("SPATIAL_OT_SUBREGION_BOUNDARY_KNN", 12))


def _subregion_partition_feature_weight() -> float:
    return max(0.0, _env_float("SPATIAL_OT_SUBREGION_FEATURE_WEIGHT", 0.0))


def _subregion_partition_feature_dims() -> int:
    return max(0, _env_int("SPATIAL_OT_SUBREGION_FEATURE_DIMS", 16))


def _deep_segmentation_knn_default() -> int:
    return max(2, _env_int("SPATIAL_OT_DEEP_SEGMENTATION_KNN", 12))


def _deep_segmentation_feature_dims_default() -> int:
    return max(1, _env_int("SPATIAL_OT_DEEP_SEGMENTATION_FEATURE_DIMS", 32))


def _deep_segmentation_feature_weight_default() -> float:
    return max(0.0, _env_float("SPATIAL_OT_DEEP_SEGMENTATION_FEATURE_WEIGHT", 1.0))


def _deep_segmentation_spatial_weight_default() -> float:
    return max(0.0, _env_float("SPATIAL_OT_DEEP_SEGMENTATION_SPATIAL_WEIGHT", 0.05))


def _deep_segmentation_refinement_iters_default() -> int:
    return max(0, _env_int("SPATIAL_OT_DEEP_SEGMENTATION_REFINEMENT_ITERS", 6))


def _leakage_rf_estimators() -> int:
    return max(20, _env_int("SPATIAL_OT_LEAKAGE_RF_ESTIMATORS", 120))


def _leakage_max_subregions() -> int:
    return max(0, _env_int("SPATIAL_OT_LEAKAGE_MAX_SUBREGIONS", 20000))


def _leakage_sample_indices(labels: np.ndarray, *, seed: int) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int32)
    n = int(y.shape[0])
    cap = _leakage_max_subregions()
    if cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    unique = np.unique(y)
    per_label = max(2, int(cap) // max(int(unique.size), 1))
    selected_parts: list[np.ndarray] = []
    selected_mask = np.zeros(n, dtype=bool)
    for label in unique:
        idx = np.flatnonzero(y == int(label))
        keep_n = min(int(idx.size), int(per_label))
        if keep_n <= 0:
            continue
        keep = rng.choice(idx, size=keep_n, replace=False) if idx.size > keep_n else idx
        keep = np.asarray(keep, dtype=np.int64)
        selected_parts.append(keep)
        selected_mask[keep] = True
    selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)
    if selected.size < cap:
        rest = np.flatnonzero(~selected_mask)
        if rest.size:
            extra_n = min(int(rest.size), int(cap) - int(selected.size))
            extra = rng.choice(rest, size=extra_n, replace=False).astype(np.int64)
            selected = np.concatenate([selected, extra])
    if selected.size > cap:
        selected = rng.choice(selected, size=int(cap), replace=False).astype(np.int64)
    return np.sort(selected.astype(np.int64))


def _prepare_partition_features(
    coords_um: np.ndarray,
    partition_features: np.ndarray | None,
    *,
    target_scale_um: float,
    feature_weight: float,
    feature_dims: int,
) -> np.ndarray:
    coords = np.asarray(coords_um, dtype=np.float32)
    centered = coords - coords.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    scale = max(scale, float(target_scale_um), 1e-6)
    spatial = centered / scale
    if partition_features is None or float(feature_weight) <= 0.0 or int(feature_dims) <= 0:
        return spatial.astype(np.float32)
    features = np.asarray(partition_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != coords.shape[0]:
        raise ValueError("partition_features must have shape (n_cells, n_features).")
    keep_dims = min(int(feature_dims), int(features.shape[1]))
    if keep_dims <= 0:
        return spatial.astype(np.float32)
    feature_view = _standardize_features(features[:, :keep_dims])
    joint = np.concatenate([spatial, float(feature_weight) * feature_view], axis=1)
    return joint.astype(np.float32)


def _fit_coordinate_seed_partition(
    coords_um: np.ndarray,
    *,
    partition_features: np.ndarray | None = None,
    target_scale_um: float,
    feature_weight: float,
    feature_dims: int,
    target_count: int,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    coords = np.asarray(coords_um, dtype=np.float32)
    n_cells = int(coords.shape[0])
    target_count = max(1, min(int(target_count), n_cells))
    if target_count == 1:
        members = [np.arange(n_cells, dtype=np.int32)]
        return np.asarray([coords.mean(axis=0)], dtype=np.float32), members

    x = _prepare_partition_features(
        coords,
        partition_features,
        target_scale_um=float(target_scale_um),
        feature_weight=float(feature_weight),
        feature_dims=int(feature_dims),
    )
    batch_size = min(n_cells, max(8192, int(target_count) * _subregion_seed_kmeans_batch_multiplier()))
    model = MiniBatchKMeans(
        n_clusters=target_count,
        random_state=int(seed),
        batch_size=batch_size,
        n_init=_subregion_seed_kmeans_n_init(),
        max_iter=_subregion_seed_kmeans_max_iter(),
        reassignment_ratio=0.0,
    )
    labels = model.fit_predict(x).astype(np.int32)
    centers: list[np.ndarray] = []
    members: list[np.ndarray] = []
    for label in np.unique(labels).tolist():
        member = np.flatnonzero(labels == int(label)).astype(np.int32)
        if member.size == 0:
            continue
        members.append(member)
        centers.append(_center_from_members(coords, member))
    return np.vstack(centers).astype(np.float32), members


def _reindex_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    unique = np.unique(labels)
    remap = np.full(int(unique.max()) + 1, -1, dtype=np.int32)
    remap[unique] = np.arange(unique.size, dtype=np.int32)
    return remap[labels].astype(np.int32)


def _members_from_labels(labels: np.ndarray) -> list[np.ndarray]:
    labels = _reindex_labels(labels)
    return [np.flatnonzero(labels == label).astype(np.int32) for label in range(int(labels.max()) + 1)]


def _refine_partition_by_feature_boundaries(
    coords_um: np.ndarray,
    members: list[np.ndarray],
    partition_features: np.ndarray | None,
    *,
    target_scale_um: float,
    feature_weight: float,
    spatial_weight: float = 1.0,
    n_iters: int | None = None,
    n_neighbors: int | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    refinement_iters = _subregion_boundary_refinement_iters() if n_iters is None else max(0, int(n_iters))
    refinement_knn = _subregion_boundary_refinement_knn() if n_neighbors is None else max(2, int(n_neighbors))
    if partition_features is None or float(feature_weight) <= 0.0 or refinement_iters <= 0:
        centers = np.vstack([_center_from_members(coords_um, member) for member in members]).astype(np.float32)
        return centers, members
    coords = np.asarray(coords_um, dtype=np.float32)
    features = np.asarray(partition_features, dtype=np.float32)
    labels = _labels_from_partition(coords.shape[0], members)
    labels = _reindex_labels(labels)
    nn = NearestNeighbors(
        n_neighbors=min(refinement_knn + 1, coords.shape[0]),
        metric="euclidean",
    )
    nn.fit(coords)
    neighbors = nn.kneighbors(coords, return_distance=False)
    chunk_size = max(50000, _env_int("SPATIAL_OT_SUBREGION_BOUNDARY_CHUNK_SIZE", 100000))
    scale_sq = max(float(target_scale_um) ** 2, 1e-6)
    spatial_weight = max(0.0, float(spatial_weight))
    for _ in range(refinement_iters):
        members_iter = _members_from_labels(labels)
        centers = np.vstack([_center_from_members(coords, member) for member in members_iter]).astype(np.float32)
        feature_centers = np.vstack(_feature_centroids_from_members(features, members_iter)).astype(np.float32)
        changed = 0
        new_labels = labels.copy()
        for start in range(0, coords.shape[0], chunk_size):
            stop = min(start + chunk_size, coords.shape[0])
            own = labels[start:stop, None]
            neighbor_labels = labels[neighbors[start:stop]]
            candidates = np.concatenate([own, neighbor_labels], axis=1)
            spatial_delta = coords[start:stop, None, :] - centers[candidates]
            spatial_score = np.sum(spatial_delta * spatial_delta, axis=2) / scale_sq
            feature_delta = features[start:stop, None, :] - feature_centers[candidates]
            feature_score = np.mean(feature_delta * feature_delta, axis=2)
            scores = spatial_weight * spatial_score + float(feature_weight) * feature_score
            assigned = candidates[np.arange(stop - start), np.argmin(scores, axis=1)].astype(np.int32)
            changed += int(np.sum(assigned != labels[start:stop]))
            new_labels[start:stop] = assigned
        labels = _reindex_labels(new_labels)
        if changed == 0:
            break
    out_members = _members_from_labels(labels)
    out_centers = np.vstack([_center_from_members(coords, member) for member in out_members]).astype(np.float32)
    return out_centers, out_members


def _estimate_connectivity_radius_um(
    coords_um: np.ndarray,
    *,
    target_scale_um: float,
    sample_size: int = 50000,
) -> float:
    coords = np.asarray(coords_um, dtype=np.float32)
    if coords.shape[0] < 2:
        return float("inf")
    if coords.shape[0] > int(sample_size):
        keep = np.linspace(0, coords.shape[0] - 1, num=int(sample_size), dtype=np.int64)
        sample = coords[keep]
    else:
        sample = coords
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(sample)
    distances = nn.kneighbors(sample, return_distance=True)[0][:, 1]
    finite = distances[np.isfinite(distances) & (distances > 0)]
    if finite.size == 0:
        return max(float(target_scale_um) * 0.5, 1e-6)
    local_radius = float(np.quantile(finite, 0.95)) * 4.0
    scale_radius = float(target_scale_um) * 0.5
    return max(local_radius, scale_radius, 1e-6)


def _split_members_by_spatial_connectivity(
    *,
    coords_um: np.ndarray,
    members: list[np.ndarray],
    connectivity_radius_um: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    coords = np.asarray(coords_um, dtype=np.float32)
    out_members: list[np.ndarray] = []
    out_centers: list[np.ndarray] = []
    radius = float(connectivity_radius_um)
    for member in members:
        member_arr = np.asarray(member, dtype=np.int32)
        if member_arr.size <= 2 or not np.isfinite(radius):
            out_members.append(np.sort(member_arr).astype(np.int32))
            out_centers.append(_center_from_members(coords, member_arr))
            continue
        local = coords[member_arr]
        nn = NearestNeighbors(radius=radius, metric="euclidean")
        nn.fit(local)
        graph = nn.radius_neighbors_graph(local, mode="connectivity")
        graph = graph.maximum(graph.T)
        n_components, component_labels = connected_components(graph, directed=False, return_labels=True)
        if n_components <= 1:
            out_members.append(np.sort(member_arr).astype(np.int32))
            out_centers.append(_center_from_members(coords, member_arr))
            continue
        for cid in range(int(n_components)):
            component = np.sort(member_arr[np.flatnonzero(component_labels == cid)]).astype(np.int32)
            if component.size == 0:
                continue
            out_members.append(component)
            out_centers.append(_center_from_members(coords, component))
    return np.vstack(out_centers).astype(np.float32), out_members


def _split_members_by_max_area(
    *,
    coords_um: np.ndarray,
    members: list[np.ndarray],
    max_area_um2: float | None,
    min_cells: int,
    partition_features: np.ndarray | None = None,
    feature_weight: float = 0.0,
    target_scale_um: float = 1.0,
    allow_below_min_cells: bool = False,
    seed: int = 1337,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if max_area_um2 is None or float(max_area_um2) <= 0.0:
        centers = np.vstack([_center_from_members(coords_um, member) for member in members]).astype(np.float32)
        return centers, [np.asarray(member, dtype=np.int32) for member in members]

    def _balanced_long_axis_split(member: np.ndarray, n_parts: int) -> list[np.ndarray]:
        member_arr = np.asarray(member, dtype=np.int32)
        local = coords[member_arr]
        centered_local = local - local.mean(axis=0, keepdims=True)
        axis = int(np.argmax(np.ptp(local, axis=0)))
        projection = centered_local[:, axis]
        order = np.argsort(projection, kind="mergesort")
        chunks = np.array_split(member_arr[order], int(n_parts))
        return [np.sort(chunk).astype(np.int32) for chunk in chunks if chunk.size > 0]

    coords = np.asarray(coords_um, dtype=np.float32)
    features = np.asarray(partition_features, dtype=np.float32) if partition_features is not None else None
    max_area = float(max_area_um2)
    min_cells = max(int(min_cells), 1)
    split_min_cells = 1 if bool(allow_below_min_cells) else min_cells
    out_members: list[np.ndarray] = []
    rng = np.random.default_rng(int(seed))

    for original in members:
        queue: list[np.ndarray] = [np.asarray(original, dtype=np.int32)]
        while queue:
            member = np.asarray(queue.pop(), dtype=np.int32)
            area = _member_area_um2(coords, member)
            if area <= max_area * (1.0 + 1e-6) or member.size < 2 * split_min_cells:
                out_members.append(np.sort(member).astype(np.int32))
                continue
            max_parts_by_size = max(1, int(member.size) // split_min_cells)
            requested_parts = max(2, int(np.ceil(area / max(max_area, 1e-8))))
            n_parts = min(max_parts_by_size, requested_parts, int(member.size))
            if n_parts <= 1:
                out_members.append(np.sort(member).astype(np.int32))
                continue
            if bool(allow_below_min_cells):
                split_members = _balanced_long_axis_split(member, int(n_parts))
            else:
                local_coords = coords[member]
                centered = local_coords - local_coords.mean(axis=0, keepdims=True)
                scale = max(float(target_scale_um), float(np.sqrt(max_area)), 1e-6)
                local_view = centered / scale
                if features is not None and float(feature_weight) > 0.0:
                    local_features = features[member]
                    local_view = np.concatenate(
                        [local_view, float(feature_weight) * _standardize_features(local_features)],
                        axis=1,
                    )
                split_members = []
                for trial_parts in range(int(n_parts), 1, -1):
                    model = MiniBatchKMeans(
                        n_clusters=int(trial_parts),
                        random_state=int(rng.integers(1, 2_147_483_647)),
                        batch_size=min(max(1024, int(trial_parts) * 8), int(member.size)),
                        n_init=_subregion_seed_kmeans_n_init(),
                        max_iter=max(_subregion_seed_kmeans_max_iter(), 20),
                        reassignment_ratio=0.0,
                    )
                    labels = model.fit_predict(local_view.astype(np.float32)).astype(np.int32)
                    trial_members = [
                        np.sort(member[np.flatnonzero(labels == label)]).astype(np.int32)
                        for label in range(int(labels.max()) + 1)
                        if np.any(labels == label)
                    ]
                    if len(trial_members) > 1 and all(child.size >= split_min_cells for child in trial_members):
                        split_members = trial_members
                        break
            if len(split_members) <= 1 or any(
                _member_area_um2(coords, child) >= area * 0.999 for child in split_members
            ):
                balanced = _balanced_long_axis_split(member, int(n_parts))
                if len(balanced) > 1 and all(child.size >= split_min_cells for child in balanced):
                    split_members = balanced
            if len(split_members) <= 1:
                out_members.append(np.sort(member).astype(np.int32))
                continue
            for child in split_members:
                if child.size == 0:
                    continue
                if _member_area_um2(coords, child) < area * 0.999:
                    queue.append(child)
                else:
                    out_members.append(child)

    centers = np.vstack([_center_from_members(coords, member) for member in out_members]).astype(np.float32)
    return centers, out_members


def _max_member_area_um2(coords_um: np.ndarray, members: list[np.ndarray]) -> float:
    if not members:
        return 0.0
    return float(max(_member_area_um2(coords_um, member) for member in members))


def _validate_max_subregion_area(
    coords_um: np.ndarray,
    members: list[np.ndarray],
    *,
    max_area_um2: float | None,
) -> None:
    if max_area_um2 is None or float(max_area_um2) <= 0.0:
        return
    max_area = float(max_area_um2)
    violating = [
        (idx, _member_area_um2(coords_um, member), int(np.asarray(member, dtype=np.int32).size))
        for idx, member in enumerate(members)
        if _member_area_um2(coords_um, member) > max_area * (1.0 + 1e-6)
    ]
    if violating:
        idx, area, n_cells = max(violating, key=lambda item: item[1])
        raise RuntimeError(
            "Constructed subregions violate max_subregion_area_um2 after area-aware splitting/merging: "
            f"{len(violating)} subregions exceed {max_area:g} um^2; largest is subregion {idx} "
            f"with area={area:g} um^2 and n_cells={n_cells}. Reduce min_cells, increase max_subregions, "
            "or use a less restrictive area cap."
        )


def _labels_from_partition(n_cells: int, members: list[np.ndarray]) -> np.ndarray:
    labels = np.full(int(n_cells), -1, dtype=np.int32)
    for idx, member in enumerate(members):
        labels[np.asarray(member, dtype=np.int64)] = int(idx)
    return labels


def _feature_centroids_from_members(
    features: np.ndarray | None,
    members: list[np.ndarray],
) -> list[np.ndarray] | None:
    if features is None:
        return None
    x = np.asarray(features, dtype=np.float32)
    return [
        x[np.asarray(member, dtype=np.int32)].mean(axis=0).astype(np.float32)
        for member in members
    ]


def _sort_partition_by_center(
    centers_um: np.ndarray,
    members: list[np.ndarray],
    component_ids: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    centers = np.asarray(centers_um, dtype=np.float32)
    if centers.shape[0] == 0:
        return centers, members, component_ids
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return (
        centers[order].astype(np.float32),
        [np.asarray(members[int(idx)], dtype=np.int32) for idx in order.tolist()],
        [np.asarray(component_ids[int(idx)], dtype=np.int32) for idx in order.tolist()],
    )


def _region_adjacency_from_knn(
    coords_um: np.ndarray,
    members: list[np.ndarray],
    *,
    n_neighbors: int = 8,
) -> list[set[int]]:
    coords = np.asarray(coords_um, dtype=np.float32)
    n_cells = int(coords.shape[0])
    n_regions = len(members)
    adjacency = [set() for _ in range(n_regions)]
    if n_cells < 2 or n_regions < 2:
        return adjacency
    labels = _labels_from_partition(n_cells, members)
    nn = NearestNeighbors(n_neighbors=min(int(n_neighbors) + 1, n_cells), metric="euclidean")
    nn.fit(coords)
    neigh = nn.kneighbors(coords, return_distance=False)[:, 1:]
    row_labels = np.repeat(labels, neigh.shape[1])
    col_labels = labels[neigh.reshape(-1)]
    mask = (row_labels >= 0) & (col_labels >= 0) & (row_labels != col_labels)
    for src, dst in zip(row_labels[mask].tolist(), col_labels[mask].tolist(), strict=False):
        adjacency[int(src)].add(int(dst))
        adjacency[int(dst)].add(int(src))
    return adjacency


def _merge_data_driven_regions(
    *,
    coords_um: np.ndarray,
    centers_um: np.ndarray,
    members: list[np.ndarray],
    component_ids: list[np.ndarray],
    min_cells: int,
    max_subregions: int,
    partition_features: np.ndarray | None = None,
    feature_weight: float = 0.0,
    target_scale_um: float = 1.0,
    max_area_um2: float | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    min_cells = int(min_cells)
    if min_cells < 1:
        raise ValueError("min_cells must be at least 1.")
    total_cells = int(sum(np.asarray(member, dtype=np.int32).size for member in members))
    if total_cells < min_cells:
        raise RuntimeError("No valid subregions can be created because the full partition is smaller than min_cells.")

    coords = np.asarray(coords_um, dtype=np.float32)
    centers = [np.asarray(center, dtype=np.float32) for center in np.asarray(centers_um, dtype=np.float32)]
    member_list = [np.asarray(member, dtype=np.int32) for member in members]
    id_list = [np.asarray(ids, dtype=np.int32) for ids in component_ids]
    feature_centers = _feature_centroids_from_members(partition_features, member_list)
    active = np.ones(len(member_list), dtype=bool)
    blocked_by_area = np.zeros(len(member_list), dtype=bool)
    adjacency = _region_adjacency_from_knn(coords, member_list)

    def active_indices() -> np.ndarray:
        return np.flatnonzero(active)

    while int(np.sum(active)) > 1:
        active_idx = active_indices()
        sizes = np.asarray([member_list[idx].size for idx in active_idx], dtype=np.int64)
        small = active_idx[(sizes < min_cells) & (~blocked_by_area[active_idx])]
        if small.size:
            source = int(small[np.argmin([member_list[idx].size for idx in small])])
        elif int(max_subregions) > 0 and active_idx.size > int(max_subregions):
            source = int(active_idx[np.argmin(sizes)])
        else:
            break

        candidates = [idx for idx in adjacency[source] if active[int(idx)] and int(idx) != source]
        if candidates:
            target_pool = np.asarray(candidates, dtype=np.int32)
        else:
            target_pool = active_idx[active_idx != source]
        if max_area_um2 is not None and float(max_area_um2) > 0.0:
            feasible_targets = []
            for candidate in target_pool.tolist():
                merged_candidate = np.unique(
                    np.concatenate([member_list[int(candidate)], member_list[source]])
                ).astype(np.int32)
                if _member_area_um2(coords, merged_candidate) <= float(max_area_um2) * (1.0 + 1e-6):
                    feasible_targets.append(int(candidate))
            if feasible_targets:
                target_pool = np.asarray(feasible_targets, dtype=np.int32)
            elif member_list[source].size < min_cells and int(max_subregions) <= 0:
                blocked_by_area[source] = True
                continue
        center_arr = np.vstack([centers[int(idx)] for idx in target_pool]).astype(np.float32)
        delta = center_arr - centers[source]
        score = np.sum(delta * delta, axis=1) / max(float(target_scale_um) ** 2, 1e-6)
        if feature_centers is not None and float(feature_weight) > 0.0:
            feat_arr = np.vstack([feature_centers[int(idx)] for idx in target_pool]).astype(np.float32)
            feat_delta = feat_arr - feature_centers[source]
            score = score + float(feature_weight) * np.mean(feat_delta * feat_delta, axis=1)
        target = int(target_pool[int(np.argmin(score))])

        merged_members = np.unique(np.concatenate([member_list[target], member_list[source]])).astype(np.int32)
        merged_ids = np.unique(np.concatenate([id_list[target], id_list[source]])).astype(np.int32)
        member_list[target] = merged_members
        id_list[target] = merged_ids
        centers[target] = _center_from_members(coords, merged_members)
        if feature_centers is not None:
            x = np.asarray(partition_features, dtype=np.float32)
            feature_centers[target] = x[merged_members].mean(axis=0).astype(np.float32)

        active[source] = False
        source_neighbors = set(adjacency[source])
        adjacency[target].update(source_neighbors)
        adjacency[target].discard(source)
        adjacency[target].discard(target)
        for neighbor in source_neighbors:
            adjacency[int(neighbor)].discard(source)
            if active[int(neighbor)] and int(neighbor) != target:
                adjacency[int(neighbor)].add(target)
        adjacency[source].clear()

    active_idx = active_indices()
    out_members = [np.asarray(member_list[int(idx)], dtype=np.int32) for idx in active_idx.tolist()]
    if (max_area_um2 is None or float(max_area_um2) <= 0.0) and any(member.size < min_cells for member in out_members):
        raise RuntimeError("No valid subregions remain after enforcing min_cells.")
    out_ids = [np.asarray(id_list[int(idx)], dtype=np.int32) for idx in active_idx.tolist()]
    out_centers = np.vstack([centers[int(idx)] for idx in active_idx.tolist()]).astype(np.float32)
    return out_centers, out_members, out_ids


def build_deep_graph_segmentation_subregions(
    coords_um: np.ndarray,
    segmentation_features: np.ndarray,
    *,
    target_scale_um: float,
    min_cells: int,
    max_subregions: int,
    max_area_um2: float | None = None,
    segmentation_knn: int | None = None,
    segmentation_feature_dims: int | None = None,
    segmentation_feature_weight: float | None = None,
    segmentation_spatial_weight: float | None = None,
    seed: int = 1337,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray], list[np.ndarray]]:
    coords = np.asarray(coords_um, dtype=np.float32)
    if coords.shape[0] == 0:
        raise RuntimeError("No cells were provided, so no subregions can be created.")
    if float(target_scale_um) <= 0:
        raise ValueError("target_scale_um must be positive for deep graph segmentation.")
    area_scale_um = float(target_scale_um)
    if max_area_um2 is not None and float(max_area_um2) > 0.0:
        area_scale_um = min(area_scale_um, float(np.sqrt(float(max_area_um2))))
    feature_arr = np.asarray(segmentation_features, dtype=np.float32)
    if feature_arr.ndim != 2 or feature_arr.shape[0] != coords.shape[0]:
        raise ValueError("segmentation_features must have shape (n_cells, n_features).")
    feature_dims = (
        _deep_segmentation_feature_dims_default()
        if segmentation_feature_dims is None
        else int(segmentation_feature_dims)
    )
    keep_dims = min(max(int(feature_dims), 1), int(feature_arr.shape[1]))
    feature_view = _standardize_features(feature_arr[:, :keep_dims])
    target_count = _target_data_driven_region_count(
        coords,
        target_scale_um=float(area_scale_um),
        min_cells=int(min_cells),
        max_subregions=int(max_subregions),
        max_area_um2=max_area_um2,
    )
    _geometry_progress(
        f"deep segmentation seed partition target_count={int(target_count)}, area_scale={float(area_scale_um):g}um"
    )
    _, seed_members = _fit_coordinate_seed_partition(
        coords,
        partition_features=feature_view,
        target_scale_um=float(area_scale_um),
        feature_weight=(
            _deep_segmentation_feature_weight_default()
            if segmentation_feature_weight is None
            else float(segmentation_feature_weight)
        ),
        feature_dims=int(feature_view.shape[1]),
        target_count=target_count,
        seed=int(seed),
    )
    _geometry_progress(f"deep segmentation seed partition produced {len(seed_members)} regions")
    _, refined_members = _refine_partition_by_feature_boundaries(
        coords,
        seed_members,
        feature_view,
        target_scale_um=float(area_scale_um),
        feature_weight=(
            _deep_segmentation_feature_weight_default()
            if segmentation_feature_weight is None
            else float(segmentation_feature_weight)
        ),
        spatial_weight=(
            _deep_segmentation_spatial_weight_default()
            if segmentation_spatial_weight is None
            else float(segmentation_spatial_weight)
        ),
        n_iters=_deep_segmentation_refinement_iters_default(),
        n_neighbors=_deep_segmentation_knn_default() if segmentation_knn is None else int(segmentation_knn),
    )
    _geometry_progress(f"deep segmentation boundary refinement produced {len(refined_members)} regions")
    connectivity_radius = _estimate_connectivity_radius_um(coords, target_scale_um=float(area_scale_um))
    _geometry_progress(f"splitting {len(refined_members)} regions by spatial connectivity")
    basic_centers, basic_members = _split_members_by_spatial_connectivity(
        coords_um=coords,
        members=refined_members,
        connectivity_radius_um=connectivity_radius,
    )
    _geometry_progress(f"spatial connectivity split produced {len(basic_members)} basic regions")
    _geometry_progress(
        "pre-merge max-area split deferred until after min-cell merging "
        f"({len(basic_members)} basic regions)"
    )
    basic_ids = [np.asarray([idx], dtype=np.int32) for idx in range(len(basic_members))]
    _geometry_progress(f"merging {len(basic_members)} basic regions with min_cells={int(min_cells)}")
    subregion_centers, subregion_members, subregion_basic_ids = _merge_data_driven_regions(
        coords_um=coords,
        centers_um=basic_centers,
        members=basic_members,
        component_ids=basic_ids,
        min_cells=int(min_cells),
        max_subregions=int(max_subregions),
        partition_features=feature_view,
        feature_weight=(
            _deep_segmentation_feature_weight_default()
            if segmentation_feature_weight is None
            else float(segmentation_feature_weight)
        ),
        target_scale_um=float(area_scale_um),
        max_area_um2=max_area_um2,
    )
    _geometry_progress(f"area-aware merge produced {len(subregion_members)} subregions")
    if max_area_um2 is not None and float(max_area_um2) > 0.0:
        _geometry_progress(f"final max-area split on {len(subregion_members)} subregions")
        subregion_centers, subregion_members = _split_members_by_max_area(
            coords_um=coords,
            members=subregion_members,
            max_area_um2=max_area_um2,
            min_cells=int(min_cells),
            partition_features=feature_view,
            feature_weight=(
                _deep_segmentation_feature_weight_default()
                if segmentation_feature_weight is None
                else float(segmentation_feature_weight)
            ),
            target_scale_um=float(area_scale_um),
            allow_below_min_cells=True,
            seed=int(seed) + 31,
        )
        subregion_basic_ids = [np.asarray([idx], dtype=np.int32) for idx in range(len(subregion_members))]
        _geometry_progress(f"final max-area split produced {len(subregion_members)} subregions; validating")
        _validate_max_subregion_area(coords, subregion_members, max_area_um2=max_area_um2)
        _geometry_progress("max-area validation passed")
    subregion_centers, subregion_members, subregion_basic_ids = _sort_partition_by_center(
        subregion_centers,
        subregion_members,
        subregion_basic_ids,
    )
    _validate_mutually_exclusive_memberships(coords.shape[0], subregion_members, require_full_coverage=True)
    return (
        subregion_centers.astype(np.float32),
        [np.asarray(member, dtype=np.int32) for member in subregion_members],
        basic_centers.astype(np.float32),
        [np.asarray(member, dtype=np.int32) for member in basic_members],
        [np.asarray(ids, dtype=np.int32) for ids in subregion_basic_ids],
    )


def build_data_driven_subregions(
    coords_um: np.ndarray,
    radius_um: float,
    stride_um: float,
    min_cells: int,
    max_subregions: int,
    *,
    target_scale_um: float | None = None,
    partition_features: np.ndarray | None = None,
    partition_feature_weight: float | None = None,
    partition_feature_dims: int | None = None,
    max_area_um2: float | None = None,
    seed: int = 1337,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray], list[np.ndarray]]:
    del radius_um
    coords = np.asarray(coords_um, dtype=np.float32)
    if coords.shape[0] == 0:
        raise RuntimeError("No cells were provided, so no subregions can be created.")
    scale_um = float(target_scale_um) if target_scale_um is not None else float(stride_um)
    if scale_um <= 0 or float(stride_um) <= 0:
        raise ValueError("target scale and stride_um must be positive.")
    area_scale_um = scale_um
    if max_area_um2 is not None and float(max_area_um2) > 0.0:
        area_scale_um = min(area_scale_um, float(np.sqrt(float(max_area_um2))))
    feature_weight = (
        _subregion_partition_feature_weight()
        if partition_feature_weight is None
        else max(0.0, float(partition_feature_weight))
    )
    feature_dims = (
        _subregion_partition_feature_dims()
        if partition_feature_dims is None
        else max(0, int(partition_feature_dims))
    )
    feature_view = None
    if partition_features is not None and feature_weight > 0.0 and feature_dims > 0:
        feature_arr = np.asarray(partition_features, dtype=np.float32)
        if feature_arr.ndim != 2 or feature_arr.shape[0] != coords.shape[0]:
            raise ValueError("partition_features must have shape (n_cells, n_features).")
        feature_view = _standardize_features(feature_arr[:, : min(feature_dims, feature_arr.shape[1])])

    target_count = _target_data_driven_region_count(
        coords,
        target_scale_um=area_scale_um,
        min_cells=int(min_cells),
        max_subregions=int(max_subregions),
        max_area_um2=max_area_um2,
    )
    _geometry_progress(
        f"data-driven seed partition target_count={int(target_count)}, area_scale={float(area_scale_um):g}um"
    )
    _, seed_members = _fit_coordinate_seed_partition(
        coords,
        partition_features=feature_view,
        target_scale_um=area_scale_um,
        feature_weight=feature_weight,
        feature_dims=feature_dims,
        target_count=target_count,
        seed=int(seed),
    )
    _geometry_progress(f"data-driven seed partition produced {len(seed_members)} regions")
    _, seed_members = _refine_partition_by_feature_boundaries(
        coords,
        seed_members,
        feature_view,
        target_scale_um=area_scale_um,
        feature_weight=feature_weight,
    )
    _geometry_progress(f"data-driven boundary refinement produced {len(seed_members)} regions")
    connectivity_radius = _estimate_connectivity_radius_um(coords, target_scale_um=area_scale_um)
    _geometry_progress(f"splitting {len(seed_members)} regions by spatial connectivity")
    basic_centers, basic_members = _split_members_by_spatial_connectivity(
        coords_um=coords,
        members=seed_members,
        connectivity_radius_um=connectivity_radius,
    )
    _geometry_progress(f"spatial connectivity split produced {len(basic_members)} basic regions")
    _geometry_progress(
        "pre-merge max-area split deferred until after min-cell merging "
        f"({len(basic_members)} basic regions)"
    )
    basic_ids = [np.asarray([idx], dtype=np.int32) for idx in range(len(basic_members))]
    _geometry_progress(f"merging {len(basic_members)} basic regions with min_cells={int(min_cells)}")
    subregion_centers, subregion_members, subregion_basic_ids = _merge_data_driven_regions(
        coords_um=coords,
        centers_um=basic_centers,
        members=basic_members,
        component_ids=basic_ids,
        min_cells=int(min_cells),
        max_subregions=int(max_subregions),
        partition_features=feature_view,
        feature_weight=feature_weight,
        target_scale_um=area_scale_um,
        max_area_um2=max_area_um2,
    )
    _geometry_progress(f"area-aware merge produced {len(subregion_members)} subregions")
    if max_area_um2 is not None and float(max_area_um2) > 0.0:
        _geometry_progress(f"final max-area split on {len(subregion_members)} subregions")
        subregion_centers, subregion_members = _split_members_by_max_area(
            coords_um=coords,
            members=subregion_members,
            max_area_um2=max_area_um2,
            min_cells=int(min_cells),
            partition_features=feature_view,
            feature_weight=feature_weight,
            target_scale_um=area_scale_um,
            allow_below_min_cells=True,
            seed=int(seed) + 31,
        )
        subregion_basic_ids = [np.asarray([idx], dtype=np.int32) for idx in range(len(subregion_members))]
        _geometry_progress(f"final max-area split produced {len(subregion_members)} subregions; validating")
        _validate_max_subregion_area(coords, subregion_members, max_area_um2=max_area_um2)
        _geometry_progress("max-area validation passed")
    subregion_centers, subregion_members, subregion_basic_ids = _sort_partition_by_center(
        subregion_centers,
        subregion_members,
        subregion_basic_ids,
    )
    _validate_mutually_exclusive_memberships(coords.shape[0], subregion_members, require_full_coverage=True)
    return (
        subregion_centers.astype(np.float32),
        [np.asarray(member, dtype=np.int32) for member in subregion_members],
        basic_centers.astype(np.float32),
        [np.asarray(member, dtype=np.int32) for member in basic_members],
        [np.asarray(ids, dtype=np.int32) for ids in subregion_basic_ids],
    )

def _coerce_membership_indices(
    member: np.ndarray,
    *,
    n_cells: int,
    subregion_index: int,
) -> np.ndarray:
    raw = np.asarray(member)
    if raw.ndim != 1:
        raise RuntimeError(
            f"Subregion {int(subregion_index)} membership must be a one-dimensional array of integer cell indices."
        )
    if raw.size == 0:
        raise RuntimeError("Constructed subregions must not be empty.")

    if np.issubdtype(raw.dtype, np.integer):
        member_arr = raw.astype(np.int64, copy=False)
    elif raw.dtype == object:
        flat = raw.reshape(-1)
        if not all(isinstance(value, (int, np.integer)) and not isinstance(value, bool) for value in flat.tolist()):
            raise RuntimeError("Constructed subregions must use integer cell indices.")
        member_arr = raw.astype(np.int64, copy=False)
    else:
        raise RuntimeError("Constructed subregions must use integer cell indices.")

    if int(member_arr.min()) < 0 or int(member_arr.max()) >= int(n_cells):
        raise RuntimeError("Constructed subregions contain out-of-range cell indices.")
    if np.unique(member_arr).size != member_arr.size:
        raise RuntimeError("Constructed subregions contain duplicate cell indices.")
    return member_arr


def _validate_mutually_exclusive_memberships(
    n_cells: int,
    members: list[np.ndarray],
    *,
    require_full_coverage: bool = False,
) -> np.ndarray:
    counts = np.zeros(int(n_cells), dtype=np.int32)
    for idx, member in enumerate(members):
        member_arr = _coerce_membership_indices(member, n_cells=int(n_cells), subregion_index=int(idx))
        np.add.at(counts, member_arr, 1)
    if int(counts.max(initial=0)) > 1:
        raise RuntimeError("Constructed subregions are not mutually exclusive.")
    if bool(require_full_coverage) and (counts.size == 0 or int(counts.min()) < 1):
        raise RuntimeError("Constructed subregions do not cover every cell exactly once.")
    return counts


def build_subregions(
    coords_um: np.ndarray,
    radius_um: float,
    stride_um: float,
    min_cells: int,
    max_subregions: int,
    max_area_um2: float | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    subregion_centers, subregion_members, _, _, _ = build_data_driven_subregions(
        coords_um=coords_um,
        radius_um=radius_um,
        stride_um=stride_um,
        min_cells=min_cells,
        max_subregions=max_subregions,
        max_area_um2=max_area_um2,
    )
    return subregion_centers, subregion_members


def build_partition_subregions_from_grid_tiles(
    coords_um: np.ndarray,
    radius_um: float,
    stride_um: float,
    min_cells: int,
    max_subregions: int,
    *,
    partition_features: np.ndarray | None = None,
    partition_feature_weight: float | None = None,
    partition_feature_dims: int | None = None,
    max_area_um2: float | None = None,
    seed: int = 1337,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Backward-compatible alias for data-driven subregion construction.

    The name is retained so older scripts keep running; no grid tiles or square
    proposal regions are constructed here.
    """
    return build_data_driven_subregions(
        coords_um=coords_um,
        radius_um=radius_um,
        stride_um=stride_um,
        min_cells=min_cells,
        max_subregions=max_subregions,
        target_scale_um=float(stride_um),
        partition_features=partition_features,
        partition_feature_weight=partition_feature_weight,
        partition_feature_dims=partition_feature_dims,
        max_area_um2=max_area_um2,
        seed=int(seed),
    )


def build_basic_niches(
    coords_um: np.ndarray,
    niche_size_um: float,
    min_cells: int,
    max_subregions: int,
    *,
    partition_features: np.ndarray | None = None,
    partition_feature_weight: float | None = None,
    partition_feature_dims: int | None = None,
    max_area_um2: float | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build mutually exclusive data-driven atomic subregions.

    ``niche_size_um`` is a scale hint for the seed partition, not a fixed
    square/circular window size.
    """
    centers, members, _, _, _ = build_data_driven_subregions(
        coords_um=coords_um,
        radius_um=float(niche_size_um),
        stride_um=float(niche_size_um),
        min_cells=int(min_cells),
        max_subregions=int(max_subregions),
        target_scale_um=float(niche_size_um),
        partition_features=partition_features,
        partition_feature_weight=partition_feature_weight,
        partition_feature_dims=partition_feature_dims,
        max_area_um2=max_area_um2,
    )
    return centers, members


def build_composite_subregions_from_basic_niches(
    coords_um: np.ndarray,
    radius_um: float,
    stride_um: float,
    min_cells: int,
    max_subregions: int,
    basic_niche_size_um: float,
    *,
    partition_features: np.ndarray | None = None,
    partition_feature_weight: float | None = None,
    partition_feature_dims: int | None = None,
    max_area_um2: float | None = None,
    seed: int = 1337,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Build data-driven subregions plus atomic seed provenance."""
    return build_data_driven_subregions(
        coords_um=coords_um,
        radius_um=radius_um,
        stride_um=stride_um,
        min_cells=min_cells,
        max_subregions=max_subregions,
        target_scale_um=float(basic_niche_size_um),
        partition_features=partition_features,
        partition_feature_weight=partition_feature_weight,
        partition_feature_dims=partition_feature_dims,
        max_area_um2=max_area_um2,
        seed=int(seed),
    )


def _region_geometries_from_observed_points(
    subregion_members: list[np.ndarray],
) -> list[RegionGeometry]:
    return [
        RegionGeometry(
            region_id=f"region_{idx:04d}",
            members=np.asarray(members, dtype=np.int32),
            use_observed_geometry=True,
        )
        for idx, members in enumerate(subregion_members)
    ]


def _triangle_area(tri: np.ndarray) -> float:
    x1, y1 = tri[0]
    x2, y2 = tri[1]
    x3, y3 = tri[2]
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) * 0.5


def _sample_points_in_triangle(tri: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return tri[0] + u * (tri[1] - tri[0]) + v * (tri[2] - tri[0])


def _sample_uniform_points_in_convex_hull(coords: np.ndarray, n_points: int, seed: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if coords.shape[0] <= n_points:
        return coords.astype(np.float32)

    rng = np.random.default_rng(seed)
    try:
        hull = ConvexHull(coords)
        hull_pts = coords[hull.vertices]
    except Exception:
        hull_pts = coords

    if hull_pts.shape[0] < 3:
        take = rng.choice(coords.shape[0], size=min(n_points, coords.shape[0]), replace=False)
        return coords[take].astype(np.float32)

    anchor = hull_pts[0]
    triangles = []
    areas = []
    for i in range(1, hull_pts.shape[0] - 1):
        tri = np.vstack([anchor, hull_pts[i], hull_pts[i + 1]])
        area = _triangle_area(tri)
        if area > 1e-10:
            triangles.append(tri)
            areas.append(area)

    if not triangles:
        take = rng.choice(coords.shape[0], size=min(n_points, coords.shape[0]), replace=False)
        return coords[take].astype(np.float32)

    areas_arr = np.asarray(areas, dtype=np.float64)
    probs = areas_arr / areas_arr.sum()
    chosen = rng.choice(len(triangles), size=n_points, replace=True, p=probs)
    counts = np.bincount(chosen, minlength=len(triangles))
    samples = []
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        samples.append(_sample_points_in_triangle(triangles[idx], count, rng))
    return np.vstack(samples).astype(np.float32)


def _sample_observed_point_cloud_geometry(
    observed_coords: np.ndarray,
    n_points: int,
    seed: int,
) -> np.ndarray:
    coords = np.asarray(observed_coords, dtype=np.float64)
    if coords.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    coords = coords[np.all(np.isfinite(coords), axis=1)]
    if coords.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    coords = np.unique(coords, axis=0)
    if coords.shape[0] <= int(n_points):
        return coords.astype(np.float32)
    rng = np.random.default_rng(seed)
    keep = rng.choice(coords.shape[0], size=int(n_points), replace=False)
    return coords[np.sort(keep)].astype(np.float32)


def _sample_uniform_points_in_polygon_components(
    polygon_components: list[np.ndarray],
    n_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    components = [np.asarray(poly, dtype=np.float64) for poly in polygon_components if np.asarray(poly).shape[0] >= 3]
    if not components:
        return np.zeros((0, 2), dtype=np.float32)

    bboxes = []
    areas = []
    paths = []
    for poly in components:
        xmin, ymin = poly.min(axis=0)
        xmax, ymax = poly.max(axis=0)
        bboxes.append((xmin, xmax, ymin, ymax))
        x = poly[:, 0]
        y = poly[:, 1]
        poly_area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        areas.append(max(float(poly_area), 1e-12))
        paths.append(MplPath(poly))
    probs = np.asarray(areas, dtype=np.float64)
    probs = probs / probs.sum()

    points = []
    max_attempts = max(32, n_points // 8)
    attempts = 0
    while len(points) < n_points and attempts < max_attempts:
        attempts += 1
        comp_idx = int(rng.choice(len(components), p=probs))
        xmin, xmax, ymin, ymax = bboxes[comp_idx]
        batch_n = max(128, 2 * (n_points - len(points)))
        cand = np.column_stack(
            [
                rng.uniform(xmin, xmax, size=batch_n),
                rng.uniform(ymin, ymax, size=batch_n),
            ]
        )
        inside = paths[comp_idx].contains_points(cand)
        if np.any(inside):
            points.extend(cand[inside].tolist())
    if len(points) < n_points:
        raise ValueError("Unable to sample enough points from polygon geometry.")
    return np.asarray(points[:n_points], dtype=np.float32)


def _sample_uniform_points_in_mask(
    mask: np.ndarray,
    n_points: int,
    seed: int,
    affine: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.asarray(mask, dtype=bool)
    ij = np.argwhere(mask)
    if ij.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pick = rng.choice(ij.shape[0], size=n_points, replace=ij.shape[0] < n_points)
    sampled = ij[pick].astype(np.float64)
    sampled += rng.random(sampled.shape)
    xy = sampled[:, ::-1]
    if affine is not None:
        affine = np.asarray(affine, dtype=np.float64)
        if affine.shape == (3, 3):
            homo = np.column_stack([xy, np.ones(xy.shape[0])])
            xy = (homo @ affine.T)[:, :2]
    return xy.astype(np.float32)


def sample_geometry_points(
    region_geometry: RegionGeometry,
    observed_coords: np.ndarray,
    n_points: int,
    seed: int,
    allow_convex_hull_fallback: bool = True,
    warn_on_fallback: bool = True,
) -> tuple[np.ndarray, str, bool]:
    if region_geometry.mask is not None:
        pts = _sample_uniform_points_in_mask(region_geometry.mask, n_points=n_points, seed=seed, affine=region_geometry.affine)
        if pts.shape[0] == 0:
            raise ValueError(f"Region '{region_geometry.region_id}' provided an empty mask geometry.")
        return pts, "mask", False
    if region_geometry.polygon_components:
        pts = _sample_uniform_points_in_polygon_components(region_geometry.polygon_components, n_points=n_points, seed=seed)
        if pts.shape[0] > 0:
            return pts, "polygon_components", False
        raise ValueError(f"Region '{region_geometry.region_id}' provided polygon components but sampling produced no points.")
    if region_geometry.polygon_vertices is not None and np.asarray(region_geometry.polygon_vertices).shape[0] >= 3:
        pts = _sample_uniform_points_in_polygon_components([np.asarray(region_geometry.polygon_vertices)], n_points=n_points, seed=seed)
        if pts.shape[0] > 0:
            return pts, "polygon", False
        raise ValueError(f"Region '{region_geometry.region_id}' provided polygon geometry but sampling produced no points.")
    if bool(getattr(region_geometry, "use_observed_geometry", False)):
        pts = _sample_observed_point_cloud_geometry(observed_coords, n_points=n_points, seed=seed)
        if pts.shape[0] == 0:
            raise ValueError(f"Region '{region_geometry.region_id}' has no observed coordinates for data-driven geometry.")
        return pts, "observed_point_cloud", False
    if not allow_convex_hull_fallback:
        raise ValueError(
            f"Region '{region_geometry.region_id}' has no explicit geometry. "
            "Pass polygon/mask geometry or allow convex hull fallback explicitly."
        )
    if warn_on_fallback:
        warnings.warn(
            f"Region '{region_geometry.region_id}' has no explicit geometry; using convex hull of observed coordinates for shape normalization.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _sample_uniform_points_in_convex_hull(observed_coords, n_points=n_points, seed=seed), "convex_hull_fallback", True


def _region_geometries_from_members(
    subregion_members: list[np.ndarray],
) -> list[RegionGeometry]:
    return [
        RegionGeometry(
            region_id=f"region_{idx:04d}",
            members=np.asarray(members, dtype=np.int32),
        )
        for idx, members in enumerate(subregion_members)
    ]


def _ordered_hull_points(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] < 3:
        return coords
    try:
        hull = ConvexHull(coords)
        return coords[hull.vertices]
    except Exception:
        return coords


def _subregion_shape_descriptors(coords: np.ndarray) -> dict[str, float]:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return {
            "shape_area": 0.0,
            "shape_perimeter": 0.0,
            "shape_compactness": 0.0,
            "shape_aspect_ratio": 1.0,
            "shape_eccentricity": 0.0,
            "shape_radius_mean": 0.0,
            "shape_radius_std": 0.0,
        }

    hull_pts = _ordered_hull_points(coords)
    if hull_pts.shape[0] >= 3:
        x = hull_pts[:, 0]
        y = hull_pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        perimeter = float(np.sum(np.sqrt(np.sum((np.roll(hull_pts, -1, axis=0) - hull_pts) ** 2, axis=1))))
    else:
        area = 0.0
        perimeter = float(np.sum(np.sqrt(np.sum((coords - coords.mean(axis=0, keepdims=True)) ** 2, axis=1))))

    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T) if coords.shape[0] > 1 else np.eye(2)
    eigvals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 1e-12))[::-1]
    major = float(np.sqrt(eigvals[0]))
    minor = float(np.sqrt(eigvals[1]))
    aspect_ratio = major / max(minor, 1e-8)
    eccentricity = float(np.sqrt(max(0.0, 1.0 - eigvals[1] / max(eigvals[0], 1e-12))))
    radius = np.sqrt(np.sum(centered**2, axis=1))
    compactness = float(4.0 * np.pi * area / max(perimeter**2, 1e-12)) if perimeter > 0 else 0.0
    return {
        "shape_area": float(area),
        "shape_perimeter": float(perimeter),
        "shape_compactness": compactness,
        "shape_aspect_ratio": float(aspect_ratio),
        "shape_eccentricity": eccentricity,
        "shape_radius_mean": float(radius.mean()) if radius.size else 0.0,
        "shape_radius_std": float(radius.std()) if radius.size else 0.0,
    }


def _shape_descriptor_frame(
    subregion_members: list[np.ndarray],
    coords_um: np.ndarray,
    region_geometries: list[RegionGeometry] | None = None,
) -> pd.DataFrame:
    if region_geometries is not None and len(region_geometries) != len(subregion_members):
        raise ValueError("region_geometries must have the same length as subregion_members for shape descriptors.")
    large_observed_only = len(subregion_members) > 50000 and (
        region_geometries is None
        or all(
            region.mask is None
            and not region.polygon_components
            and region.polygon_vertices is None
            for region in region_geometries
        )
    )
    rows = []
    for rid, members in enumerate(subregion_members):
        source = "observed_coordinate_hull"
        member_arr = np.asarray(members, dtype=np.int64)
        if large_observed_only and member_arr.size <= 3:
            if member_arr.size == 0:
                desc = _subregion_shape_descriptors(np.zeros((0, 2), dtype=np.float64))
            elif member_arr.size == 1:
                desc = {
                    "shape_area": 0.0,
                    "shape_perimeter": 0.0,
                    "shape_compactness": 0.0,
                    "shape_aspect_ratio": 1.0,
                    "shape_eccentricity": 0.0,
                    "shape_radius_mean": 0.0,
                    "shape_radius_std": 0.0,
                }
            elif member_arr.size == 2:
                pts = np.asarray(coords_um[member_arr], dtype=np.float64)
                dist = float(np.linalg.norm(pts[1] - pts[0]))
                desc = {
                    "shape_area": 0.0,
                    "shape_perimeter": dist,
                    "shape_compactness": 0.0,
                    "shape_aspect_ratio": float(max(dist, 1e-8) / 1e-8),
                    "shape_eccentricity": 1.0 if dist > 0 else 0.0,
                    "shape_radius_mean": float(dist * 0.5),
                    "shape_radius_std": 0.0,
                }
            else:
                pts = np.asarray(coords_um[member_arr], dtype=np.float64)
                centered = pts - pts.mean(axis=0, keepdims=True)
                x = pts[:, 0]
                y = pts[:, 1]
                area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
                perimeter = float(np.sum(np.sqrt(np.sum((np.roll(pts, -1, axis=0) - pts) ** 2, axis=1))))
                radius = np.sqrt(np.sum(centered * centered, axis=1))
                if np.allclose(centered, 0.0):
                    aspect_ratio = 1.0
                    eccentricity = 0.0
                else:
                    cov = (centered.T @ centered) / max(float(pts.shape[0] - 1), 1.0)
                    eigvals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 1e-12))[::-1]
                    aspect_ratio = float(np.sqrt(eigvals[0]) / max(float(np.sqrt(eigvals[1])), 1e-8))
                    eccentricity = float(np.sqrt(max(0.0, 1.0 - eigvals[1] / max(eigvals[0], 1e-12))))
                desc = {
                    "shape_area": float(area),
                    "shape_perimeter": float(perimeter),
                    "shape_compactness": float(4.0 * np.pi * area / max(perimeter**2, 1e-12)) if perimeter > 0 else 0.0,
                    "shape_aspect_ratio": aspect_ratio,
                    "shape_eccentricity": eccentricity,
                    "shape_radius_mean": float(radius.mean()) if radius.size else 0.0,
                    "shape_radius_std": float(radius.std()) if radius.size else 0.0,
                }
            source = "observed_point_cloud_fast_large_partition"
        elif region_geometries is not None:
            region = region_geometries[rid]
            if region.mask is not None:
                geom = _sample_uniform_points_in_mask(region.mask, n_points=512, seed=rid, affine=region.affine)
                if geom.shape[0] > 0:
                    desc = _subregion_shape_descriptors(geom)
                    source = "explicit_mask"
                else:
                    desc = _subregion_shape_descriptors(coords_um[members])
            elif region.polygon_components:
                pts = np.vstack([np.asarray(poly, dtype=np.float64) for poly in region.polygon_components if np.asarray(poly).shape[0] >= 3])
                if pts.shape[0] > 0:
                    desc = _subregion_shape_descriptors(pts)
                    source = "explicit_polygon"
                else:
                    desc = _subregion_shape_descriptors(coords_um[members])
            elif region.polygon_vertices is not None and np.asarray(region.polygon_vertices).shape[0] >= 3:
                desc = _subregion_shape_descriptors(np.asarray(region.polygon_vertices, dtype=np.float64))
                source = "explicit_polygon"
            elif bool(getattr(region, "use_observed_geometry", False)):
                desc = _subregion_shape_descriptors(coords_um[members])
                source = "observed_point_cloud"
            else:
                desc = _subregion_shape_descriptors(coords_um[members])
        else:
            desc = _subregion_shape_descriptors(coords_um[members])
        desc["subregion_id"] = int(rid)
        desc["shape_descriptor_source"] = source
        rows.append(desc)
    return pd.DataFrame(rows)


def _shape_leakage_balanced_accuracy(shape_df: pd.DataFrame, labels: np.ndarray, seed: int) -> float | None:
    if shape_df.empty:
        return None
    y = np.asarray(labels, dtype=np.int32)
    sample_idx = _leakage_sample_indices(y, seed=seed)
    if sample_idx.size != y.shape[0]:
        y = y[sample_idx]
        shape_df = shape_df.iloc[sample_idx]
    counts = np.bincount(y)
    counts = counts[counts > 0]
    if counts.size < 2 or counts.min() < 2:
        return None
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        return None
    numeric_cols = [c for c in shape_df.columns if c != "subregion_id" and np.issubdtype(shape_df[c].dtype, np.number)]
    x = shape_df[numeric_cols].to_numpy(dtype=np.float32)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = RandomForestClassifier(n_estimators=_leakage_rf_estimators(), random_state=seed)
    scores = cross_val_score(clf, x, y, cv=cv, scoring="balanced_accuracy")
    return float(np.mean(scores))


def _shape_leakage_spatial_block_accuracy(
    shape_df: pd.DataFrame,
    labels: np.ndarray,
    centers_um: np.ndarray,
    seed: int,
    n_blocks: int = 5,
) -> float | None:
    if shape_df.empty:
        return None
    y = np.asarray(labels, dtype=np.int32)
    unique_labels = np.unique(y)
    if unique_labels.size < 2:
        return None
    centers = np.asarray(centers_um, dtype=np.float32)
    if centers.shape[0] != y.shape[0]:
        return None
    sample_idx = _leakage_sample_indices(y, seed=seed)
    if sample_idx.size != y.shape[0]:
        y = y[sample_idx]
        centers = centers[sample_idx]
        shape_df = shape_df.iloc[sample_idx]
        unique_labels = np.unique(y)
        if unique_labels.size < 2:
            return None
    block_count = min(int(n_blocks), centers.shape[0])
    if block_count < 2:
        return None
    block_km = KMeans(n_clusters=block_count, n_init=20, random_state=seed)
    blocks = block_km.fit_predict(centers)
    numeric_cols = [c for c in shape_df.columns if c != "subregion_id" and np.issubdtype(shape_df[c].dtype, np.number)]
    x = shape_df[numeric_cols].to_numpy(dtype=np.float32)
    clf = RandomForestClassifier(n_estimators=_leakage_rf_estimators(), random_state=seed)
    scores: list[float] = []
    for block in np.unique(blocks):
        test_mask = blocks == block
        train_mask = ~test_mask
        if np.unique(y[train_mask]).size < 2 or np.unique(y[test_mask]).size < 2:
            continue
        clf.fit(x[train_mask], y[train_mask])
        pred = clf.predict(x[test_mask])
        denom = 0.0
        score = 0.0
        for label in unique_labels:
            mask = y[test_mask] == label
            if not np.any(mask):
                continue
            denom += 1.0
            score += float(np.mean(pred[mask] == label))
        if denom > 0:
            scores.append(score / denom)
    if not scores:
        return None
    return float(np.mean(scores))


def _shape_leakage_permutation_baseline(
    shape_df: pd.DataFrame,
    labels: np.ndarray,
    seed: int,
    n_perm: int = 64,
) -> dict[str, float] | None:
    observed = _shape_leakage_balanced_accuracy(shape_df, labels, seed=seed)
    if observed is None:
        return None
    if int(n_perm) <= 0:
        return {
            "observed": float(observed),
            "perm_mean": float("nan"),
            "perm_p95": float("nan"),
            "excess": float("nan"),
            "max_subregions": float(_leakage_max_subregions()),
        }
    rng = np.random.default_rng(seed)
    perm_scores = []
    for _ in range(int(n_perm)):
        score = _shape_leakage_balanced_accuracy(shape_df, rng.permutation(labels), seed=int(rng.integers(1_000_000)))
        if score is not None:
            perm_scores.append(score)
    if not perm_scores:
        return {
            "observed": float(observed),
            "perm_mean": float("nan"),
            "perm_p95": float("nan"),
            "excess": float("nan"),
            "max_subregions": float(_leakage_max_subregions()),
        }
    perm = np.asarray(perm_scores, dtype=np.float64)
    return {
        "observed": float(observed),
        "perm_mean": float(np.mean(perm)),
        "perm_p95": float(np.percentile(perm, 95)),
        "excess": float(observed - np.mean(perm)),
        "max_subregions": float(_leakage_max_subregions()),
    }


def _validate_fit_inputs(
    features: np.ndarray,
    coords_um: np.ndarray,
    n_clusters: int,
    atoms_per_cluster: int,
    radius_um: float,
    stride_um: float,
    basic_niche_size_um: float | None,
    min_cells: int,
    max_subregions: int,
    max_subregion_area_um2: float | None,
    lambda_x: float,
    lambda_y: float,
    geometry_eps: float,
    ot_eps: float,
    rho: float,
    geometry_samples: int,
    compressed_support_size: int,
    align_iters: int,
    max_iter: int,
    tol: float,
    n_init: int,
    min_scale: float,
    max_scale: float,
) -> None:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    if coords_um.ndim != 2 or coords_um.shape[1] != 2:
        raise ValueError("coords_um must be a 2D array with shape (n, 2).")
    if features.shape[0] != coords_um.shape[0]:
        raise ValueError("features and coords_um must have the same number of rows.")
    if not np.all(np.isfinite(features)):
        raise ValueError("features contains NaN or Inf.")
    if not np.all(np.isfinite(coords_um)):
        raise ValueError("coords_um contains NaN or Inf.")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    if atoms_per_cluster < 1:
        raise ValueError("atoms_per_cluster must be at least 1.")
    if radius_um <= 0 or stride_um <= 0:
        raise ValueError("radius_um and stride_um must be positive.")
    if basic_niche_size_um is not None and basic_niche_size_um <= 0:
        raise ValueError("basic_niche_size_um must be positive when set.")
    if min_cells < 1:
        raise ValueError("min_cells must be at least 1.")
    if max_subregions != 0 and max_subregions < 1:
        raise ValueError("max_subregions must be positive or 0.")
    if max_subregion_area_um2 is not None and float(max_subregion_area_um2) <= 0.0:
        raise ValueError("max_subregion_area_um2 must be positive when set.")
    if lambda_x < 0 or lambda_y < 0:
        raise ValueError("lambda_x and lambda_y must be non-negative.")
    if lambda_x == 0 and lambda_y == 0:
        raise ValueError("at least one of lambda_x or lambda_y must be positive.")
    if geometry_eps <= 0 or ot_eps <= 0:
        raise ValueError("geometry_eps and ot_eps must be positive.")
    if rho <= 0:
        raise ValueError("rho must be positive.")
    if geometry_samples < 32:
        raise ValueError("geometry_samples must be at least 32.")
    if compressed_support_size < 2:
        raise ValueError("compressed_support_size must be at least 2.")
    if align_iters < 1 or max_iter < 1 or n_init < 1:
        raise ValueError("align_iters, max_iter, and n_init must be at least 1.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if min_scale <= 0 or max_scale <= 0 or min_scale > max_scale:
        raise ValueError("min_scale and max_scale must be positive and min_scale <= max_scale.")


def make_reference_points_unit_disk(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    n_points = max(int(n_points), 32)
    idx = np.arange(n_points, dtype=np.float64)
    radius = np.sqrt((idx + 0.5) / n_points)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    theta = idx * golden
    q = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    w = np.full(n_points, 1.0 / n_points, dtype=np.float64)
    return q.astype(np.float32), w


def _normalize_coords_basic(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    center = x.mean(axis=0, keepdims=True)
    x0 = x - center
    scale = float(np.sqrt(np.mean(np.sum(x0**2, axis=1))))
    scale = max(scale, 1e-8)
    return x0 / scale, center.astype(np.float64), scale


def _gpu_balanced_sinkhorn_transport(
    g_norm: np.ndarray,
    q: np.ndarray,
    w_geom: np.ndarray,
    w_ref: np.ndarray,
    eps_geom: float,
    compute_device: torch.device,
) -> tuple[np.ndarray, float, bool]:
    """GPU balanced Sinkhorn for the geometry shape normalizer.

    Returns (T, ot_cost, converged).
    """
    from .gpu_ot import sinkhorn_balanced_log_torch

    dtype = torch.float32
    g_t = torch.as_tensor(g_norm, dtype=dtype, device=compute_device)
    q_t = torch.as_tensor(q, dtype=dtype, device=compute_device)
    c = torch.cdist(g_t, q_t, p=2).pow(2)
    positive = c[c > 0]
    scale_cost = float(positive.median().item()) if positive.numel() else 1.0
    c = c / max(scale_cost, 1e-12)
    a_t = torch.as_tensor(w_geom, dtype=dtype, device=compute_device)
    b_t = torch.as_tensor(w_ref, dtype=dtype, device=compute_device)
    t, transport_cost, converged, _ = sinkhorn_balanced_log_torch(
        a_t,
        b_t,
        c,
        eps=max(float(eps_geom), 1e-5),
        num_iter=_sinkhorn_max_iter(),
        tol=_sinkhorn_tol(),
    )
    if not bool(torch.isfinite(t).all().item()):
        # Re-try with a more relaxed regulariser before falling back to numpy.
        for fallback_eps in (max(float(eps_geom), 1e-5) * 4.0, max(float(eps_geom), 1e-5) * 16.0):
            t, transport_cost, converged, _ = sinkhorn_balanced_log_torch(
                a_t,
                b_t,
                c,
                eps=fallback_eps,
                num_iter=_sinkhorn_max_iter(),
                tol=_sinkhorn_tol(),
            )
            if bool(torch.isfinite(t).all().item()):
                break
        else:
            raise RuntimeError("GPU balanced Sinkhorn produced non-finite plan even after regulariser fallback.")
    return (
        t.detach().cpu().numpy().astype(np.float64),
        float(transport_cost.detach().item()),
        bool(converged),
    )


def _fit_degenerate_shape_normalizer(
    geometry_points: np.ndarray,
) -> tuple[ShapeNormalizer, ShapeNormalizerDiagnostics]:
    """Fallback normalizer for subregions with fewer than 3 geometry points.

    There is not enough geometry to estimate a stable OT map in this case, so we
    keep the coordinates in a centered-and-scaled local frame and skip the OT
    interpolation step entirely.
    """

    g = np.asarray(geometry_points, dtype=np.float64)
    g_norm, center, scale = _normalize_coords_basic(g)
    radius = np.sqrt(np.sum(g_norm**2, axis=1))
    normalizer = ShapeNormalizer(center=center, scale=scale, interpolator=None)
    diagnostics = ShapeNormalizerDiagnostics(
        geometry_source="unknown",
        used_fallback=False,
        ot_cost=None,
        sinkhorn_converged=None,
        mapped_radius_p95=float(np.percentile(radius, 95)) if radius.size else 0.0,
        mapped_radius_max=float(radius.max()) if radius.size else 0.0,
        interpolation_residual=0.0,
    )
    return normalizer, diagnostics


def fit_ot_shape_normalizer(
    geometry_points: np.ndarray,
    reference_points: np.ndarray,
    reference_weights: np.ndarray | None = None,
    eps_geom: float = 0.03,
    rbf_smoothing: float = 1e-3,
    rbf_neighbors: int = 64,
    compute_device: torch.device | None = None,
) -> tuple[ShapeNormalizer, ShapeNormalizerDiagnostics]:
    g = np.asarray(geometry_points, dtype=np.float64)
    q = np.asarray(reference_points, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError("geometry_points must have shape (n_points, 2).")
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError("reference_points must have shape (n_points, 2).")
    if g.shape[0] < 1:
        raise ValueError("At least 1 geometry point is required for shape normalization.")
    if not np.all(np.isfinite(g)):
        raise ValueError("geometry_points contains NaN or Inf.")
    if not np.all(np.isfinite(q)):
        raise ValueError("reference_points contains NaN or Inf.")
    centered_geometry = g - g.mean(axis=0, keepdims=True)
    if g.shape[0] < 3 or int(np.linalg.matrix_rank(centered_geometry, tol=1e-8)) < 2:
        return _fit_degenerate_shape_normalizer(g)

    g_norm, center, scale = _normalize_coords_basic(g)
    if reference_weights is None:
        w_ref = np.full(q.shape[0], 1.0 / max(q.shape[0], 1), dtype=np.float64)
    else:
        w_ref = _normalize_hist(reference_weights)
    w_geom = np.full(g_norm.shape[0], 1.0 / max(g_norm.shape[0], 1), dtype=np.float64)

    interpolator: RBFInterpolator | None = None
    ot_cost = None
    sinkhorn_converged = None
    mapped_radius_p95 = None
    mapped_radius_max = None
    interpolation_residual = None
    try:
        use_gpu = compute_device is not None and torch.device(compute_device).type == "cuda"
        if use_gpu:
            t, ot_cost, sinkhorn_converged = _gpu_balanced_sinkhorn_transport(
                g_norm=g_norm,
                q=q,
                w_geom=w_geom,
                w_ref=w_ref,
                eps_geom=eps_geom,
                compute_device=torch.device(compute_device),
            )
            row_mass = np.maximum(t.sum(axis=1, keepdims=True), 1e-12)
            g_mapped = (t @ q) / row_mass
        else:
            c = ot.dist(g_norm, q, metric="sqeuclidean")
            positive = c[c > 0]
            scale_cost = float(np.median(positive)) if positive.size else 1.0
            c = c / max(scale_cost, 1e-12)
            t, log = ot.sinkhorn(
                w_geom,
                w_ref,
                c,
                reg=max(float(eps_geom), 1e-5),
                method="sinkhorn_stabilized",
                numItermax=_cpu_sinkhorn_max_iter(),
                stopThr=_cpu_sinkhorn_tol(),
                warn=False,
                log=True,
            )
            row_mass = np.maximum(t.sum(axis=1, keepdims=True), 1e-12)
            g_mapped = (t @ q) / row_mass
            ot_cost = float(np.sum(t * c))
            err_hist = np.asarray(log.get("err", []), dtype=np.float64)
            sinkhorn_converged = bool(err_hist.size == 0 or err_hist[-1] < _cpu_sinkhorn_tol())
        radius = np.sqrt(np.sum(g_mapped**2, axis=1))
        mapped_radius_p95 = float(np.percentile(radius, 95)) if radius.size else 0.0
        mapped_radius_max = float(radius.max()) if radius.size else 0.0
        interpolator = RBFInterpolator(
            g_norm,
            g_mapped,
            smoothing=rbf_smoothing,
            neighbors=min(int(rbf_neighbors), g_norm.shape[0]),
        )
        pred = np.asarray(interpolator(g_norm), dtype=np.float64)
        interpolation_residual = float(np.sqrt(np.mean(np.sum((pred - g_mapped) ** 2, axis=1))))
    except Exception as exc:
        raise RuntimeError("Shape normalization failed.") from exc
    normalizer = ShapeNormalizer(center=center, scale=scale, interpolator=interpolator)
    diagnostics = ShapeNormalizerDiagnostics(
        geometry_source="unknown",
        used_fallback=False,
        ot_cost=ot_cost,
        sinkhorn_converged=sinkhorn_converged,
        mapped_radius_p95=mapped_radius_p95,
        mapped_radius_max=mapped_radius_max,
        interpolation_residual=interpolation_residual,
    )
    return normalizer, diagnostics
