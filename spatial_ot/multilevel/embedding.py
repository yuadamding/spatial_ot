from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .geometry import _validate_mutually_exclusive_memberships
from .types import MultilevelOTResult


def compute_subregion_embedding(weights: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    weights_arr = np.asarray(weights, dtype=np.float32)
    if weights_arr.ndim != 2:
        raise ValueError("weights must be a 2D array.")
    if weights_arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), "empty"
    if weights_arr.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32), "constant"
    try:
        import umap.umap_ as umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(30, max(3, weights_arr.shape[0] - 1)),
            min_dist=0.2,
            metric="euclidean",
            random_state=seed,
            transform_seed=seed,
        )
        return reducer.fit_transform(weights_arr).astype(np.float32), "UMAP"
    except Exception:
        if min(weights_arr.shape) >= 2:
            pca = PCA(n_components=2, random_state=seed)
            return pca.fit_transform(weights_arr).astype(np.float32), "PCA"
        values = weights_arr[:, 0] if weights_arr.shape[1] else np.zeros(weights_arr.shape[0], dtype=np.float32)
        values = values.astype(np.float32, copy=False)
        values = (values - float(values.mean())) / max(float(values.std()), 1e-6)
        return np.column_stack([values, np.zeros_like(values)]).astype(np.float32), "1D"


def _zscore_columns(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return arr.astype(np.float32, copy=False)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((arr - mean) / std).astype(np.float32, copy=False)


def native_subregion_embedding(result: MultilevelOTResult) -> np.ndarray:
    summaries = np.asarray(result.subregion_measure_summaries, dtype=np.float32)
    if summaries.ndim == 2 and summaries.shape[0] == result.subregion_cluster_labels.shape[0] and summaries.shape[1] > 0:
        return _zscore_columns(summaries)
    weights = np.asarray(result.subregion_atom_weights, dtype=np.float32)
    return _zscore_columns(weights)


def subregion_embedding_compactness(result: MultilevelOTResult) -> dict[str, object]:
    embedding = native_subregion_embedding(result)
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
    valid = labels_arr >= 0
    if not np.all(valid):
        coords = coords[valid]
        labels_arr = labels_arr[valid]
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


def _cell_labels_from_subregions(
    *,
    n_cells: int,
    subregion_members: list[np.ndarray],
    subregion_labels: np.ndarray,
) -> np.ndarray:
    try:
        _validate_mutually_exclusive_memberships(int(n_cells), subregion_members)
    except RuntimeError as exc:
        raise ValueError("Cannot compute cell adjacency diagnostics because subregion memberships overlap or are invalid.") from exc
    labels = np.full(int(n_cells), -1, dtype=np.int32)
    membership_counts = np.zeros(int(n_cells), dtype=np.int32)
    for idx, members in enumerate(subregion_members):
        member_arr = np.asarray(members, dtype=np.int64)
        if member_arr.size == 0:
            continue
        labels[member_arr] = int(subregion_labels[idx])
        np.add.at(membership_counts, member_arr, 1)
    if int(membership_counts.max(initial=0)) > 1:
        raise ValueError("Cannot compute cell adjacency diagnostics because subregion memberships overlap.")
    return labels


def subregion_graph_metrics(
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
    cell_labels = _cell_labels_from_subregions(
        n_cells=int(n_cells),
        subregion_members=result.subregion_members,
        subregion_labels=labels,
    )
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
            "cell_adjacency_same_label_fraction": _cell_adjacency_same_label_fraction(coords_um, cell_labels),
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
            "cell_adjacency_same_label_fraction": _cell_adjacency_same_label_fraction(coords_um, cell_labels),
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
        "cell_adjacency_same_label_fraction": _cell_adjacency_same_label_fraction(coords_um, cell_labels),
    }
