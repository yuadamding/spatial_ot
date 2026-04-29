from __future__ import annotations

import numpy as np
import torch

from .numerics import pairwise_sqdist_array
from .transforms import apply_similarity
from .types import SubregionMeasure

DEFAULT_LOCAL_POSTERIOR_RADII: tuple[float, ...] = (0.25, 0.5, 1.0)


def empty_spot_level_latent_charts(
    *,
    n_cells: int,
    atoms_per_cluster: int,
) -> dict[str, np.ndarray]:
    return {
        "spot_latent_cell_indices": np.zeros(0, dtype=np.int32),
        "spot_latent_subregion_ids": np.zeros(0, dtype=np.int32),
        "spot_latent_cluster_labels": np.zeros(0, dtype=np.int32),
        "spot_latent_coords": np.zeros((0, 2), dtype=np.float32),
        "spot_latent_aligned_coords": np.zeros((0, 2), dtype=np.float32),
        "spot_latent_cluster_probs": np.zeros(0, dtype=np.float32),
        "spot_latent_atom_confidence": np.zeros(0, dtype=np.float32),
        "spot_latent_weights": np.zeros(0, dtype=np.float32),
        "spot_latent_atom_posteriors": np.zeros((0, int(atoms_per_cluster)), dtype=np.float32),
        "cell_spot_latent_coords": np.full((int(n_cells), 2), np.nan, dtype=np.float32),
        "cell_spot_latent_cluster_labels": np.full(int(n_cells), -1, dtype=np.int32),
        "cell_spot_latent_weights": np.zeros(int(n_cells), dtype=np.float32),
    }


def weighted_atom_posteriors(
    total_cost: np.ndarray,
    prototype_weights: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    temp = max(float(temperature), 1e-5)
    weights = np.clip(np.asarray(prototype_weights, dtype=np.float32), 1e-8, None)
    logits = -np.asarray(total_cost, dtype=np.float32) / temp + np.log(weights).astype(np.float32)[None, :]
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(logits, -80.0, 80.0)).astype(np.float32)
    return (exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-8)).astype(np.float32)


def _local_posterior_features(
    aligned: np.ndarray,
    posterior: np.ndarray,
    *,
    radii: tuple[float, ...],
) -> np.ndarray:
    n_obs, n_atoms = posterior.shape
    if n_obs == 0 or not radii:
        return np.zeros((n_obs, 0), dtype=np.float32)
    d2 = ((aligned[:, None, :] - aligned[None, :, :]) ** 2).sum(axis=2)
    features = np.zeros((n_obs, n_atoms * len(radii)), dtype=np.float32)
    for ridx, radius in enumerate(radii):
        neighbors = d2 <= float(radius) ** 2
        counts = np.maximum(neighbors.sum(axis=1, keepdims=True), 1)
        local_mean = (neighbors.astype(np.float32) @ posterior) / counts.astype(np.float32)
        start = ridx * n_atoms
        features[:, start : start + n_atoms] = local_mean.astype(np.float32)
    return features


def _cluster_local_pca_chart(
    chart_features: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    n_clusters: int,
) -> np.ndarray:
    latent = np.zeros((chart_features.shape[0], 2), dtype=np.float32)
    for cluster_id in range(int(n_clusters)):
        idx = np.flatnonzero(cluster_labels == cluster_id)
        if idx.size <= 1:
            continue
        x = np.asarray(chart_features[idx], dtype=np.float64)
        x_mean = x.mean(axis=0, keepdims=True)
        x_std = x.std(axis=0, keepdims=True)
        x_std[x_std < 1e-6] = 1.0
        xz = (x - x_mean) / x_std
        cov = (xz.T @ xz) / max(idx.size - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        basis = eigvecs[:, order[:2]]
        for dim in range(basis.shape[1]):
            anchor = int(np.argmax(np.abs(basis[:, dim])))
            if basis[anchor, dim] < 0:
                basis[:, dim] *= -1.0
        z = xz @ basis
        if z.shape[1] == 1:
            z = np.column_stack([z[:, 0], np.zeros(z.shape[0], dtype=np.float64)])
        latent[idx] = z[:, :2].astype(np.float32)
    return latent


def _weighted_standardize(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim != 2 or x_arr.shape[0] == 0:
        return x_arr.astype(np.float32, copy=True)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != x_arr.shape[0] or not np.any(w > 0):
        w = np.ones(x_arr.shape[0], dtype=np.float64)
    w = np.clip(w, 0.0, None)
    w_sum = max(float(w.sum()), 1e-12)
    mean = (w[:, None] * x_arr.astype(np.float64)).sum(axis=0) / w_sum
    centered = x_arr.astype(np.float64) - mean[None, :]
    var = (w[:, None] * centered * centered).sum(axis=0) / w_sum
    std = np.sqrt(np.maximum(var, 1e-8))
    std[std < 1e-4] = 1.0
    return ((x_arr - mean.astype(np.float32)[None, :]) / std.astype(np.float32)[None, :]).astype(np.float32)


def _normalize_local_latent(
    local_latent: np.ndarray,
    labels: np.ndarray,
    *,
    n_clusters: int,
    local_radius: float,
) -> np.ndarray:
    local = np.asarray(local_latent, dtype=np.float32)
    label_arr = np.asarray(labels, dtype=np.int32)
    normalized = np.zeros_like(local, dtype=np.float32)
    for cluster_id in range(int(n_clusters)):
        idx = np.flatnonzero(label_arr == cluster_id)
        if idx.size == 0:
            continue
        values = local[idx].astype(np.float32)
        center = np.nanmedian(values, axis=0).astype(np.float32)
        centered = values - center[None, :]
        radius = np.sqrt(np.sum(centered * centered, axis=1))
        scale = float(np.nanpercentile(radius, 95.0)) if radius.size else 1.0
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        normalized[idx] = (float(local_radius) * centered / scale).astype(np.float32)
    return normalized


def _cluster_centroids(
    x: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float32)
    label_arr = np.asarray(labels, dtype=np.int32)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != x_arr.shape[0] or not np.any(w > 0):
        w = np.ones(x_arr.shape[0], dtype=np.float64)
    w = np.clip(w, 0.0, None)
    global_center = np.average(x_arr.astype(np.float64), axis=0, weights=np.maximum(w, 1e-8))
    centroids = np.tile(global_center[None, :], (int(n_clusters), 1)).astype(np.float32)
    masses = np.zeros(int(n_clusters), dtype=np.float64)
    for cluster_id in range(int(n_clusters)):
        idx = np.flatnonzero(label_arr == cluster_id)
        if idx.size == 0:
            continue
        local_w = np.maximum(w[idx], 1e-8)
        masses[cluster_id] = float(local_w.sum())
        centroids[cluster_id] = np.average(x_arr[idx].astype(np.float64), axis=0, weights=local_w).astype(np.float32)
    return centroids, masses.astype(np.float32)


def _weighted_pca_scores(x: np.ndarray, weights: np.ndarray, *, n_components: int = 2) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 2 or x_arr.shape[0] == 0:
        return np.zeros((x_arr.shape[0], int(n_components)), dtype=np.float32)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != x_arr.shape[0] or not np.any(w > 0):
        w = np.ones(x_arr.shape[0], dtype=np.float64)
    w = np.maximum(w, 1e-8)
    mean = np.average(x_arr, axis=0, weights=w)
    centered = x_arr - mean[None, :]
    cov = (centered.T * w[None, :]) @ centered / max(float(w.sum()), 1e-12)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[: max(1, int(n_components))]]
    scores = centered @ basis
    if scores.shape[1] < int(n_components):
        scores = np.column_stack([scores, np.zeros((scores.shape[0], int(n_components) - scores.shape[1]))])
    for dim in range(int(n_components)):
        anchor = int(np.argmax(np.abs(scores[:, dim]))) if scores.shape[0] else 0
        if scores.shape[0] and scores[anchor, dim] < 0:
            scores[:, dim] *= -1.0
    return scores[:, : int(n_components)].astype(np.float32)


def _weighted_fisher_scores(
    x: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    n_components: int = 2,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int32)
    if x_arr.ndim != 2 or x_arr.shape[0] == 0 or label_arr.shape[0] != x_arr.shape[0]:
        return np.zeros((x_arr.shape[0], int(n_components)), dtype=np.float32)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != x_arr.shape[0] or not np.any(w > 0):
        w = np.ones(x_arr.shape[0], dtype=np.float64)
    w = np.maximum(w, 1e-8)
    valid_labels = np.unique(label_arr[label_arr >= 0])
    if valid_labels.size < 2:
        return _weighted_pca_scores(x_arr, w, n_components=int(n_components))

    global_mean = np.average(x_arr, axis=0, weights=w)
    dim = int(x_arr.shape[1])
    sw = np.zeros((dim, dim), dtype=np.float64)
    sb = np.zeros((dim, dim), dtype=np.float64)
    total_mass = 0.0
    for cluster_id in valid_labels.tolist():
        idx = np.flatnonzero(label_arr == int(cluster_id))
        if idx.size == 0:
            continue
        wk = w[idx]
        mass = float(wk.sum())
        if mass <= 0.0:
            continue
        mean_k = np.average(x_arr[idx], axis=0, weights=wk)
        centered = x_arr[idx] - mean_k[None, :]
        sw += (centered.T * wk[None, :]) @ centered
        diff = (mean_k - global_mean)[:, None]
        sb += mass * (diff @ diff.T)
        total_mass += mass
    if total_mass <= 0.0:
        return _weighted_pca_scores(x_arr, w, n_components=int(n_components))
    sw /= total_mass
    sb /= total_mass
    ridge = max(float(np.trace(sw)) / max(dim, 1), 1.0) * 1e-3
    sw = sw + ridge * np.eye(dim, dtype=np.float64)
    try:
        eigvals_w, eigvecs_w = np.linalg.eigh(sw)
        keep = eigvals_w > max(float(np.max(eigvals_w)) * 1e-8, 1e-10)
        if not np.any(keep):
            return _weighted_pca_scores(x_arr, w, n_components=int(n_components))
        whitening = eigvecs_w[:, keep] @ np.diag(1.0 / np.sqrt(eigvals_w[keep])) @ eigvecs_w[:, keep].T
        fisher = whitening @ sb @ whitening
        eigvals_f, eigvecs_f = np.linalg.eigh((fisher + fisher.T) * 0.5)
        order = np.argsort(eigvals_f)[::-1]
        basis = whitening @ eigvecs_f[:, order[: max(1, int(n_components))]]
        scores = (x_arr - global_mean[None, :]) @ basis
    except np.linalg.LinAlgError:
        return _weighted_pca_scores(x_arr, w, n_components=int(n_components))

    if scores.shape[1] < int(n_components):
        scores = np.column_stack([scores, np.zeros((scores.shape[0], int(n_components) - scores.shape[1]))])
    if not np.all(np.isfinite(scores)) or float(np.nanmax(np.abs(scores))) < 1e-8:
        return _weighted_pca_scores(x_arr, w, n_components=int(n_components))
    for dim_idx in range(int(n_components)):
        anchor = int(np.argmax(np.abs(scores[:, dim_idx]))) if scores.shape[0] else 0
        if scores.shape[0] and scores[anchor, dim_idx] < 0:
            scores[:, dim_idx] *= -1.0
    return scores[:, : int(n_components)].astype(np.float32)


def _global_discriminative_latent_chart(
    chart_features: np.ndarray,
    cluster_labels: np.ndarray,
    weights: np.ndarray,
    local_latent: np.ndarray,
    *,
    n_clusters: int,
    local_radius: float = 0.85,
) -> np.ndarray:
    label_arr = np.asarray(cluster_labels, dtype=np.int32)
    if chart_features.shape[0] == 0 or int(n_clusters) <= 0:
        return np.zeros((chart_features.shape[0], 2), dtype=np.float32)
    x = _weighted_standardize(chart_features, weights)
    local = _normalize_local_latent(
        local_latent,
        label_arr,
        n_clusters=int(n_clusters),
        local_radius=float(local_radius),
    )
    valid = (label_arr >= 0) & (label_arr < int(n_clusters))
    if not np.any(valid):
        return local.astype(np.float32)
    global_scores = np.zeros((label_arr.shape[0], 2), dtype=np.float32)
    global_scores[valid] = _weighted_fisher_scores(
        x[valid],
        label_arr[valid],
        np.asarray(weights, dtype=np.float32)[valid],
        n_components=2,
    )
    anchors, masses = _cluster_centroids(
        global_scores[valid],
        label_arr[valid],
        np.asarray(weights, dtype=np.float32)[valid],
        n_clusters=int(n_clusters),
    )
    del masses
    final = np.full((label_arr.shape[0], 2), np.nan, dtype=np.float32)
    final[valid] = anchors[label_arr[valid]] + local[valid]
    final[~np.all(np.isfinite(final), axis=1)] = 0.0
    return final.astype(np.float32)


def spot_latent_separation_diagnostics(
    latent_coords: np.ndarray,
    cluster_labels: np.ndarray,
    weights: np.ndarray | None = None,
    subregion_ids: np.ndarray | None = None,
) -> dict[str, object]:
    latent = np.asarray(latent_coords, dtype=np.float32)
    labels = np.asarray(cluster_labels, dtype=np.int32)
    if latent.ndim != 2 or latent.shape[1] != 2 or labels.ndim != 1 or labels.shape[0] != latent.shape[0]:
        return {
            "n_occurrences": int(latent.shape[0]) if latent.ndim >= 1 else 0,
            "n_clusters": 0,
            "n_present_clusters": 0,
            "mean_within_cluster_radius": None,
            "median_between_cluster_center_distance": None,
            "min_between_cluster_center_distance": None,
            "separation_ratio_median_between_over_mean_within": None,
            "minimum_between_cluster_distance_forced": False,
        }
    if weights is None:
        w = np.ones(labels.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.shape[0] != labels.shape[0] or not np.any(w > 0):
            w = np.ones(labels.shape[0], dtype=np.float64)
    w = np.maximum(w, 1e-8)
    valid = (labels >= 0) & np.all(np.isfinite(latent), axis=1)
    present_labels = np.unique(labels[valid])
    n_clusters = int(labels[valid].max()) + 1 if np.any(valid) else 0
    cluster_occurrence_counts = [
        int(np.sum(labels[valid] == cluster_id))
        for cluster_id in range(n_clusters)
    ]
    centers = []
    within = []
    for cluster_id in range(n_clusters):
        idx = np.flatnonzero(valid & (labels == cluster_id))
        if idx.size == 0:
            centers.append(np.full(2, np.nan, dtype=np.float32))
            continue
        local_w = w[idx]
        center = np.average(latent[idx].astype(np.float64), axis=0, weights=local_w)
        centers.append(center.astype(np.float32))
        radius = np.sqrt(np.sum((latent[idx].astype(np.float64) - center[None, :]) ** 2, axis=1))
        within.append(float(np.average(radius, weights=local_w)))
    center_arr = np.vstack(centers).astype(np.float32) if centers else np.zeros((0, 2), dtype=np.float32)
    finite_centers = np.all(np.isfinite(center_arr), axis=1)
    if int(np.sum(finite_centers)) >= 2:
        distances = np.sqrt(
            np.maximum(
                ((center_arr[finite_centers, None, :] - center_arr[finite_centers][None, :, :]) ** 2).sum(axis=2),
                0.0,
            )
        )
        between = distances[distances > 1e-8]
    else:
        between = np.zeros(0, dtype=np.float32)
    mean_within = float(np.mean(within)) if within else None
    out: dict[str, object] = {
        "n_occurrences": int(latent.shape[0]),
        "n_clusters": int(n_clusters),
        "n_present_clusters": int(present_labels.size),
        "cluster_occurrence_counts": cluster_occurrence_counts,
        "mean_within_cluster_radius": mean_within,
        "median_between_cluster_center_distance": float(np.median(between)) if between.size else None,
        "min_between_cluster_center_distance": float(np.min(between)) if between.size else None,
        "separation_ratio_median_between_over_mean_within": (
            float(np.median(between) / max(float(mean_within), 1e-8))
            if between.size and mean_within is not None
            else None
        ),
        "minimum_between_cluster_distance_forced": False,
    }
    if subregion_ids is not None:
        subregion_arr = np.asarray(subregion_ids)
        out["n_subregions"] = int(np.unique(subregion_arr).size) if subregion_arr.shape[0] == labels.shape[0] else None
    return out


def compute_spot_level_latent_charts(
    *,
    features: np.ndarray,
    coords_um: np.ndarray,
    measures: list[SubregionMeasure],
    subregion_labels: np.ndarray,
    subregion_cluster_probs: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    prototype_weights: np.ndarray,
    assigned_transforms: list[dict[str, np.ndarray | float]],
    lambda_x: float,
    lambda_y: float,
    cost_scale_x: float,
    cost_scale_y: float,
    assignment_temperature: float,
    local_posterior_radii: tuple[float, ...] = DEFAULT_LOCAL_POSTERIOR_RADII,
    compute_device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    compute_device = compute_device or torch.device("cpu")
    features = np.asarray(features, dtype=np.float32)
    coords_um = np.asarray(coords_um, dtype=np.float32)
    subregion_labels = np.asarray(subregion_labels, dtype=np.int32)
    subregion_cluster_probs = np.asarray(subregion_cluster_probs, dtype=np.float32)
    atom_coords = np.asarray(atom_coords, dtype=np.float32)
    atom_features = np.asarray(atom_features, dtype=np.float32)
    prototype_weights = np.asarray(prototype_weights, dtype=np.float32)

    n_cells = int(features.shape[0])
    n_clusters = int(atom_coords.shape[0])
    atoms_per_cluster = int(atom_coords.shape[1])
    n_occurrences = int(sum(np.asarray(measure.members).size for measure in measures))
    if n_occurrences == 0:
        return empty_spot_level_latent_charts(n_cells=n_cells, atoms_per_cluster=atoms_per_cluster)

    cell_indices = np.zeros(n_occurrences, dtype=np.int32)
    subregion_ids = np.zeros(n_occurrences, dtype=np.int32)
    cluster_labels = np.zeros(n_occurrences, dtype=np.int32)
    aligned_coords = np.zeros((n_occurrences, 2), dtype=np.float32)
    cluster_probs = np.zeros(n_occurrences, dtype=np.float32)
    atom_confidence = np.zeros(n_occurrences, dtype=np.float32)
    weights = np.zeros(n_occurrences, dtype=np.float32)
    atom_posteriors = np.zeros((n_occurrences, atoms_per_cluster), dtype=np.float32)
    chart_dim = 2 + atoms_per_cluster * (1 + len(local_posterior_radii))
    chart_features = np.zeros((n_occurrences, chart_dim), dtype=np.float32)

    offset = 0
    for r, measure in enumerate(measures):
        members = np.asarray(measure.members, dtype=np.int32)
        n_members = int(members.size)
        if n_members == 0:
            continue
        k = int(subregion_labels[r])
        canonical = measure.normalizer.transform(coords_um[members]).astype(np.float32)
        aligned = apply_similarity(canonical, assigned_transforms[r]).astype(np.float32)
        cx = pairwise_sqdist_array(aligned, atom_coords[k], device=compute_device) / max(float(cost_scale_x), 1e-5)
        cy = pairwise_sqdist_array(features[members], atom_features[k], device=compute_device) / max(float(cost_scale_y), 1e-5)
        total_cost = float(lambda_x) * cx + float(lambda_y) * cy
        posterior = weighted_atom_posteriors(
            total_cost,
            prototype_weights[k],
            temperature=assignment_temperature,
        )
        local_features = _local_posterior_features(
            aligned,
            posterior,
            radii=local_posterior_radii,
        )
        assigned_cluster_prob = np.full(n_members, float(subregion_cluster_probs[r, k]), dtype=np.float32)
        confidence = posterior.max(axis=1).astype(np.float32)
        occurrence_weight = (assigned_cluster_prob * confidence).astype(np.float32)

        stop = offset + n_members
        cell_indices[offset:stop] = members
        subregion_ids[offset:stop] = int(r)
        cluster_labels[offset:stop] = int(k)
        aligned_coords[offset:stop] = aligned
        cluster_probs[offset:stop] = assigned_cluster_prob
        atom_confidence[offset:stop] = confidence
        weights[offset:stop] = occurrence_weight
        atom_posteriors[offset:stop] = posterior
        chart_features[offset:stop, :2] = aligned
        chart_features[offset:stop, 2 : 2 + atoms_per_cluster] = posterior
        if local_features.shape[1] > 0:
            chart_features[offset:stop, 2 + atoms_per_cluster :] = local_features
        offset = stop

    if offset != n_occurrences:
        cell_indices = cell_indices[:offset]
        subregion_ids = subregion_ids[:offset]
        cluster_labels = cluster_labels[:offset]
        aligned_coords = aligned_coords[:offset]
        cluster_probs = cluster_probs[:offset]
        atom_confidence = atom_confidence[:offset]
        weights = weights[:offset]
        atom_posteriors = atom_posteriors[:offset]
        chart_features = chart_features[:offset]

    local_latent_coords = _cluster_local_pca_chart(
        chart_features,
        cluster_labels,
        n_clusters=n_clusters,
    )
    latent_coords = _global_discriminative_latent_chart(
        chart_features,
        cluster_labels,
        np.maximum(weights, 1e-6),
        local_latent_coords,
        n_clusters=n_clusters,
    )

    cell_latent_num = np.zeros((n_cells, n_clusters, 2), dtype=np.float32)
    cell_latent_den = np.zeros((n_cells, n_clusters), dtype=np.float32)
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if not np.any(mask):
            continue
        members = cell_indices[mask]
        occurrence_weight = weights[mask]
        weighted_latent = occurrence_weight[:, None] * latent_coords[mask]
        np.add.at(cell_latent_num[:, cluster_id, 0], members, weighted_latent[:, 0])
        np.add.at(cell_latent_num[:, cluster_id, 1], members, weighted_latent[:, 1])
        np.add.at(cell_latent_den[:, cluster_id], members, occurrence_weight)

    cell_spot_labels = cell_latent_den.argmax(axis=1).astype(np.int32)
    row_idx = np.arange(n_cells, dtype=np.int64)
    cell_spot_weights = cell_latent_den[row_idx, cell_spot_labels.astype(np.int64)].astype(np.float32)
    cell_spot_coords = np.full((n_cells, 2), np.nan, dtype=np.float32)
    covered = cell_spot_weights > 0
    if np.any(covered):
        covered_idx = np.flatnonzero(covered)
        covered_labels = cell_spot_labels[covered_idx].astype(np.int64)
        cell_spot_coords[covered_idx] = (
            cell_latent_num[covered_idx, covered_labels]
            / np.maximum(cell_spot_weights[covered_idx, None], 1e-8)
        ).astype(np.float32)
    cell_spot_labels[~covered] = -1

    return {
        "spot_latent_cell_indices": cell_indices.astype(np.int32),
        "spot_latent_subregion_ids": subregion_ids.astype(np.int32),
        "spot_latent_cluster_labels": cluster_labels.astype(np.int32),
        "spot_latent_coords": latent_coords.astype(np.float32),
        "spot_latent_aligned_coords": aligned_coords.astype(np.float32),
        "spot_latent_cluster_probs": cluster_probs.astype(np.float32),
        "spot_latent_atom_confidence": atom_confidence.astype(np.float32),
        "spot_latent_weights": weights.astype(np.float32),
        "spot_latent_atom_posteriors": atom_posteriors.astype(np.float32),
        "cell_spot_latent_coords": cell_spot_coords.astype(np.float32),
        "cell_spot_latent_cluster_labels": cell_spot_labels.astype(np.int32),
        "cell_spot_latent_weights": cell_spot_weights.astype(np.float32),
    }
