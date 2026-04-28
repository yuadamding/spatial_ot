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

    latent_coords = _cluster_local_pca_chart(
        chart_features,
        cluster_labels,
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
