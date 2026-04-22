from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances

from ..config import ExperimentConfig
from .preprocessing import PreparedSpatialOTData


@dataclass
class NicheResult:
    state_centroids: np.ndarray
    state_probs: np.ndarray
    neighborhood_hist: np.ndarray
    covariance_features: np.ndarray
    shell_features: np.ndarray
    prototype_hist: np.ndarray
    prototype_cov: np.ndarray
    prototype_shell: np.ndarray
    niche_probs: np.ndarray
    niche_labels: np.ndarray
    distance_matrix: np.ndarray


def _normalize_mass(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, None)
    x = x + 1e-6
    return x / x.sum()


def _row_softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    x = -x / max(temperature, 1e-4)
    x = x - x.max(axis=1, keepdims=True)
    out = np.exp(x)
    out = out / (out.sum(axis=1, keepdims=True) + 1e-8)
    return out.astype(np.float32)


def _sinkhorn_distance_or_empty(a: np.ndarray, b: np.ndarray, cost: np.ndarray, reg: float, num_iter: int = 400) -> float:
    a_sum = float(np.sum(a))
    b_sum = float(np.sum(b))
    if a_sum <= 1e-8 and b_sum <= 1e-8:
        return 0.0
    if a_sum <= 1e-8 or b_sum <= 1e-8:
        return 1.0
    return float(
        ot.sinkhorn2(
            _normalize_mass(a),
            _normalize_mass(b),
            cost,
            reg=reg,
            numItermax=num_iter,
            warn=False,
        )
    )


def _build_shell_ground(n_shells: int, n_atoms: int) -> np.ndarray:
    shell_idx = np.repeat(np.arange(n_shells), n_atoms)
    atom_idx = np.tile(np.arange(n_atoms), n_shells)
    shell_cost = np.abs(shell_idx[:, None] - shell_idx[None, :]).astype(np.float32)
    atom_cost = (atom_idx[:, None] != atom_idx[None, :]).astype(np.float32)
    cost = shell_cost + 0.25 * atom_cost
    cost = cost / (cost.max() + 1e-6)
    np.fill_diagonal(cost, 0.0)
    return cost


def fit_state_atoms(z: np.ndarray, n_atoms: int, seed: int, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    kmeans = MiniBatchKMeans(n_clusters=n_atoms, random_state=seed, batch_size=min(2048, len(z)))
    kmeans.fit_predict(z)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    distances = pairwise_distances(z, centroids, metric="euclidean").astype(np.float32)
    state_probs = _row_softmax(distances, temperature=temperature)
    return centroids, state_probs


def _covariance_upper(features: np.ndarray, global_mean: np.ndarray) -> np.ndarray:
    centered = features - global_mean
    if centered.shape[0] <= 1:
        cov = np.zeros((centered.shape[1], centered.shape[1]), dtype=np.float32)
    else:
        cov = centered.T @ centered / float(centered.shape[0] - 1)
    tri = np.triu_indices(cov.shape[0])
    return cov[tri].astype(np.float32)


def _build_shell_profiles(data: PreparedSpatialOTData, state_probs: np.ndarray) -> np.ndarray:
    profiles = []
    for edge_index in data.shell_edge_indices:
        if edge_index.size == 0:
            profiles.append(np.zeros((data.n_cells, state_probs.shape[1]), dtype=np.float32))
            continue
        src, dst = edge_index
        out = np.zeros((data.n_cells, state_probs.shape[1]), dtype=np.float32)
        deg = np.zeros((data.n_cells, 1), dtype=np.float32)
        np.add.at(out, src, state_probs[dst])
        np.add.at(deg, src, 1.0)
        deg[deg == 0] = 1.0
        profiles.append(out / deg)
    return np.concatenate(profiles, axis=1) if profiles else np.zeros((data.n_cells, 0), dtype=np.float32)


def build_neighborhood_objects(data: PreparedSpatialOTData, z: np.ndarray, s: np.ndarray, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_atoms = config.model.state_atoms
    centroids, state_probs = fit_state_atoms(
        z,
        n_atoms=n_atoms,
        seed=config.data.seed,
        temperature=config.loss.state_temperature,
    )
    neighbors = data.ot_neighbor_index
    hist = np.zeros((data.n_cells, n_atoms), dtype=np.float32)

    if s.shape[1] > 0:
        program_rank = np.argsort(-np.var(s, axis=0))[: min(config.data.top_program_features, s.shape[1])]
        s_top = s[:, program_rank]
    else:
        s_top = np.zeros((data.n_cells, 0), dtype=np.float32)

    feature_basis = np.concatenate([state_probs, s_top], axis=1)
    global_mean = feature_basis.mean(axis=0, keepdims=True)
    cov_features = []
    for i in range(data.n_cells):
        nbr = neighbors[i]
        hist[i] = state_probs[nbr].mean(axis=0)
        cov_features.append(_covariance_upper(feature_basis[nbr], global_mean))
    cov_features = np.vstack(cov_features).astype(np.float32)
    shell_features = _build_shell_profiles(data, state_probs)
    return centroids, hist, np.concatenate([cov_features, shell_features], axis=1).astype(np.float32), state_probs


def _prototype_means(values: np.ndarray, labels: np.ndarray, n_prototypes: int) -> np.ndarray:
    means = np.zeros((n_prototypes, values.shape[1]), dtype=np.float32)
    for k in range(n_prototypes):
        mask = labels == k
        if np.any(mask):
            means[k] = values[mask].mean(axis=0)
    return means


def fit_niche_prototypes(data: PreparedSpatialOTData, z: np.ndarray, s: np.ndarray, config: ExperimentConfig) -> NicheResult:
    centroids, hist, object_features, state_probs = build_neighborhood_objects(data, z, s, config)
    cov_dim = object_features.shape[1]
    shell_dim = len(data.shell_edge_indices) * centroids.shape[0]
    cov_features = object_features[:, : cov_dim - shell_dim]
    shell_features = object_features[:, cov_dim - shell_dim :] if shell_dim > 0 else np.zeros((data.n_cells, 0), dtype=np.float32)
    concat = np.concatenate([hist, cov_features, shell_features], axis=1)
    kmeans = MiniBatchKMeans(
        n_clusters=config.model.niche_prototypes,
        random_state=config.data.seed,
        batch_size=min(2048, len(concat)),
    )
    labels = kmeans.fit_predict(concat)
    prototype_hist = _prototype_means(hist, labels, config.model.niche_prototypes)
    prototype_cov = _prototype_means(cov_features, labels, config.model.niche_prototypes)
    prototype_shell = _prototype_means(shell_features, labels, config.model.niche_prototypes)

    atom_cost = pairwise_distances(centroids, centroids, metric="euclidean").astype(np.float32)
    atom_cost = atom_cost / (atom_cost.max() + 1e-6)
    shell_ground = _build_shell_ground(len(data.shell_edge_indices), centroids.shape[0]) if shell_features.shape[1] > 0 else None
    distance_matrix = np.zeros((data.n_cells, config.model.niche_prototypes), dtype=np.float32)
    for i in range(data.n_cells):
        for k in range(config.model.niche_prototypes):
            hist_dist = _sinkhorn_distance_or_empty(hist[i], prototype_hist[k], atom_cost, reg=config.loss.ot_temperature)
            cov_dist = np.mean((cov_features[i] - prototype_cov[k]) ** 2)
            if shell_features.shape[1] > 0:
                shell_dist = _sinkhorn_distance_or_empty(shell_features[i], prototype_shell[k], shell_ground, reg=config.loss.ot_temperature)
            else:
                shell_dist = 0.0
            distance_matrix[i, k] = (
                config.loss.ot_hist * hist_dist
                + config.loss.ot_cov * cov_dist
                + config.loss.ot_shell * shell_dist
            )

    niche_probs = _row_softmax(distance_matrix, temperature=config.loss.niche_temperature)
    niche_labels = niche_probs.argmax(axis=1).astype(np.int64)
    return NicheResult(
        state_centroids=centroids,
        state_probs=state_probs,
        neighborhood_hist=hist,
        covariance_features=cov_features,
        shell_features=shell_features,
        prototype_hist=prototype_hist,
        prototype_cov=prototype_cov,
        prototype_shell=prototype_shell,
        niche_probs=niche_probs,
        niche_labels=niche_labels,
        distance_matrix=distance_matrix,
    )
