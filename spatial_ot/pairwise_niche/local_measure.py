from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class LocalMeasureSet:
    tokens: np.ndarray
    weights: np.ndarray
    mask: np.ndarray
    neighbor_indices: np.ndarray
    full_neighbor_counts: np.ndarray
    retained_neighbor_counts: np.ndarray
    metadata: dict[str, object]


def _kernel_weights(distances: np.ndarray, *, radius_um: float, kernel: str) -> np.ndarray:
    dist = np.asarray(distances, dtype=np.float32)
    requested = str(kernel or "gaussian").strip().lower()
    if requested in {"uniform", "binary", "none"}:
        return np.ones_like(dist, dtype=np.float32)
    if requested in {"inverse", "inverse_distance"}:
        return (1.0 / np.maximum(dist, 1e-6)).astype(np.float32)
    if requested in {"gaussian", "rbf"}:
        sigma = max(float(radius_um) / 2.0, 1e-6)
        return np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)
    raise ValueError("graph_kernel must be gaussian, uniform, or inverse_distance.")


def _fit_state_labels(z: np.ndarray, *, n_states: int, seed: int) -> np.ndarray:
    if int(n_states) <= 1 or z.shape[0] <= 1:
        return np.zeros(z.shape[0], dtype=np.int32)
    k = min(int(n_states), int(z.shape[0]))
    model = MiniBatchKMeans(
        n_clusters=k,
        batch_size=min(max(4096, k * 256), max(int(z.shape[0]), 4096)),
        n_init=3,
        random_state=int(seed),
    )
    return np.asarray(model.fit_predict(z), dtype=np.int32)


def _cap_neighbors(
    *,
    local_indices: np.ndarray,
    distances: np.ndarray,
    weights: np.ndarray,
    state_labels: np.ndarray,
    radius_um: float,
    max_neighbors: int,
    radial_shells: int,
    cap_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if local_indices.size <= int(max_neighbors):
        return np.arange(local_indices.size, dtype=np.int64)
    shells = max(int(radial_shells), 1)
    shell_idx = np.floor(
        np.clip(distances / max(float(radius_um), 1e-8), 0.0, 0.999999) * shells
    ).astype(np.int32)
    requested = str(cap_mode or "radial_shell").strip().lower()
    if requested == "radial_shell_state":
        strata = [
            (int(shell), int(state_labels[int(idx)]))
            for shell, idx in zip(shell_idx, local_indices, strict=False)
        ]
    else:
        strata = [(int(shell), 0) for shell in shell_idx]

    selected: list[int] = []
    unique = sorted(set(strata))
    base = max(int(max_neighbors) // max(len(unique), 1), 1)
    for pos, stratum in enumerate(unique):
        remaining_slots = int(max_neighbors) - len(selected)
        if remaining_slots <= 0:
            break
        same = np.asarray(
            [idx for idx, value in enumerate(strata) if value == stratum],
            dtype=np.int64,
        )
        quota = min(
            base + (1 if pos < int(max_neighbors) % max(len(unique), 1) else 0),
            remaining_slots,
        )
        if same.size <= quota:
            selected.extend(int(idx) for idx in same)
            continue
        p = np.asarray(weights[same], dtype=np.float64)
        p = p / p.sum() if np.isfinite(p).all() and p.sum() > 0 else None
        selected.extend(int(idx) for idx in rng.choice(same, size=quota, replace=False, p=p))
    if len(selected) < int(max_neighbors):
        remaining = np.setdiff1d(
            np.arange(local_indices.size),
            np.asarray(selected),
            assume_unique=False,
        )
        need = min(int(max_neighbors) - len(selected), remaining.size)
        if need > 0:
            p = np.asarray(weights[remaining], dtype=np.float64)
            p = p / p.sum() if np.isfinite(p).all() and p.sum() > 0 else None
            selected.extend(int(idx) for idx in rng.choice(remaining, size=need, replace=False, p=p))
    chosen = np.asarray(selected[: int(max_neighbors)], dtype=np.int64)
    return chosen[np.argsort(distances[chosen], kind="stable")]


def build_local_measures(
    *,
    expression_embedding: np.ndarray,
    coords_um: np.ndarray,
    sample_ids: np.ndarray,
    radius_um: float = 50.0,
    max_neighbors: int = 32,
    include_anchor: bool = True,
    graph_kernel: str = "gaussian",
    cap_mode: str = "radial_shell_state",
    cap_state_clusters: int = 16,
    radial_shells: int = 3,
    expression_weight: float = 1.0,
    spatial_weight: float = 0.25,
    distance_weight: float = 0.10,
    seed: int = 1337,
) -> LocalMeasureSet:
    """Build padded cell-centered local measures, sample-isolated by construction."""

    z = np.asarray(expression_embedding, dtype=np.float32)
    xy = np.asarray(coords_um, dtype=np.float32)
    samples = np.asarray(sample_ids, dtype=object).astype(str)
    if z.ndim != 2 or xy.ndim != 2 or xy.shape[1] != 2 or z.shape[0] != xy.shape[0]:
        raise ValueError("expression_embedding and coords_um must align with shape n_cells.")
    n = int(z.shape[0])
    support = max(int(max_neighbors), 1) + (1 if bool(include_anchor) else 0)
    token_dim = int(z.shape[1]) + 3
    tokens = np.zeros((n, support, token_dim), dtype=np.float32)
    weights = np.zeros((n, support), dtype=np.float32)
    mask = np.zeros((n, support), dtype=bool)
    neighbor_indices = np.full((n, support), -1, dtype=np.int64)
    full_counts = np.zeros(n, dtype=np.int32)
    retained_counts = np.zeros(n, dtype=np.int32)

    state_labels = _fit_state_labels(
        z,
        n_states=int(cap_state_clusters) if str(cap_mode).endswith("state") else 1,
        seed=int(seed),
    )
    rng = np.random.default_rng(int(seed))
    radius = float(radius_um)
    sample_order = list(dict.fromkeys(samples.tolist()))
    for sample in sample_order:
        global_idx = np.flatnonzero(samples == sample)
        if global_idx.size == 0:
            continue
        local_xy = xy[global_idx]
        model = NearestNeighbors(radius=radius, algorithm="auto")
        model.fit(local_xy)
        dist_list, ind_list = model.radius_neighbors(
            local_xy,
            return_distance=True,
            sort_results=True,
        )
        for row_local, (dist, ind) in enumerate(zip(dist_list, ind_list, strict=True)):
            anchor = int(global_idx[row_local])
            dist = np.asarray(dist, dtype=np.float32)
            ind = np.asarray(ind, dtype=np.int64)
            keep = np.isfinite(dist) & (ind != int(row_local))
            dist = dist[keep]
            ind = ind[keep]
            global_neighbors = global_idx[ind]
            full_counts[anchor] = int(global_neighbors.size)
            raw_weights = _kernel_weights(dist, radius_um=radius, kernel=graph_kernel)
            if global_neighbors.size > int(max_neighbors):
                chosen = _cap_neighbors(
                    local_indices=global_neighbors,
                    distances=dist,
                    weights=raw_weights,
                    state_labels=state_labels,
                    radius_um=radius,
                    max_neighbors=int(max_neighbors),
                    radial_shells=int(radial_shells),
                    cap_mode=str(cap_mode),
                    rng=rng,
                )
                global_neighbors = global_neighbors[chosen]
                dist = dist[chosen]
                raw_weights = raw_weights[chosen]
            retained_counts[anchor] = int(global_neighbors.size)

            ids: list[int] = []
            dists: list[float] = []
            w_values: list[float] = []
            if bool(include_anchor):
                ids.append(anchor)
                dists.append(0.0)
                w_values.append(1.0)
            ids.extend(int(idx) for idx in global_neighbors)
            dists.extend(float(value) for value in dist)
            w_values.extend(float(value) for value in raw_weights)
            if not ids:
                weights[anchor, 0] = 1.0
                mask[anchor, 0] = True
                continue
            ids_arr = np.asarray(ids[:support], dtype=np.int64)
            dist_arr = np.asarray(dists[:support], dtype=np.float32)
            w_arr = np.asarray(w_values[:support], dtype=np.float32)
            w_arr = w_arr / np.sum(w_arr).clip(min=1e-12)

            rel = (xy[ids_arr] - xy[anchor]) / max(radius, 1e-8)
            radial = dist_arr[:, None] / max(radius, 1e-8)
            token = np.hstack(
                [
                    np.sqrt(max(float(expression_weight), 0.0)) * z[ids_arr],
                    np.sqrt(max(float(spatial_weight), 0.0)) * rel,
                    np.sqrt(max(float(distance_weight), 0.0)) * radial,
                ]
            ).astype(np.float32)
            count = int(ids_arr.size)
            tokens[anchor, :count, :] = token
            weights[anchor, :count] = w_arr
            mask[anchor, :count] = True
            neighbor_indices[anchor, :count] = ids_arr

    metadata = {
        "radius_um": float(radius),
        "max_neighbors": int(max_neighbors),
        "support_size": int(support),
        "include_anchor": bool(include_anchor),
        "graph_kernel": str(graph_kernel),
        "cap_mode": str(cap_mode),
        "cap_state_clusters": int(cap_state_clusters),
        "radial_shells": int(radial_shells),
        "expression_weight": float(expression_weight),
        "spatial_weight": float(spatial_weight),
        "distance_weight": float(distance_weight),
        "n_cells": int(n),
        "mean_full_neighbors": float(np.mean(full_counts)) if n else 0.0,
        "mean_retained_neighbors": float(np.mean(retained_counts)) if n else 0.0,
        "isolated_without_anchor_uses_zero_dummy": True,
        "cross_sample_edges_allowed": False,
        "samples": {
            str(sample): int(np.sum(samples == sample)) for sample in sample_order
        },
    }
    return LocalMeasureSet(
        tokens=tokens,
        weights=weights,
        mask=mask,
        neighbor_indices=neighbor_indices,
        full_neighbor_counts=full_counts,
        retained_neighbor_counts=retained_counts,
        metadata=metadata,
    )
