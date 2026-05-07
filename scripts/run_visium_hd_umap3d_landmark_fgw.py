from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from spatial_ot.pairwise_niche import (
    LocalMeasureSet,
    assign_high_contrast_colors,
)
from spatial_ot.pairwise_niche.cluster import (
    cluster_from_distance,
)
from spatial_ot.pairwise_niche.expression_embedding import (
    fit_expression_embedding,
    save_expression_embedding_state,
)
from spatial_ot.pairwise_niche.distance_matrix import _fit_fgw_structure_scale
from spatial_ot.pairwise_niche.fgw import (
    batched_fused_gromov_wasserstein_cost,
    fused_gromov_wasserstein_block,
)
from spatial_ot.pairwise_niche.local_measure import (
    _cap_neighbors,
    _canonical_fgw_structure_mode,
    _fgw_structure_disconnected_metadata,
    _fit_ground_cost_scales,
    _fit_state_labels,
    _kernel_weights,
    _local_structure_matrix_with_diagnostics,
)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_fgw_structures(
    structures: np.ndarray,
    mask: np.ndarray,
    *,
    normalization: str,
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    scale = _fit_fgw_structure_scale(
        np.asarray(structures, dtype=np.float32),
        np.asarray(mask, dtype=bool),
        normalization=str(normalization),
        n_pairs=int(n_pairs),
        seed=int(seed),
    )
    normalized = np.asarray(structures, dtype=np.float32) / np.float32(
        np.sqrt(max(float(scale), 1e-8))
    )
    return normalized, {
        "fgw_structure_normalization": str(normalization),
        "fgw_structure_sample_pairs": int(n_pairs),
        "fgw_structure_cost_scale": float(scale),
    }


def _log(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with log_path.open("a") as handle:
        handle.write(f"{message}\n")


def _load_per_sample_umap(
    *,
    pooled_h5ad: Path,
    input_dir: Path,
    umap_key: str,
    source_obs_key: str,
) -> tuple[ad.AnnData, np.ndarray]:
    pooled = ad.read_h5ad(pooled_h5ad, backed="r")
    obs = pooled.obs.copy()
    out = np.empty((pooled.n_obs, 3), dtype=np.float32)
    sources = obs[source_obs_key].astype(str).to_numpy()
    for source_name in sorted(np.unique(sources)):
        rows = np.flatnonzero(sources == source_name)
        source_path = input_dir / str(source_name)
        source = ad.read_h5ad(source_path, backed="r")
        try:
            if umap_key not in source.obsm:
                raise KeyError(f"{umap_key} missing from {source_path}")
            target_cell_ids = obs.iloc[rows]["cell_id"].to_numpy()
            source_cell_ids = source.obs["cell_id"].to_numpy()
            if rows.size != source.n_obs or not np.array_equal(target_cell_ids, source_cell_ids):
                raise ValueError(f"Cannot align {source_name} by row order and cell_id.")
            out[rows] = np.asarray(source.obsm[umap_key], dtype=np.float32)
        finally:
            source.file.close()
    pooled.file.close()
    return ad.AnnData(obs=obs), out


def _build_compact_local_measures(
    *,
    expression_embedding: np.ndarray,
    coords_um: np.ndarray,
    sample_ids: np.ndarray,
    radius_um: float,
    max_neighbors: int,
    include_anchor: bool,
    graph_kernel: str,
    cap_mode: str,
    cap_state_clusters: int,
    radial_shells: int,
    isolated_policy: str,
    fgw_structure_mode: str,
    fgw_structure_knn: int,
    fgw_structure_radius_fraction: float,
    expression_weight: float,
    spatial_weight: float,
    distance_weight: float,
    ground_cost_normalization: str,
    ground_cost_sample_pairs: int,
    seed: int,
) -> LocalMeasureSet:
    z = np.asarray(expression_embedding, dtype=np.float32)
    xy = np.asarray(coords_um, dtype=np.float32)
    samples = np.asarray(sample_ids, dtype=object).astype(str)
    n = int(z.shape[0])
    rng = np.random.default_rng(int(seed))
    state_labels = _fit_state_labels(
        z,
        n_states=int(cap_state_clusters) if str(cap_mode).endswith("state") else 1,
        seed=int(seed),
    )
    selected_ids: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n)]
    selected_dist: list[np.ndarray] = [np.empty(0, dtype=np.float32) for _ in range(n)]
    selected_weights: list[np.ndarray] = [np.empty(0, dtype=np.float32) for _ in range(n)]
    requested_structure_mode = str(fgw_structure_mode)
    canonical_structure_mode = _canonical_fgw_structure_mode(requested_structure_mode)
    structure_disconnected = np.zeros(n, dtype=bool)
    full_counts = np.zeros(n, dtype=np.int32)
    retained_counts = np.zeros(n, dtype=np.int32)
    max_active = 1 if bool(include_anchor) else 0
    for sample in list(dict.fromkeys(samples.tolist())):
        global_idx = np.flatnonzero(samples == sample)
        model = NearestNeighbors(radius=float(radius_um), algorithm="auto")
        model.fit(xy[global_idx])
        dist_list, ind_list = model.radius_neighbors(
            xy[global_idx],
            return_distance=True,
            sort_results=True,
        )
        for row_local, (dist, ind) in enumerate(zip(dist_list, ind_list, strict=True)):
            anchor = int(global_idx[row_local])
            dist = np.asarray(dist, dtype=np.float32)
            ind = np.asarray(ind, dtype=np.int64)
            keep = np.isfinite(dist) & (ind != row_local)
            local = global_idx[ind[keep]]
            dist = dist[keep]
            full_counts[anchor] = int(local.size)
            if local.size:
                weights = _kernel_weights(dist, radius_um=float(radius_um), kernel=str(graph_kernel))
                chosen = _cap_neighbors(
                    local_indices=local,
                    distances=dist,
                    weights=weights,
                    state_labels=state_labels,
                    radius_um=float(radius_um),
                    max_neighbors=int(max_neighbors),
                    radial_shells=int(radial_shells),
                    cap_mode=str(cap_mode),
                    rng=rng,
                )
                local = local[chosen]
                dist = dist[chosen]
                weights = weights[chosen]
            elif not include_anchor and isolated_policy == "anchor_fallback":
                local = np.asarray([anchor], dtype=np.int64)
                dist = np.asarray([0.0], dtype=np.float32)
                weights = np.asarray([1.0], dtype=np.float32)
            else:
                local = np.empty(0, dtype=np.int64)
                dist = np.empty(0, dtype=np.float32)
                weights = np.empty(0, dtype=np.float32)
            selected_ids[anchor] = local.astype(np.int64, copy=False)
            selected_dist[anchor] = dist.astype(np.float32, copy=False)
            selected_weights[anchor] = weights.astype(np.float32, copy=False)
            retained_counts[anchor] = int(local.size if not (local.size == 1 and local[0] == anchor and not include_anchor) else 0)
            max_active = max(max_active, int(local.size) + (1 if include_anchor else 0))
    support = max(max_active, 1)
    token_dim = int(z.shape[1]) + 3
    tokens = np.zeros((n, support, token_dim), dtype=np.float32)
    weights = np.zeros((n, support), dtype=np.float32)
    mask = np.zeros((n, support), dtype=bool)
    neighbor_indices = np.full((n, support), -1, dtype=np.int64)
    structure = np.zeros((n, support, support), dtype=np.float32)
    for row in range(n):
        pos = 0
        support_ids: list[int] = []
        support_weights: list[float] = []
        if include_anchor:
            support_ids.append(row)
            support_weights.append(1.0)
        support_ids.extend(int(idx) for idx in selected_ids[row])
        support_weights.extend(float(value) for value in selected_weights[row])
        if not support_ids:
            support_ids = [-1]
            support_weights = [1.0]
        total = float(np.sum(support_weights))
        if not np.isfinite(total) or total <= 0:
            support_weights = [1.0 / len(support_weights)] * len(support_weights)
        else:
            support_weights = [value / total for value in support_weights]
        rel_coords: list[np.ndarray] = []
        for idx, weight in zip(support_ids, support_weights, strict=True):
            if idx >= 0:
                rel = (xy[idx] - xy[row]) / max(float(radius_um), 1e-8)
                radial = np.asarray([np.linalg.norm(rel)], dtype=np.float32)
                tokens[row, pos, : z.shape[1]] = z[idx]
                tokens[row, pos, z.shape[1] : z.shape[1] + 2] = rel
                tokens[row, pos, z.shape[1] + 2] = radial[0]
                neighbor_indices[row, pos] = int(idx)
                rel_coords.append(rel.astype(np.float32, copy=False))
            else:
                rel_coords.append(np.zeros(2, dtype=np.float32))
            weights[row, pos] = float(weight)
            mask[row, pos] = True
            pos += 1
        coords = np.vstack(rel_coords).astype(np.float32, copy=False)
        graph_dist, disconnected, _ = _local_structure_matrix_with_diagnostics(
            coords,
            mode=canonical_structure_mode,
            knn=int(fgw_structure_knn),
            radius_fraction=float(fgw_structure_radius_fraction),
        )
        structure_disconnected[row] = bool(disconnected)
        structure[row, : coords.shape[0], : coords.shape[0]] = graph_dist
    scales = _fit_ground_cost_scales(
        tokens,
        mask,
        expression_dim=int(z.shape[1]),
        normalization=str(ground_cost_normalization),
        n_pairs=int(ground_cost_sample_pairs),
        seed=int(seed),
    )
    tokens[:, :, : z.shape[1]] *= np.sqrt(float(expression_weight) / max(scales["expression"], 1e-8))
    tokens[:, :, z.shape[1] : z.shape[1] + 2] *= np.sqrt(
        float(spatial_weight) / max(scales["relative_xy"], 1e-8)
    )
    tokens[:, :, z.shape[1] + 2 :] *= np.sqrt(
        float(distance_weight) / max(scales["relative_distance"], 1e-8)
    )
    metadata = {
        "radius_um": float(radius_um),
        "max_neighbors": int(max_neighbors),
        "support_size_requested": int(max_neighbors) + (1 if include_anchor else 0),
        "support_size": int(support),
        "max_radius_um": float(radius_um),
        "max_neighbors_included": int(max_neighbors),
        "include_anchor": bool(include_anchor),
        "graph_kernel": str(graph_kernel),
        "cap_mode": str(cap_mode),
        "cap_state_clusters": int(cap_state_clusters),
        "radial_shells": int(radial_shells),
        "isolated_policy": str(isolated_policy),
        "fgw_structure_mode": str(canonical_structure_mode),
        "fgw_structure_requested_mode": str(requested_structure_mode),
        "fgw_structure_knn": int(fgw_structure_knn),
        "fgw_structure_radius_fraction": float(fgw_structure_radius_fraction),
        **_fgw_structure_disconnected_metadata(
            structure_disconnected,
            mode=canonical_structure_mode,
        ),
        "uses_graph_topology_structure": str(canonical_structure_mode)
        != "complete_euclidean",
        "expression_weight": float(expression_weight),
        "spatial_weight": float(spatial_weight),
        "distance_weight": float(distance_weight),
        "ground_cost_normalization": str(ground_cost_normalization),
        "ground_cost_sample_pairs": int(ground_cost_sample_pairs),
        "ground_cost_component_scales": scales,
        "n_cells": int(n),
        "mean_full_neighbors": float(np.mean(full_counts)),
        "mean_retained_neighbors": float(np.mean(retained_counts)),
        "actual_max_active_support": int(mask.sum(axis=1).max()),
        "actual_mean_active_support": float(np.mean(mask.sum(axis=1))),
        "compacted_support_width": int(support),
        "cross_sample_edges_allowed": False,
        "fgw_node_feature_term": "umap3d_only",
    }
    return LocalMeasureSet(
        tokens=tokens,
        weights=weights,
        mask=mask,
        neighbor_indices=neighbor_indices,
        full_neighbor_counts=full_counts,
        retained_neighbor_counts=retained_counts,
        metadata=metadata,
        structure_matrices=structure,
    )


def _select_sample_balanced_landmarks(
    *,
    embedding: np.ndarray,
    sample_ids: np.ndarray,
    n_landmarks: int,
    seed: int,
) -> np.ndarray:
    z = np.asarray(embedding, dtype=np.float32)
    samples = np.asarray(sample_ids, dtype=object).astype(str)
    sample_names = list(dict.fromkeys(samples.tolist()))
    base = int(n_landmarks) // len(sample_names)
    remainder = int(n_landmarks) % len(sample_names)
    selected: list[int] = []
    for pos, sample in enumerate(sample_names):
        target = base + (1 if pos < remainder else 0)
        idx = np.flatnonzero(samples == sample)
        if idx.size <= target:
            selected.extend(int(value) for value in idx)
            continue
        k = min(target, int(idx.size))
        model = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(max(4096, k * 64), int(idx.size)),
            n_init=3,
            random_state=int(seed) + pos,
        )
        model.fit(z[idx])
        centers = np.asarray(model.cluster_centers_, dtype=np.float32)
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(z[idx])
        nearest = nn.kneighbors(centers, return_distance=False).reshape(-1)
        chosen = idx[nearest]
        if np.unique(chosen).size < k:
            chosen = list(dict.fromkeys(int(value) for value in chosen))
            remaining = np.setdiff1d(idx, np.asarray(chosen, dtype=np.int64), assume_unique=False)
            if remaining.size:
                rng = np.random.default_rng(int(seed) + pos)
                fill = rng.choice(remaining, size=min(k - len(chosen), remaining.size), replace=False)
                chosen = np.concatenate([np.asarray(chosen, dtype=np.int64), fill])
            else:
                chosen = np.asarray(chosen, dtype=np.int64)
        selected.extend(int(value) for value in np.asarray(chosen, dtype=np.int64)[:k])
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def _compute_landmark_fgw(
    *,
    features: np.ndarray,
    structures: np.ndarray,
    weights: np.ndarray,
    device: torch.device,
    block_size: int,
    alpha: float,
    epsilon: float,
    sinkhorn_iters: int,
    fgw_iters: int,
    log_path: Path,
) -> np.ndarray:
    n = int(features.shape[0])
    out = np.zeros((n, n), dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, int(block_size)):
        stop = min(start + int(block_size), n)
        fa = torch.as_tensor(features[start:stop], dtype=torch.float32, device=device)
        ca = torch.as_tensor(structures[start:stop], dtype=torch.float32, device=device)
        wa = torch.as_tensor(weights[start:stop], dtype=torch.float32, device=device)
        for j_start in range(start, n, int(block_size)):
            j_stop = min(j_start + int(block_size), n)
            fb = torch.as_tensor(features[j_start:j_stop], dtype=torch.float32, device=device)
            cb = torch.as_tensor(structures[j_start:j_stop], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[j_start:j_stop], dtype=torch.float32, device=device)
            block = fused_gromov_wasserstein_block(
                fa,
                ca,
                wa,
                fb,
                cb,
                wb,
                alpha=float(alpha),
                epsilon=float(epsilon),
                sinkhorn_iters=int(sinkhorn_iters),
                fgw_iters=int(fgw_iters),
            ).detach().cpu().numpy().astype(np.float32)
            if j_start == start:
                block = (0.5 * (block + block.T)).astype(np.float32, copy=False)
                np.fill_diagonal(block, 0.0)
            out[start:stop, j_start:j_stop] = block
            if j_start != start:
                out[j_start:j_stop, start:stop] = block.T
        if start == 0 or (start // int(block_size)) % 5 == 0 or stop == n:
            _log(log_path, f"  landmark rows {start}-{stop}/{n}; elapsed {time.time() - t0:.1f}s")
    np.fill_diagonal(out, 0.0)
    return out


def _assign_all_cells_to_landmarks(
    *,
    features: np.ndarray,
    structures: np.ndarray,
    weights: np.ndarray,
    landmark_features: np.ndarray,
    landmark_structures: np.ndarray,
    landmark_weights: np.ndarray,
    expression_embedding: np.ndarray,
    landmark_indices: np.ndarray,
    candidate_landmarks: int,
    query_batch: int,
    device: torch.device,
    alpha: float,
    epsilon: float,
    sinkhorn_iters: int,
    fgw_iters: int,
    log_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(features.shape[0])
    landmark_z = expression_embedding[np.asarray(landmark_indices, dtype=np.int64)]
    nn = NearestNeighbors(n_neighbors=min(int(candidate_landmarks), landmark_z.shape[0]))
    nn.fit(landmark_z)
    assigned_landmark = np.zeros(n, dtype=np.int32)
    assigned_distance = np.zeros(n, dtype=np.float32)
    assignment_score = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    q = int(query_batch)
    start = 0
    while start < n:
        stop = min(start + q, n)
        cand = nn.kneighbors(
            expression_embedding[start:stop],
            return_distance=False,
        ).astype(np.int64)
        queries = stop - start
        try:
            query_features = np.repeat(features[start:stop], cand.shape[1], axis=0)
            query_structures = np.repeat(structures[start:stop], cand.shape[1], axis=0)
            query_weights = np.repeat(weights[start:stop], cand.shape[1], axis=0)
            flat_cand = cand.reshape(-1)
            target_features = landmark_features[flat_cand]
            target_structures = landmark_structures[flat_cand]
            target_weights = landmark_weights[flat_cand]
            fq = torch.as_tensor(query_features, dtype=torch.float32, device=device)
            ft = torch.as_tensor(target_features, dtype=torch.float32, device=device)
            cq = torch.as_tensor(query_structures, dtype=torch.float32, device=device)
            ct = torch.as_tensor(target_structures, dtype=torch.float32, device=device)
            wq = torch.as_tensor(query_weights, dtype=torch.float32, device=device)
            wt = torch.as_tensor(target_weights, dtype=torch.float32, device=device)
            diff = fq[:, :, None, :] - ft[:, None, :, :]
            feature_cost = diff.pow(2).sum(dim=-1)
            values = batched_fused_gromov_wasserstein_cost(
                feature_cost,
                cq,
                ct,
                wq,
                wt,
                alpha=float(alpha),
                epsilon=float(epsilon),
                sinkhorn_iters=int(sinkhorn_iters),
                fgw_iters=int(fgw_iters),
            ).reshape(queries, cand.shape[1])
            distances = values.detach().cpu().numpy().astype(np.float32)
            order = np.argsort(distances, axis=1)
            best_pos = order[:, 0]
            second_pos = order[:, 1] if cand.shape[1] > 1 else order[:, 0]
            best = distances[np.arange(queries), best_pos]
            second = distances[np.arange(queries), second_pos]
            assigned_landmark[start:stop] = cand[np.arange(queries), best_pos].astype(np.int32)
            assigned_distance[start:stop] = best
            assignment_score[start:stop] = np.clip(
                (second - best) / np.maximum(second, 1e-8),
                0.0,
                1.0,
            ).astype(np.float32)
            start = stop
            if start == stop and start % max(q * 20, 1) == 0:
                pass
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or q <= 16:
                raise
            torch.cuda.empty_cache()
            q = max(q // 2, 16)
            _log(log_path, f"  CUDA OOM; reducing assignment query batch to {q}")
            continue
        if start == n or start % max(q * 64, 1) == 0 or start <= q:
            elapsed = time.time() - t0
            rate = start / max(elapsed, 1e-8)
            _log(
                log_path,
                f"  assigned {start:,}/{n:,} ({100 * start / n:.1f}%), "
                f"rate={rate:.1f} cells/s, q={q}, elapsed={elapsed:.1f}s",
            )
    return assigned_landmark, assigned_distance, assignment_score


def _connected_instances_sparse(
    *,
    coords_um: np.ndarray,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    radius_um: float,
    log_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(coords_um, dtype=np.float32)
    samples = np.asarray(sample_ids, dtype=object).astype(str)
    y = np.asarray(labels, dtype=np.int32)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for sample in list(dict.fromkeys(samples.tolist())):
        global_idx = np.flatnonzero(samples == sample)
        model = NearestNeighbors(radius=float(radius_um), algorithm="auto")
        model.fit(xy[global_idx])
        dist_list, ind_list = model.radius_neighbors(
            xy[global_idx],
            return_distance=True,
            sort_results=True,
        )
        sample_edges = 0
        for row_local, (dist, ind) in enumerate(zip(dist_list, ind_list, strict=True)):
            anchor = int(global_idx[row_local])
            dist = np.asarray(dist, dtype=np.float32)
            ind = np.asarray(ind, dtype=np.int64)
            keep = np.isfinite(dist) & (ind != row_local)
            neighbors = global_idx[ind[keep]]
            if neighbors.size:
                same = neighbors[y[neighbors] == y[anchor]]
                if same.size:
                    rows.append(np.full(same.size, anchor, dtype=np.int64))
                    cols.append(same.astype(np.int64, copy=False))
                    sample_edges += int(same.size)
        _log(log_path, f"  {sample}: same-label instance edges={sample_edges:,}")
    n = int(y.size)
    row_idx = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    col_idx = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    graph = sparse.coo_matrix(
        (np.ones(row_idx.size, dtype=np.float32), (row_idx, col_idx)),
        shape=(n, n),
    ).tocsr()
    graph = graph.maximum(graph.T).tocsr()
    _, components = connected_components(graph, directed=False, return_labels=True)
    components = components.astype(np.int32, copy=False)
    instance_ids = np.empty(n, dtype=np.int32)
    instance_names = np.empty(n, dtype=object)
    next_id = 0
    for label in sorted(np.unique(y)):
        members = np.flatnonzero(y == int(label))
        local_components = pd.factorize(components[members], sort=True)[0].astype(
            np.int32,
            copy=False,
        )
        for local_component in np.unique(local_components):
            rows_for_component = members[local_components == int(local_component)]
            instance_ids[rows_for_component] = next_id
            instance_names[rows_for_component] = f"ON{int(label)}_{int(local_component)}"
            next_id += 1
    return instance_ids, instance_names


def _plot_outputs(
    *,
    adata: ad.AnnData,
    out_dir: Path,
    plot_prefix: str,
) -> None:
    plots = out_dir / "plots"
    spatial_dir = plots / "ot_niche_by_sample"
    umap_dir = plots / "cell_umap3d_by_sample"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    umap_dir.mkdir(parents=True, exist_ok=True)
    colors = dict(adata.uns["pairwise_niche_color_map"])
    categories = list(adata.obs["ot_niche"].cat.categories)
    labels = adata.obs["ot_niche"].astype(str).to_numpy()
    samples = adata.obs["sample_id"].astype(str).to_numpy()
    sample_values = sorted(np.unique(samples))
    umap = np.asarray(adata.obsm["X_cell_umap_3d"], dtype=np.float32)
    x_key = "original_cell_x" if "original_cell_x" in adata.obs else "cell_x"
    y_key = "original_cell_y" if "original_cell_y" in adata.obs else "cell_y"
    point_size = 0.18

    fig = plt.figure(figsize=(8, 7), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    for cat in categories:
        mask = labels == cat
        ax.scatter(
            umap[mask, 0],
            umap[mask, 1],
            umap[mask, 2],
            s=point_size,
            c=colors[cat],
            label=cat,
            linewidths=0,
            alpha=0.65,
            depthshade=False,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.set_title("Full cohort cell 3D UMAP colored by OT niche")
    ax.view_init(elev=22, azim=35)
    ax.legend(markerscale=10, frameon=False, loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plots / f"{plot_prefix}_cell_umap3d_by_niche.png")
    plt.close(fig)

    palette = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(8, 7), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    for idx, sample in enumerate(sample_values):
        mask = samples == sample
        ax.scatter(
            umap[mask, 0],
            umap[mask, 1],
            umap[mask, 2],
            s=point_size,
            c=[palette(idx % 10)],
            label=sample,
            linewidths=0,
            alpha=0.55,
            depthshade=False,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.set_title("Full cohort cell 3D UMAP colored by sample")
    ax.view_init(elev=22, azim=35)
    ax.legend(markerscale=10, frameon=False, loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / f"{plot_prefix}_cell_umap3d_by_sample.png")
    plt.close(fig)

    for sample in sample_values:
        sample_mask = samples == sample
        sample_labels = labels[sample_mask]
        sample_obs = adata.obs.loc[sample_mask]
        sample_umap = umap[sample_mask]
        fig, ax = plt.subplots(figsize=(8, 7), dpi=220)
        for cat in categories:
            local = sample_labels == cat
            if np.any(local):
                ax.scatter(
                    sample_obs.loc[local, x_key],
                    sample_obs.loc[local, y_key],
                    s=0.25,
                    c=colors[cat],
                    label=cat,
                    linewidths=0,
                    alpha=0.75,
                    rasterized=True,
                )
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(f"{sample} spatial OT niches")
        ax.legend(markerscale=8, frameon=False, loc="best", fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(spatial_dir / f"{sample}_ot_niche_{plot_prefix}.png")
        plt.close(fig)

        fig = plt.figure(figsize=(8, 7), dpi=220)
        ax = fig.add_subplot(111, projection="3d")
        for cat in categories:
            local = sample_labels == cat
            if np.any(local):
                ax.scatter(
                    sample_umap[local, 0],
                    sample_umap[local, 1],
                    sample_umap[local, 2],
                    s=0.25,
                    c=colors[cat],
                    label=cat,
                    linewidths=0,
                    alpha=0.7,
                    depthshade=False,
                )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        ax.set_title(f"{sample} cell 3D UMAP colored by OT niche")
        ax.view_init(elev=22, azim=35)
        ax.legend(markerscale=8, frameon=False, loc="best", fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(umap_dir / f"{sample}_cell_umap3d_{plot_prefix}.png")
        plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_visium_hd_umap3d_landmark_fgw.log"
    if log_path.exists():
        log_path.unlink()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    _log(log_path, json.dumps({
        "event": "start",
        "input_h5ad": str(args.input_h5ad),
        "output_dir": str(out_dir),
        "feature_obsm_key": args.umap_key,
        "device": str(device),
        "radius_um": float(args.radius_um),
        "max_neighbors": int(args.max_neighbors),
        "n_landmarks": int(args.n_landmarks),
        "candidate_landmarks": int(args.candidate_landmarks),
    }, indent=2))
    t0 = time.time()
    _log(log_path, "[1/9] Loading pooled obs and per-sample 3D UMAP...")
    adata, umap_raw = _load_per_sample_umap(
        pooled_h5ad=Path(args.input_h5ad),
        input_dir=Path(args.input_dir),
        umap_key=str(args.umap_key),
        source_obs_key=str(args.source_obs_key),
    )
    sample_ids = adata.obs[str(args.sample_obs_key)].astype(str).to_numpy()
    coords = np.column_stack([
        np.asarray(adata.obs[str(args.spatial_x_key)], dtype=np.float32),
        np.asarray(adata.obs[str(args.spatial_y_key)], dtype=np.float32),
    ])
    _log(log_path, f"Input cells={adata.n_obs:,}, raw_umap_shape={umap_raw.shape}")

    _log(log_path, "[2/9] Using standardized 3D UMAP as the cell feature embedding...")
    embedding = fit_expression_embedding(
        umap_raw,
        method="precomputed",
        standardize_precomputed=not bool(args.no_standardize_umap),
        random_state=int(args.seed),
    )
    z = embedding.values.astype(np.float32, copy=False)
    np.save(out_dir / "X_cell_umap_3d_raw.npy", umap_raw)
    np.save(out_dir / "X_cell_umap_3d.npy", z)
    save_expression_embedding_state(embedding.state, out_dir / "expression_embedding_state_umap3d.npz")
    _log(log_path, json.dumps(embedding.metadata, indent=2, default=_json_default))

    _log(log_path, "[3/9] Building compact same-sample local cell graphs...")
    measures = _build_compact_local_measures(
        expression_embedding=z,
        coords_um=coords,
        sample_ids=sample_ids,
        radius_um=float(args.radius_um),
        max_neighbors=int(args.max_neighbors),
        include_anchor=True,
        graph_kernel="gaussian",
        cap_mode="radial_shell_state",
        cap_state_clusters=int(args.cap_state_clusters),
        radial_shells=int(args.radial_shells),
        isolated_policy="anchor_fallback",
        fgw_structure_mode=str(args.fgw_structure_mode),
        fgw_structure_knn=int(args.fgw_structure_knn),
        fgw_structure_radius_fraction=float(args.fgw_structure_radius_fraction),
        expression_weight=float(args.expression_weight),
        spatial_weight=float(args.spatial_weight),
        distance_weight=float(args.distance_weight),
        ground_cost_normalization=str(args.ground_cost_normalization),
        ground_cost_sample_pairs=int(args.ground_cost_sample_pairs),
        seed=int(args.seed),
    )
    _log(log_path, json.dumps(measures.metadata, indent=2, default=_json_default))
    feature_width = z.shape[1]
    features = measures.tokens[:, :, :feature_width].astype(np.float32, copy=False)
    structures, structure_norm = _normalize_fgw_structures(
        np.asarray(measures.structure_matrices, dtype=np.float32),
        measures.mask,
        normalization=str(args.fgw_structure_normalization),
        n_pairs=int(args.fgw_structure_sample_pairs),
        seed=int(args.seed),
    )
    weights = measures.weights.astype(np.float32, copy=False)

    _log(log_path, "[4/9] Selecting sample-balanced UMAP landmarks...")
    landmark_indices = _select_sample_balanced_landmarks(
        embedding=z,
        sample_ids=sample_ids,
        n_landmarks=int(args.n_landmarks),
        seed=int(args.seed),
    )
    np.save(out_dir / "landmark_indices.npy", landmark_indices)
    landmark_counts = {
        str(sample): int(np.sum(sample_ids[landmark_indices] == str(sample)))
        for sample in sorted(set(sample_ids.tolist()))
    }
    _log(log_path, json.dumps({"n_landmarks": int(landmark_indices.size), "landmarks_per_sample": landmark_counts}, indent=2))

    _log(log_path, "[5/9] Computing landmark FGW matrix from UMAP cell graphs...")
    lm_features = features[landmark_indices]
    lm_structures = structures[landmark_indices]
    lm_weights = weights[landmark_indices]
    landmark_distance = _compute_landmark_fgw(
        features=lm_features,
        structures=lm_structures,
        weights=lm_weights,
        device=device,
        block_size=int(args.landmark_block_size),
        alpha=float(args.fgw_alpha),
        epsilon=float(args.fgw_epsilon),
        sinkhorn_iters=int(args.fgw_sinkhorn_iters),
        fgw_iters=int(args.fgw_iters),
        log_path=log_path,
    )
    np.save(out_dir / "landmark_fgw_dissimilarity.npy", landmark_distance)
    _log(log_path, json.dumps({
        "shape": list(landmark_distance.shape),
        "min": float(np.min(landmark_distance)),
        "max": float(np.max(landmark_distance)),
        "symmetric_max_abs_diff": float(np.max(np.abs(landmark_distance - landmark_distance.T))),
    }, indent=2))

    _log(log_path, "[6/9] Selecting landmark cluster count from K=5..30...")
    cluster = cluster_from_distance(
        landmark_distance,
        method="agglomerative",
        candidate_n_clusters=tuple(range(5, 31)),
        model_selection_metrics=(
            "silhouette",
            "pseudo_calinski_harabasz",
            "medoid_davies_bouldin",
            "percentile_dunn",
        ),
    )
    landmark_labels = cluster.labels.astype(np.int32)
    np.save(out_dir / "landmark_labels_model_selected.npy", landmark_labels)
    (out_dir / "landmark_model_selection_k5_30.json").write_text(
        json.dumps(cluster.metadata["model_selection"], indent=2, sort_keys=True, default=_json_default)
    )
    _log(log_path, json.dumps(cluster.metadata["model_selection"], indent=2, default=_json_default)[:5000])

    _log(log_path, f"[7/9] Assigning all cells to nearest of {args.candidate_landmarks} UMAP-candidate landmarks by FGW...")
    assigned_landmark, assigned_distance, assignment_score = _assign_all_cells_to_landmarks(
        features=features,
        structures=structures,
        weights=weights,
        landmark_features=lm_features,
        landmark_structures=lm_structures,
        landmark_weights=lm_weights,
        expression_embedding=z,
        landmark_indices=landmark_indices,
        candidate_landmarks=int(args.candidate_landmarks),
        query_batch=int(args.assignment_query_batch),
        device=device,
        alpha=float(args.fgw_alpha),
        epsilon=float(args.fgw_epsilon),
        sinkhorn_iters=int(args.assignment_sinkhorn_iters),
        fgw_iters=int(args.assignment_fgw_iters),
        log_path=log_path,
    )
    full_labels = landmark_labels[assigned_landmark].astype(np.int32)
    np.save(out_dir / "assignment_landmark_index.npy", assigned_landmark)
    np.save(out_dir / "assignment_fgw_distance.npy", assigned_distance)
    np.save(out_dir / "ot_niche_assignment_score.npy", assignment_score)
    np.save(out_dir / "ot_niche_int.npy", full_labels)

    _log(log_path, "[8/9] Writing H5AD, summary, and plots...")
    categories = [f"ON{idx}" for idx in sorted(np.unique(full_labels))]
    label_names = np.asarray([f"ON{idx}" for idx in full_labels], dtype=object)
    colors = assign_high_contrast_colors(categories)
    # The raw 3D UMAP is already saved as X_cell_umap_3d_raw.npy; keep the H5AD
    # lean and store only the standardized FGW input embedding.
    adata.obsm["X_cell_umap_3d"] = z.astype(np.float32, copy=False)
    adata.obs["ot_niche"] = pd.Categorical(label_names, categories=categories)
    adata.obs["ot_niche_int"] = full_labels
    adata.obs["ot_niche_assignment_score"] = assignment_score
    adata.obs["assignment_landmark_index"] = assigned_landmark
    adata.obs["assignment_fgw_distance"] = assigned_distance
    suffix = f"r{float(args.radius_um):g}".replace(".", "p")
    adata.obs[f"n_neighbors_full_{suffix}"] = measures.full_neighbor_counts
    adata.obs[f"n_neighbors_retained_{suffix}"] = measures.retained_neighbor_counts
    adata.obs[f"neighbor_retention_fraction_{suffix}"] = (
        measures.retained_neighbor_counts.astype(np.float32)
        / np.maximum(measures.full_neighbor_counts.astype(np.float32), 1.0)
    )
    area = np.pi * float(args.radius_um) ** 2
    adata.obs[f"local_density_full_per_um2_{suffix}"] = measures.full_neighbor_counts.astype(np.float32) / max(area, 1e-12)
    adata.obs[f"local_density_retained_per_um2_{suffix}"] = measures.retained_neighbor_counts.astype(np.float32) / max(area, 1e-12)
    adata.obs[f"is_isolated_{suffix}"] = measures.full_neighbor_counts == 0
    instance_ids, instance_names = _connected_instances_sparse(
        coords_um=coords,
        sample_ids=sample_ids,
        radius_um=float(args.radius_um),
        labels=full_labels,
        log_path=log_path,
    )
    adata.obs["ot_niche_instance"] = pd.Categorical(instance_names)
    adata.obs["ot_niche_instance_int"] = instance_ids.astype(np.int32)
    adata.uns["pairwise_niche_color_map"] = colors
    adata.uns["ot_niche_colors"] = [colors[cat] for cat in categories]
    cluster_counts = {
        cat: int(np.sum(label_names == cat))
        for cat in categories
    }
    summary = {
        "active_path": "pairwise-niche-landmark-fgw-full-cohort-umap3d",
        "input_h5ad": str(args.input_h5ad),
        "output_dir": str(out_dir),
        "n_cells": int(adata.n_obs),
        "feature_obsm_key": str(args.umap_key),
        "cell_feature_space": {
            "method": "precomputed_3d_umap",
            "standardized_for_fgw": not bool(args.no_standardize_umap),
            "allow_umap_as_feature": True,
            "uses_spatial_coordinates": False,
            "warning": "UMAP is used as requested for this exploratory run; it is not generally metric-preserving.",
        },
        "sample_counts": {
            str(k): int(v)
            for k, v in adata.obs[str(args.sample_obs_key)].astype(str).value_counts().items()
        },
        "local_graph": measures.metadata,
        "distance_summary": {
            "mode": "landmark_fused_gromov_wasserstein_assignment",
            "uses_graph_topology": bool(
                measures.metadata.get("fgw_structure_mode") != "complete_euclidean"
            ),
            "uses_complete_spatial_structure": bool(
                measures.metadata.get("fgw_structure_mode") == "complete_euclidean"
            ),
            "all_pairs_dense_distance_matrix_materialized": False,
            "n_landmarks": int(landmark_indices.size),
            "candidate_landmarks": int(args.candidate_landmarks),
            "radius_um": float(args.radius_um),
            "max_neighbors": int(args.max_neighbors),
            "fgw_alpha": float(args.fgw_alpha),
            "fgw_epsilon": float(args.fgw_epsilon),
            "fgw_sinkhorn_iters": int(args.fgw_sinkhorn_iters),
            "fgw_iters": int(args.fgw_iters),
            "assignment_sinkhorn_iters": int(args.assignment_sinkhorn_iters),
            "assignment_fgw_iters": int(args.assignment_fgw_iters),
            "node_feature_term": "standardized_3d_umap_only",
            "graph_structure_term": str(measures.metadata.get("fgw_structure_mode")),
            **structure_norm,
            "landmark_distance_shape": list(landmark_distance.shape),
            "landmark_distance_min": float(np.min(landmark_distance)),
            "landmark_distance_max": float(np.max(landmark_distance)),
        },
        "clustering_summary": {
            "method": "agglomerative_landmark_model_selection",
            "n_clusters": int(len(categories)),
            "cluster_counts": cluster_counts,
            "assignment_score_type": "nearest_landmark_precomputed_distance_margin",
            "landmark_model_selection": cluster.metadata["model_selection"],
            "n_niche_instances": int(np.unique(instance_ids).size),
        },
        "niche_colors": colors,
        "outputs": {
            "h5ad": str(out_dir / "cells_pairwise_niche_full_cohort_umap3d_fgw_k100_r50_n100.h5ad"),
            "summary": str(out_dir / "summary.json"),
            "landmark_distance": str(out_dir / "landmark_fgw_dissimilarity.npy"),
            "cell_umap_3d": str(out_dir / "X_cell_umap_3d.npy"),
            "cell_umap_3d_raw": str(out_dir / "X_cell_umap_3d_raw.npy"),
        },
        "runtime_seconds": float(time.time() - t0),
        "seed": int(args.seed),
    }
    adata.uns["pairwise_niche_config"] = {
        "feature_obsm_key": str(args.umap_key),
        "embedding_method": "precomputed",
        "allow_umap_as_feature": True,
        "radius_um": float(args.radius_um),
        "max_neighbors": int(args.max_neighbors),
        "candidate_n_clusters": list(range(5, 31)),
        "model_selection_metrics": [
            "silhouette",
            "pseudo_calinski_harabasz",
            "medoid_davies_bouldin",
            "percentile_dunn",
        ],
    }
    adata.uns["pairwise_niche_clustering_summary"] = summary["clustering_summary"]
    adata.uns["pairwise_niche_distance_summary"] = summary["distance_summary"]
    h5ad_path = out_dir / "cells_pairwise_niche_full_cohort_umap3d_fgw_k100_r50_n100.h5ad"
    adata.write_h5ad(h5ad_path, compression="gzip")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    (out_dir / "ot_niche_colors.json").write_text(json.dumps(colors, indent=2, sort_keys=True))
    _plot_outputs(adata=adata, out_dir=out_dir, plot_prefix="full_cohort_umap3d_fgw_k100_r50_n100")

    _log(log_path, "[9/9] Validating saved output...")
    saved = ad.read_h5ad(h5ad_path, backed="r")
    try:
        _log(log_path, f"Saved H5AD shape: {saved.shape}")
        _log(log_path, f"Cluster counts: {json.dumps(cluster_counts, indent=2)}")
    finally:
        saved.file.close()
    _log(log_path, f"Finished seconds: {time.time() - t0:.1f}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5ad", default="/storage/hackathon_2026/spatial_ot_input/visium_hd_spatial_ot_input_pooled.h5ad")
    parser.add_argument("--input-dir", default="/storage/hackathon_2026/spatial_ot_input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--umap-key", default="X_umap_marker_genes_3d")
    parser.add_argument("--source-obs-key", default="source_h5ad")
    parser.add_argument("--sample-obs-key", default="sample_id")
    parser.add_argument("--spatial-x-key", default="cell_x")
    parser.add_argument("--spatial-y-key", default="cell_y")
    parser.add_argument("--radius-um", type=float, default=50.0)
    parser.add_argument("--max-neighbors", type=int, default=100)
    parser.add_argument("--n-landmarks", type=int, default=1000)
    parser.add_argument("--candidate-landmarks", type=int, default=100)
    parser.add_argument("--cap-state-clusters", type=int, default=16)
    parser.add_argument("--radial-shells", type=int, default=3)
    parser.add_argument("--expression-weight", type=float, default=1.0)
    parser.add_argument("--spatial-weight", type=float, default=0.25)
    parser.add_argument("--distance-weight", type=float, default=0.10)
    parser.add_argument("--ground-cost-normalization", default="sampled_median")
    parser.add_argument("--ground-cost-sample-pairs", type=int, default=10000)
    parser.add_argument("--fgw-alpha", type=float, default=0.25)
    parser.add_argument(
        "--fgw-structure-mode",
        default="local_knn_shortest_path",
        choices=[
            "complete_euclidean",
            "local_knn_shortest_path",
            "radius_graph_shortest_path",
            "binary_edge_distance",
            "adjacency",
        ],
    )
    parser.add_argument("--fgw-structure-knn", type=int, default=6)
    parser.add_argument("--fgw-structure-radius-fraction", type=float, default=0.5)
    parser.add_argument("--fgw-structure-normalization", default="sampled_median")
    parser.add_argument("--fgw-structure-sample-pairs", type=int, default=10000)
    parser.add_argument("--fgw-epsilon", type=float, default=0.05)
    parser.add_argument("--fgw-sinkhorn-iters", type=int, default=8)
    parser.add_argument("--fgw-iters", type=int, default=2)
    parser.add_argument("--assignment-sinkhorn-iters", type=int, default=8)
    parser.add_argument("--assignment-fgw-iters", type=int, default=2)
    parser.add_argument("--landmark-block-size", type=int, default=32)
    parser.add_argument("--assignment-query-batch", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-standardize-umap", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    summary = run(parser.parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
