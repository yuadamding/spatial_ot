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
    structure_matrices: np.ndarray | None = None


def _component_slices(expression_dim: int) -> dict[str, tuple[int, int]]:
    z_stop = int(expression_dim)
    xy_stop = z_stop + 2
    return {
        "expression": (0, z_stop),
        "relative_xy": (z_stop, xy_stop),
        "relative_distance": (xy_stop, xy_stop + 1),
    }


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

    by_stratum: dict[tuple[int, int], np.ndarray] = {}
    for stratum in dict.fromkeys(strata):
        by_stratum[stratum] = np.asarray(
            [idx for idx, value in enumerate(strata) if value == stratum],
            dtype=np.int64,
        )

    unique = list(by_stratum)
    masses = np.asarray(
        [
            np.sum(np.asarray(weights[by_stratum[stratum]], dtype=np.float64))
            for stratum in unique
        ],
        dtype=np.float64,
    )
    masses = np.where(np.isfinite(masses) & (masses > 0), masses, 1.0)
    stratum_p = masses / masses.sum()
    if len(unique) > int(max_neighbors):
        chosen_strata = [
            unique[int(idx)]
            for idx in rng.choice(
                len(unique),
                size=int(max_neighbors),
                replace=False,
                p=stratum_p,
            )
        ]
    else:
        chosen_strata = unique

    selected: list[int] = []
    for stratum in chosen_strata:
        same = by_stratum[stratum]
        p = np.asarray(weights[same], dtype=np.float64)
        p = p / p.sum() if np.isfinite(p).all() and p.sum() > 0 else None
        selected.append(int(rng.choice(same, size=1, replace=False, p=p)[0]))
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


def _pairwise_spatial_distance(values: np.ndarray) -> np.ndarray:
    coords = np.asarray(values, dtype=np.float32)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2).astype(
        np.float32
    )


def _canonical_fgw_structure_mode(mode: str) -> str:
    requested = str(mode or "local_knn_shortest_path").strip().lower()
    if requested == "adjacency":
        return "binary_edge_distance"
    return requested


def _all_pairs_shortest_path(graph: np.ndarray) -> tuple[np.ndarray, bool]:
    path = np.asarray(graph, dtype=np.float32).copy()
    for mid in range(path.shape[0]):
        path = np.minimum(path, path[:, mid, None] + path[None, mid, :])
    disconnected = bool(np.any(~np.isfinite(path)))
    finite = path[np.isfinite(path)]
    fill = float(np.max(finite)) * 2.0 if finite.size else 1.0
    path[~np.isfinite(path)] = max(fill, 1.0)
    np.fill_diagonal(path, 0.0)
    return path.astype(np.float32, copy=False), disconnected


def _local_structure_matrix_with_diagnostics(
    relative_coords: np.ndarray,
    *,
    mode: str,
    knn: int,
    radius_fraction: float,
) -> tuple[np.ndarray, bool, str]:
    rel = np.asarray(relative_coords, dtype=np.float32)
    count = int(rel.shape[0])
    requested = _canonical_fgw_structure_mode(mode)
    if count <= 1:
        return np.zeros((count, count), dtype=np.float32), False, requested
    pairwise = _pairwise_spatial_distance(rel)
    if requested == "complete_euclidean":
        return pairwise, False, requested

    graph = np.full((count, count), np.inf, dtype=np.float32)
    np.fill_diagonal(graph, 0.0)
    if requested == "local_knn_shortest_path":
        k = min(max(int(knn), 1), count - 1)
        for row in range(count):
            order = np.argsort(pairwise[row], kind="stable")
            order = order[order != row][:k]
            graph[row, order] = pairwise[row, order]
        graph = np.minimum(graph, graph.T)
        path, disconnected = _all_pairs_shortest_path(graph)
        return path, disconnected, requested

    if requested in {"radius_graph_shortest_path", "binary_edge_distance"}:
        edge_radius = max(float(radius_fraction), 1e-8)
        edge = (pairwise <= edge_radius) & (pairwise > 0)
        if requested == "binary_edge_distance":
            graph[edge] = 1.0
            graph[~edge & ~np.eye(count, dtype=bool)] = 2.0
            return graph.astype(np.float32, copy=False), False, requested
        graph[edge] = pairwise[edge]
        graph = np.minimum(graph, graph.T)
        path, disconnected = _all_pairs_shortest_path(graph)
        return path, disconnected, requested

    raise ValueError(
        "fgw_structure_mode must be complete_euclidean, local_knn_shortest_path, "
        "radius_graph_shortest_path, binary_edge_distance, or adjacency."
    )


def _local_structure_matrix(
    relative_coords: np.ndarray,
    *,
    mode: str,
    knn: int,
    radius_fraction: float,
) -> np.ndarray:
    matrix, _, _ = _local_structure_matrix_with_diagnostics(
        relative_coords,
        mode=mode,
        knn=knn,
        radius_fraction=radius_fraction,
    )
    return matrix


def _fgw_structure_disconnected_metadata(
    disconnected: np.ndarray,
    *,
    mode: str,
) -> dict[str, object]:
    values = np.asarray(disconnected, dtype=bool)
    fraction = float(np.mean(values)) if values.size else 0.0
    shortest_path = str(mode) in {"local_knn_shortest_path", "radius_graph_shortest_path"}
    return {
        "fgw_structure_disconnected_count": int(np.sum(values)),
        "fgw_structure_disconnected_fraction": fraction,
        "fgw_structure_disconnected_warning": (
            "more_than_20_percent_disconnected_shortest_path_structures"
            if shortest_path and fraction > 0.2
            else None
        ),
        "fgw_structure_inf_fill_policy": "2x_max_finite_path" if shortest_path else "none",
    }


def _median_component_cost(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_pairs: int,
) -> float:
    if values.shape[0] <= 1 or values.shape[1] == 0:
        return 1.0
    count = min(max(int(n_pairs), 1), max(int(values.shape[0]) ** 2, 1))
    left = rng.integers(0, values.shape[0], size=count)
    right = rng.integers(0, values.shape[0], size=count)
    costs = np.sum((values[left] - values[right]) ** 2, axis=1)
    costs = costs[np.isfinite(costs) & (costs > 0)]
    if costs.size == 0:
        return 1.0
    return float(max(np.median(costs), 1e-8))


def _fit_ground_cost_scales(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    expression_dim: int,
    normalization: str,
    n_pairs: int,
    seed: int,
) -> dict[str, float]:
    slices = _component_slices(expression_dim)
    requested = str(normalization or "none").strip().lower()
    if requested == "none":
        return {"expression": 1.0, "relative_xy": 1.0, "relative_distance": 1.0}
    if requested == "dimension":
        return {
            "expression": float(max(expression_dim, 1)),
            "relative_xy": 2.0,
            "relative_distance": 1.0,
        }
    if requested != "sampled_median":
        raise ValueError("ground_cost_normalization must be none, dimension, or sampled_median.")
    active = tokens[np.asarray(mask, dtype=bool)]
    if active.size == 0:
        return {"expression": 1.0, "relative_xy": 1.0, "relative_distance": 1.0}
    rng = np.random.default_rng(int(seed))
    scales = {}
    for name, (start, stop) in slices.items():
        scales[name] = _median_component_cost(
            active[:, start:stop],
            rng=rng,
            n_pairs=int(n_pairs),
        )
    return scales


def build_instance_neighbor_indices(
    *,
    coords_um: np.ndarray,
    sample_ids: np.ndarray,
    radius_um: float,
    max_neighbors: int = 512,
) -> np.ndarray:
    xy = np.asarray(coords_um, dtype=np.float32)
    samples = np.asarray(sample_ids, dtype=object).astype(str)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coords_um must have shape (n_cells, 2).")
    if float(radius_um) <= 0:
        raise ValueError("radius_um must be positive.")
    if int(max_neighbors) <= 0:
        raise ValueError("max_neighbors must be positive.")
    n = int(xy.shape[0])
    width = max(int(max_neighbors), 1)
    out = np.full((n, width), -1, dtype=np.int64)
    radius = float(radius_um)
    for sample in list(dict.fromkeys(samples.tolist())):
        global_idx = np.flatnonzero(samples == sample)
        if global_idx.size == 0:
            continue
        model = NearestNeighbors(radius=radius, algorithm="auto")
        model.fit(xy[global_idx])
        dist_list, ind_list = model.radius_neighbors(
            xy[global_idx],
            return_distance=True,
            sort_results=True,
        )
        for row_local, (dist, ind) in enumerate(zip(dist_list, ind_list, strict=True)):
            anchor = int(global_idx[row_local])
            ind = np.asarray(ind, dtype=np.int64)
            dist = np.asarray(dist, dtype=np.float32)
            keep = np.isfinite(dist) & (ind != int(row_local))
            neighbors = global_idx[ind[keep]][:width]
            out[anchor, : neighbors.size] = neighbors.astype(np.int64, copy=False)
    return out


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
    isolated_policy: str = "anchor_fallback",
    fgw_structure_mode: str = "local_knn_shortest_path",
    fgw_structure_knn: int = 6,
    fgw_structure_radius_fraction: float = 0.5,
    expression_weight: float = 1.0,
    spatial_weight: float = 0.25,
    distance_weight: float = 0.10,
    ground_cost_normalization: str = "sampled_median",
    ground_cost_sample_pairs: int = 10000,
    seed: int = 1337,
) -> LocalMeasureSet:
    """Build padded cell-centered local measures, sample-isolated by construction."""

    z = np.asarray(expression_embedding, dtype=np.float32)
    xy = np.asarray(coords_um, dtype=np.float32)
    samples = np.asarray(sample_ids, dtype=object).astype(str)
    if z.ndim != 2 or xy.ndim != 2 or xy.shape[1] != 2 or z.shape[0] != xy.shape[0]:
        raise ValueError("expression_embedding and coords_um must align with shape n_cells.")
    if float(radius_um) <= 0:
        raise ValueError("radius_um must be positive.")
    if int(max_neighbors) <= 0:
        raise ValueError("max_neighbors must be positive.")
    n = int(z.shape[0])
    support = max(int(max_neighbors), 1) + (1 if bool(include_anchor) else 0)
    token_dim = int(z.shape[1]) + 3
    tokens = np.zeros((n, support, token_dim), dtype=np.float32)
    weights = np.zeros((n, support), dtype=np.float32)
    mask = np.zeros((n, support), dtype=bool)
    neighbor_indices = np.full((n, support), -1, dtype=np.int64)
    full_counts = np.zeros(n, dtype=np.int32)
    retained_counts = np.zeros(n, dtype=np.int32)
    structure = np.zeros((n, support, support), dtype=np.float32)
    structure_disconnected = np.zeros(n, dtype=bool)
    requested_structure_mode = str(fgw_structure_mode)
    canonical_structure_mode = _canonical_fgw_structure_mode(requested_structure_mode)

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
                if str(isolated_policy).strip().lower() == "anchor_fallback":
                    ids = [anchor]
                    dists = [0.0]
                    w_values = [1.0]
                elif str(isolated_policy).strip().lower() == "zero_dummy":
                    weights[anchor, 0] = 1.0
                    mask[anchor, 0] = True
                    continue
                else:
                    raise ValueError("isolated_policy must be zero_dummy or anchor_fallback.")
            ids_arr = np.asarray(ids[:support], dtype=np.int64)
            dist_arr = np.asarray(dists[:support], dtype=np.float32)
            w_arr = np.asarray(w_values[:support], dtype=np.float32)
            w_arr = w_arr / np.sum(w_arr).clip(min=1e-12)

            rel = (xy[ids_arr] - xy[anchor]) / max(radius, 1e-8)
            radial = dist_arr[:, None] / max(radius, 1e-8)
            token = np.hstack(
                [
                    z[ids_arr],
                    rel,
                    radial,
                ]
            ).astype(np.float32)
            count = int(ids_arr.size)
            tokens[anchor, :count, :] = token
            weights[anchor, :count] = w_arr
            mask[anchor, :count] = True
            neighbor_indices[anchor, :count] = ids_arr
            rel_graph = (xy[ids_arr] - xy[anchor]) / max(radius, 1e-8)
            graph_dist, disconnected, _ = _local_structure_matrix_with_diagnostics(
                rel_graph,
                mode=canonical_structure_mode,
                knn=int(fgw_structure_knn),
                radius_fraction=float(fgw_structure_radius_fraction),
            )
            structure_disconnected[anchor] = bool(disconnected)
            structure[anchor, :count, :count] = graph_dist

    slices = _component_slices(z.shape[1])
    scales = _fit_ground_cost_scales(
        tokens,
        mask,
        expression_dim=int(z.shape[1]),
        normalization=str(ground_cost_normalization),
        n_pairs=int(ground_cost_sample_pairs),
        seed=int(seed),
    )
    weights_by_component = {
        "expression": float(expression_weight),
        "relative_xy": float(spatial_weight),
        "relative_distance": float(distance_weight),
    }
    for name, (start, stop) in slices.items():
        factor = np.sqrt(
            max(weights_by_component[name], 0.0) / max(float(scales[name]), 1e-8)
        )
        tokens[:, :, start:stop] *= np.float32(factor)

    metadata = {
        "radius_um": float(radius),
        "max_neighbors": int(max_neighbors),
        "support_size": int(support),
        "max_radius_um": float(radius),
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
        "ground_cost_component_scales": {
            str(key): float(value) for key, value in scales.items()
        },
        "ground_cost_component_slices": {
            str(key): [int(start), int(stop)] for key, (start, stop) in slices.items()
        },
        "n_cells": int(n),
        "mean_full_neighbors": float(np.mean(full_counts)) if n else 0.0,
        "mean_retained_neighbors": float(np.mean(retained_counts)) if n else 0.0,
        "isolated_without_anchor_uses_zero_dummy": str(isolated_policy) == "zero_dummy",
        "cross_sample_edges_allowed": False,
        "samples": {
            str(sample): int(np.sum(samples == sample)) for sample in sample_order
        },
        "seed": int(seed),
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
