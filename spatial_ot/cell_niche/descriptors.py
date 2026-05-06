from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from .graph import NeighborhoodGraph


DEFAULT_BLOCK_WEIGHTS = {
    "self": 0.25,
    "composition": 0.25,
    "diversity": 0.25,
    "moments": 0.15,
    "radial": 0.15,
    "pair": 0.15,
    "covariance": 0.15,
    "gradient": 0.10,
}


@dataclass(frozen=True)
class DescriptorResult:
    raw: np.ndarray
    standardized: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingResult:
    values: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class _RawBlock:
    name: str
    family: str
    values: np.ndarray


def _normalize_blocks(blocks: list[str] | tuple[str, ...] | str | None) -> tuple[str, ...]:
    if blocks is None:
        return (
            "composition",
            "diversity",
            "moments",
            "radial",
            "pair",
            "covariance",
            "gradient",
        )
    if isinstance(blocks, str):
        items = [part.strip().lower() for part in blocks.split(",") if part.strip()]
    else:
        items = [str(part).strip().lower() for part in blocks if str(part).strip()]
    aliases = {
        "heterogeneity": "diversity",
        "local_composition": "composition",
        "neighbor_moments": "moments",
        "radial_shells": "radial",
        "cooccurrence": "pair",
        "covet": "covariance",
        "texture": "gradient",
    }
    normalized = tuple(dict.fromkeys(aliases.get(item, item) for item in items))
    allowed = {
        "composition",
        "diversity",
        "moments",
        "radial",
        "pair",
        "covariance",
        "gradient",
    }
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise ValueError(f"Unknown descriptor block(s): {', '.join(unknown)}")
    return normalized


def _row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    csr = matrix.tocsr(copy=True).astype(np.float32)
    row_sums = np.asarray(csr.sum(axis=1)).reshape(-1).astype(np.float32)
    scale = np.zeros_like(row_sums, dtype=np.float32)
    nonzero = row_sums > 1e-12
    scale[nonzero] = 1.0 / row_sums[nonzero]
    csr.data *= np.repeat(scale, np.diff(csr.indptr))
    return csr


def _composition_block(
    graph: NeighborhoodGraph,
    posteriors: np.ndarray,
) -> np.ndarray:
    weights = _row_normalize(graph.connectivities)
    return np.asarray(weights @ np.asarray(posteriors, dtype=np.float32), dtype=np.float32)


def _diversity_block(composition: np.ndarray) -> np.ndarray:
    p = np.asarray(composition, dtype=np.float64)
    total = np.sum(p, axis=1, keepdims=True)
    valid = total[:, 0] > 1e-12
    p_norm = np.zeros_like(p, dtype=np.float64)
    p_norm[valid] = p[valid] / total[valid]
    k = int(p.shape[1])
    entropy = -np.sum(p_norm * np.log(np.maximum(p_norm, 1e-12)), axis=1)
    entropy = entropy / max(float(np.log(max(k, 2))), 1e-8)
    simpson = 1.0 - np.sum(p_norm * p_norm, axis=1)
    effective_states = np.exp(
        -np.sum(p_norm * np.log(np.maximum(p_norm, 1e-12)), axis=1)
    ) / max(k, 1)
    dominance = np.max(p_norm, axis=1) if k else np.zeros(p.shape[0], dtype=np.float64)
    rare_mass = np.sum(p_norm * (p_norm < (1.0 / max(k, 1)) * 0.25), axis=1)
    out = np.vstack([entropy, simpson, effective_states, dominance, rare_mass]).T
    out[~valid] = 0.0
    return out.astype(np.float32)


def _moments_block(graph: NeighborhoodGraph, features: np.ndarray) -> np.ndarray:
    weights = _row_normalize(graph.connectivities)
    z = np.asarray(features, dtype=np.float32)
    mean = np.asarray(weights @ z, dtype=np.float32)
    second = np.asarray(weights @ (z * z), dtype=np.float32)
    var = np.maximum(second - mean * mean, 0.0)
    std = np.sqrt(var).astype(np.float32)
    local_var = np.mean(var, axis=1, keepdims=True).astype(np.float32)
    return np.hstack([mean, std, local_var]).astype(np.float32)


def _row_distance_data(graph: NeighborhoodGraph) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    conn = graph.connectivities.tocsr()
    dist = graph.distances.tocsr()
    if not (
        np.array_equal(conn.indptr, dist.indptr)
        and np.array_equal(conn.indices, dist.indices)
    ):
        dist = dist.maximum(conn.multiply(0.0)).tocsr()
    return conn, dist


def _radial_shell_composition(
    graph: NeighborhoodGraph,
    posteriors: np.ndarray,
    *,
    n_shells: int,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(posteriors, dtype=np.float32)
    n_cells, n_states = int(q.shape[0]), int(q.shape[1])
    shells = max(int(n_shells), 1)
    out = np.zeros((n_cells, shells, n_states), dtype=np.float32)
    mass = np.zeros((n_cells, shells), dtype=np.float32)
    conn, dist = _row_distance_data(graph)
    for row in range(n_cells):
        start, stop = int(conn.indptr[row]), int(conn.indptr[row + 1])
        if start == stop:
            continue
        cols = conn.indices[start:stop]
        weights = np.asarray(conn.data[start:stop], dtype=np.float32)
        distances = np.asarray(dist.data[start:stop], dtype=np.float32)
        denom = (
            float(graph.radius_um)
            if graph.radius_um is not None and float(graph.radius_um) > 0.0
            else float(np.max(distances))
            if distances.size
            else 1.0
        )
        shell_idx = np.floor(np.clip(distances / max(denom, 1e-8), 0.0, 0.999999) * shells)
        shell_idx = np.asarray(shell_idx, dtype=np.int32)
        for shell in range(shells):
            mask = shell_idx == shell
            if not np.any(mask):
                continue
            w = weights[mask].astype(np.float64)
            m = float(np.sum(w))
            if m <= 1e-12:
                continue
            out[row, shell] = np.sum(q[cols[mask]] * w[:, None], axis=0) / m
            mass[row, shell] = m
    return out.astype(np.float32), mass.astype(np.float32)


def _radial_block(shell_composition: np.ndarray) -> np.ndarray:
    shell = np.asarray(shell_composition, dtype=np.float32)
    if shell.shape[1] == 1:
        diff = np.zeros((shell.shape[0], shell.shape[2]), dtype=np.float32)
    else:
        diff = shell[:, -1, :] - shell[:, 0, :]
    return np.hstack([shell.reshape(shell.shape[0], -1), diff]).astype(np.float32)


def _gradient_block(
    graph: NeighborhoodGraph,
    coords: np.ndarray,
    shell_composition: np.ndarray,
    shell_mass: np.ndarray,
) -> np.ndarray:
    xy = np.asarray(coords, dtype=np.float32)
    conn, dist = _row_distance_data(graph)
    n_cells = int(xy.shape[0])
    out = np.zeros((n_cells, 5), dtype=np.float32)
    if shell_composition.shape[1] > 1:
        boundary = np.linalg.norm(
            shell_composition[:, -1, :] - shell_composition[:, 0, :], axis=1
        )
    else:
        boundary = np.zeros(n_cells, dtype=np.float32)
    occupied = np.mean(shell_mass > 1e-12, axis=1).astype(np.float32)
    for row in range(n_cells):
        start, stop = int(conn.indptr[row]), int(conn.indptr[row + 1])
        if start == stop:
            continue
        cols = conn.indices[start:stop]
        weights = np.asarray(conn.data[start:stop], dtype=np.float64)
        distances = np.asarray(dist.data[start:stop], dtype=np.float32)
        wsum = float(np.sum(weights))
        if wsum <= 1e-12:
            continue
        denom = (
            float(graph.radius_um)
            if graph.radius_um is not None and float(graph.radius_um) > 0.0
            else float(np.max(distances))
            if distances.size
            else 1.0
        )
        rel = (xy[cols].astype(np.float64) - xy[row].astype(np.float64)) / max(
            denom, 1e-8
        )
        w = weights / wsum
        mean_rel = np.sum(rel * w[:, None], axis=0)
        centered = rel - mean_rel[None, :]
        cov = centered.T @ (centered * w[:, None])
        eig = np.linalg.eigvalsh(cov)
        total = float(np.sum(np.maximum(eig, 0.0)))
        anisotropy = 0.0 if total <= 1e-12 else float((eig[-1] - eig[0]) / total)
        out[row, 0] = float(boundary[row])
        out[row, 1] = float(anisotropy)
        out[row, 2] = float(np.linalg.norm(mean_rel))
        out[row, 3] = float(np.sum(w * distances.astype(np.float64)) / max(denom, 1e-8))
        out[row, 4] = float(occupied[row])
    return out.astype(np.float32)


def _pair_block(
    anchor_posteriors: np.ndarray,
    composition: np.ndarray,
    *,
    top_states: int,
) -> tuple[np.ndarray, dict[str, object]]:
    q = np.asarray(anchor_posteriors, dtype=np.float32)
    p = np.asarray(composition, dtype=np.float32)
    k = int(q.shape[1])
    keep = min(max(int(top_states), 1), k)
    global_mass = np.mean(q, axis=0)
    top_idx = np.argsort(-global_mass)[:keep]
    anchor = q[:, top_idx]
    context = p[:, top_idx]
    outer = anchor[:, :, None] * context[:, None, :]
    return outer.reshape(q.shape[0], keep * keep).astype(np.float32), {
        "mode": "anchor_neighbor",
        "top_states": [int(idx) for idx in top_idx.tolist()],
    }


def _covariance_block(
    graph: NeighborhoodGraph,
    features: np.ndarray,
    *,
    covariance_dims: int,
) -> tuple[np.ndarray, dict[str, object]]:
    z = np.asarray(features, dtype=np.float32)
    keep = min(max(int(covariance_dims), 1), int(z.shape[1]))
    view = z[:, :keep].astype(np.float32, copy=False)
    n_cells = int(view.shape[0])
    tri = np.triu_indices(keep)
    out = np.zeros((n_cells, int(len(tri[0]))), dtype=np.float32)
    conn = _row_normalize(graph.connectivities)
    for row in range(n_cells):
        start, stop = int(conn.indptr[row]), int(conn.indptr[row + 1])
        if start == stop:
            continue
        cols = conn.indices[start:stop]
        weights = np.asarray(conn.data[start:stop], dtype=np.float64)
        wsum = float(np.sum(weights))
        if wsum <= 1e-12:
            continue
        w = weights / wsum
        x = view[cols].astype(np.float64, copy=False)
        mean = np.sum(x * w[:, None], axis=0)
        centered = x - mean[None, :]
        cov = centered.T @ (centered * w[:, None])
        out[row] = cov[tri].astype(np.float32)
    return out, {
        "mode": "covet_like_weighted_local_covariance",
        "covariance_dims": int(keep),
        "upper_triangular_dim": int(out.shape[1]),
    }


def _standardize_block(
    values: np.ndarray,
    *,
    weight: float,
) -> tuple[np.ndarray, dict[str, object]]:
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("descriptor block values must be 2D.")
    if x.shape[1] == 0:
        return x, {"dimension": 0, "weight": float(weight)}
    mean = np.mean(x, axis=0, dtype=np.float64)
    scale = np.std(x, axis=0, dtype=np.float64)
    finite = scale[np.isfinite(scale) & (scale > 1e-8)]
    floor = float(np.median(finite) * 1e-3) if finite.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale > max(floor, 1e-8)), scale, 1.0)
    z = ((x.astype(np.float64, copy=False) - mean[None, :]) / scale[None, :]).astype(
        np.float32
    )
    weighted = z * (float(weight) / float(np.sqrt(max(int(x.shape[1]), 1))))
    return weighted.astype(np.float32), {
        "dimension": int(x.shape[1]),
        "weight": float(weight),
        "standardization": "mean_std_then_weight_over_sqrt_dimension",
        "mean_weighted_l2": float(np.mean(np.linalg.norm(weighted, axis=1)))
        if weighted.size
        else 0.0,
    }


def compute_cell_heterogeneity_descriptors(
    *,
    features: np.ndarray,
    posteriors: np.ndarray,
    graphs: dict[str, NeighborhoodGraph],
    coords: np.ndarray,
    blocks: list[str] | tuple[str, ...] | str | None = None,
    block_weights: dict[str, float] | None = None,
    self_weight: float = 0.25,
    radial_shells: int = 3,
    pair_mode: str = "anchor_neighbor",
    pair_top_states: int = 16,
    covariance_dims: int = 8,
) -> DescriptorResult:
    """Compute deterministic cell-centered multi-scale heterogeneity descriptors."""

    z = np.asarray(features, dtype=np.float32)
    q = np.asarray(posteriors, dtype=np.float32)
    if z.ndim != 2 or q.ndim != 2 or z.shape[0] != q.shape[0]:
        raise ValueError("features and posteriors must be 2D with the same number of rows.")
    if not graphs:
        raise ValueError("At least one neighborhood graph is required.")
    requested_blocks = _normalize_blocks(blocks)
    raw_blocks: list[_RawBlock] = []
    graph_diagnostics: dict[str, object] = {}

    if float(self_weight) > 0.0:
        raw_blocks.append(_RawBlock(name="self", family="self", values=z))

    for graph_key, graph in graphs.items():
        composition = _composition_block(graph, q)
        graph_diagnostics[graph_key] = dict(graph.metadata)
        shell_composition = None
        shell_mass = None
        if "composition" in requested_blocks:
            raw_blocks.append(
                _RawBlock(
                    name=f"composition:{graph_key}",
                    family="composition",
                    values=composition,
                )
            )
        if "diversity" in requested_blocks:
            raw_blocks.append(
                _RawBlock(
                    name=f"diversity:{graph_key}",
                    family="diversity",
                    values=_diversity_block(composition),
                )
            )
        if "moments" in requested_blocks:
            raw_blocks.append(
                _RawBlock(
                    name=f"moments:{graph_key}",
                    family="moments",
                    values=_moments_block(graph, z),
                )
            )
        if "radial" in requested_blocks or "gradient" in requested_blocks:
            shell_composition, shell_mass = _radial_shell_composition(
                graph, q, n_shells=max(int(radial_shells), 1)
            )
        if "radial" in requested_blocks:
            raw_blocks.append(
                _RawBlock(
                    name=f"radial:{graph_key}",
                    family="radial",
                    values=_radial_block(shell_composition),
                )
            )
        if "pair" in requested_blocks:
            requested_pair = str(pair_mode or "anchor_neighbor").strip().lower()
            if requested_pair not in {"anchor_neighbor", "none", "disabled"}:
                raise ValueError("pair_mode currently supports anchor_neighbor or none.")
            if requested_pair == "anchor_neighbor":
                pair_values, pair_meta = _pair_block(
                    q, composition, top_states=int(pair_top_states)
                )
                graph_diagnostics[graph_key]["pair_block"] = pair_meta
                raw_blocks.append(
                    _RawBlock(
                        name=f"pair:{graph_key}",
                        family="pair",
                        values=pair_values,
                    )
                )
        if "covariance" in requested_blocks:
            cov_values, cov_meta = _covariance_block(
                graph,
                z,
                covariance_dims=int(covariance_dims),
            )
            graph_diagnostics[graph_key]["covariance_block"] = cov_meta
            raw_blocks.append(
                _RawBlock(
                    name=f"covariance:{graph_key}",
                    family="covariance",
                    values=cov_values,
                )
            )
        if "gradient" in requested_blocks:
            if shell_composition is None or shell_mass is None:
                shell_composition, shell_mass = _radial_shell_composition(
                    graph, q, n_shells=max(int(radial_shells), 1)
                )
            raw_blocks.append(
                _RawBlock(
                    name=f"gradient:{graph_key}",
                    family="gradient",
                    values=_gradient_block(graph, np.asarray(coords), shell_composition, shell_mass),
                )
            )

    if not raw_blocks:
        raise ValueError("No descriptor blocks were enabled.")

    raw = np.hstack([block.values for block in raw_blocks]).astype(np.float32)
    family_counts: dict[str, int] = {}
    for block in raw_blocks:
        family_counts[block.family] = int(family_counts.get(block.family, 0)) + 1
    incoming = dict(DEFAULT_BLOCK_WEIGHTS)
    if block_weights:
        for key, value in block_weights.items():
            if key in incoming:
                incoming[key] = float(value)
    incoming["self"] = max(float(self_weight), 0.0)
    present_families = list(family_counts)
    total_weight = float(sum(max(incoming.get(family, 0.0), 0.0) for family in present_families))
    if total_weight <= 1e-12:
        total_weight = float(len(present_families))
        incoming = {family: 1.0 for family in present_families}

    standardized_parts: list[np.ndarray] = []
    block_meta: dict[str, object] = {}
    slices: dict[str, list[int]] = {}
    cursor = 0
    for block in raw_blocks:
        family_weight = max(float(incoming.get(block.family, 0.0)), 0.0) / total_weight
        block_weight = family_weight / max(int(family_counts[block.family]), 1)
        weighted, stats = _standardize_block(block.values, weight=block_weight)
        standardized_parts.append(weighted)
        slices[block.name] = [int(cursor), int(cursor + weighted.shape[1])]
        cursor += int(weighted.shape[1])
        block_meta[block.name] = {
            **stats,
            "family": str(block.family),
            "family_weight": float(family_weight),
        }
    standardized = np.hstack(standardized_parts).astype(np.float32)
    metadata = {
        "mode": "cell_centered_spatial_heterogeneity_descriptor",
        "n_cells": int(z.shape[0]),
        "feature_dim": int(z.shape[1]),
        "state_codebook_dim": int(q.shape[1]),
        "raw_dim": int(raw.shape[1]),
        "standardized_dim": int(standardized.shape[1]),
        "requested_blocks": list(requested_blocks),
        "block_slices": slices,
        "blocks": block_meta,
        "family_counts": dict(family_counts),
        "radial_shells": int(radial_shells),
        "pair_mode": str(pair_mode),
        "pair_top_states": int(pair_top_states),
        "covariance_dims": int(covariance_dims),
        "graph_diagnostics": graph_diagnostics,
        "primary_unit": "cell",
        "uses_fitted_region_boundaries": False,
        "uses_raw_absolute_coordinates_for_clustering": False,
    }
    return DescriptorResult(raw=raw, standardized=standardized, metadata=metadata)


def reduce_descriptor_embedding(
    descriptors: np.ndarray,
    *,
    n_components: int = 64,
    method: str = "descriptor_pca",
    random_state: int = 1337,
) -> EmbeddingResult:
    x = np.asarray(descriptors, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("descriptors must be a 2D matrix.")
    requested = str(method or "descriptor_pca").strip().lower()
    if requested in {"none", "descriptor", "identity"}:
        values = x.astype(np.float32, copy=True)
        reduction = "none"
        explained = None
    elif requested in {"descriptor_pca", "pca", "svd", "descriptor_svd"}:
        target = min(max(int(n_components), 1), int(x.shape[1]), max(int(x.shape[0]) - 1, 1))
        if target >= int(x.shape[1]) or int(x.shape[0]) <= target + 1:
            values = x.astype(np.float32, copy=True)
            reduction = "none_dim_not_reduced"
            explained = None
        else:
            svd = TruncatedSVD(
                n_components=int(target),
                n_iter=7,
                random_state=int(random_state),
            )
            values = svd.fit_transform(x).astype(np.float32, copy=False)
            reduction = "truncated_svd"
            explained = float(
                np.sum(np.asarray(svd.explained_variance_ratio_, dtype=np.float64))
            )
    else:
        raise ValueError("embedding method must be descriptor_pca or none.")

    center = np.mean(values, axis=0, dtype=np.float64)
    scale = np.std(values, axis=0, dtype=np.float64)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    values = ((values.astype(np.float64, copy=False) - center[None, :]) / scale[None, :]).astype(
        np.float32
    )
    return EmbeddingResult(
        values=values,
        metadata={
            "method": str(requested),
            "reduction": reduction,
            "requested_dim": int(n_components),
            "embedding_dim": int(values.shape[1]),
            "explained_variance_ratio_sum": explained,
            "standardization": "mean_std_after_reduction",
        },
    )


__all__ = [
    "DEFAULT_BLOCK_WEIGHTS",
    "DescriptorResult",
    "EmbeddingResult",
    "compute_cell_heterogeneity_descriptors",
    "reduce_descriptor_embedding",
]
