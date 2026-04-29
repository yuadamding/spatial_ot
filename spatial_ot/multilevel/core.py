from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np
import ot
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
import torch

from .._runtime import runtime_memory_snapshot as _runtime_memory_snapshot
from .runtime import (
    configure_local_thread_budget as _configure_local_thread_budget,
    cuda_target_bytes as _cuda_target_bytes,
    env_float as _env_float,
    env_int as _env_int,
    relative_change as _relative_change,
    resolve_compute_device as _resolve_compute_device,
    resolve_cuda_device_pool as _resolve_cuda_device_pool,
    resolve_parallel_restart_workers as _resolve_parallel_restart_workers,
)
from .geometry import (
    _normalize_hist,
    _region_geometries_from_members,
    _region_geometries_from_observed_points,
    _softmax_over_negative_costs,
    _standardize_features,
    _validate_fit_inputs,
    _validate_mutually_exclusive_memberships,
    build_deep_graph_segmentation_subregions,
    build_partition_subregions_from_grid_tiles,
    build_composite_subregions_from_basic_niches,
    fit_ot_shape_normalizer,
    make_reference_points_unit_disk,
    refine_subregions_by_cluster_coherence,
    sample_geometry_points,
)
from .gpu_ot import sinkhorn_semirelaxed_unbalanced_log_torch
from .heterogeneity import (
    HETEROGENEITY_DESCRIPTOR_ALIASES,
    HETEROGENEITY_DESCRIPTOR_MODE,
    HETEROGENEITY_FGW_MODE,
    HETEROGENEITY_FUSED_OT_MODE,
    LEGACY_HETEROGENEITY_OT_ALIAS,
    TRANSPORT_HETEROGENEITY_MODES,
    build_internal_heterogeneity_embeddings,
    build_subregion_fgw_measures,
    pairwise_transport_distance_matrix,
)
from .model_selection import (
    comprehensive_select_k_from_latent_embeddings,
    effective_min_cluster_size,
    fit_kmeans_on_latent_embeddings,
    repair_labels_to_minimum_size,
    sanitize_candidate_n_clusters,
    select_k_from_ot_landmark_costs,
)
from .numerics import pairwise_sqdist_array as _pairwise_sqdist_array
from .spot_latent import (
    compute_spot_level_latent_charts,
    empty_spot_level_latent_charts,
)
from .transforms import apply_similarity
from .types import (
    MultilevelOTResult,
    OTSolveDiagnostics,
    OptimizationMeasure,
    RegionGeometry,
    SubregionMeasure,
)


_RESTART_WORKER_MEASURES: list[OptimizationMeasure] | None = None
_RESTART_WORKER_SUMMARIES: np.ndarray | None = None
_RESTART_WORKER_PARAMS: dict[str, object] | None = None
_PROGRESS_START = time.perf_counter()


def _progress(message: str) -> None:
    raw = os.environ.get("SPATIAL_OT_PROGRESS", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        elapsed = time.perf_counter() - _PROGRESS_START
        print(f"[spatial_ot {elapsed:8.1f}s] {message}", file=sys.stderr, flush=True)


def _sinkhorn_max_iter() -> int:
    return _env_int("SPATIAL_OT_SINKHORN_MAX_ITER", 600)


def _sinkhorn_tol() -> float:
    return _env_float("SPATIAL_OT_SINKHORN_TOL", 1e-5)


def _cpu_sinkhorn_max_iter() -> int:
    return _env_int("SPATIAL_OT_CPU_SINKHORN_MAX_ITER", 3000)


def _cpu_sinkhorn_tol() -> float:
    return _env_float("SPATIAL_OT_CPU_SINKHORN_TOL", 1e-8)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _make_optimization_measures(
    measures: list[SubregionMeasure],
) -> list[OptimizationMeasure]:
    return [
        OptimizationMeasure(
            subregion_id=int(measure.subregion_id),
            canonical_coords=np.asarray(measure.canonical_coords, dtype=np.float32),
            features=np.asarray(measure.features, dtype=np.float32),
            weights=np.asarray(measure.weights, dtype=np.float32),
        )
        for measure in measures
    ]


def _build_overlap_consistency_graph(
    measures: list[SubregionMeasure],
    summaries: np.ndarray,
    *,
    min_jaccard: float,
    contrast_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_subregions = len(measures)
    if n_subregions < 2:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
        )
    rows = np.concatenate(
        [np.asarray(measure.members, dtype=np.int64) for measure in measures],
        dtype=np.int64,
    )
    cols = np.concatenate(
        [
            np.full(
                np.asarray(measure.members, dtype=np.int64).shape[0],
                rid,
                dtype=np.int64,
            )
            for rid, measure in enumerate(measures)
        ],
        dtype=np.int64,
    )
    if rows.size == 0:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
        )
    n_cells = int(rows.max()) + 1
    incidence = sparse.csr_matrix(
        (np.ones(rows.shape[0], dtype=np.int8), (rows, cols)),
        shape=(n_cells, n_subregions),
        dtype=np.int8,
    )
    overlap = (incidence.T @ incidence).tocoo()
    member_sizes = np.asarray(
        [len(measure.members) for measure in measures], dtype=np.int64
    )
    summary_arr = np.asarray(summaries, dtype=np.float32)
    edge_i: list[int] = []
    edge_j: list[int] = []
    edge_weight: list[float] = []
    for i, j, intersection in zip(
        overlap.row.tolist(), overlap.col.tolist(), overlap.data.tolist(), strict=False
    ):
        if int(i) >= int(j) or int(intersection) <= 0:
            continue
        union = int(member_sizes[int(i)] + member_sizes[int(j)] - int(intersection))
        if union <= 0:
            continue
        jaccard = float(int(intersection) / union)
        if jaccard < float(min_jaccard):
            continue
        weight = jaccard
        if (
            summary_arr.ndim == 2
            and summary_arr.shape[0] == n_subregions
            and summary_arr.shape[1] > 0
        ):
            contrast = float(np.linalg.norm(summary_arr[int(i)] - summary_arr[int(j)]))
            weight *= float(np.exp(-contrast / max(float(contrast_scale), 1e-6)))
        if weight <= 0:
            continue
        edge_i.append(int(i))
        edge_j.append(int(j))
        edge_weight.append(float(weight))
    return (
        np.asarray(edge_i, dtype=np.int32),
        np.asarray(edge_j, dtype=np.int32),
        np.asarray(edge_weight, dtype=np.float32),
    )


def _compute_overlap_penalty_matrix(
    probs: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_weight: np.ndarray,
) -> np.ndarray:
    probs_arr = np.asarray(probs, dtype=np.float64)
    n_subregions, n_clusters = probs_arr.shape
    if n_subregions == 0 or n_clusters == 0 or edge_weight.size == 0:
        return np.zeros((n_subregions, n_clusters), dtype=np.float32)
    penalty = np.zeros((n_subregions, n_clusters), dtype=np.float64)
    norm = np.zeros(n_subregions, dtype=np.float64)
    for i, j, weight in zip(
        edge_i.tolist(), edge_j.tolist(), edge_weight.tolist(), strict=False
    ):
        w = float(weight)
        penalty[int(i)] += w * (1.0 - probs_arr[int(j)])
        penalty[int(j)] += w * (1.0 - probs_arr[int(i)])
        norm[int(i)] += w
        norm[int(j)] += w
    active = norm > 0
    if np.any(active):
        penalty[active] /= norm[active, None]
    return penalty.astype(np.float32)


def _apply_overlap_consistency_regularization(
    transport_costs: np.ndarray,
    *,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_weight: np.ndarray,
    overlap_consistency_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.asarray(transport_costs, dtype=np.float32)
    if float(overlap_consistency_weight) <= 0 or edge_weight.size == 0:
        zero = np.zeros_like(costs, dtype=np.float32)
        return costs.astype(np.float32, copy=True), zero
    temperature = max(float(np.std(costs)), 1e-3)
    probs = _softmax_over_negative_costs(costs, temperature=temperature)
    penalties = _compute_overlap_penalty_matrix(
        probs,
        np.asarray(edge_i, dtype=np.int32),
        np.asarray(edge_j, dtype=np.int32),
        np.asarray(edge_weight, dtype=np.float32),
    )
    weighted_penalties = float(overlap_consistency_weight) * penalties
    return (costs + weighted_penalties).astype(np.float32), weighted_penalties.astype(
        np.float32
    )


def _init_restart_worker(
    measures: list[OptimizationMeasure],
    summaries: np.ndarray,
    params: dict[str, object],
    worker_threads: int,
    worker_interop_threads: int,
) -> None:
    global _RESTART_WORKER_MEASURES, _RESTART_WORKER_SUMMARIES, _RESTART_WORKER_PARAMS
    _configure_local_thread_budget(worker_threads, worker_interop_threads)
    _RESTART_WORKER_MEASURES = measures
    _RESTART_WORKER_SUMMARIES = summaries
    _RESTART_WORKER_PARAMS = params


def _run_restart_worker(run: int, compute_device: str) -> dict[str, object]:
    if (
        _RESTART_WORKER_MEASURES is None
        or _RESTART_WORKER_SUMMARIES is None
        or _RESTART_WORKER_PARAMS is None
    ):
        raise RuntimeError("Restart worker was not initialized.")
    return _execute_restart(
        measures=_RESTART_WORKER_MEASURES,
        summaries=_RESTART_WORKER_SUMMARIES,
        run=run,
        compute_device=compute_device,
        **_RESTART_WORKER_PARAMS,
    )


def _cluster_cost_matrix(
    u_aligned: np.ndarray,
    y: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    *,
    cost_scale_x: float,
    cost_scale_y: float,
    lambda_x: float,
    lambda_y: float,
    compute_device: torch.device,
    atom_coords_t: torch.Tensor | None = None,
    atom_features_t: torch.Tensor | None = None,
    y_t: torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    if compute_device.type == "cpu":
        cx = ot.dist(u_aligned, atom_coords, metric="sqeuclidean") / max(
            cost_scale_x, 1e-12
        )
        cy = ot.dist(y, atom_features, metric="sqeuclidean") / max(cost_scale_y, 1e-12)
        return lambda_x * cx + lambda_y * cy

    u_aligned_t = torch.as_tensor(u_aligned, dtype=torch.float64, device=compute_device)
    if atom_coords_t is None:
        atom_coords_t = torch.as_tensor(
            atom_coords, dtype=torch.float64, device=compute_device
        )
    if atom_features_t is None:
        atom_features_t = torch.as_tensor(
            atom_features, dtype=torch.float64, device=compute_device
        )
    if y_t is None:
        y_t = torch.as_tensor(y, dtype=torch.float64, device=compute_device)
    cx_t = torch.cdist(u_aligned_t, atom_coords_t, p=2).pow(2) / max(
        cost_scale_x, 1e-12
    )
    cy_t = torch.cdist(y_t, atom_features_t, p=2).pow(2) / max(cost_scale_y, 1e-12)
    return float(lambda_x) * cx_t + float(lambda_y) * cy_t


def _compress_measure(
    canonical_coords: np.ndarray,
    features: np.ndarray,
    weights: np.ndarray,
    m: int,
    lambda_x: float,
    lambda_y: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = canonical_coords.shape[0]
    if n <= m:
        return (
            canonical_coords.astype(np.float32),
            features.astype(np.float32),
            _normalize_hist(weights).astype(np.float32),
        )

    z = np.hstack(
        [np.sqrt(lambda_x) * canonical_coords, np.sqrt(lambda_y) * features]
    ).astype(np.float32)
    km = MiniBatchKMeans(
        n_clusters=m,
        random_state=random_state,
        batch_size=min(4096, n),
        n_init="auto",
    )
    labels = km.fit_predict(z, sample_weight=weights)
    u_centers = np.zeros((m, canonical_coords.shape[1]), dtype=np.float64)
    y_centers = np.zeros((m, features.shape[1]), dtype=np.float64)
    a_centers = np.zeros(m, dtype=np.float64)
    for j in range(m):
        mask = labels == j
        if not np.any(mask):
            continue
        a_j = float(weights[mask].sum())
        a_centers[j] = a_j
        norm = max(a_j, 1e-12)
        u_centers[j] = (weights[mask, None] * canonical_coords[mask]).sum(axis=0) / norm
        y_centers[j] = (weights[mask, None] * features[mask]).sum(axis=0) / norm
    keep = a_centers > 1e-12
    if not np.any(keep):
        return (
            canonical_coords[:m].astype(np.float32),
            features[:m].astype(np.float32),
            _normalize_hist(weights[:m]).astype(np.float32),
        )
    return (
        u_centers[keep].astype(np.float32),
        y_centers[keep].astype(np.float32),
        _normalize_hist(a_centers[keep]).astype(np.float32),
    )


def _build_subregion_measures(
    features: np.ndarray,
    coords_um: np.ndarray,
    centers_um: np.ndarray,
    region_geometries: list[RegionGeometry],
    geometry_reference_points: np.ndarray,
    geometry_reference_weights: np.ndarray,
    geometry_eps: float,
    geometry_samples: int,
    compressed_support_size: int,
    lambda_x: float,
    lambda_y: float,
    seed: int,
    allow_convex_hull_fallback: bool,
    compute_device: torch.device | None = None,
) -> list[SubregionMeasure]:
    measures: list[SubregionMeasure] = []
    for rid, region in enumerate(region_geometries):
        members = np.asarray(region.members, dtype=np.int32)
        local_coords = np.asarray(coords_um[members], dtype=np.float32)
        local_features = np.asarray(features[members], dtype=np.float32)
        weights = np.full(
            local_coords.shape[0], 1.0 / max(local_coords.shape[0], 1), dtype=np.float32
        )
        geom_points, geometry_source, used_fallback = sample_geometry_points(
            region,
            observed_coords=local_coords,
            n_points=max(int(geometry_samples), 32),
            seed=seed + rid,
            allow_convex_hull_fallback=allow_convex_hull_fallback,
            warn_on_fallback=False,
        )
        normalizer, diagnostics = fit_ot_shape_normalizer(
            geometry_points=geom_points,
            reference_points=geometry_reference_points,
            reference_weights=geometry_reference_weights,
            eps_geom=geometry_eps,
            compute_device=compute_device,
        )
        diagnostics.geometry_source = geometry_source
        diagnostics.used_fallback = used_fallback
        canonical_coords = normalizer.transform(local_coords)
        uc, yc, ac = _compress_measure(
            canonical_coords=canonical_coords,
            features=local_features,
            weights=weights,
            m=max(int(compressed_support_size), 1),
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            random_state=seed + 10000 + rid,
        )
        measures.append(
            SubregionMeasure(
                subregion_id=rid,
                center_um=centers_um[rid].astype(np.float32),
                members=np.asarray(members, dtype=np.int32),
                canonical_coords=uc,
                features=yc,
                weights=ac,
                geometry_point_count=int(geom_points.shape[0]),
                compressed_point_count=int(uc.shape[0]),
                normalizer=normalizer,
                normalizer_diagnostics=diagnostics,
            )
        )
    return measures


def _measure_summary(measure: SubregionMeasure) -> np.ndarray:
    z = np.hstack([measure.canonical_coords, measure.features]).astype(np.float64)
    mean = np.average(z, axis=0, weights=measure.weights)
    var = np.average((z - mean) ** 2, axis=0, weights=measure.weights)
    return np.hstack([mean, np.sqrt(np.maximum(var, 0.0))]).astype(np.float32)


_SUBREGION_LATENT_EMBEDDING_MODES = {
    "mean_std",
    "mean_std_shrunk",
    "mean_std_skew_count",
    "mean_std_quantile",
    "codebook_histogram",
    "mean_std_codebook",
}

_SUBREGION_LATENT_HETEROGENEITY_WEIGHT = 0.5


def _normalize_subregion_latent_embedding_mode(mode: str | None) -> str:
    normalized = str(mode or "mean_std_shrunk").strip().lower().replace("-", "_")
    aliases = {
        "legacy": "mean_std",
        "mean_std_legacy": "mean_std",
        "shrunk": "mean_std_shrunk",
        "moments": "mean_std_skew_count",
        "quantile": "mean_std_quantile",
        "quantiles": "mean_std_quantile",
        "codebook": "codebook_histogram",
        "histogram": "codebook_histogram",
        "mean_std_histogram": "mean_std_codebook",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _SUBREGION_LATENT_EMBEDDING_MODES:
        raise ValueError(
            "subregion_latent_embedding_mode must be one of "
            f"{sorted(_SUBREGION_LATENT_EMBEDDING_MODES)}, got '{mode}'"
        )
    return normalized


def _feature_distribution_latent_embedding(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    mode: str | None = "mean_std_shrunk",
    shrinkage_tau: float = 25.0,
    heterogeneity_weight: float = _SUBREGION_LATENT_HETEROGENEITY_WEIGHT,
) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("feature values must be a 2D matrix.")
    if y.shape[0] == 0:
        raise ValueError("feature values must contain at least one row.")
    if weights is None:
        weights_arr = np.full(y.shape[0], 1.0 / float(y.shape[0]), dtype=np.float64)
    else:
        weights_arr = _normalize_hist(np.asarray(weights, dtype=np.float64))
        if weights_arr.shape[0] != y.shape[0]:
            raise ValueError("feature weights must have one value per feature row.")
    mean = np.average(y, axis=0, weights=weights_arr)
    centered = y - mean
    var = np.maximum(np.average(centered * centered, axis=0, weights=weights_arr), 0.0)
    std = np.sqrt(var)
    normalized_mode = _normalize_subregion_latent_embedding_mode(mode)
    if normalized_mode == "mean_std":
        return np.hstack([mean, std]).astype(np.float32)
    if normalized_mode == "mean_std_shrunk":
        tau = max(float(shrinkage_tau), 0.0)
        if tau <= 0.0:
            return np.hstack([mean, std]).astype(np.float32)
        h_weight = max(float(heterogeneity_weight), 0.0)
        global_mean = y.mean(axis=0, dtype=np.float64)
        global_var = np.maximum(y.var(axis=0, dtype=np.float64), 0.0)
        alpha = float(y.shape[0]) / (float(y.shape[0]) + tau)
        shrunk_mean = alpha * mean + (1.0 - alpha) * global_mean
        mixed_var = alpha * (var + (mean - shrunk_mean) ** 2) + (1.0 - alpha) * (
            global_var + (global_mean - shrunk_mean) ** 2
        )
        return np.hstack(
            [shrunk_mean, h_weight * np.sqrt(np.maximum(mixed_var, 0.0))]
        ).astype(np.float32)
    if normalized_mode == "mean_std_skew_count":
        skew = (
            np.average(centered * centered * centered, axis=0, weights=weights_arr)
            / np.maximum(std, 1e-8) ** 3
        )
        reliability = float(y.shape[0]) / (
            float(y.shape[0]) + max(float(shrinkage_tau), 1e-8)
        )
        return np.hstack(
            [mean, std, skew, [np.log1p(float(y.shape[0])), reliability]]
        ).astype(np.float32)
    if normalized_mode == "mean_std_quantile":
        quantiles = np.quantile(y, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
        return np.hstack([mean, std, quantiles.reshape(-1)]).astype(np.float32)
    raise ValueError(
        f"Mode '{normalized_mode}' is not supported for a single compressed measure embedding."
    )


def _measure_feature_latent_embedding(measure: SubregionMeasure) -> np.ndarray:
    return _feature_distribution_latent_embedding(
        measure.features, measure.weights, mode="mean_std"
    )


def _member_feature_latent_embedding(
    features: np.ndarray,
    members: np.ndarray,
    *,
    mode: str | None = "mean_std_shrunk",
    shrinkage_tau: float = 25.0,
    heterogeneity_weight: float = _SUBREGION_LATENT_HETEROGENEITY_WEIGHT,
) -> np.ndarray:
    member_idx = np.asarray(members, dtype=np.int64)
    if member_idx.ndim != 1 or member_idx.size == 0:
        raise ValueError("subregion members must be a non-empty 1D vector.")
    return _feature_distribution_latent_embedding(
        np.asarray(features, dtype=np.float32)[member_idx],
        mode=mode,
        shrinkage_tau=shrinkage_tau,
        heterogeneity_weight=heterogeneity_weight,
    )


def _subregion_sample_ids_from_members(
    sample_ids: np.ndarray | None,
    subregion_members: list[np.ndarray],
) -> np.ndarray:
    if sample_ids is None:
        return np.full(len(subregion_members), "cohort", dtype=object)
    sample_values = np.asarray(sample_ids).astype(str)
    out = np.full(len(subregion_members), "cohort", dtype=object)
    for rid, members in enumerate(subregion_members):
        member_idx = np.asarray(members, dtype=np.int64)
        if member_idx.size == 0:
            continue
        values, counts = np.unique(sample_values[member_idx], return_counts=True)
        out[int(rid)] = str(values[int(np.argmax(counts))]) if values.size else "cohort"
    return out


def _soft_codebook_histogram_embeddings(
    features: np.ndarray,
    region_ids: np.ndarray,
    *,
    n_regions: int,
    codebook_size: int,
    codebook_sample_size: int,
    random_state: int,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    region_arr = np.asarray(region_ids, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("features must be a 2D matrix.")
    if region_arr.shape[0] != x.shape[0]:
        raise ValueError("region_ids must have one entry per feature row.")
    n_codes = min(max(int(codebook_size), 2), max(int(x.shape[0]), 1))
    hist = np.zeros((int(n_regions), int(n_codes)), dtype=np.float32)
    rng = np.random.default_rng(int(random_state))
    sample_size = min(max(int(codebook_sample_size), n_codes), int(x.shape[0]))
    sample_idx = np.sort(
        rng.choice(int(x.shape[0]), size=sample_size, replace=False).astype(np.int64)
    )
    sample = x[sample_idx].astype(np.float64, copy=False)
    center = np.mean(sample, axis=0, dtype=np.float64)
    scale = np.std(sample, axis=0, dtype=np.float64)
    finite_scale = scale[np.isfinite(scale) & (scale > 1e-8)]
    scale_floor = float(np.median(finite_scale) * 1e-3) if finite_scale.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale > max(scale_floor, 1e-8)), scale, 1.0)
    sample_z = ((sample - center[None, :]) / scale[None, :]).astype(np.float32)
    model = MiniBatchKMeans(
        n_clusters=int(n_codes),
        n_init=3,
        batch_size=min(max(int(n_codes) * 256, 2048), max(int(sample_size), 2048)),
        random_state=int(random_state),
        max_iter=40,
        max_no_improvement=5,
        reassignment_ratio=0.0,
    )
    model.fit(sample_z)
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    sample_d2 = _pairwise_sqdist_array(sample_z, centers, device=torch.device("cpu"))
    nearest_sample_d2 = (
        np.min(sample_d2, axis=1) if sample_d2.size else np.zeros(0, dtype=np.float32)
    )
    positive = nearest_sample_d2[
        np.isfinite(nearest_sample_d2) & (nearest_sample_d2 > 1e-8)
    ]
    temperature = float(np.median(positive)) if positive.size else 1.0
    temperature = max(temperature, 1e-4)
    chunk_size = max(
        4096,
        min(
            32768,
            int(os.environ.get("SPATIAL_OT_CODEBOOK_ASSIGNMENT_CHUNK_SIZE", "32768")),
        ),
    )
    for start in range(0, int(x.shape[0]), chunk_size):
        stop = min(start + chunk_size, int(x.shape[0]))
        chunk_z = (
            (x[start:stop].astype(np.float64, copy=False) - center[None, :])
            / scale[None, :]
        ).astype(np.float32)
        d2 = _pairwise_sqdist_array(chunk_z, centers, device=torch.device("cpu"))
        logits = -d2 / temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits).astype(np.float32)
        probs /= np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-8)
        np.add.at(hist, region_arr[start:stop], probs)
    denom = np.maximum(
        np.bincount(region_arr, minlength=int(n_regions)).astype(np.float32)[:, None],
        1.0,
    )
    return (hist / denom).astype(np.float32)


def _subregion_latent_embedding_metadata(
    *,
    mode: str | None,
    shrinkage_tau: float,
    heterogeneity_weight: float,
    sample_prior_weight: float,
    codebook_size: int,
    codebook_sample_size: int,
    feature_dim: int,
    embedding_dim: int,
    sample_aware_shrinkage: bool = False,
) -> dict[str, object]:
    normalized = _normalize_subregion_latent_embedding_mode(mode)
    blocks_by_mode = {
        "mean_std": ["feature_mean", "feature_std"],
        "mean_std_shrunk": ["shrunk_feature_mean", "shrunk_feature_std"],
        "mean_std_skew_count": [
            "feature_mean",
            "feature_std",
            "feature_skew",
            "log1p_cell_count",
            "reliability_weight",
        ],
        "mean_std_quantile": [
            "feature_mean",
            "feature_std",
            "feature_quantiles_q10_q25_q50_q75_q90",
        ],
        "codebook_histogram": ["soft_whitened_cell_state_codebook_histogram"],
        "mean_std_codebook": [
            "shrunk_feature_mean",
            "shrunk_feature_std",
            "soft_whitened_cell_state_codebook_histogram",
        ],
    }
    return {
        "mode": normalized,
        "feature_dim": int(feature_dim),
        "embedding_dim": int(embedding_dim),
        "shrinkage_tau": float(shrinkage_tau),
        "sample_prior_weight": float(sample_prior_weight),
        "cohort_prior_weight": float(
            1.0 - np.clip(float(sample_prior_weight), 0.0, 1.0)
        ),
        "sample_aware_shrinkage": bool(sample_aware_shrinkage),
        "codebook_size": int(codebook_size),
        "codebook_sample_size": int(codebook_sample_size),
        "codebook_assignment": "softmax_over_whitened_codebook_squared_distance",
        "codebook_feature_standardization": "mean_std_whitening_fit_on_codebook_training_sample",
        "blocks": blocks_by_mode[normalized],
        "uses_spatial_coordinates": False,
        "uses_compressed_ot_supports": False,
        "description": (
            "Raw member-cell feature-distribution summary used for pooled subregion clustering. "
            "This embedding is built before KMeans/model selection and intentionally excludes "
            "subregion centers, canonical coordinates, overlap edges, and OT candidate costs."
        ),
        "heterogeneity_block_weight": float(heterogeneity_weight)
        if normalized in {"mean_std_shrunk", "mean_std_codebook"}
        else 1.0,
    }


def _build_subregion_latent_embeddings_from_members(
    features: np.ndarray,
    subregion_members: list[np.ndarray],
    *,
    mode: str | None = "mean_std_shrunk",
    shrinkage_tau: float = 25.0,
    heterogeneity_weight: float = _SUBREGION_LATENT_HETEROGENEITY_WEIGHT,
    sample_ids: np.ndarray | None = None,
    sample_prior_weight: float = 0.5,
    codebook_size: int = 32,
    codebook_sample_size: int = 50000,
    random_state: int = 1337,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray | bool]]:
    def _finish(
        latent_arr: np.ndarray,
        *,
        alpha: np.ndarray | None = None,
        raw_to_shrunk: np.ndarray | None = None,
        subregion_sample_ids: np.ndarray | None = None,
        sample_aware: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray | bool]]:
        if not return_diagnostics:
            return latent_arr
        n_regions_local = int(latent_arr.shape[0])
        diagnostics: dict[str, np.ndarray | bool] = {
            "shrinkage_alpha": (
                np.asarray(alpha, dtype=np.float32)
                if alpha is not None
                else np.ones(n_regions_local, dtype=np.float32)
            ),
            "raw_to_shrunk_distance": (
                np.asarray(raw_to_shrunk, dtype=np.float32)
                if raw_to_shrunk is not None
                else np.zeros(n_regions_local, dtype=np.float32)
            ),
            "sample_ids": (
                np.asarray(subregion_sample_ids, dtype=object)
                if subregion_sample_ids is not None
                else np.full(n_regions_local, "cohort", dtype=object)
            ),
            "sample_aware_shrinkage": bool(sample_aware),
        }
        return latent_arr, diagnostics

    if not subregion_members:
        return _finish(np.zeros((0, 0), dtype=np.float32))
    normalized_mode = _normalize_subregion_latent_embedding_mode(mode)
    h_weight = max(float(heterogeneity_weight), 0.0)
    sample_prior_weight = float(np.clip(float(sample_prior_weight), 0.0, 1.0))
    x = np.asarray(features, dtype=np.float32)
    n_regions = int(len(subregion_members))
    if x.ndim != 2:
        raise ValueError("features must be a 2D matrix.")
    sample_codes: np.ndarray | None = None
    sample_names = np.asarray([], dtype=object)
    subregion_sample_codes = np.full(n_regions, -1, dtype=np.int32)
    subregion_sample_ids = np.full(n_regions, "cohort", dtype=object)
    if sample_ids is not None:
        sample_values = np.asarray(sample_ids)
        if sample_values.shape[0] != x.shape[0]:
            raise ValueError("sample_ids must have one entry per feature row.")
        sample_names, sample_codes = np.unique(
            sample_values.astype(str), return_inverse=True
        )
        sample_codes = sample_codes.astype(np.int32, copy=False)
    labels = np.full(int(x.shape[0]), -1, dtype=np.int32)
    for rid, members in enumerate(subregion_members):
        member_idx = np.asarray(members, dtype=np.int64)
        if member_idx.ndim != 1 or member_idx.size == 0:
            raise ValueError("subregion members must be non-empty 1D vectors.")
        labels[member_idx] = int(rid)
        if sample_codes is not None:
            member_codes = sample_codes[member_idx]
            member_codes = member_codes[member_codes >= 0]
            if member_codes.size:
                code = int(np.argmax(np.bincount(member_codes)))
                subregion_sample_codes[int(rid)] = code
                subregion_sample_ids[int(rid)] = str(sample_names[code])
    valid = labels >= 0
    if not np.any(valid):
        if normalized_mode == "mean_std_skew_count":
            dim = int(x.shape[1]) * 3 + 2
        elif normalized_mode == "mean_std_quantile":
            dim = int(x.shape[1]) * 7
        elif normalized_mode == "codebook_histogram":
            dim = max(int(codebook_size), 2)
        elif normalized_mode == "mean_std_codebook":
            dim = int(x.shape[1]) * 2 + max(int(codebook_size), 2)
        else:
            dim = int(x.shape[1]) * 2
        return _finish(
            np.zeros((n_regions, dim), dtype=np.float32),
            subregion_sample_ids=subregion_sample_ids,
        )
    region_ids = labels[valid].astype(np.int64, copy=False)
    counts = np.bincount(region_ids, minlength=n_regions).astype(np.float64)
    sums = np.zeros((n_regions, int(x.shape[1])), dtype=np.float64)
    sums_sq = np.zeros_like(sums)
    values = x[valid].astype(np.float64, copy=False)
    for dim in range(int(x.shape[1])):
        col = values[:, dim]
        sums[:, dim] = np.bincount(region_ids, weights=col, minlength=n_regions)
        sums_sq[:, dim] = np.bincount(
            region_ids, weights=col * col, minlength=n_regions
        )
    denom = np.maximum(counts[:, None], 1.0)
    mean = sums / denom
    var = np.maximum(sums_sq / denom - mean * mean, 0.0)
    std = np.sqrt(var)
    if normalized_mode == "mean_std":
        latent = np.hstack([mean, std]).astype(np.float32)
    elif normalized_mode in {"mean_std_shrunk", "mean_std_codebook"}:
        tau = max(float(shrinkage_tau), 0.0)
        alpha = np.ones(n_regions, dtype=np.float64)
        raw_base = np.hstack([mean, h_weight * std])
        raw_to_shrunk = np.zeros(n_regions, dtype=np.float32)
        sample_aware = False
        if tau > 0.0:
            global_mean = np.mean(values, axis=0, dtype=np.float64)
            global_var = np.maximum(np.var(values, axis=0, dtype=np.float64), 0.0)
            prior_mean = np.broadcast_to(global_mean[None, :], mean.shape).copy()
            prior_var = np.broadcast_to(global_var[None, :], var.shape).copy()
            if (
                sample_codes is not None
                and sample_names.size > 1
                and sample_prior_weight > 0.0
            ):
                valid_sample_codes = sample_codes[valid]
                sample_mean = np.zeros(
                    (int(sample_names.size), int(x.shape[1])), dtype=np.float64
                )
                sample_var = np.zeros_like(sample_mean)
                for sample_code in range(int(sample_names.size)):
                    sample_mask = valid_sample_codes == int(sample_code)
                    if not np.any(sample_mask):
                        sample_mean[sample_code] = global_mean
                        sample_var[sample_code] = global_var
                        continue
                    sample_values = values[sample_mask]
                    sample_mean[sample_code] = np.mean(
                        sample_values, axis=0, dtype=np.float64
                    )
                    sample_var[sample_code] = np.maximum(
                        np.var(sample_values, axis=0, dtype=np.float64), 0.0
                    )
                valid_region_samples = subregion_sample_codes >= 0
                if np.any(valid_region_samples):
                    sm = sample_mean[subregion_sample_codes[valid_region_samples]]
                    sv = sample_var[subregion_sample_codes[valid_region_samples]]
                    blended_mean = (
                        sample_prior_weight * sm
                        + (1.0 - sample_prior_weight) * global_mean[None, :]
                    )
                    blended_var = sample_prior_weight * (
                        sv + (sm - blended_mean) ** 2
                    ) + (1.0 - sample_prior_weight) * (
                        global_var[None, :] + (global_mean[None, :] - blended_mean) ** 2
                    )
                    prior_mean[valid_region_samples] = blended_mean
                    prior_var[valid_region_samples] = np.maximum(blended_var, 0.0)
                    sample_aware = True
            alpha = (counts / np.maximum(counts + tau, 1e-8))[:, None]
            shrunk_mean = alpha * mean + (1.0 - alpha) * prior_mean
            mixed_var = alpha * (var + (mean - shrunk_mean) ** 2) + (1.0 - alpha) * (
                prior_var + (prior_mean - shrunk_mean) ** 2
            )
            base = np.hstack(
                [shrunk_mean, h_weight * np.sqrt(np.maximum(mixed_var, 0.0))]
            )
            raw_to_shrunk = np.linalg.norm(base - raw_base, axis=1).astype(np.float32)
            alpha = alpha[:, 0]
        else:
            base = np.hstack([mean, h_weight * std])
        if normalized_mode == "mean_std_codebook":
            hist = _soft_codebook_histogram_embeddings(
                x[valid],
                region_ids,
                n_regions=n_regions,
                codebook_size=int(codebook_size),
                codebook_sample_size=int(codebook_sample_size),
                random_state=int(random_state),
            )
            latent = np.hstack([base, hist]).astype(np.float32)
        else:
            latent = base.astype(np.float32)
    elif normalized_mode == "mean_std_skew_count":
        sums_cu = np.zeros_like(sums)
        for dim in range(int(x.shape[1])):
            col = values[:, dim]
            sums_cu[:, dim] = np.bincount(
                region_ids, weights=col * col * col, minlength=n_regions
            )
        raw_third = sums_cu / denom
        central_third = (
            raw_third - 3.0 * mean * (sums_sq / denom) + 2.0 * mean * mean * mean
        )
        skew = central_third / np.maximum(std, 1e-8) ** 3
        reliability = (
            counts / np.maximum(counts + max(float(shrinkage_tau), 1e-8), 1e-8)
        )[:, None]
        count_features = np.hstack([np.log1p(counts)[:, None], reliability])
        latent = np.hstack([mean, std, skew, count_features]).astype(np.float32)
    elif normalized_mode == "mean_std_quantile":
        order = np.argsort(region_ids, kind="mergesort")
        sorted_region = region_ids[order]
        sorted_values = values[order]
        breaks = np.r_[
            0, np.flatnonzero(np.diff(sorted_region)) + 1, sorted_region.shape[0]
        ]
        quantiles = np.zeros((n_regions, int(x.shape[1]) * 5), dtype=np.float32)
        for left, right in zip(breaks[:-1], breaks[1:], strict=False):
            rid = int(sorted_region[left])
            quantiles[rid] = (
                np.quantile(
                    sorted_values[left:right], [0.10, 0.25, 0.50, 0.75, 0.90], axis=0
                )
                .reshape(-1)
                .astype(np.float32)
            )
        latent = np.hstack(
            [mean.astype(np.float32), std.astype(np.float32), quantiles]
        ).astype(np.float32)
    elif normalized_mode == "codebook_histogram":
        latent = _soft_codebook_histogram_embeddings(
            x[valid],
            region_ids,
            n_regions=n_regions,
            codebook_size=int(codebook_size),
            codebook_sample_size=int(codebook_sample_size),
            random_state=int(random_state),
        )
    else:
        raise AssertionError(
            f"Unhandled subregion latent embedding mode: {normalized_mode}"
        )
    latent[counts <= 0] = 0.0
    if normalized_mode in {"mean_std_shrunk", "mean_std_codebook"}:
        return _finish(
            latent,
            alpha=alpha,
            raw_to_shrunk=raw_to_shrunk,
            subregion_sample_ids=subregion_sample_ids,
            sample_aware=sample_aware,
        )
    return _finish(latent, subregion_sample_ids=subregion_sample_ids)


def _build_subregion_latent_embeddings(measures: list[SubregionMeasure]) -> np.ndarray:
    """Return feature-only summaries for compressed measures.

    This helper is retained for diagnostics and backwards-compatible internal
    uses. Primary pooled-latent clustering uses raw member-cell features through
    ``_build_subregion_latent_embeddings_from_members`` so the label step cannot
    inherit spatial effects from coordinate-aware measure compression.
    """
    if not measures:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(
        [_measure_feature_latent_embedding(measure) for measure in measures]
    ).astype(np.float32)


def _initialize_cluster_atoms(
    measures: list[SubregionMeasure],
    labels: np.ndarray,
    n_clusters: int,
    atoms_per_cluster: int,
    lambda_x: float,
    lambda_y: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    coord_dim = measures[0].canonical_coords.shape[1]
    feat_dim = measures[0].features.shape[1]
    atom_coords = np.zeros((n_clusters, atoms_per_cluster, coord_dim), dtype=np.float32)
    atom_features = np.zeros(
        (n_clusters, atoms_per_cluster, feat_dim), dtype=np.float32
    )
    betas = np.zeros((n_clusters, atoms_per_cluster), dtype=np.float32)
    sx = np.sqrt(max(float(lambda_x), 1e-12))
    sy = np.sqrt(max(float(lambda_y), 1e-12))

    for k in range(n_clusters):
        idx = np.flatnonzero(labels == k)
        if idx.size == 0:
            idx = np.asarray([int(rng.integers(len(measures)))], dtype=np.int32)
        u_pool = np.vstack([measures[r].canonical_coords for r in idx]).astype(
            np.float32
        )
        y_pool = np.vstack([measures[r].features for r in idx]).astype(np.float32)
        a_pool = np.concatenate([measures[r].weights for r in idx]).astype(np.float64)
        a_pool = _normalize_hist(a_pool)
        z_pool = np.hstack(
            [np.sqrt(lambda_x) * u_pool, np.sqrt(lambda_y) * y_pool]
        ).astype(np.float32)

        if z_pool.shape[0] >= atoms_per_cluster:
            km = KMeans(
                n_clusters=atoms_per_cluster,
                n_init=10,
                random_state=random_state + k,
            )
            local_labels = km.fit_predict(z_pool, sample_weight=a_pool)
            beta_k = np.zeros(atoms_per_cluster, dtype=np.float64)
            for ell in range(atoms_per_cluster):
                mask = local_labels == ell
                beta_k[ell] = a_pool[mask].sum()
                if beta_k[ell] <= 1e-12:
                    continue
                atom_coords[k, ell] = np.average(
                    u_pool[mask], axis=0, weights=a_pool[mask]
                ).astype(np.float32)
                atom_features[k, ell] = np.average(
                    y_pool[mask], axis=0, weights=a_pool[mask]
                ).astype(np.float32)
            dead = beta_k <= 1e-12
            if np.any(dead):
                centers = km.cluster_centers_
                atom_coords[k, dead] = (centers[dead, :coord_dim] / sx).astype(
                    np.float32
                )
                atom_features[k, dead] = (centers[dead, coord_dim:] / sy).astype(
                    np.float32
                )
            betas[k] = _normalize_hist(beta_k + 1e-6).astype(np.float32)
        else:
            reps = int(np.ceil(atoms_per_cluster / max(z_pool.shape[0], 1)))
            atom_coords[k] = np.tile(u_pool, (reps, 1))[:atoms_per_cluster]
            atom_features[k] = np.tile(y_pool, (reps, 1))[:atoms_per_cluster]
            betas[k] = np.full(
                atoms_per_cluster, 1.0 / atoms_per_cluster, dtype=np.float32
            )

    return atom_coords, atom_features, betas


def _estimate_cost_scales(
    measures: list[SubregionMeasure],
    max_points: int,
    random_state: int,
    compute_device: torch.device,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    u_all = np.vstack([m.canonical_coords for m in measures]).astype(np.float32)
    y_all = np.vstack([m.features for m in measures]).astype(np.float32)
    if u_all.shape[0] > max_points:
        idx = rng.choice(u_all.shape[0], size=max_points, replace=False)
        u_all = u_all[idx]
        y_all = y_all[idx]

    def _estimate_pairwise(x: np.ndarray, n_pairs: int = 200_000) -> float:
        n = x.shape[0]
        if n <= 1:
            return 1.0
        i = rng.integers(0, n, size=min(n_pairs, max(n * 8, 1024)))
        j = rng.integers(0, n, size=i.shape[0])
        if compute_device.type == "cpu":
            d2 = np.sum((x[i] - x[j]) ** 2, axis=1)
            d2 = d2[d2 > 0]
            return float(np.median(d2)) if d2.size else 1.0
        x_t = torch.as_tensor(x, dtype=torch.float32, device=compute_device)
        i_t = torch.as_tensor(i, dtype=torch.long, device=compute_device)
        j_t = torch.as_tensor(j, dtype=torch.long, device=compute_device)
        with torch.inference_mode():
            d2_t = torch.sum((x_t[i_t] - x_t[j_t]) ** 2, dim=1)
            d2_t = d2_t[d2_t > 0]
        if d2_t.numel() == 0:
            return 1.0
        return float(torch.median(d2_t).detach().cpu())

    sx = _estimate_pairwise(u_all)
    sy = _estimate_pairwise(y_all)
    return max(sx, 1e-6), max(sy, 1e-6)


def weighted_similarity_fit(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    allow_reflection: bool = True,
    allow_scale: bool = True,
    min_scale: float = 0.75,
    max_scale: float = 1.33,
) -> dict[str, np.ndarray | float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = _normalize_hist(w)
    xbar = np.sum(w[:, None] * x, axis=0)
    ybar = np.sum(w[:, None] * y, axis=0)
    x0 = x - xbar
    y0 = y - ybar
    h = x0.T @ (w[:, None] * y0)
    u, _, vt = np.linalg.svd(h)
    d = np.eye(x.shape[1], dtype=np.float64)
    if not allow_reflection and np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1.0
    r = u @ d @ vt
    if allow_scale:
        denom = float(np.sum(w * np.sum(x0**2, axis=1)))
        scale = float(np.trace(r.T @ h) / max(denom, 1e-12))
        scale = max(scale, 1e-12)
    else:
        scale = 1.0
    scale = float(np.clip(scale, min_scale, max_scale))
    t = ybar - scale * xbar @ r
    return {"R": r.astype(np.float64), "scale": scale, "t": t.astype(np.float64)}


def _assignment_cost_breakdown(
    *,
    measure: SubregionMeasure,
    gamma: np.ndarray,
    transform: dict[str, np.ndarray | float],
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    cost_scale_x: float,
    cost_scale_y: float,
    scale_penalty: float,
    shift_penalty: float,
) -> tuple[float, float, float]:
    gamma64 = np.asarray(gamma, dtype=np.float64)
    if gamma64.size == 0:
        return 0.0, 0.0, _transform_penalty(transform, scale_penalty, shift_penalty)
    u_aligned = apply_similarity(measure.canonical_coords, transform)
    cx = ot.dist(u_aligned, atom_coords, metric="sqeuclidean") / max(
        float(cost_scale_x), 1e-12
    )
    cy = ot.dist(measure.features, atom_features, metric="sqeuclidean") / max(
        float(cost_scale_y), 1e-12
    )
    geometry_cost = float(np.sum(gamma64 * (float(lambda_x) * cx)))
    feature_cost = float(np.sum(gamma64 * (float(lambda_y) * cy)))
    transform_penalty = _transform_penalty(transform, scale_penalty, shift_penalty)
    return geometry_cost, feature_cost, transform_penalty


def _compute_assigned_cost_breakdowns(
    *,
    measures: list[SubregionMeasure],
    labels: np.ndarray,
    plans: list[np.ndarray],
    transforms: list[dict[str, np.ndarray | float]],
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    cost_scale_x: float,
    cost_scale_y: float,
    scale_penalty: float,
    shift_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry_costs = np.zeros(len(measures), dtype=np.float32)
    feature_costs = np.zeros(len(measures), dtype=np.float32)
    transform_penalties = np.zeros(len(measures), dtype=np.float32)
    for r, measure in enumerate(measures):
        k = int(labels[r])
        geometry_cost, feature_cost, transform_penalty = _assignment_cost_breakdown(
            measure=measure,
            gamma=plans[r],
            transform=transforms[r],
            atom_coords=atom_coords[k],
            atom_features=atom_features[k],
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
        )
        geometry_costs[r] = float(geometry_cost)
        feature_costs[r] = float(feature_cost)
        transform_penalties[r] = float(transform_penalty)
    return geometry_costs, feature_costs, transform_penalties


def _transform_diagnostic_arrays(
    transforms: list[dict[str, np.ndarray | float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation_deg = np.zeros(len(transforms), dtype=np.float32)
    reflection = np.zeros(len(transforms), dtype=bool)
    scale = np.zeros(len(transforms), dtype=np.float32)
    translation_norm = np.zeros(len(transforms), dtype=np.float32)
    for idx, transform in enumerate(transforms):
        r = np.asarray(transform["R"], dtype=np.float64)
        t = np.asarray(transform["t"], dtype=np.float64)
        rotation_deg[idx] = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
        reflection[idx] = bool(np.linalg.det(r) < 0.0)
        scale[idx] = float(transform["scale"])
        translation_norm[idx] = float(np.linalg.norm(t))
    return rotation_deg, reflection, scale, translation_norm


def _transform_penalty(
    transform: dict[str, np.ndarray | float],
    scale_penalty: float,
    shift_penalty: float,
) -> float:
    scale = max(float(transform["scale"]), 1e-12)
    t = np.asarray(transform["t"], dtype=np.float64)
    return float(scale_penalty * (np.log(scale) ** 2) + shift_penalty * float(t @ t))


def _solve_semirelaxed_unbalanced_gpu(
    a: np.ndarray | torch.Tensor,
    beta: np.ndarray | torch.Tensor,
    c: np.ndarray | torch.Tensor,
    eps: float,
    rho: float,
    compute_device: torch.device,
) -> tuple[torch.Tensor, float, OTSolveDiagnostics]:
    regs = [max(float(eps), 1e-5)]
    regs.extend([regs[0] * 2.0, regs[0] * 4.0, regs[0] * 8.0])
    a_t = torch.as_tensor(a, dtype=torch.float32, device=compute_device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=compute_device)
    c_t = torch.as_tensor(c, dtype=torch.float32, device=compute_device)
    last_gamma: torch.Tensor | None = None
    last_objective = float("inf")
    last_reg = regs[0]
    for reg in regs:
        gamma, objective, converged, _ = sinkhorn_semirelaxed_unbalanced_log_torch(
            a_t,
            beta_t,
            c_t,
            eps=reg,
            rho=max(float(rho), 1e-6),
            num_iter=_sinkhorn_max_iter(),
            tol=_sinkhorn_tol(),
        )
        gamma_finite = bool(torch.isfinite(gamma).all().item())
        obj_val = float(objective.detach().item())
        obj_finite = bool(np.isfinite(obj_val))
        last_gamma = gamma
        last_objective = obj_val
        last_reg = reg
        if gamma_finite and obj_finite and converged:
            return (
                gamma,
                obj_val,
                OTSolveDiagnostics(
                    effective_eps=float(reg), used_fallback=not np.isclose(reg, regs[0])
                ),
            )
    if last_gamma is None or not bool(torch.isfinite(last_gamma).all().item()):
        raise FloatingPointError(
            "Unable to obtain a finite semi-relaxed unbalanced OT solution on GPU."
        )
    return (
        last_gamma,
        last_objective if np.isfinite(last_objective) else 1e12,
        OTSolveDiagnostics(
            effective_eps=float(last_reg),
            used_fallback=not np.isclose(last_reg, regs[0]),
        ),
    )


def _solve_semirelaxed_unbalanced(
    a: np.ndarray,
    beta: np.ndarray,
    c: np.ndarray,
    eps: float,
    rho: float,
    compute_device: torch.device,
) -> tuple[np.ndarray, float, OTSolveDiagnostics]:
    if compute_device.type == "cuda":
        gamma_t, objective, diag = _solve_semirelaxed_unbalanced_gpu(
            a, beta, c, eps=eps, rho=rho, compute_device=compute_device
        )
        return gamma_t.detach().cpu().numpy().astype(np.float64), float(objective), diag
    regs = [max(float(eps), 1e-5)]
    regs.extend([regs[0] * 2.0, regs[0] * 4.0, regs[0] * 8.0])
    last_gamma = None
    last_objective = 1e12
    last_reg = regs[0]
    a_backend = np.asarray(a, dtype=np.float64)
    beta_backend = np.asarray(beta, dtype=np.float64)
    for reg in regs:
        c_backend = np.asarray(c, dtype=np.float64)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gamma = ot.unbalanced.sinkhorn_unbalanced(
                a_backend,
                beta_backend,
                c_backend,
                reg=reg,
                reg_m=(float("inf"), max(float(rho), 1e-6)),
                method="sinkhorn_stabilized",
                reg_type="kl",
                numItermax=_cpu_sinkhorn_max_iter(),
                stopThr=_cpu_sinkhorn_tol(),
            )
            objective = ot.unbalanced.sinkhorn_unbalanced2(
                a_backend,
                beta_backend,
                c_backend,
                reg=reg,
                reg_m=(float("inf"), max(float(rho), 1e-6)),
                method="sinkhorn_stabilized",
                reg_type="kl",
                returnCost="total",
                numItermax=_cpu_sinkhorn_max_iter(),
                stopThr=_cpu_sinkhorn_tol(),
            )
        if torch.is_tensor(gamma):
            gamma = gamma.detach().cpu().numpy()
        gamma = np.asarray(gamma, dtype=np.float64)
        if torch.is_tensor(objective):
            objective = float(objective.detach().cpu())
        else:
            objective = float(objective)
        last_gamma = gamma
        last_objective = objective
        last_reg = reg
        numeric_warn = any(
            ("Numerical errors" in str(w.message))
            or ("did not converge" in str(w.message))
            for w in caught
        )
        if np.all(np.isfinite(gamma)) and np.isfinite(objective) and not numeric_warn:
            return (
                gamma,
                objective,
                OTSolveDiagnostics(
                    effective_eps=float(reg), used_fallback=not np.isclose(reg, regs[0])
                ),
            )
    if last_gamma is None or not np.all(np.isfinite(last_gamma)):
        raise FloatingPointError(
            "Unable to obtain a finite semi-relaxed unbalanced OT solution."
        )
    return (
        last_gamma,
        last_objective if np.isfinite(last_objective) else 1e12,
        OTSolveDiagnostics(
            effective_eps=float(last_reg),
            used_fallback=not np.isclose(last_reg, regs[0]),
        ),
    )


def aligned_semirelaxed_ot_to_cluster(
    u: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    beta: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    n_align_iter: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    compute_device: torch.device,
) -> tuple[
    float, np.ndarray, dict[str, np.ndarray | float], np.ndarray, OTSolveDiagnostics
]:
    u = np.asarray(u, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = _normalize_hist(a)
    beta = _normalize_hist(beta)
    transform: dict[str, np.ndarray | float] = {
        "R": np.eye(2, dtype=np.float64),
        "scale": 1.0,
        "t": np.zeros(2, dtype=np.float64),
    }
    atom_coords_t = (
        torch.as_tensor(atom_coords, dtype=torch.float64, device=compute_device)
        if compute_device.type != "cpu"
        else None
    )
    atom_features_t = (
        torch.as_tensor(atom_features, dtype=torch.float64, device=compute_device)
        if compute_device.type != "cpu"
        else None
    )
    y_t = (
        torch.as_tensor(y, dtype=torch.float64, device=compute_device)
        if compute_device.type != "cpu"
        else None
    )

    for _ in range(max(int(n_align_iter), 1)):
        u_aligned = apply_similarity(u, transform)
        c = _cluster_cost_matrix(
            u_aligned,
            y,
            atom_coords,
            atom_features,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            compute_device=compute_device,
            atom_coords_t=atom_coords_t,
            atom_features_t=atom_features_t,
            y_t=y_t,
        )
        gamma, _, _ = _solve_semirelaxed_unbalanced(
            a=a, beta=beta, c=c, eps=eps, rho=rho, compute_device=compute_device
        )
        row_mass = np.maximum(gamma.sum(axis=1), 1e-12)
        target_bary = (gamma @ atom_coords) / row_mass[:, None]
        transform = weighted_similarity_fit(
            u,
            target_bary,
            row_mass,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            min_scale=min_scale,
            max_scale=max_scale,
        )

    u_aligned = apply_similarity(u, transform)
    c = _cluster_cost_matrix(
        u_aligned,
        y,
        atom_coords,
        atom_features,
        cost_scale_x=cost_scale_x,
        cost_scale_y=cost_scale_y,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        compute_device=compute_device,
        atom_coords_t=atom_coords_t,
        atom_features_t=atom_features_t,
        y_t=y_t,
    )
    gamma, objective, solve_diag = _solve_semirelaxed_unbalanced(
        a=a, beta=beta, c=c, eps=eps, rho=rho, compute_device=compute_device
    )
    target_mass = np.asarray(gamma.sum(axis=0), dtype=np.float64)
    theta = _normalize_hist(target_mass)
    objective += _transform_penalty(
        transform,
        scale_penalty=scale_penalty,
        shift_penalty=shift_penalty,
    )
    if not np.isfinite(objective):
        objective = 1e12
    return (
        objective,
        gamma.astype(np.float32),
        transform,
        theta.astype(np.float32),
        solve_diag,
    )


def _batched_weighted_similarity_fit_torch(
    x: torch.Tensor,  # (K, m, 2)
    y: torch.Tensor,  # (K, m, 2)
    w: torch.Tensor,  # (K, m)
    allow_reflection: bool,
    allow_scale: bool,
    min_scale: float,
    max_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    w_norm = w / w_sum
    xbar = (w_norm.unsqueeze(-1) * x).sum(dim=-2)  # (K, 2)
    ybar = (w_norm.unsqueeze(-1) * y).sum(dim=-2)  # (K, 2)
    x0 = x - xbar.unsqueeze(-2)
    y0 = y - ybar.unsqueeze(-2)
    h = x0.transpose(-1, -2) @ (w_norm.unsqueeze(-1) * y0)  # (K, 2, 2)
    u, _s, vt = torch.linalg.svd(h)
    d = (
        torch.eye(2, device=x.device, dtype=x.dtype)
        .expand(h.shape[0], 2, 2)
        .contiguous()
    )
    if not allow_reflection:
        det = torch.linalg.det(u @ vt)
        neg = det < 0
        if torch.any(neg):
            d = d.clone()
            d[:, -1, -1] = torch.where(
                neg, torch.full_like(det, -1.0), torch.full_like(det, 1.0)
            )
    r = u @ d @ vt  # (K, 2, 2)
    if allow_scale:
        denom = (w_norm * (x0**2).sum(dim=-1)).sum(dim=-1).clamp_min(1e-12)  # (K,)
        trace = torch.einsum("kij,kij->k", r, h)
        scale = (trace / denom).clamp_min(1e-12)
    else:
        scale = torch.ones(h.shape[0], device=x.device, dtype=x.dtype)
    scale = scale.clamp(min=float(min_scale), max=float(max_scale))
    xbar_r = torch.einsum("ki,kij->kj", xbar, r)  # (K, 2)
    t = ybar - scale.unsqueeze(-1) * xbar_r
    return r, scale, t


def _apply_similarity_batched_torch(
    x: torch.Tensor, r: torch.Tensor, scale: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    # x: (K, m, 2), r: (K, 2, 2), scale: (K,), t: (K, 2)
    return scale.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
        "kmi,kij->kmj", x, r
    ) + t.unsqueeze(-2)


def _aligned_semirelaxed_ot_costs_all_clusters_gpu(
    u: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    betas: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    n_align_iter: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    compute_device: torch.device,
    *,
    atom_coords_t: torch.Tensor | None = None,
    atom_features_t: torch.Tensor | None = None,
    betas_t: torch.Tensor | None = None,
) -> np.ndarray:
    """All-K batched aligned semi-relaxed OT (costs only). Returns shape (K,) numpy."""
    dtype = torch.float32
    dev = compute_device
    # tensors
    u_t = torch.as_tensor(u, dtype=dtype, device=dev)  # (m, 2)
    y_t = torch.as_tensor(y, dtype=dtype, device=dev)  # (m, f)
    a_vec = torch.as_tensor(_normalize_hist(a), dtype=dtype, device=dev)  # (m,)
    if atom_coords_t is None:
        atom_coords_t = torch.as_tensor(atom_coords, dtype=dtype, device=dev)
    if atom_features_t is None:
        atom_features_t = torch.as_tensor(atom_features, dtype=dtype, device=dev)
    if betas_t is None:
        betas_np = np.stack([_normalize_hist(b) for b in betas], axis=0)
        betas_t = torch.as_tensor(betas_np, dtype=dtype, device=dev)
    K = atom_coords_t.shape[0]
    # Broadcast u, y to (K, m, *)
    u_kb = u_t.unsqueeze(0).expand(K, -1, -1).contiguous()
    y_kb = y_t.unsqueeze(0).expand(K, -1, -1).contiguous()
    a_kb = a_vec.unsqueeze(0).expand(K, -1).contiguous()

    r = torch.eye(2, device=dev, dtype=dtype).unsqueeze(0).expand(K, 2, 2).contiguous()
    scale = torch.ones(K, device=dev, dtype=dtype)
    t = torch.zeros(K, 2, device=dev, dtype=dtype)

    sx = max(float(cost_scale_x), 1e-12)
    sy = max(float(cost_scale_y), 1e-12)
    cy_full = torch.cdist(y_kb, atom_features_t, p=2).pow(2) / sy
    cy_scaled = float(lambda_y) * cy_full  # (K, m, p) — fixed over align iters

    eps_base = max(float(eps), 1e-5)
    reg_schedule = [eps_base, 2.0 * eps_base, 4.0 * eps_base, 8.0 * eps_base]

    def _solve_once(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        last_gamma = None
        last_obj = None
        last_reg = reg_schedule[0]
        for reg in reg_schedule:
            gamma, obj, converged, _ = sinkhorn_semirelaxed_unbalanced_log_torch(
                a_kb,
                betas_t,
                cost,
                eps=reg,
                rho=max(float(rho), 1e-6),
                num_iter=_sinkhorn_max_iter(),
                tol=_sinkhorn_tol(),
            )
            finite = torch.isfinite(gamma).all() and torch.isfinite(obj).all()
            last_gamma = gamma
            last_obj = obj
            last_reg = reg
            if bool(finite.item()) and converged:
                return gamma, obj, float(reg)
        if last_gamma is None:
            raise FloatingPointError(
                "Unable to obtain finite batched semirelaxed OT on GPU."
            )
        return last_gamma, last_obj, float(last_reg)

    for _ in range(max(int(n_align_iter), 1)):
        u_aligned = _apply_similarity_batched_torch(u_kb, r, scale, t)  # (K, m, 2)
        cx = torch.cdist(u_aligned, atom_coords_t, p=2).pow(2) / sx  # (K, m, p)
        cost = float(lambda_x) * cx + cy_scaled
        gamma, _, _ = _solve_once(cost)
        row_mass = gamma.sum(dim=-1).clamp_min(1e-12)  # (K, m)
        target_bary = torch.einsum(
            "kmp,kpd->kmd", gamma, atom_coords_t
        ) / row_mass.unsqueeze(-1)
        r, scale, t = _batched_weighted_similarity_fit_torch(
            u_kb,
            target_bary,
            row_mass,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            min_scale=min_scale,
            max_scale=max_scale,
        )

    u_aligned = _apply_similarity_batched_torch(u_kb, r, scale, t)
    cx = torch.cdist(u_aligned, atom_coords_t, p=2).pow(2) / sx
    cost = float(lambda_x) * cx + cy_scaled
    _, obj, _ = _solve_once(cost)

    # transform penalty: scale_penalty * log(scale)^2 + shift_penalty * (t @ t)
    penalty = float(scale_penalty) * torch.log(scale.clamp_min(1e-12)).pow(2) + float(
        shift_penalty
    ) * (t * t).sum(dim=-1)
    total = obj + penalty
    total = torch.where(torch.isfinite(total), total, torch.full_like(total, 1e12))
    return total.detach().cpu().numpy().astype(np.float64)


def _pack_measures_padded(
    measures: list[SubregionMeasure],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad all subregion measures to shape (R, m_max, *). Zero-mass rows mark padding."""
    R = len(measures)
    if R == 0:
        raise ValueError("no subregion measures to pack")
    m_max = max(int(m.canonical_coords.shape[0]) for m in measures)
    coord_dim = int(measures[0].canonical_coords.shape[1])
    feat_dim = int(measures[0].features.shape[1])
    u = np.zeros((R, m_max, coord_dim), dtype=np.float32)
    y = np.zeros((R, m_max, feat_dim), dtype=np.float32)
    a = np.zeros((R, m_max), dtype=np.float32)
    m_r = np.zeros(R, dtype=np.int32)
    for r, meas in enumerate(measures):
        mr = int(meas.canonical_coords.shape[0])
        m_r[r] = mr
        u[r, :mr] = np.asarray(meas.canonical_coords, dtype=np.float32)
        y[r, :mr] = np.asarray(meas.features, dtype=np.float32)
        a[r, :mr] = _normalize_hist(np.asarray(meas.weights)).astype(np.float32)
    return (
        torch.as_tensor(u, dtype=dtype, device=device),
        torch.as_tensor(y, dtype=dtype, device=device),
        torch.as_tensor(a, dtype=dtype, device=device),
        torch.as_tensor(m_r, dtype=torch.int32, device=device),
    )


def _gpu_assignment_subregion_batch_size(
    *,
    n_subregions: int,
    n_clusters: int,
    measure_size: int,
    support_size: int,
    feature_dim: int,
    device: torch.device,
) -> int:
    requested = _env_int("SPATIAL_OT_GPU_ASSIGNMENT_SUBREGION_BATCH_SIZE", 0)
    if requested > 0:
        return max(1, min(int(n_subregions), int(requested)))
    bytes_per_float = 4
    # Candidate evaluation materializes broadcast feature blocks with shape
    # (R, K, m, feature_dim) plus several (R, K, m, p) work arrays. Include m in
    # the estimate; otherwise large cohorts can OOM before chunking.
    m = max(int(measure_size), 1)
    k = max(int(n_clusters), 1)
    p = max(int(support_size), 1)
    f = max(int(feature_dim), 1)
    per_subregion = bytes_per_float * max(
        k * m * f * 2 + k * p * f + k * m * p * 12 + k * m * 8,
        1,
    )
    budget = max(_cuda_target_bytes(device=device) // 4, 1 << 26)
    auto_batch = int(max(1, budget // max(per_subregion, 1)))
    return max(1, min(int(n_subregions), auto_batch))


def _batched_weighted_similarity_fit_rk_torch(
    x: torch.Tensor,  # (R, K, m, 2)  --- x is u replicated across K
    y: torch.Tensor,  # (R, K, m, 2)  --- barycentric target
    w: torch.Tensor,  # (R, K, m)
    allow_reflection: bool,
    allow_scale: bool,
    min_scale: float,
    max_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    w_norm = w / w_sum
    xbar = (w_norm.unsqueeze(-1) * x).sum(dim=-2)  # (R, K, 2)
    ybar = (w_norm.unsqueeze(-1) * y).sum(dim=-2)  # (R, K, 2)
    x0 = x - xbar.unsqueeze(-2)
    y0 = y - ybar.unsqueeze(-2)
    # h[r,k] = x0^T @ (diag(w_norm) @ y0)
    h = torch.einsum("rkmi,rkmj->rkij", x0, w_norm.unsqueeze(-1) * y0)
    # batched 2x2 SVD over (R, K, 2, 2)
    u_svd, _s, vt = torch.linalg.svd(h)
    if not allow_reflection:
        det = torch.linalg.det(u_svd @ vt)
        d = (
            torch.eye(2, device=x.device, dtype=x.dtype)
            .expand(*h.shape[:-2], 2, 2)
            .contiguous()
        )
        neg = det < 0
        if torch.any(neg):
            d = d.clone()
            # set last diagonal entry to -1 where det<0
            fix = torch.where(
                neg, torch.full_like(det, -1.0), torch.full_like(det, 1.0)
            )
            d[..., -1, -1] = fix
        r_mat = u_svd @ d @ vt
    else:
        r_mat = u_svd @ vt
    if allow_scale:
        denom = (w_norm * (x0**2).sum(dim=-1)).sum(dim=-1).clamp_min(1e-12)
        trace = torch.einsum("rkij,rkij->rk", r_mat, h)
        scale = (trace / denom).clamp_min(1e-12)
    else:
        scale = torch.ones(h.shape[:-2], device=x.device, dtype=x.dtype)
    scale = scale.clamp(min=float(min_scale), max=float(max_scale))
    xbar_r = torch.einsum("rki,rkij->rkj", xbar, r_mat)
    t = ybar - scale.unsqueeze(-1) * xbar_r
    return r_mat, scale, t


def _apply_similarity_rk_torch(
    x: torch.Tensor, r: torch.Tensor, scale: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    return scale.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
        "rkmi,rkij->rkmj", x, r
    ) + t.unsqueeze(-2)


def _compute_assignment_costs_rk_gpu(
    measures: list[SubregionMeasure],
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    betas: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    compute_device: torch.device,
    *,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fully batched (R*K, m, *) aligned semi-relaxed OT on GPU.

    This amortizes kernel-launch overhead across all (subregion, cluster) pairs.
    Returns costs of shape (R, K) as float32 numpy.
    """
    dtype = torch.float32
    dev = compute_device
    u_rm, y_rm, a_rm, _m_r = _pack_measures_padded(measures, dtype=dtype, device=dev)
    R = u_rm.shape[0]
    m = u_rm.shape[1]
    K = int(atom_coords.shape[0])
    p = int(atom_coords.shape[1])

    atom_coords_t = torch.as_tensor(atom_coords, dtype=dtype, device=dev)  # (K, p, 2)
    atom_features_t = torch.as_tensor(
        atom_features, dtype=dtype, device=dev
    )  # (K, p, f)
    betas_np = np.stack([_normalize_hist(b) for b in betas], axis=0)
    beta_k = torch.as_tensor(betas_np, dtype=dtype, device=dev)  # (K, p)

    # Broadcast to (R, K, *)
    u_rkm = u_rm.unsqueeze(1).expand(R, K, m, u_rm.shape[-1]).contiguous()
    y_rkm = y_rm.unsqueeze(1).expand(R, K, m, y_rm.shape[-1]).contiguous()
    a_rkm = a_rm.unsqueeze(1).expand(R, K, m).contiguous()
    atom_coords_rk = (
        atom_coords_t.unsqueeze(0).expand(R, K, p, atom_coords_t.shape[-1]).contiguous()
    )
    atom_features_rk = (
        atom_features_t.unsqueeze(0)
        .expand(R, K, p, atom_features_t.shape[-1])
        .contiguous()
    )
    beta_rk = beta_k.unsqueeze(0).expand(R, K, p).contiguous()

    sx = max(float(cost_scale_x), 1e-12)
    sy = max(float(cost_scale_y), 1e-12)
    cy_full = (
        torch.cdist(
            y_rkm.reshape(R * K, m, y_rm.shape[-1]),
            atom_features_rk.reshape(R * K, p, y_rm.shape[-1]),
            p=2,
        )
        .pow(2)
        .reshape(R, K, m, p)
        / sy
    )
    cy_scaled = float(lambda_y) * cy_full

    r_mat = torch.eye(2, device=dev, dtype=dtype).expand(R, K, 2, 2).contiguous()
    scale_t = torch.ones(R, K, device=dev, dtype=dtype)
    t_t = torch.zeros(R, K, 2, device=dev, dtype=dtype)

    eps_base = max(float(eps), 1e-5)
    reg_schedule = (eps_base, 2.0 * eps_base, 4.0 * eps_base, 8.0 * eps_base)
    rho_val = max(float(rho), 1e-6)

    def _solve(
        cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # cost shape: (R, K, m, p). Flatten batch dims for Sinkhorn.
        flat_cost = cost.reshape(R * K, m, p)
        flat_a = a_rkm.reshape(R * K, m)
        flat_b = beta_rk.reshape(R * K, p)
        last_gamma = None
        last_obj = None
        eff_reg = torch.full((R, K), reg_schedule[0], dtype=dtype, device=dev)
        found_finite = torch.zeros((R, K), dtype=torch.bool, device=dev)
        converged_solution = torch.zeros((R, K), dtype=torch.bool, device=dev)
        for reg in reg_schedule:
            gamma, obj, converged, _err = sinkhorn_semirelaxed_unbalanced_log_torch(
                flat_a,
                flat_b,
                flat_cost,
                eps=reg,
                rho=rho_val,
                num_iter=_sinkhorn_max_iter(),
                tol=_sinkhorn_tol(),
            )
            gamma = gamma.reshape(R, K, m, p)
            obj = obj.reshape(R, K)
            finite = torch.isfinite(gamma).all(dim=(-1, -2)) & torch.isfinite(obj)
            if last_gamma is None:
                last_gamma = torch.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
                last_obj = torch.where(
                    torch.isfinite(obj), obj, torch.full_like(obj, 1e12)
                )
                eff_reg = torch.full_like(obj, reg)
            update = finite if bool(converged) else finite & ~converged_solution
            if update.any():
                last_gamma = torch.where(
                    update.unsqueeze(-1).unsqueeze(-1), gamma, last_gamma
                )
                last_obj = torch.where(update, obj, last_obj)
                eff_reg = torch.where(update, torch.full_like(eff_reg, reg), eff_reg)
                found_finite = found_finite | update
                if bool(converged):
                    converged_solution = converged_solution | update
            if bool(converged) and bool(finite.all().item()):
                break
        if (
            last_gamma is None
            or last_obj is None
            or not bool(found_finite.any().item())
        ):
            raise FloatingPointError("GPU batched Sinkhorn failed")
        used_fallback = (~converged_solution) | ~torch.isclose(
            eff_reg, torch.full_like(eff_reg, eps_base)
        )
        return last_gamma, last_obj, eff_reg, used_fallback

    identity_r = torch.eye(2, device=dev, dtype=dtype).expand(R, K, 2, 2).contiguous()
    for _ in range(max(int(align_iters), 1)):
        u_aligned = _apply_similarity_rk_torch(
            u_rkm, r_mat, scale_t, t_t
        )  # (R, K, m, 2)
        cx = (
            torch.cdist(
                u_aligned.reshape(R * K, m, u_aligned.shape[-1]),
                atom_coords_rk.reshape(R * K, p, u_aligned.shape[-1]),
                p=2,
            )
            .pow(2)
            .reshape(R, K, m, p)
            / sx
        )
        cost = float(lambda_x) * cx + cy_scaled
        gamma, _obj, _reg, _used_fallback = _solve(cost)
        gamma = torch.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
        row_mass = gamma.sum(dim=-1).clamp_min(1e-12)  # (R, K, m)
        target_bary = torch.einsum(
            "rkmp,rkpd->rkmd", gamma, atom_coords_rk
        ) / row_mass.unsqueeze(-1)
        r_new, scale_new, t_new = _batched_weighted_similarity_fit_rk_torch(
            u_rkm,
            target_bary,
            row_mass,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            min_scale=min_scale,
            max_scale=max_scale,
        )
        # Reset any NaN/Inf transforms to identity so we don't poison subsequent iters.
        bad = (
            ~torch.isfinite(r_new).all(dim=(-1, -2))
            | ~torch.isfinite(scale_new)
            | ~torch.isfinite(t_new).all(dim=-1)
        )
        if bool(bad.any().item()):
            r_new = torch.where(bad.unsqueeze(-1).unsqueeze(-1), identity_r, r_new)
            scale_new = torch.where(bad, torch.ones_like(scale_new), scale_new)
            t_new = torch.where(bad.unsqueeze(-1), torch.zeros_like(t_new), t_new)
        r_mat, scale_t, t_t = r_new, scale_new, t_new

    u_aligned = _apply_similarity_rk_torch(u_rkm, r_mat, scale_t, t_t)
    cx = (
        torch.cdist(
            u_aligned.reshape(R * K, m, u_aligned.shape[-1]),
            atom_coords_rk.reshape(R * K, p, u_aligned.shape[-1]),
            p=2,
        )
        .pow(2)
        .reshape(R, K, m, p)
        / sx
    )
    cost = float(lambda_x) * cx + cy_scaled
    _gamma, obj, eff_reg, used_fallback = _solve(cost)

    penalty = float(scale_penalty) * torch.log(scale_t.clamp_min(1e-12)).pow(2) + float(
        shift_penalty
    ) * (t_t * t_t).sum(dim=-1)
    total = obj + penalty
    total = torch.where(torch.isfinite(total), total, torch.full_like(total, 1e12))
    total_np = total.detach().cpu().numpy().astype(np.float32)
    if not return_diagnostics:
        return total_np
    eff_reg_np = eff_reg.detach().cpu().numpy().astype(np.float32)
    used_fallback_np = used_fallback.detach().cpu().numpy().astype(bool)
    return total_np, eff_reg_np, used_fallback_np.astype(bool)


def _compute_assigned_artifacts_r_gpu(
    measures: list[SubregionMeasure],
    labels: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    betas: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    compute_device: torch.device,
) -> tuple[
    list[np.ndarray],
    list[dict[str, np.ndarray | float]],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fully batched (over R) aligned semirelaxed OT on GPU, returning per-subregion artifacts."""
    dtype = torch.float32
    dev = compute_device
    R = len(measures)
    label_idx = torch.as_tensor(
        np.asarray(labels, dtype=np.int64), dtype=torch.long, device=dev
    )
    u_rm, y_rm, a_rm, m_r = _pack_measures_padded(measures, dtype=dtype, device=dev)
    atom_coords_t = torch.as_tensor(atom_coords, dtype=dtype, device=dev)  # (K, p, 2)
    atom_features_t = torch.as_tensor(
        atom_features, dtype=dtype, device=dev
    )  # (K, p, f)
    betas_np = np.stack([_normalize_hist(b) for b in betas], axis=0)
    beta_k = torch.as_tensor(betas_np, dtype=dtype, device=dev)  # (K, p)

    # Gather per-r atoms using labels
    atom_coords_r = atom_coords_t[label_idx]  # (R, p, 2)
    atom_features_r = atom_features_t[label_idx]  # (R, p, f)
    beta_r = beta_k[label_idx]  # (R, p)

    sx = max(float(cost_scale_x), 1e-12)
    sy = max(float(cost_scale_y), 1e-12)
    cy_full = torch.cdist(y_rm, atom_features_r, p=2).pow(2) / sy
    cy_scaled = float(lambda_y) * cy_full

    r_mat = torch.eye(2, device=dev, dtype=dtype).expand(R, 2, 2).contiguous()
    scale_t = torch.ones(R, device=dev, dtype=dtype)
    t_t = torch.zeros(R, 2, device=dev, dtype=dtype)

    eps_base = max(float(eps), 1e-5)
    reg_schedule = (eps_base, 2.0 * eps_base, 4.0 * eps_base, 8.0 * eps_base)
    rho_val = max(float(rho), 1e-6)

    def _solve(
        cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        last_gamma = None
        last_obj = None
        eff_reg = torch.full((R,), reg_schedule[0], dtype=dtype, device=dev)
        found_finite = torch.zeros((R,), dtype=torch.bool, device=dev)
        converged_solution = torch.zeros((R,), dtype=torch.bool, device=dev)
        for reg in reg_schedule:
            gamma, obj, converged, _err = sinkhorn_semirelaxed_unbalanced_log_torch(
                a_rm,
                beta_r,
                cost,
                eps=reg,
                rho=rho_val,
                num_iter=_sinkhorn_max_iter(),
                tol=_sinkhorn_tol(),
            )
            finite = torch.isfinite(gamma).all(dim=(-1, -2)) & torch.isfinite(obj)
            if last_gamma is None:
                last_gamma = torch.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
                last_obj = torch.where(
                    torch.isfinite(obj), obj, torch.full_like(obj, 1e12)
                )
                eff_reg = torch.full_like(obj, reg)
            update = finite if bool(converged) else finite & ~converged_solution
            if update.any():
                last_gamma = torch.where(
                    update.unsqueeze(-1).unsqueeze(-1), gamma, last_gamma
                )
                last_obj = torch.where(update, obj, last_obj)
                eff_reg = torch.where(update, torch.full_like(eff_reg, reg), eff_reg)
                found_finite = found_finite | update
                if bool(converged):
                    converged_solution = converged_solution | update
            if bool(converged) and bool(finite.all().item()):
                break
        if (
            last_gamma is None
            or last_obj is None
            or not bool(found_finite.any().item())
        ):
            raise FloatingPointError("GPU batched-R Sinkhorn failed")
        used_fallback = (~converged_solution) | ~torch.isclose(
            eff_reg, torch.full_like(eff_reg, eps_base)
        )
        return last_gamma, last_obj, eff_reg, used_fallback

    # R-wise barycentric alignment
    identity_r = torch.eye(2, device=dev, dtype=dtype).expand(R, 2, 2).contiguous()
    for _ in range(max(int(align_iters), 1)):
        u_aligned = scale_t.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
            "rmi,rij->rmj", u_rm, r_mat
        ) + t_t.unsqueeze(-2)
        cx = torch.cdist(u_aligned, atom_coords_r, p=2).pow(2) / sx
        cost = float(lambda_x) * cx + cy_scaled
        gamma, _obj, _reg, _used_fallback = _solve(cost)
        gamma = torch.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
        row_mass = gamma.sum(dim=-1).clamp_min(1e-12)
        target_bary = torch.einsum(
            "rmp,rpd->rmd", gamma, atom_coords_r
        ) / row_mass.unsqueeze(-1)
        # Batched weighted similarity fit over R (2D).
        w_sum = row_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        w_norm = row_mass / w_sum
        xbar = (w_norm.unsqueeze(-1) * u_rm).sum(dim=-2)
        ybar = (w_norm.unsqueeze(-1) * target_bary).sum(dim=-2)
        x0 = u_rm - xbar.unsqueeze(-2)
        y0 = target_bary - ybar.unsqueeze(-2)
        h = torch.einsum("rmi,rmj->rij", x0, w_norm.unsqueeze(-1) * y0)
        u_svd, _s, vt = torch.linalg.svd(h)
        if not allow_reflection:
            det = torch.linalg.det(u_svd @ vt)
            d = torch.eye(2, device=dev, dtype=dtype).expand(R, 2, 2).contiguous()
            neg = det < 0
            if torch.any(neg):
                d = d.clone()
                fix = torch.where(
                    neg, torch.full_like(det, -1.0), torch.full_like(det, 1.0)
                )
                d[..., -1, -1] = fix
            r_new = u_svd @ d @ vt
        else:
            r_new = u_svd @ vt
        if allow_scale:
            denom = (w_norm * (x0**2).sum(dim=-1)).sum(dim=-1).clamp_min(1e-12)
            trace = torch.einsum("rij,rij->r", r_new, h)
            scale_new = (trace / denom).clamp_min(1e-12)
        else:
            scale_new = torch.ones(R, device=dev, dtype=dtype)
        scale_new = scale_new.clamp(min=float(min_scale), max=float(max_scale))
        xbar_r = torch.einsum("ri,rij->rj", xbar, r_new)
        t_new = ybar - scale_new.unsqueeze(-1) * xbar_r
        bad = (
            ~torch.isfinite(r_new).all(dim=(-1, -2))
            | ~torch.isfinite(scale_new)
            | ~torch.isfinite(t_new).all(dim=-1)
        )
        if bool(bad.any().item()):
            r_new = torch.where(bad.unsqueeze(-1).unsqueeze(-1), identity_r, r_new)
            scale_new = torch.where(bad, torch.ones_like(scale_new), scale_new)
            t_new = torch.where(bad.unsqueeze(-1), torch.zeros_like(t_new), t_new)
        r_mat, scale_t, t_t = r_new, scale_new, t_new

    # Final solve
    u_aligned = scale_t.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
        "rmi,rij->rmj", u_rm, r_mat
    ) + t_t.unsqueeze(-2)
    cx = torch.cdist(u_aligned, atom_coords_r, p=2).pow(2) / sx
    cost = float(lambda_x) * cx + cy_scaled
    gamma, obj, eff_reg, used_fallback_t = _solve(cost)
    gamma = torch.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
    col_mass = gamma.sum(dim=-2)  # (R, p)
    theta = col_mass / col_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    penalty = float(scale_penalty) * torch.log(scale_t.clamp_min(1e-12)).pow(2) + float(
        shift_penalty
    ) * (t_t * t_t).sum(dim=-1)
    total = obj + penalty
    total = torch.where(torch.isfinite(total), total, torch.full_like(total, 1e12))

    # Move to CPU once
    gamma_np = gamma.detach().cpu().numpy()
    r_np = r_mat.detach().cpu().numpy()
    scale_np = scale_t.detach().cpu().numpy()
    t_np = t_t.detach().cpu().numpy()
    theta_np = theta.detach().cpu().numpy()
    total_np = total.detach().cpu().numpy().astype(np.float32)
    eff_reg_np = eff_reg.detach().cpu().numpy().astype(np.float32)
    m_r_np = m_r.detach().cpu().numpy()

    plans: list[np.ndarray] = []
    transforms: list[dict[str, np.ndarray | float]] = []
    thetas: list[np.ndarray] = []
    used_fallback_np = used_fallback_t.detach().cpu().numpy().astype(bool)
    for r in range(R):
        mr = int(m_r_np[r])
        plans.append(gamma_np[r, :mr, :].astype(np.float32))
        transforms.append(
            {
                "R": r_np[r].astype(np.float64),
                "scale": float(scale_np[r]),
                "t": t_np[r].astype(np.float64),
            }
        )
        thetas.append(theta_np[r].astype(np.float32))
    return (
        plans,
        transforms,
        thetas,
        total_np,
        eff_reg_np,
        used_fallback_np.astype(bool),
    )


def _ensure_nonempty_clusters(
    labels: np.ndarray, costs: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    return _ensure_minimum_cluster_size(
        labels, costs, n_clusters=n_clusters, min_subregions_per_cluster=1
    )


def _ensure_minimum_cluster_size(
    labels: np.ndarray,
    costs: np.ndarray,
    *,
    n_clusters: int,
    min_subregions_per_cluster: int,
) -> tuple[np.ndarray, np.ndarray]:
    return repair_labels_to_minimum_size(
        labels,
        costs,
        n_clusters=int(n_clusters),
        min_cluster_size=int(min_subregions_per_cluster),
    )


def _cluster_precomputed_transport_distances(
    distances: np.ndarray,
    *,
    n_clusters: int,
    min_subregions_per_cluster: int,
) -> dict[str, object]:
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("transport distance matrix must be square.")
    n = int(d.shape[0])
    k = int(n_clusters)
    if k < 1 or k > n:
        raise ValueError(f"n_clusters={k} is invalid for {n} subregions.")
    if k == 1:
        labels = np.zeros(n, dtype=np.int32)
    else:
        try:
            model = AgglomerativeClustering(
                n_clusters=k,
                metric="precomputed",
                linkage="average",
            )
        except TypeError:
            model = AgglomerativeClustering(
                n_clusters=k,
                affinity="precomputed",
                linkage="average",
            )
        labels = np.asarray(model.fit_predict(d), dtype=np.int32)

    def _medoids_for(current_labels: np.ndarray) -> np.ndarray:
        medoids = np.zeros(k, dtype=np.int32)
        for cluster_id in range(k):
            idx = np.flatnonzero(current_labels == cluster_id)
            if idx.size == 0:
                medoids[cluster_id] = int(np.argmin(np.mean(d, axis=1)))
                continue
            local = d[np.ix_(idx, idx)]
            medoids[cluster_id] = int(idx[int(np.argmin(np.mean(local, axis=1)))])
        return medoids

    medoids = _medoids_for(labels)
    costs = d[:, medoids].astype(np.float32)
    argmin_labels = np.argmin(costs, axis=1).astype(np.int32)
    labels, forced_mask = _ensure_minimum_cluster_size(
        labels,
        costs,
        n_clusters=k,
        min_subregions_per_cluster=int(min_subregions_per_cluster),
    )
    medoids = _medoids_for(labels)
    costs = d[:, medoids].astype(np.float32)
    argmin_labels = np.argmin(costs, axis=1).astype(np.int32)
    inertia = float(np.sum(costs[np.arange(n), labels]))
    return {
        "labels": labels.astype(np.int32),
        "argmin_labels": argmin_labels.astype(np.int32),
        "costs": costs.astype(np.float32),
        "forced_label_mask": forced_mask.astype(bool),
        "medoid_indices": medoids.astype(np.int32),
        "inertia": inertia,
    }


def _compute_assignment_costs(
    measures: list[SubregionMeasure],
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    betas: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    compute_device: torch.device,
    *,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_subregions = len(measures)
    n_clusters = atom_coords.shape[0]
    costs = np.zeros((n_subregions, n_clusters), dtype=np.float64)
    effective_eps = (
        np.zeros((n_subregions, n_clusters), dtype=np.float32)
        if return_diagnostics
        else None
    )
    used_fallback = (
        np.zeros((n_subregions, n_clusters), dtype=bool) if return_diagnostics else None
    )

    if compute_device.type == "cuda":
        # Fully batched (R*K) GPU solve — amortizes kernel launch latency.
        batch_size = _gpu_assignment_subregion_batch_size(
            n_subregions=n_subregions,
            n_clusters=n_clusters,
            measure_size=max(
                (int(measure.canonical_coords.shape[0]) for measure in measures),
                default=1,
            ),
            support_size=int(atom_coords.shape[1]),
            feature_dim=int(atom_features.shape[2]),
            device=compute_device,
        )
        if n_subregions > batch_size:
            cost_chunks: list[np.ndarray] = []
            eps_chunks: list[np.ndarray] = []
            fallback_chunks: list[np.ndarray] = []
            for start in range(0, n_subregions, batch_size):
                chunk_result = _compute_assignment_costs_rk_gpu(
                    measures=measures[start : start + batch_size],
                    atom_coords=atom_coords,
                    atom_features=atom_features,
                    betas=betas,
                    lambda_x=lambda_x,
                    lambda_y=lambda_y,
                    eps=eps,
                    rho=rho,
                    align_iters=align_iters,
                    allow_reflection=allow_reflection,
                    allow_scale=allow_scale,
                    cost_scale_x=cost_scale_x,
                    cost_scale_y=cost_scale_y,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    scale_penalty=scale_penalty,
                    shift_penalty=shift_penalty,
                    compute_device=compute_device,
                    return_diagnostics=return_diagnostics,
                )
                if return_diagnostics:
                    chunk_costs, chunk_eps, chunk_fallback = chunk_result
                    cost_chunks.append(np.asarray(chunk_costs, dtype=np.float32))
                    eps_chunks.append(np.asarray(chunk_eps, dtype=np.float32))
                    fallback_chunks.append(np.asarray(chunk_fallback, dtype=bool))
                else:
                    cost_chunks.append(np.asarray(chunk_result, dtype=np.float32))
            costs_np = np.vstack(cost_chunks).astype(np.float32)
            if not return_diagnostics:
                return np.clip(costs_np, -1e12, 1e12).astype(np.float32)
            return (
                np.clip(costs_np, -1e12, 1e12).astype(np.float32),
                np.vstack(eps_chunks).astype(np.float32),
                np.vstack(fallback_chunks).astype(bool),
            )
        costs_rk = _compute_assignment_costs_rk_gpu(
            measures=measures,
            atom_coords=atom_coords,
            atom_features=atom_features,
            betas=betas,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            eps=eps,
            rho=rho,
            align_iters=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
            compute_device=compute_device,
            return_diagnostics=return_diagnostics,
        )
        if not return_diagnostics:
            return np.clip(np.asarray(costs_rk), -1e12, 1e12).astype(np.float32)
        costs_np, eff_reg_np, used_fallback_np = costs_rk
        return (
            np.clip(np.asarray(costs_np), -1e12, 1e12).astype(np.float32),
            np.asarray(eff_reg_np, dtype=np.float32),
            np.asarray(used_fallback_np, dtype=bool),
        )

    for r, measure in enumerate(measures):
        for k in range(n_clusters):
            cost, _, _, _, solve_diag = aligned_semirelaxed_ot_to_cluster(
                u=measure.canonical_coords,
                y=measure.features,
                a=measure.weights,
                atom_coords=atom_coords[k],
                atom_features=atom_features[k],
                beta=betas[k],
                lambda_x=lambda_x,
                lambda_y=lambda_y,
                eps=eps,
                rho=rho,
                n_align_iter=align_iters,
                allow_reflection=allow_reflection,
                allow_scale=allow_scale,
                cost_scale_x=cost_scale_x,
                cost_scale_y=cost_scale_y,
                min_scale=min_scale,
                max_scale=max_scale,
                scale_penalty=scale_penalty,
                shift_penalty=shift_penalty,
                compute_device=compute_device,
            )
            costs[r, k] = float(np.clip(cost, -1e12, 1e12))
            if (
                return_diagnostics
                and effective_eps is not None
                and used_fallback is not None
            ):
                effective_eps[r, k] = float(solve_diag.effective_eps)
                used_fallback[r, k] = bool(solve_diag.used_fallback)
    if not return_diagnostics:
        return costs.astype(np.float32)
    assert effective_eps is not None and used_fallback is not None
    return (
        costs.astype(np.float32),
        effective_eps.astype(np.float32),
        used_fallback.astype(bool),
    )


def _stabilize_mixed_candidate_assignment_costs(
    *,
    measures: list[SubregionMeasure],
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    betas: np.ndarray,
    transport_costs: np.ndarray,
    candidate_effective_eps_matrix: np.ndarray,
    candidate_used_fallback_matrix: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    ot_eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    compute_device: torch.device,
    max_attempts: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transport = np.asarray(transport_costs, dtype=np.float32).copy()
    effective_eps = np.asarray(candidate_effective_eps_matrix, dtype=np.float32).copy()
    used_fallback = np.asarray(candidate_used_fallback_matrix, dtype=bool).copy()
    if transport.ndim != 2 or transport.shape[0] == 0:
        return transport, effective_eps, used_fallback
    mixed_mask = (
        np.max(effective_eps, axis=1) - np.min(effective_eps, axis=1) > 1e-8
    ) | np.any(used_fallback != used_fallback[:, :1], axis=1)
    if not np.any(mixed_mask):
        return transport, effective_eps, used_fallback

    for rid in np.flatnonzero(mixed_mask).tolist():
        common_eps = float(max(np.max(effective_eps[rid]), float(ot_eps)))
        for _ in range(max(int(max_attempts), 1)):
            row_costs, row_eps, row_fallback = _compute_assignment_costs(
                measures=[measures[int(rid)]],
                atom_coords=atom_coords,
                atom_features=atom_features,
                betas=betas,
                lambda_x=lambda_x,
                lambda_y=lambda_y,
                eps=common_eps,
                rho=rho,
                align_iters=align_iters,
                allow_reflection=allow_reflection,
                allow_scale=allow_scale,
                cost_scale_x=cost_scale_x,
                cost_scale_y=cost_scale_y,
                min_scale=min_scale,
                max_scale=max_scale,
                scale_penalty=scale_penalty,
                shift_penalty=shift_penalty,
                compute_device=compute_device,
                return_diagnostics=True,
            )
            transport[rid] = row_costs[0]
            effective_eps[rid] = row_eps[0]
            used_fallback[rid] = row_fallback[0]
            next_eps = float(np.max(row_eps[0]))
            row_is_mixed = (
                float(np.max(row_eps[0]) - np.min(row_eps[0])) > 1e-8
            ) or bool(np.any(row_fallback[0] != row_fallback[0][:1]))
            if not row_is_mixed or next_eps <= common_eps + 1e-8:
                break
            common_eps = next_eps
    return (
        transport.astype(np.float32),
        effective_eps.astype(np.float32),
        used_fallback.astype(bool),
    )


def _compute_assigned_artifacts(
    measures: list[SubregionMeasure],
    labels: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    betas: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    compute_device: torch.device,
) -> tuple[
    list[np.ndarray],
    list[dict[str, np.ndarray | float]],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if compute_device.type == "cuda":
        batch_size = _gpu_assignment_subregion_batch_size(
            n_subregions=len(measures),
            n_clusters=1,
            measure_size=max(
                (int(measure.canonical_coords.shape[0]) for measure in measures),
                default=1,
            ),
            support_size=int(atom_coords.shape[1]),
            feature_dim=int(atom_features.shape[2]),
            device=compute_device,
        )
        if len(measures) > batch_size:
            all_plans: list[np.ndarray] = []
            all_transforms: list[dict[str, np.ndarray | float]] = []
            all_thetas: list[np.ndarray] = []
            cost_chunks: list[np.ndarray] = []
            eps_chunks: list[np.ndarray] = []
            fallback_chunks: list[np.ndarray] = []
            for start in range(0, len(measures), batch_size):
                chunk = _compute_assigned_artifacts_r_gpu(
                    measures=measures[start : start + batch_size],
                    labels=labels[start : start + batch_size],
                    atom_coords=atom_coords,
                    atom_features=atom_features,
                    betas=betas,
                    lambda_x=lambda_x,
                    lambda_y=lambda_y,
                    eps=eps,
                    rho=rho,
                    align_iters=align_iters,
                    allow_reflection=allow_reflection,
                    allow_scale=allow_scale,
                    cost_scale_x=cost_scale_x,
                    cost_scale_y=cost_scale_y,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    scale_penalty=scale_penalty,
                    shift_penalty=shift_penalty,
                    compute_device=compute_device,
                )
                (
                    plans,
                    transforms,
                    thetas,
                    assigned_costs,
                    effective_eps,
                    used_fallback,
                ) = chunk
                all_plans.extend(plans)
                all_transforms.extend(transforms)
                all_thetas.extend(thetas)
                cost_chunks.append(assigned_costs)
                eps_chunks.append(effective_eps)
                fallback_chunks.append(used_fallback)
            return (
                all_plans,
                all_transforms,
                all_thetas,
                np.concatenate(cost_chunks).astype(np.float32),
                np.concatenate(eps_chunks).astype(np.float32),
                np.concatenate(fallback_chunks).astype(bool),
            )
        return _compute_assigned_artifacts_r_gpu(
            measures=measures,
            labels=labels,
            atom_coords=atom_coords,
            atom_features=atom_features,
            betas=betas,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            eps=eps,
            rho=rho,
            align_iters=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
            compute_device=compute_device,
        )
    plans: list[np.ndarray] = []
    transforms: list[dict[str, np.ndarray | float]] = []
    thetas: list[np.ndarray] = []
    assigned_costs = np.zeros(len(measures), dtype=np.float32)
    effective_eps = np.zeros(len(measures), dtype=np.float32)
    used_fallback = np.zeros(len(measures), dtype=bool)
    for r, measure in enumerate(measures):
        k = int(labels[r])
        cost, gamma, transform, theta, solve_diag = aligned_semirelaxed_ot_to_cluster(
            u=measure.canonical_coords,
            y=measure.features,
            a=measure.weights,
            atom_coords=atom_coords[k],
            atom_features=atom_features[k],
            beta=betas[k],
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            eps=eps,
            rho=rho,
            n_align_iter=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
            compute_device=compute_device,
        )
        assigned_costs[r] = cost
        effective_eps[r] = float(solve_diag.effective_eps)
        used_fallback[r] = bool(solve_diag.used_fallback)
        plans.append(gamma)
        transforms.append(transform)
        thetas.append(theta)
    return plans, transforms, thetas, assigned_costs, effective_eps, used_fallback


def _update_atoms(
    measures: list[SubregionMeasure],
    labels: np.ndarray,
    plans: list[np.ndarray],
    transforms: list[dict[str, np.ndarray | float]],
    thetas: list[np.ndarray],
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    beta_smoothing: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_clusters, atoms_per_cluster, coord_dim = atom_coords.shape
    feat_dim = atom_features.shape[2]
    new_coords = atom_coords.copy().astype(np.float32)
    new_features = atom_features.copy().astype(np.float32)
    new_betas = np.zeros((n_clusters, atoms_per_cluster), dtype=np.float32)

    for k in range(n_clusters):
        idx = np.flatnonzero(labels == k)
        if idx.size == 0:
            r = int(rng.integers(len(measures)))
            measure = measures[r]
            pick = rng.choice(
                measure.canonical_coords.shape[0],
                size=atoms_per_cluster,
                replace=measure.canonical_coords.shape[0] < atoms_per_cluster,
                p=measure.weights,
            )
            new_coords[k] = measure.canonical_coords[pick]
            new_features[k] = measure.features[pick]
            new_betas[k] = np.full(
                atoms_per_cluster, 1.0 / atoms_per_cluster, dtype=np.float32
            )
            continue

        coord_num = np.zeros((atoms_per_cluster, coord_dim), dtype=np.float64)
        feat_num = np.zeros((atoms_per_cluster, feat_dim), dtype=np.float64)
        denom = np.zeros(atoms_per_cluster, dtype=np.float64)
        beta_num = np.zeros(atoms_per_cluster, dtype=np.float64)

        for r in idx:
            measure = measures[r]
            gamma = np.asarray(plans[r], dtype=np.float64)
            theta = np.asarray(thetas[r], dtype=np.float64)
            transform = transforms[r]
            aligned = apply_similarity(measure.canonical_coords, transform)
            coord_num += gamma.T @ aligned
            feat_num += gamma.T @ measure.features
            denom += gamma.sum(axis=0)
            beta_num += theta

        active = denom > 1e-10
        if np.any(active):
            new_coords[k, active] = (coord_num[active] / denom[active, None]).astype(
                np.float32
            )
            new_features[k, active] = (feat_num[active] / denom[active, None]).astype(
                np.float32
            )

        if np.any(~active):
            u_pool = np.vstack([measures[r].canonical_coords for r in idx]).astype(
                np.float32
            )
            y_pool = np.vstack([measures[r].features for r in idx]).astype(np.float32)
            a_pool = _normalize_hist(np.concatenate([measures[r].weights for r in idx]))
            dead = np.where(~active)[0]
            chosen = rng.choice(
                u_pool.shape[0],
                size=dead.size,
                replace=u_pool.shape[0] < dead.size,
                p=a_pool,
            )
            new_coords[k, dead] = u_pool[chosen]
            new_features[k, dead] = y_pool[chosen]

        new_betas[k] = _normalize_hist(beta_num + beta_smoothing).astype(np.float32)

    return new_coords, new_features, new_betas


def _cell_cluster_feature_costs(
    features: np.ndarray,
    support_features: np.ndarray,
    prototype_weights: np.ndarray,
    temperature: float,
    compute_device: torch.device,
) -> np.ndarray:
    n_cells = features.shape[0]
    n_clusters = support_features.shape[0]
    temp = max(float(temperature), 1e-5)
    if compute_device.type == "cuda" and n_cells > 0 and n_clusters > 0:
        dtype = torch.float32
        feats_t = torch.as_tensor(features, dtype=dtype, device=compute_device)
        support_t = torch.as_tensor(
            support_features, dtype=dtype, device=compute_device
        )
        weights_t = torch.as_tensor(
            np.clip(prototype_weights, 1e-8, None), dtype=dtype, device=compute_device
        )
        costs_t = torch.empty((n_cells, n_clusters), dtype=dtype, device=compute_device)
        feat_dim = int(features.shape[1]) if features.ndim == 2 else 1
        support_size = (
            int(support_features.shape[1]) if support_features.ndim >= 2 else 1
        )
        support_flat = support_t.reshape(
            n_clusters * support_size, feat_dim
        ).contiguous()
        support_norm = (support_flat * support_flat).sum(dim=1)
        # Compute all cluster/atom distances with one GEMM per cell batch. This
        # avoids a Python loop over clusters and is much faster for K*p << n_cells.
        denom = max((n_clusters * support_size * 8 + feat_dim) * 16, 1)
        batch = max(1, min(n_cells, _cuda_target_bytes(device=compute_device) // denom))
        with torch.inference_mode():
            for start in range(0, n_cells, batch):
                stop = min(start + batch, n_cells)
                chunk_done = False
                while not chunk_done:
                    try:
                        f_chunk = feats_t[start:stop]
                        chunk_norm = (f_chunk * f_chunk).sum(dim=1, keepdim=True)
                        dist = (
                            chunk_norm
                            + support_norm.unsqueeze(0)
                            - 2.0 * (f_chunk @ support_flat.T)
                        )
                        dist = dist.clamp_min(0.0).reshape(
                            stop - start, n_clusters, support_size
                        )
                        scaled = torch.exp(-dist / temp) * weights_t.unsqueeze(0)
                        costs_t[start:stop] = -temp * torch.log(
                            scaled.sum(dim=-1).clamp_min(1e-8)
                        )
                        chunk_done = True
                    except torch.OutOfMemoryError:
                        if stop - start <= 1:
                            raise
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        stop = start + max((stop - start) // 2, 1)
        return costs_t.detach().cpu().numpy().astype(np.float32)
    costs = np.zeros((n_cells, n_clusters), dtype=np.float32)
    for k in range(n_clusters):
        dist = _pairwise_sqdist_array(
            features, support_features[k], device=compute_device
        )
        weights = np.clip(prototype_weights[k], 1e-8, None).astype(np.float32)
        scores = np.exp(-dist / temp) * weights[None, :]
        costs[:, k] = -temp * np.log(np.maximum(scores.sum(axis=1), 1e-8))
    return costs


def _project_cells_from_subregions(
    features: np.ndarray,
    coords_um: np.ndarray,
    measures: list[SubregionMeasure],
    subregion_labels: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    prototype_weights: np.ndarray,
    assigned_transforms: list[dict[str, np.ndarray | float]],
    subregion_cluster_costs: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    cost_scale_x: float,
    cost_scale_y: float,
    assignment_temperature: float,
    context_weight: float = 0.5,
    compute_device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    compute_device = compute_device or torch.device("cpu")
    support_features = atom_features
    feature_costs = _cell_cluster_feature_costs(
        features=features,
        support_features=support_features,
        prototype_weights=prototype_weights,
        temperature=assignment_temperature,
        compute_device=compute_device,
    )
    feature_probs = _softmax_over_negative_costs(
        feature_costs, temperature=assignment_temperature
    )
    subregion_probs = _softmax_over_negative_costs(
        subregion_cluster_costs, temperature=max(assignment_temperature, 1e-4)
    )

    n_cells = features.shape[0]
    n_clusters = atom_coords.shape[0]
    context_probs = np.zeros((n_cells, n_clusters), dtype=np.float32)
    membership_counts = np.zeros((n_cells, 1), dtype=np.float32)
    local_model_probs = np.zeros((n_cells, n_clusters), dtype=np.float32)
    local_model_counts = np.zeros((n_cells, 1), dtype=np.float32)
    for r, measure in enumerate(measures):
        members = measure.members
        context_probs[members] += subregion_probs[r]
        membership_counts[members] += 1.0
        k = int(subregion_labels[r])
        canonical = measure.normalizer.transform(coords_um[members])
        aligned = apply_similarity(canonical, assigned_transforms[r]).astype(np.float32)
        cx = _pairwise_sqdist_array(
            aligned, atom_coords[k], device=compute_device
        ) / max(float(cost_scale_x), 1e-5)
        cy = _pairwise_sqdist_array(
            features[members], atom_features[k], device=compute_device
        ) / max(float(cost_scale_y), 1e-5)
        total_cost = float(lambda_x) * cx + float(lambda_y) * cy
        atom_scores = (
            np.exp(-total_cost / max(float(assignment_temperature), 1e-5))
            * np.clip(prototype_weights[k], 1e-8, None)[None, :]
        )
        local_support = atom_scores.sum(axis=1).astype(np.float32)
        local_model_probs[members] += (
            local_support[:, None] * subregion_probs[r][None, :]
        )
        local_model_counts[members] += 1.0
    covered = membership_counts[:, 0] > 0
    if np.any(covered):
        context_probs[covered] /= membership_counts[covered]
    if np.any(~covered):
        context_probs[~covered] = feature_probs[~covered]
    local_covered = local_model_counts[:, 0] > 0
    if np.any(local_covered):
        local_model_probs[local_covered] /= local_model_counts[local_covered]
        local_model_probs[local_covered] /= np.maximum(
            local_model_probs[local_covered].sum(axis=1, keepdims=True), 1e-8
        )
    if np.any(~local_covered):
        local_model_probs[~local_covered] = feature_probs[~local_covered]

    combined = (
        feature_probs
        * np.power(np.clip(context_probs, 1e-8, None), float(context_weight))
        * np.power(np.clip(local_model_probs, 1e-8, None), 1.0 - float(context_weight))
    )
    combined = combined / np.maximum(combined.sum(axis=1, keepdims=True), 1e-8)
    labels = combined.argmax(axis=1).astype(np.int32)
    return (
        labels,
        combined.astype(np.float32),
        local_model_probs.astype(np.float32),
        context_probs.astype(np.float32),
    )


def _execute_restart(
    measures: list[OptimizationMeasure],
    summaries: np.ndarray,
    *,
    run: int,
    n_clusters: int,
    atoms_per_cluster: int,
    lambda_x: float,
    lambda_y: float,
    ot_eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    max_iter: int,
    tol: float,
    min_subregions_per_cluster: int,
    seed: int,
    compute_device: str,
    overlap_edge_i: np.ndarray | None = None,
    overlap_edge_j: np.ndarray | None = None,
    overlap_edge_weight: np.ndarray | None = None,
    overlap_consistency_weight: float = 0.0,
) -> dict[str, object]:
    resolved_compute_device = _resolve_compute_device(compute_device)
    if resolved_compute_device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(resolved_compute_device)
        except Exception:
            pass
    run_seed = seed + 1000 * int(run)
    init = KMeans(n_clusters=n_clusters, n_init=20, random_state=run_seed)
    labels = init.fit_predict(summaries).astype(np.int32)
    labels, init_forced_label_mask = _ensure_minimum_cluster_size(
        labels,
        init.transform(summaries),
        n_clusters=n_clusters,
        min_subregions_per_cluster=min_subregions_per_cluster,
    )
    atom_coords, atom_features, betas = _initialize_cluster_atoms(
        measures=measures,
        labels=labels,
        n_clusters=n_clusters,
        atoms_per_cluster=atoms_per_cluster,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        random_state=run_seed,
    )
    objective_history: list[dict[str, float]] = []
    overlap_edge_i = np.asarray(
        overlap_edge_i if overlap_edge_i is not None else np.zeros(0, dtype=np.int32),
        dtype=np.int32,
    )
    overlap_edge_j = np.asarray(
        overlap_edge_j if overlap_edge_j is not None else np.zeros(0, dtype=np.int32),
        dtype=np.int32,
    )
    overlap_edge_weight = np.asarray(
        overlap_edge_weight
        if overlap_edge_weight is not None
        else np.zeros(0, dtype=np.float32),
        dtype=np.float32,
    )

    for iteration in range(int(max_iter)):
        prev_coords = atom_coords.copy()
        prev_features = atom_features.copy()
        prev_labels = labels.copy()
        transport_costs = _compute_assignment_costs(
            measures=measures,
            atom_coords=atom_coords,
            atom_features=atom_features,
            betas=betas,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            eps=ot_eps,
            rho=rho,
            align_iters=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
            compute_device=resolved_compute_device,
        )
        costs, overlap_penalties = _apply_overlap_consistency_regularization(
            transport_costs,
            edge_i=overlap_edge_i,
            edge_j=overlap_edge_j,
            edge_weight=overlap_edge_weight,
            overlap_consistency_weight=overlap_consistency_weight,
        )
        argmin_labels = costs.argmin(axis=1).astype(np.int32)
        labels, forced_label_mask = _ensure_minimum_cluster_size(
            argmin_labels,
            costs,
            n_clusters=n_clusters,
            min_subregions_per_cluster=min_subregions_per_cluster,
        )
        plans, transforms, thetas, assigned_costs, _, assigned_used_fallback = (
            _compute_assigned_artifacts(
                measures=measures,
                labels=labels,
                atom_coords=atom_coords,
                atom_features=atom_features,
                betas=betas,
                lambda_x=lambda_x,
                lambda_y=lambda_y,
                eps=ot_eps,
                rho=rho,
                align_iters=align_iters,
                allow_reflection=allow_reflection,
                allow_scale=allow_scale,
                cost_scale_x=cost_scale_x,
                cost_scale_y=cost_scale_y,
                min_scale=min_scale,
                max_scale=max_scale,
                scale_penalty=scale_penalty,
                shift_penalty=shift_penalty,
                compute_device=resolved_compute_device,
            )
        )
        assigned_overlap_penalties = overlap_penalties[
            np.arange(labels.shape[0], dtype=np.int64),
            labels.astype(np.int64),
        ].astype(np.float32)
        atom_coords, atom_features, betas = _update_atoms(
            measures=measures,
            labels=labels,
            plans=plans,
            transforms=transforms,
            thetas=thetas,
            atom_coords=atom_coords,
            atom_features=atom_features,
            beta_smoothing=1e-3,
            random_state=run_seed + iteration,
        )

        label_change_rate = float(np.mean(labels != prev_labels))
        coord_shift = _relative_change(atom_coords, prev_coords)
        feat_shift = _relative_change(atom_features, prev_features)
        mean_transport_obj = float(np.mean(assigned_costs))
        mean_overlap_penalty = (
            float(np.mean(assigned_overlap_penalties))
            if assigned_overlap_penalties.size
            else 0.0
        )
        mean_obj = float(np.mean(assigned_costs + assigned_overlap_penalties))
        sorted_costs = np.sort(costs, axis=1)
        mean_margin = (
            float(np.mean(sorted_costs[:, 1] - sorted_costs[:, 0]))
            if sorted_costs.shape[1] >= 2
            else float("nan")
        )
        objective_history.append(
            {
                "iteration": int(iteration + 1),
                "mean_objective": mean_obj,
                "mean_transport_objective": mean_transport_obj,
                "mean_overlap_consistency_penalty": mean_overlap_penalty,
                "label_change_rate": label_change_rate,
                "coord_shift": coord_shift,
                "feature_shift": feat_shift,
                "mean_assignment_margin": mean_margin,
                "forced_label_count": int(forced_label_mask.sum()),
                "min_subregions_per_cluster": int(
                    effective_min_cluster_size(
                        len(labels), n_clusters, int(min_subregions_per_cluster)
                    )
                ),
                "assigned_ot_fallback_fraction": float(
                    np.mean(assigned_used_fallback.astype(np.float32))
                ),
            }
        )
        if label_change_rate < 0.005 and max(coord_shift, feat_shift) < tol:
            break

    (
        final_transport_costs,
        _candidate_effective_eps_matrix,
        candidate_used_fallback_matrix,
    ) = _compute_assignment_costs(
        measures=measures,
        atom_coords=atom_coords,
        atom_features=atom_features,
        betas=betas,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        eps=ot_eps,
        rho=rho,
        align_iters=align_iters,
        allow_reflection=allow_reflection,
        allow_scale=allow_scale,
        cost_scale_x=cost_scale_x,
        cost_scale_y=cost_scale_y,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_penalty=scale_penalty,
        shift_penalty=shift_penalty,
        compute_device=resolved_compute_device,
        return_diagnostics=True,
    )
    (
        final_transport_costs,
        _candidate_effective_eps_matrix,
        candidate_used_fallback_matrix,
    ) = _stabilize_mixed_candidate_assignment_costs(
        measures=measures,
        atom_coords=atom_coords,
        atom_features=atom_features,
        betas=betas,
        transport_costs=final_transport_costs,
        candidate_effective_eps_matrix=_candidate_effective_eps_matrix,
        candidate_used_fallback_matrix=candidate_used_fallback_matrix,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        ot_eps=ot_eps,
        rho=rho,
        align_iters=align_iters,
        allow_reflection=allow_reflection,
        allow_scale=allow_scale,
        cost_scale_x=cost_scale_x,
        cost_scale_y=cost_scale_y,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_penalty=scale_penalty,
        shift_penalty=shift_penalty,
        compute_device=resolved_compute_device,
    )
    final_costs, final_overlap_penalties = _apply_overlap_consistency_regularization(
        final_transport_costs,
        edge_i=overlap_edge_i,
        edge_j=overlap_edge_j,
        edge_weight=overlap_edge_weight,
        overlap_consistency_weight=overlap_consistency_weight,
    )
    final_argmin_labels = final_costs.argmin(axis=1).astype(np.int32)
    final_labels, final_forced_label_mask = _ensure_minimum_cluster_size(
        final_argmin_labels,
        final_costs,
        n_clusters=n_clusters,
        min_subregions_per_cluster=min_subregions_per_cluster,
    )
    row_idx = np.arange(final_labels.shape[0], dtype=np.int64)
    final_assigned_costs = final_transport_costs[
        row_idx, final_labels.astype(np.int64)
    ].astype(np.float32)
    final_assigned_used_fallback = candidate_used_fallback_matrix[
        row_idx, final_labels.astype(np.int64)
    ].astype(bool)
    final_assigned_overlap_penalties = final_overlap_penalties[
        row_idx,
        final_labels.astype(np.int64),
    ].astype(np.float32)
    final_objective = float(
        np.sum(final_assigned_costs + final_assigned_overlap_penalties)
    )
    return {
        "run": int(run),
        "seed": int(run_seed),
        "objective": final_objective,
        "n_iter": int(len(objective_history)),
        "mean_assigned_cost": float(
            np.mean(final_assigned_costs + final_assigned_overlap_penalties)
        ),
        "mean_assigned_transport_cost": float(np.mean(final_assigned_costs)),
        "mean_assigned_overlap_penalty": float(
            np.mean(final_assigned_overlap_penalties)
        )
        if final_assigned_overlap_penalties.size
        else 0.0,
        "labels": final_labels.astype(np.int32),
        "costs": final_costs.astype(np.float32),
        "transport_costs": final_transport_costs.astype(np.float32),
        "overlap_penalties": final_overlap_penalties.astype(np.float32),
        "atom_coords": atom_coords.astype(np.float32),
        "atom_features": atom_features.astype(np.float32),
        "betas": betas.astype(np.float32),
        "objective_history": objective_history,
        "device": str(resolved_compute_device),
        "assigned_ot_fallback_fraction": float(
            np.mean(final_assigned_used_fallback.astype(np.float32))
        ),
        "candidate_ot_fallback_fraction": float(
            np.mean(candidate_used_fallback_matrix.astype(np.float32))
        ),
        "initial_min_size_forced_label_count": int(init_forced_label_mask.sum()),
        "final_min_size_forced_label_count": int(final_forced_label_mask.sum()),
        "min_subregions_per_cluster": int(
            effective_min_cluster_size(
                len(final_labels), n_clusters, int(min_subregions_per_cluster)
            )
        ),
        "runtime_memory": _runtime_memory_snapshot(resolved_compute_device),
    }


def _fit_restart_bundles(
    *,
    measures: list[OptimizationMeasure],
    summaries: np.ndarray,
    n_clusters: int,
    atoms_per_cluster: int,
    lambda_x: float,
    lambda_y: float,
    ot_eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    max_iter: int,
    tol: float,
    min_subregions_per_cluster: int,
    seed: int,
    overlap_edge_i: np.ndarray,
    overlap_edge_j: np.ndarray,
    overlap_edge_weight: np.ndarray,
    overlap_consistency_weight: float,
    n_init: int,
    compute_device: str,
    resolved_compute_device: torch.device,
    progress_label: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    restart_params = {
        "n_clusters": int(n_clusters),
        "atoms_per_cluster": int(atoms_per_cluster),
        "lambda_x": float(lambda_x),
        "lambda_y": float(lambda_y),
        "ot_eps": float(ot_eps),
        "rho": float(rho),
        "align_iters": int(align_iters),
        "allow_reflection": bool(allow_reflection),
        "allow_scale": bool(allow_scale),
        "cost_scale_x": float(cost_scale_x),
        "cost_scale_y": float(cost_scale_y),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "scale_penalty": float(scale_penalty),
        "shift_penalty": float(shift_penalty),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "min_subregions_per_cluster": int(min_subregions_per_cluster),
        "seed": int(seed),
        "overlap_edge_i": overlap_edge_i,
        "overlap_edge_j": overlap_edge_j,
        "overlap_edge_weight": overlap_edge_weight,
        "overlap_consistency_weight": float(overlap_consistency_weight),
    }
    device_pool = (
        _resolve_cuda_device_pool(str(compute_device), int(n_init))
        if resolved_compute_device.type == "cuda"
        else [str(resolved_compute_device)]
    )
    parallel_restart_workers = _resolve_parallel_restart_workers(
        device_pool, int(n_init)
    )
    restart_results: list[dict[str, object]] = []
    _progress(
        f"{progress_label}: running {int(n_init)} restart(s) for K={int(n_clusters)} on {','.join(device_pool)} "
        f"with max_iter={int(max_iter)}, align_iters={int(align_iters)}"
    )
    if parallel_restart_workers > 1:
        total_torch_threads = _env_int(
            "SPATIAL_OT_TORCH_NUM_THREADS", max(torch.get_num_threads(), 1)
        )
        worker_threads = max(1, total_torch_threads // parallel_restart_workers)
        worker_interop_threads = 1
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=parallel_restart_workers,
            mp_context=ctx,
            initializer=_init_restart_worker,
            initargs=(
                measures,
                summaries,
                restart_params,
                worker_threads,
                worker_interop_threads,
            ),
        ) as executor:
            future_map = {
                executor.submit(
                    _run_restart_worker, run, device_pool[run % len(device_pool)]
                ): run
                for run in range(int(n_init))
            }
            for future in as_completed(future_map):
                result = future.result()
                restart_results.append(result)
                _progress(
                    f"{progress_label}: restart {int(result['run'])} finished: "
                    f"objective={float(result['objective']):.4g}, iterations={int(result['n_iter'])}"
                )
    else:
        for run in range(int(n_init)):
            result = _execute_restart(
                measures=measures,
                summaries=summaries,
                run=run,
                compute_device=device_pool[run % len(device_pool)],
                **restart_params,
            )
            restart_results.append(result)
            _progress(
                f"{progress_label}: restart {int(result['run'])} finished: "
                f"objective={float(result['objective']):.4g}, iterations={int(result['n_iter'])}"
            )
    restart_results.sort(key=lambda result: int(result["run"]))
    restart_summaries = [
        {
            "run": int(result["run"]),
            "seed": int(result["seed"]),
            "objective": float(result["objective"]),
            "n_iter": int(result["n_iter"]),
            "mean_assigned_cost": float(result["mean_assigned_cost"]),
            "mean_assigned_transport_cost": float(
                result.get("mean_assigned_transport_cost", result["mean_assigned_cost"])
            ),
            "mean_assigned_overlap_penalty": float(
                result.get("mean_assigned_overlap_penalty", 0.0)
            ),
            "device": str(result["device"]),
            "assigned_ot_fallback_fraction": float(
                result["assigned_ot_fallback_fraction"]
            ),
            "candidate_ot_fallback_fraction": float(
                result.get(
                    "candidate_ot_fallback_fraction",
                    result["assigned_ot_fallback_fraction"],
                )
            ),
            "initial_min_size_forced_label_count": int(
                result.get("initial_min_size_forced_label_count", 0)
            ),
            "final_min_size_forced_label_count": int(
                result.get("final_min_size_forced_label_count", 0)
            ),
            "min_subregions_per_cluster": int(
                result.get("min_subregions_per_cluster", 1)
            ),
            "runtime_memory": dict(result.get("runtime_memory", {})),
        }
        for result in restart_results
    ]
    best_bundle = min(restart_results, key=lambda result: float(result["objective"]))
    return restart_results, restart_summaries, best_bundle


def _fit_fixed_label_atom_dictionary(
    *,
    measures: list[OptimizationMeasure],
    labels: np.ndarray,
    n_clusters: int,
    atoms_per_cluster: int,
    lambda_x: float,
    lambda_y: float,
    ot_eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    max_iter: int,
    tol: float,
    seed: int,
    compute_device: torch.device,
) -> dict[str, object]:
    fixed_labels = np.asarray(labels, dtype=np.int32)
    atom_coords, atom_features, betas = _initialize_cluster_atoms(
        measures=measures,
        labels=fixed_labels,
        n_clusters=int(n_clusters),
        atoms_per_cluster=int(atoms_per_cluster),
        lambda_x=float(lambda_x),
        lambda_y=float(lambda_y),
        random_state=int(seed),
    )
    objective_history: list[dict[str, float]] = []
    final_plans: list[np.ndarray] = []
    final_transforms: list[dict[str, np.ndarray | float]] = []
    final_thetas: list[np.ndarray] = []
    final_costs = np.zeros(len(measures), dtype=np.float32)
    final_eps = np.full(len(measures), float(ot_eps), dtype=np.float32)
    final_fallback = np.zeros(len(measures), dtype=bool)

    for iteration in range(max(int(max_iter), 1)):
        prev_coords = atom_coords.copy()
        prev_features = atom_features.copy()
        plans, transforms, thetas, assigned_costs, assigned_eps, assigned_fallback = (
            _compute_assigned_artifacts(
                measures=measures,
                labels=fixed_labels,
                atom_coords=atom_coords,
                atom_features=atom_features,
                betas=betas,
                lambda_x=lambda_x,
                lambda_y=lambda_y,
                eps=ot_eps,
                rho=rho,
                align_iters=align_iters,
                allow_reflection=allow_reflection,
                allow_scale=allow_scale,
                cost_scale_x=cost_scale_x,
                cost_scale_y=cost_scale_y,
                min_scale=min_scale,
                max_scale=max_scale,
                scale_penalty=scale_penalty,
                shift_penalty=shift_penalty,
                compute_device=compute_device,
            )
        )
        atom_coords, atom_features, betas = _update_atoms(
            measures=measures,
            labels=fixed_labels,
            plans=plans,
            transforms=transforms,
            thetas=thetas,
            atom_coords=atom_coords,
            atom_features=atom_features,
            beta_smoothing=1e-3,
            random_state=int(seed) + int(iteration),
        )
        coord_shift = _relative_change(atom_coords, prev_coords)
        feat_shift = _relative_change(atom_features, prev_features)
        objective_history.append(
            {
                "iteration": int(iteration + 1),
                "mean_objective": float(np.mean(assigned_costs)),
                "mean_transport_objective": float(np.mean(assigned_costs)),
                "mean_overlap_consistency_penalty": 0.0,
                "label_change_rate": 0.0,
                "coord_shift": coord_shift,
                "feature_shift": feat_shift,
                "mean_assignment_margin": float("nan"),
                "forced_label_count": 0,
                "assigned_ot_fallback_fraction": float(
                    np.mean(np.asarray(assigned_fallback, dtype=np.float32))
                ),
            }
        )
        final_plans = plans
        final_transforms = transforms
        final_thetas = thetas
        final_costs = np.asarray(assigned_costs, dtype=np.float32)
        final_eps = np.asarray(assigned_eps, dtype=np.float32)
        final_fallback = np.asarray(assigned_fallback, dtype=bool)
        if max(coord_shift, feat_shift) < float(tol):
            break

    (
        final_plans,
        final_transforms,
        final_thetas,
        final_costs,
        final_eps,
        final_fallback,
    ) = _compute_assigned_artifacts(
        measures=measures,
        labels=fixed_labels,
        atom_coords=atom_coords,
        atom_features=atom_features,
        betas=betas,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        eps=ot_eps,
        rho=rho,
        align_iters=align_iters,
        allow_reflection=allow_reflection,
        allow_scale=allow_scale,
        cost_scale_x=cost_scale_x,
        cost_scale_y=cost_scale_y,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_penalty=scale_penalty,
        shift_penalty=shift_penalty,
        compute_device=compute_device,
    )
    return {
        "atom_coords": atom_coords.astype(np.float32),
        "atom_features": atom_features.astype(np.float32),
        "betas": betas.astype(np.float32),
        "plans": final_plans,
        "transforms": final_transforms,
        "thetas": final_thetas,
        "assigned_costs": np.asarray(final_costs, dtype=np.float32),
        "assigned_effective_eps": np.asarray(final_eps, dtype=np.float32),
        "assigned_used_fallback": np.asarray(final_fallback, dtype=bool),
        "objective_history": objective_history,
        "runtime_memory": _runtime_memory_snapshot(compute_device),
    }


def _cell_projection_from_subregion_probabilities(
    *,
    n_cells: int,
    subregion_members: list[np.ndarray],
    subregion_labels: np.ndarray,
    subregion_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(int(n_cells), -1, dtype=np.int32)
    probs = np.zeros((int(n_cells), int(subregion_probs.shape[1])), dtype=np.float32)
    for rid, members in enumerate(subregion_members):
        member_idx = np.asarray(members, dtype=np.int64)
        if member_idx.size == 0:
            continue
        labels[member_idx] = int(subregion_labels[int(rid)])
        probs[member_idx] = np.asarray(subregion_probs[int(rid)], dtype=np.float32)
    return labels, probs


def _cluster_feature_centroids_from_subregions(
    subregion_latent_embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    n_clusters: int,
    feature_dim: int,
) -> np.ndarray:
    latent = np.asarray(subregion_latent_embeddings, dtype=np.float32)
    label_arr = np.asarray(labels, dtype=np.int32)
    feature_dim = int(feature_dim)
    if latent.ndim != 2 or latent.shape[1] < feature_dim:
        return np.zeros((int(n_clusters), feature_dim), dtype=np.float32)
    mean_features = latent[:, :feature_dim]
    centroids = np.zeros((int(n_clusters), feature_dim), dtype=np.float32)
    global_mean = (
        mean_features.mean(axis=0).astype(np.float32)
        if mean_features.size
        else np.zeros(feature_dim, dtype=np.float32)
    )
    for cid in range(int(n_clusters)):
        idx = label_arr == int(cid)
        centroids[cid] = (
            mean_features[idx].mean(axis=0).astype(np.float32)
            if np.any(idx)
            else global_mean
        )
    return centroids


def _fast_pooled_latent_only_threshold() -> int:
    return max(_env_int("SPATIAL_OT_FAST_POOLED_LATENT_MAX_SUBREGIONS", 50000), 1)


def _fit_pooled_latent_only_result(
    *,
    features: np.ndarray,
    sample_ids: np.ndarray | None,
    centers_um: np.ndarray,
    subregion_members: list[np.ndarray],
    n_clusters: int,
    atoms_per_cluster: int,
    n_init: int,
    auto_n_clusters: bool,
    candidate_ks: tuple[int, ...],
    auto_k_max_score_subregions: int,
    auto_k_gap_references: int,
    min_subregions_per_cluster: int,
    requested_min_subregions_per_cluster: int,
    subregion_latent_embedding_mode: str,
    subregion_latent_shrinkage_tau: float,
    subregion_latent_heterogeneity_weight: float,
    subregion_latent_sample_prior_weight: float,
    subregion_latent_codebook_size: int,
    subregion_latent_codebook_sample_size: int,
    ot_eps: float,
    seed: int,
    basic_niche_size_um: float | None,
    used_basic_niches: bool,
    joint_refinement_summary: dict[str, object] | None = None,
) -> MultilevelOTResult:
    _progress(
        "using fast pooled-subregion-latent-only fit; skipping fixed-label OT atom diagnostics "
        "because the constructed run has a very large subregion set"
    )
    n_subregions = int(len(subregion_members))
    feature_dim = int(np.asarray(features).shape[1])
    subregion_latent_embeddings, subregion_latent_diagnostics = (
        _build_subregion_latent_embeddings_from_members(
            features,
            subregion_members,
            mode=subregion_latent_embedding_mode,
            shrinkage_tau=float(subregion_latent_shrinkage_tau),
            heterogeneity_weight=float(subregion_latent_heterogeneity_weight),
            sample_ids=sample_ids,
            sample_prior_weight=float(subregion_latent_sample_prior_weight),
            codebook_size=int(subregion_latent_codebook_size),
            codebook_sample_size=int(subregion_latent_codebook_sample_size),
            random_state=int(seed),
            return_diagnostics=True,
        )
    )
    subregion_latent_embedding_mode = _normalize_subregion_latent_embedding_mode(
        subregion_latent_embedding_mode
    )
    subregion_latent_embedding_metadata = _subregion_latent_embedding_metadata(
        mode=subregion_latent_embedding_mode,
        shrinkage_tau=float(subregion_latent_shrinkage_tau),
        heterogeneity_weight=float(subregion_latent_heterogeneity_weight),
        sample_prior_weight=float(subregion_latent_sample_prior_weight),
        codebook_size=int(subregion_latent_codebook_size),
        codebook_sample_size=int(subregion_latent_codebook_sample_size),
        feature_dim=feature_dim,
        embedding_dim=int(subregion_latent_embeddings.shape[1]),
        sample_aware_shrinkage=bool(
            subregion_latent_diagnostics.get("sample_aware_shrinkage", False)
        ),
    )
    if joint_refinement_summary is not None:
        subregion_latent_embedding_metadata["joint_refinement"] = dict(
            joint_refinement_summary
        )
    auto_k_selection: dict[str, object] | None = None
    selected_n_clusters = int(n_clusters)
    if bool(auto_n_clusters):
        _progress(
            "automatic K selection enabled on pooled subregion latent embeddings; candidate K="
            + ",".join(str(k) for k in candidate_ks)
        )
        stability_seed_count = max(
            3, min(8, _env_int("SPATIAL_OT_AUTO_K_STABILITY_SEEDS", 5))
        )
        auto_k_selection = comprehensive_select_k_from_latent_embeddings(
            subregion_latent_embeddings,
            candidate_n_clusters=candidate_ks,
            fallback_n_clusters=int(n_clusters),
            seeds=tuple(int(seed) + offset for offset in range(stability_seed_count)),
            n_init=max(1, int(n_init)),
            max_silhouette_subregions=int(auto_k_max_score_subregions),
            gap_references=int(auto_k_gap_references),
            bootstrap_repeats=max(
                0, _env_int("SPATIAL_OT_AUTO_K_BOOTSTRAP_REPEATS", 3)
            ),
            bootstrap_fraction=float(
                _env_float("SPATIAL_OT_AUTO_K_BOOTSTRAP_FRACTION", 0.8)
            ),
            min_cluster_size=int(min_subregions_per_cluster),
            random_state=int(seed),
        )
        selected_n_clusters = int(auto_k_selection["selected_k"])
        auto_k_selection.update(
            {
                "requested_n_clusters": int(n_clusters),
                "selection_clustering_method": "pooled_subregion_latent",
                "pilot_n_clusters": None,
                "pilot_n_init": None,
                "pilot_max_iter": None,
                "final_refit_n_init": int(n_init),
                "requested_min_subregions_per_cluster": int(
                    requested_min_subregions_per_cluster
                ),
                "reused_pilot_fit_for_final": False,
                "fast_pooled_latent_only": True,
            }
        )
        _progress(
            f"automatic K selection chose K={selected_n_clusters} "
            f"from votes={auto_k_selection.get('criterion_votes', {})}"
        )

    _progress(
        f"fitting selected K={selected_n_clusters} pooled subregion latent clusters"
    )
    latent_fit = fit_kmeans_on_latent_embeddings(
        subregion_latent_embeddings,
        n_clusters=int(selected_n_clusters),
        n_init=max(int(n_init), 1),
        min_cluster_size=int(min_subregions_per_cluster),
        random_state=int(seed),
    )
    _progress(
        "pooled subregion latent clustering finished; assembling large-run result"
    )
    labels = np.asarray(latent_fit["labels"], dtype=np.int32)
    final_argmin_labels = np.asarray(latent_fit["argmin_labels"], dtype=np.int32)
    final_costs = np.asarray(latent_fit["costs"], dtype=np.float32)
    subregion_cluster_probs = _softmax_over_negative_costs(
        final_costs,
        temperature=max(float(np.std(final_costs)), 1e-3),
    )
    cell_labels, cell_probs = _cell_projection_from_subregion_probabilities(
        n_cells=int(features.shape[0]),
        subregion_members=subregion_members,
        subregion_labels=labels,
        subregion_probs=subregion_cluster_probs,
    )

    atom_count = max(int(atoms_per_cluster), 1)
    cluster_feature_centroids = _cluster_feature_centroids_from_subregions(
        subregion_latent_embeddings,
        labels,
        n_clusters=int(selected_n_clusters),
        feature_dim=feature_dim,
    )
    atom_coords = np.zeros((int(selected_n_clusters), atom_count, 2), dtype=np.float32)
    atom_features = np.repeat(
        cluster_feature_centroids[:, None, :], atom_count, axis=1
    ).astype(np.float32)
    betas = np.full(
        (int(selected_n_clusters), atom_count),
        1.0 / float(atom_count),
        dtype=np.float32,
    )
    cluster_supports = np.concatenate([atom_coords, atom_features], axis=2).astype(
        np.float32
    )

    standardized = np.asarray(
        latent_fit.get("standardized_embeddings", subregion_latent_embeddings),
        dtype=np.float32,
    )
    measure_summaries = standardized.astype(np.float32)
    atom_weights = standardized[:, : min(standardized.shape[1], 32)].astype(np.float32)
    if atom_weights.shape[1] == 0:
        atom_weights = np.zeros((n_subregions, 1), dtype=np.float32)

    assigned = final_costs[
        np.arange(n_subregions, dtype=np.int64), labels.astype(np.int64)
    ].astype(np.float32)
    spot_latent = empty_spot_level_latent_charts(
        n_cells=int(features.shape[0]),
        atoms_per_cluster=atom_count,
        n_clusters=int(selected_n_clusters),
    )
    restart_summaries = [
        {
            "run": 0,
            "seed": int(seed),
            "objective": float(latent_fit["inertia"]),
            "n_iter": 0,
            "mean_assigned_cost": float(np.mean(assigned)) if assigned.size else 0.0,
            "mean_assigned_transport_cost": 0.0,
            "mean_assigned_overlap_penalty": 0.0,
            "device": "cpu",
            "assigned_ot_fallback_fraction": 0.0,
            "candidate_ot_fallback_fraction": 0.0,
            "initial_min_size_forced_label_count": int(
                np.asarray(latent_fit["forced_label_mask"], dtype=bool).sum()
            ),
            "final_min_size_forced_label_count": int(
                np.asarray(latent_fit["forced_label_mask"], dtype=bool).sum()
            ),
            "min_subregions_per_cluster": int(
                effective_min_cluster_size(
                    n_subregions,
                    int(selected_n_clusters),
                    int(min_subregions_per_cluster),
                )
            ),
            "runtime_memory": _runtime_memory_snapshot(_resolve_compute_device("cpu")),
            "label_assignment_source": "pooled_subregion_latent_embeddings",
            "fast_pooled_latent_only": True,
            "fixed_label_ot_atom_diagnostics": "skipped_large_area_capped_partition",
        }
    ]
    zero_sub = np.zeros(n_subregions, dtype=np.float32)
    false_sub = np.zeros(n_subregions, dtype=bool)
    candidate_eps = np.full(
        (n_subregions, int(selected_n_clusters)), float(ot_eps), dtype=np.float32
    )
    candidate_fallback = np.zeros((n_subregions, int(selected_n_clusters)), dtype=bool)
    subregion_basic_niche_ids = [
        np.zeros(0, dtype=np.int32) for _ in range(n_subregions)
    ]
    return MultilevelOTResult(
        basic_niche_size_um=float(basic_niche_size_um)
        if used_basic_niches and basic_niche_size_um is not None
        else None,
        basic_niche_centers_um=np.zeros((0, 2), dtype=np.float32),
        basic_niche_members=[],
        subregion_basic_niche_ids=subregion_basic_niche_ids,
        subregion_centers_um=np.asarray(centers_um, dtype=np.float32),
        subregion_members=[
            np.asarray(members, dtype=np.int32) for members in subregion_members
        ],
        subregion_argmin_labels=final_argmin_labels,
        subregion_forced_label_mask=np.asarray(
            latent_fit["forced_label_mask"], dtype=bool
        ),
        subregion_geometry_point_counts=np.asarray(
            [len(members) for members in subregion_members], dtype=np.int32
        ),
        subregion_geometry_sources=["observed_point_cloud_fast_area_capped"]
        * n_subregions,
        subregion_geometry_used_fallback=np.zeros(n_subregions, dtype=bool),
        subregion_normalizer_radius_p95=np.full(n_subregions, np.nan, dtype=np.float32),
        subregion_normalizer_radius_max=np.full(n_subregions, np.nan, dtype=np.float32),
        subregion_normalizer_interpolation_residual=np.full(
            n_subregions, np.nan, dtype=np.float32
        ),
        subregion_cluster_labels=labels,
        subregion_cluster_probs=subregion_cluster_probs.astype(np.float32),
        subregion_cluster_costs=final_costs.astype(np.float32),
        subregion_cluster_transport_costs=np.zeros_like(final_costs, dtype=np.float32),
        subregion_cluster_overlap_penalties=np.zeros_like(
            final_costs, dtype=np.float32
        ),
        subregion_atom_weights=atom_weights.astype(np.float32),
        subregion_measure_summaries=measure_summaries.astype(np.float32),
        subregion_latent_embeddings=subregion_latent_embeddings.astype(np.float32),
        subregion_latent_embedding_mode=subregion_latent_embedding_mode,
        subregion_latent_embedding_metadata=subregion_latent_embedding_metadata,
        subregion_latent_shrinkage_alpha=np.asarray(
            subregion_latent_diagnostics["shrinkage_alpha"], dtype=np.float32
        ),
        subregion_latent_raw_to_shrunk_distance=np.asarray(
            subregion_latent_diagnostics["raw_to_shrunk_distance"], dtype=np.float32
        ),
        subregion_sample_ids=np.asarray(
            subregion_latent_diagnostics["sample_ids"], dtype=object
        ),
        subregion_clustering_method="pooled_subregion_latent",
        subregion_clustering_uses_spatial=False,
        subregion_assigned_effective_eps=np.full(
            n_subregions, float(ot_eps), dtype=np.float32
        ),
        subregion_assigned_used_ot_fallback=false_sub,
        subregion_candidate_effective_eps_matrix=candidate_eps,
        subregion_candidate_used_ot_fallback_matrix=candidate_fallback,
        subregion_assigned_geometry_transport_costs=zero_sub,
        subregion_assigned_feature_transport_costs=assigned,
        subregion_assigned_transform_penalties=zero_sub,
        subregion_assigned_overlap_consistency_penalties=zero_sub,
        subregion_assigned_transform_rotation_deg=zero_sub,
        subregion_assigned_transform_reflection=false_sub,
        subregion_assigned_transform_scale=np.ones(n_subregions, dtype=np.float32),
        subregion_assigned_transform_translation_norm=zero_sub,
        cluster_supports=cluster_supports,
        cluster_atom_coords=atom_coords,
        cluster_atom_features=atom_features,
        cluster_prototype_weights=betas,
        cell_feature_cluster_probs=cell_probs,
        cell_context_cluster_probs=cell_probs,
        cell_cluster_probs=cell_probs,
        cell_cluster_labels=cell_labels,
        spot_latent_cell_indices=spot_latent["spot_latent_cell_indices"].astype(
            np.int32
        ),
        spot_latent_subregion_ids=spot_latent["spot_latent_subregion_ids"].astype(
            np.int32
        ),
        spot_latent_cluster_labels=spot_latent["spot_latent_cluster_labels"].astype(
            np.int32
        ),
        spot_latent_coords=spot_latent["spot_latent_coords"].astype(np.float32),
        spot_latent_within_coords=spot_latent["spot_latent_within_coords"].astype(
            np.float32
        ),
        spot_latent_cluster_anchors=spot_latent["spot_latent_cluster_anchors"].astype(
            np.float32
        ),
        spot_latent_atom_embedding=spot_latent["spot_latent_atom_embedding"].astype(
            np.float32
        ),
        spot_latent_aligned_coords=spot_latent["spot_latent_aligned_coords"].astype(
            np.float32
        ),
        spot_latent_cluster_probs=spot_latent["spot_latent_cluster_probs"].astype(
            np.float32
        ),
        spot_latent_atom_confidence=spot_latent["spot_latent_atom_confidence"].astype(
            np.float32
        ),
        spot_latent_posterior_entropy=spot_latent[
            "spot_latent_posterior_entropy"
        ].astype(np.float32),
        spot_latent_normalized_posterior_entropy=spot_latent[
            "spot_latent_normalized_posterior_entropy"
        ].astype(np.float32),
        spot_latent_atom_argmax=spot_latent["spot_latent_atom_argmax"].astype(np.int32),
        spot_latent_temperature_used=spot_latent["spot_latent_temperature_used"].astype(
            np.float32
        ),
        spot_latent_temperature_cost_gap=spot_latent[
            "spot_latent_temperature_cost_gap"
        ].astype(np.float32),
        spot_latent_temperature_fixed=spot_latent[
            "spot_latent_temperature_fixed"
        ].astype(np.float32),
        spot_latent_weights=spot_latent["spot_latent_weights"].astype(np.float32),
        spot_latent_atom_posteriors=spot_latent["spot_latent_atom_posteriors"].astype(
            np.float32
        ),
        spot_latent_posterior_entropy_cost_gap=spot_latent[
            "spot_latent_posterior_entropy_cost_gap"
        ].astype(np.float32),
        spot_latent_normalized_posterior_entropy_cost_gap=spot_latent[
            "spot_latent_normalized_posterior_entropy_cost_gap"
        ].astype(np.float32),
        spot_latent_posterior_entropy_fixed=spot_latent[
            "spot_latent_posterior_entropy_fixed"
        ].astype(np.float32),
        spot_latent_normalized_posterior_entropy_fixed=spot_latent[
            "spot_latent_normalized_posterior_entropy_fixed"
        ].astype(np.float32),
        spot_latent_cluster_anchor_distance=spot_latent[
            "spot_latent_cluster_anchor_distance"
        ].astype(np.float32),
        spot_latent_cluster_anchor_ot_fallback_matrix=spot_latent[
            "spot_latent_cluster_anchor_ot_fallback_matrix"
        ].astype(bool),
        spot_latent_cluster_anchor_solver_status_matrix=spot_latent[
            "spot_latent_cluster_anchor_solver_status_matrix"
        ].astype(np.int8),
        spot_latent_cluster_anchor_ot_fallback_fraction=float(
            spot_latent["spot_latent_cluster_anchor_ot_fallback_fraction"].item()
        ),
        spot_latent_atom_mds_stress=spot_latent["spot_latent_atom_mds_stress"].astype(
            np.float32
        ),
        spot_latent_atom_mds_positive_eigenvalue_mass_2d=spot_latent[
            "spot_latent_atom_mds_positive_eigenvalue_mass_2d"
        ].astype(np.float32),
        spot_latent_atom_mds_negative_eigenvalue_mass_fraction=spot_latent[
            "spot_latent_atom_mds_negative_eigenvalue_mass_fraction"
        ].astype(np.float32),
        cell_spot_latent_unweighted_coords=spot_latent[
            "cell_spot_latent_unweighted_coords"
        ].astype(np.float32),
        cell_spot_latent_confidence_weighted_coords=spot_latent[
            "cell_spot_latent_confidence_weighted_coords"
        ].astype(np.float32),
        cell_spot_latent_coords=spot_latent["cell_spot_latent_coords"].astype(
            np.float32
        ),
        cell_spot_latent_cluster_labels=spot_latent[
            "cell_spot_latent_cluster_labels"
        ].astype(np.int32),
        cell_spot_latent_weights=spot_latent["cell_spot_latent_weights"].astype(
            np.float32
        ),
        cell_spot_latent_posterior_entropy=spot_latent[
            "cell_spot_latent_posterior_entropy"
        ].astype(np.float32),
        spot_latent_mode=str(spot_latent["spot_latent_mode"].item()),
        spot_latent_chart_learning_mode=str(
            spot_latent["spot_latent_chart_learning_mode"].item()
        ),
        spot_latent_projection_mode=str(
            spot_latent["spot_latent_projection_mode"].item()
        ),
        spot_latent_validation_role=str(
            spot_latent["spot_latent_validation_role"].item()
        ),
        spot_latent_global_within_scale=float(
            spot_latent["spot_latent_global_within_scale"].item()
        ),
        spot_latent_assignment_temperature=float(
            spot_latent["spot_latent_assignment_temperature"].item()
        ),
        spot_latent_temperature_mode=str(
            spot_latent["spot_latent_temperature_mode"].item()
        ),
        spot_latent_cluster_anchor_distance_method=str(
            spot_latent["spot_latent_cluster_anchor_distance_method"].item()
        ),
        spot_latent_cluster_anchor_distance_requested_method=str(
            spot_latent["spot_latent_cluster_anchor_distance_requested_method"].item()
        ),
        spot_latent_cluster_anchor_distance_effective_method=str(
            spot_latent["spot_latent_cluster_anchor_distance_effective_method"].item()
        ),
        spot_latent_cluster_mds_stress=float(
            spot_latent["spot_latent_cluster_mds_stress"].item()
        ),
        spot_latent_cluster_mds_positive_eigenvalue_mass_2d=float(
            spot_latent["spot_latent_cluster_mds_positive_eigenvalue_mass_2d"].item()
        ),
        spot_latent_cluster_mds_negative_eigenvalue_mass_fraction=float(
            spot_latent[
                "spot_latent_cluster_mds_negative_eigenvalue_mass_fraction"
            ].item()
        ),
        cost_scale_x=1.0,
        cost_scale_y=1.0,
        objective_history=[
            {
                "iteration": 0.0,
                "objective": float(latent_fit["inertia"]),
                "label_assignment_source": "pooled_subregion_latent_embeddings",
                "fast_pooled_latent_only": 1.0,
            }
        ],
        selected_restart=0,
        restart_summaries=restart_summaries,
        min_subregions_per_cluster=int(requested_min_subregions_per_cluster),
        effective_min_subregions_per_cluster=int(
            effective_min_cluster_size(
                n_subregions, int(selected_n_clusters), int(min_subregions_per_cluster)
            )
        ),
        auto_k_selection=auto_k_selection,
    )


def fit_multilevel_ot(
    features: np.ndarray,
    coords_um: np.ndarray,
    *,
    sample_ids: np.ndarray | None = None,
    subregion_members: list[np.ndarray] | None = None,
    subregion_centers_um: np.ndarray | None = None,
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
    region_geometries: list[RegionGeometry] | None = None,
    build_grid_subregions: bool | None = None,
    build_generated_subregions: bool = True,
    allow_convex_hull_fallback: bool = False,
    max_iter: int = 10,
    tol: float = 1e-4,
    overlap_consistency_weight: float = 0.0,
    overlap_jaccard_min: float = 0.15,
    overlap_contrast_scale: float = 1.0,
    basic_niche_size_um: float | None = 200.0,
    max_subregion_area_um2: float | None = None,
    subregion_construction_method: str = "data_driven",
    subregion_feature_weight: float = 0.0,
    subregion_feature_dims: int = 16,
    deep_segmentation_knn: int = 12,
    deep_segmentation_feature_dims: int = 32,
    deep_segmentation_feature_weight: float = 1.0,
    deep_segmentation_spatial_weight: float = 0.05,
    joint_refinement_iters: int = 2,
    joint_refinement_knn: int = 12,
    joint_refinement_feature_dims: int = 32,
    joint_refinement_cluster_weight: float = 1.0,
    joint_refinement_spatial_weight: float = 0.25,
    joint_refinement_cut_weight: float = 0.5,
    joint_refinement_max_move_fraction: float = 0.05,
    compute_spot_latent: bool = True,
    subregion_clustering_method: str = HETEROGENEITY_DESCRIPTOR_MODE,
    subregion_latent_embedding_mode: str = "mean_std_shrunk",
    subregion_latent_shrinkage_tau: float = 25.0,
    subregion_latent_heterogeneity_weight: float = _SUBREGION_LATENT_HETEROGENEITY_WEIGHT,
    subregion_latent_sample_prior_weight: float = 0.5,
    subregion_latent_codebook_size: int = 32,
    subregion_latent_codebook_sample_size: int = 50000,
    heterogeneity_composition_weight: float = 0.20,
    heterogeneity_diversity_weight: float = 0.15,
    heterogeneity_spatial_field_weight: float = 0.35,
    heterogeneity_pair_cooccurrence_weight: float = 0.30,
    heterogeneity_pair_distance_bins: str | tuple[float, ...] | None = None,
    heterogeneity_pair_graph_mode: str = "all_pairs",
    heterogeneity_pair_graph_k: int = 8,
    heterogeneity_pair_graph_radius: float | None = None,
    heterogeneity_pair_bin_normalization: str = "per_bin",
    heterogeneity_transport_max_subregions: int = 800,
    heterogeneity_transport_feature_mode: str = "soft_codebook",
    heterogeneity_transport_feature_cost: str = "hellinger_codebook",
    heterogeneity_fused_ot_feature_weight: float = 0.5,
    heterogeneity_fused_ot_coordinate_weight: float = 0.5,
    heterogeneity_fgw_alpha: float = 0.5,
    heterogeneity_fgw_solver: str = "conditional_gradient",
    heterogeneity_fgw_epsilon: float = 0.05,
    heterogeneity_fgw_loss_fun: str = "square_loss",
    heterogeneity_fgw_max_iter: int = 500,
    heterogeneity_fgw_tol: float = 1e-7,
    heterogeneity_fgw_structure_scale: str | float = "global_median",
    heterogeneity_fgw_structure_clip: float | None = 3.0,
    heterogeneity_fgw_partial: bool = False,
    heterogeneity_fgw_partial_mass: float = 0.85,
    heterogeneity_fgw_partial_reg: float = 0.05,
    auto_n_clusters: bool = False,
    candidate_n_clusters: tuple[int, ...] | list[int] | str | None = None,
    auto_k_max_score_subregions: int = 2500,
    auto_k_gap_references: int = 8,
    auto_k_mds_components: int = 8,
    auto_k_pilot_n_init: int = 1,
    auto_k_pilot_max_iter: int = 3,
    min_subregions_per_cluster: int = 1,
    seed: int = 1337,
    compute_device: str = "auto",
) -> MultilevelOTResult:
    features = np.asarray(features, dtype=np.float32)
    coords_um = np.asarray(coords_um, dtype=np.float32)
    resolved_compute_device = _resolve_compute_device(compute_device)
    if build_grid_subregions is not None:
        build_generated_subregions = bool(build_grid_subregions)
    _validate_fit_inputs(
        features=features,
        coords_um=coords_um,
        n_clusters=n_clusters,
        atoms_per_cluster=atoms_per_cluster,
        radius_um=radius_um,
        stride_um=stride_um,
        basic_niche_size_um=basic_niche_size_um,
        min_cells=min_cells,
        max_subregions=max_subregions,
        max_subregion_area_um2=max_subregion_area_um2,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        geometry_eps=geometry_eps,
        ot_eps=ot_eps,
        rho=rho,
        geometry_samples=geometry_samples,
        compressed_support_size=compressed_support_size,
        align_iters=align_iters,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        min_scale=min_scale,
        max_scale=max_scale,
    )
    if int(auto_k_max_score_subregions) < 0:
        raise ValueError("auto_k_max_score_subregions must be >= 0")
    if int(auto_k_gap_references) < 0:
        raise ValueError("auto_k_gap_references must be >= 0")
    if int(auto_k_mds_components) < 1:
        raise ValueError("auto_k_mds_components must be at least 1")
    if int(auto_k_pilot_n_init) < 1 or int(auto_k_pilot_max_iter) < 1:
        raise ValueError(
            "auto_k_pilot_n_init and auto_k_pilot_max_iter must be at least 1"
        )
    if int(min_subregions_per_cluster) < 1:
        raise ValueError("min_subregions_per_cluster must be at least 1")
    construction_method = str(subregion_construction_method).strip().lower()
    if construction_method not in {
        "data_driven",
        "deep_segmentation",
        "joint_refinement",
    }:
        raise ValueError(
            "subregion_construction_method must be 'data_driven', 'deep_segmentation', or 'joint_refinement'."
        )
    requested_clustering_method = str(subregion_clustering_method).strip().lower()
    clustering_method = (
        HETEROGENEITY_DESCRIPTOR_MODE
        if requested_clustering_method == LEGACY_HETEROGENEITY_OT_ALIAS
        else requested_clustering_method
    )
    subregion_latent_embedding_mode = _normalize_subregion_latent_embedding_mode(
        subregion_latent_embedding_mode
    )
    subregion_latent_shrinkage_tau = max(float(subregion_latent_shrinkage_tau), 0.0)
    subregion_latent_heterogeneity_weight = max(
        float(subregion_latent_heterogeneity_weight), 0.0
    )
    subregion_latent_sample_prior_weight = float(
        np.clip(float(subregion_latent_sample_prior_weight), 0.0, 1.0)
    )
    subregion_latent_codebook_size = max(int(subregion_latent_codebook_size), 2)
    subregion_latent_codebook_sample_size = max(
        int(subregion_latent_codebook_sample_size), subregion_latent_codebook_size
    )
    heterogeneity_block_weights = {
        "composition": max(float(heterogeneity_composition_weight), 0.0),
        "diversity": max(float(heterogeneity_diversity_weight), 0.0),
        "spatial_field": max(float(heterogeneity_spatial_field_weight), 0.0),
        "pair_cooccurrence": max(float(heterogeneity_pair_cooccurrence_weight), 0.0),
    }
    if sum(heterogeneity_block_weights.values()) <= 1e-12:
        raise ValueError("at least one heterogeneity descriptor block weight must be positive")
    heterogeneity_pair_graph_mode = (
        str(heterogeneity_pair_graph_mode).strip().lower().replace("-", "_")
    )
    if heterogeneity_pair_graph_mode not in {"all_pairs", "knn", "radius"}:
        raise ValueError(
            "heterogeneity_pair_graph_mode must be one of all_pairs, knn, radius"
        )
    heterogeneity_pair_graph_k = max(int(heterogeneity_pair_graph_k), 1)
    if (
        heterogeneity_pair_graph_radius is not None
        and float(heterogeneity_pair_graph_radius) <= 0
    ):
        heterogeneity_pair_graph_radius = None
    heterogeneity_transport_max_subregions = max(
        int(heterogeneity_transport_max_subregions), 0
    )
    heterogeneity_transport_feature_mode = (
        str(heterogeneity_transport_feature_mode).strip().lower()
    )
    heterogeneity_transport_feature_cost = (
        str(heterogeneity_transport_feature_cost).strip().lower()
    )
    heterogeneity_fused_ot_feature_weight = max(
        float(heterogeneity_fused_ot_feature_weight), 0.0
    )
    heterogeneity_fused_ot_coordinate_weight = max(
        float(heterogeneity_fused_ot_coordinate_weight), 0.0
    )
    if (
        heterogeneity_fused_ot_feature_weight
        + heterogeneity_fused_ot_coordinate_weight
        <= 1e-12
    ):
        raise ValueError("at least one heterogeneity fused-OT weight must be positive")
    heterogeneity_fgw_alpha = float(np.clip(float(heterogeneity_fgw_alpha), 0.0, 1.0))
    heterogeneity_fgw_solver = str(heterogeneity_fgw_solver).strip().lower()
    heterogeneity_fgw_epsilon = max(float(heterogeneity_fgw_epsilon), 1e-8)
    heterogeneity_fgw_loss_fun = str(heterogeneity_fgw_loss_fun).strip().lower()
    heterogeneity_fgw_max_iter = max(int(heterogeneity_fgw_max_iter), 1)
    heterogeneity_fgw_tol = max(float(heterogeneity_fgw_tol), 1e-12)
    if (
        heterogeneity_fgw_structure_clip is not None
        and float(heterogeneity_fgw_structure_clip) <= 0
    ):
        heterogeneity_fgw_structure_clip = None
    heterogeneity_fgw_partial_mass = float(
        np.clip(float(heterogeneity_fgw_partial_mass), 1e-8, 1.0)
    )
    heterogeneity_fgw_partial_reg = max(float(heterogeneity_fgw_partial_reg), 1e-8)
    if clustering_method not in {
        "pooled_subregion_latent",
        HETEROGENEITY_DESCRIPTOR_MODE,
        HETEROGENEITY_FUSED_OT_MODE,
        HETEROGENEITY_FGW_MODE,
        "ot_dictionary",
    }:
        raise ValueError(
            "subregion_clustering_method must be 'pooled_subregion_latent', "
            f"'{HETEROGENEITY_DESCRIPTOR_MODE}', '{HETEROGENEITY_FUSED_OT_MODE}', "
            f"'{HETEROGENEITY_FGW_MODE}', legacy alias "
            f"'{LEGACY_HETEROGENEITY_OT_ALIAS}', or 'ot_dictionary'."
        )
    subregion_feature_weight = max(0.0, float(subregion_feature_weight))
    subregion_feature_dims = max(0, int(subregion_feature_dims))
    joint_refinement_iters = max(0, int(joint_refinement_iters))
    joint_refinement_knn = max(2, int(joint_refinement_knn))
    joint_refinement_feature_dims = max(1, int(joint_refinement_feature_dims))
    joint_refinement_cluster_weight = max(0.0, float(joint_refinement_cluster_weight))
    joint_refinement_spatial_weight = max(0.0, float(joint_refinement_spatial_weight))
    joint_refinement_cut_weight = max(0.0, float(joint_refinement_cut_weight))
    joint_refinement_max_move_fraction = float(
        np.clip(float(joint_refinement_max_move_fraction), 0.0, 1.0)
    )

    _progress(
        f"standardizing feature matrix for {features.shape[0]} cells and {features.shape[1]} features"
    )
    features = _standardize_features(features)
    used_basic_niches = False
    basic_niche_centers_um = np.zeros((0, 2), dtype=np.float32)
    basic_niche_members: list[np.ndarray] = []
    subregion_basic_niche_ids: list[np.ndarray] = []
    proposal_region_geometries: list[RegionGeometry] | None = None
    if subregion_members is None:
        if region_geometries is not None:
            _progress(f"using {len(region_geometries)} explicit region geometries")
            subregion_members = [
                np.asarray(region.members, dtype=np.int32)
                for region in region_geometries
            ]
            subregion_basic_niche_ids = [
                np.asarray([], dtype=np.int32) for _ in subregion_members
            ]
        elif build_generated_subregions:
            target_scale_for_subregions = (
                float(basic_niche_size_um)
                if basic_niche_size_um is not None
                else float(stride_um)
            )
            if construction_method in {"deep_segmentation", "joint_refinement"}:
                refinement_suffix = (
                    "; cluster-aware joint refinement will run after initial pooled-latent clustering"
                    if construction_method == "joint_refinement"
                    else ""
                )
                _progress(
                    "building mutually exclusive deep-graph segmentation subregions "
                    f"(target_scale={target_scale_for_subregions:g}um, knn={int(deep_segmentation_knn)}, "
                    f"cap={int(max_subregions)}, soft_area_target={max_subregion_area_um2}; "
                    "learned embedding affinity drives boundary cuts; min_cells and connectivity are hard constraints"
                    f"{refinement_suffix})"
                )
                used_basic_niches = True
                (
                    subregion_centers_um,
                    subregion_members,
                    basic_niche_centers_um,
                    basic_niche_members,
                    subregion_basic_niche_ids,
                ) = build_deep_graph_segmentation_subregions(
                    coords_um=coords_um,
                    segmentation_features=features,
                    target_scale_um=target_scale_for_subregions,
                    min_cells=min_cells,
                    max_subregions=max_subregions,
                    max_area_um2=max_subregion_area_um2,
                    segmentation_knn=int(deep_segmentation_knn),
                    segmentation_feature_dims=int(deep_segmentation_feature_dims),
                    segmentation_feature_weight=float(deep_segmentation_feature_weight),
                    segmentation_spatial_weight=float(deep_segmentation_spatial_weight),
                    seed=seed,
                )
                proposal_region_geometries = _region_geometries_from_observed_points(
                    subregion_members
                )
            elif basic_niche_size_um is not None:
                _progress(
                    "building mutually exclusive data-driven subregions "
                    f"(target_scale={float(basic_niche_size_um):g}um, stride={float(stride_um):g}um, "
                    f"feature_weight={subregion_feature_weight:g}, cap={int(max_subregions)}, "
                    f"soft_area_target={max_subregion_area_um2}; "
                    "radius_um is not a membership radius)"
                )
                used_basic_niches = True
                (
                    subregion_centers_um,
                    subregion_members,
                    basic_niche_centers_um,
                    basic_niche_members,
                    subregion_basic_niche_ids,
                ) = build_composite_subregions_from_basic_niches(
                    coords_um=coords_um,
                    radius_um=radius_um,
                    stride_um=stride_um,
                    min_cells=min_cells,
                    max_subregions=max_subregions,
                    basic_niche_size_um=float(basic_niche_size_um),
                    partition_features=features,
                    partition_feature_weight=subregion_feature_weight,
                    partition_feature_dims=subregion_feature_dims,
                    max_area_um2=max_subregion_area_um2,
                    seed=seed,
                )
                proposal_region_geometries = _region_geometries_from_observed_points(
                    subregion_members
                )
            else:
                _progress(
                    "building mutually exclusive data-driven subregions "
                    f"(target_scale={float(stride_um):g}um, feature_weight={subregion_feature_weight:g}, "
                    f"cap={int(max_subregions)}, soft_area_target={max_subregion_area_um2}; "
                    "radius_um is not a membership radius); geometry is data-driven"
                )
                (
                    subregion_centers_um,
                    subregion_members,
                    _grid_tile_centers_um,
                    _grid_tile_members,
                    _subregion_grid_tile_ids,
                ) = build_partition_subregions_from_grid_tiles(
                    coords_um=coords_um,
                    radius_um=radius_um,
                    stride_um=stride_um,
                    min_cells=min_cells,
                    max_subregions=max_subregions,
                    max_area_um2=max_subregion_area_um2,
                    partition_features=features,
                    partition_feature_weight=subregion_feature_weight,
                    partition_feature_dims=subregion_feature_dims,
                    seed=seed,
                )
                subregion_basic_niche_ids = [
                    np.asarray([], dtype=np.int32) for _ in subregion_members
                ]
                proposal_region_geometries = _region_geometries_from_observed_points(
                    subregion_members
                )
        else:
            raise ValueError(
                "Explicit region_geometries or subregion_members are required when generated subregion construction is disabled."
            )
    else:
        subregion_members = [
            np.asarray(members, dtype=np.int32) for members in subregion_members
        ]
        subregion_basic_niche_ids = [
            np.asarray([], dtype=np.int32) for _ in subregion_members
        ]
    if region_geometries is None:
        region_geometries = (
            proposal_region_geometries
            or _region_geometries_from_members(subregion_members)
        )
    if len(subregion_members) == 0:
        raise RuntimeError("No valid subregions were provided or constructed.")
    if len(region_geometries) != len(subregion_members):
        raise ValueError(
            "region_geometries must have the same length as the constructed subregions."
        )
    centers_um: np.ndarray | None = None
    if subregion_centers_um is not None:
        centers_um = np.asarray(subregion_centers_um, dtype=np.float32)
        if centers_um.ndim != 2 or centers_um.shape[1] != 2:
            raise ValueError(
                "subregion_centers_um must be a 2D array with shape (n_subregions, 2)."
            )
        if centers_um.shape[0] != len(subregion_members):
            raise ValueError("subregion_centers_um must have one row per subregion.")
    try:
        _validate_mutually_exclusive_memberships(features.shape[0], subregion_members)
    except RuntimeError as exc:
        raise ValueError(
            "Subregion memberships must be mutually exclusive and contain valid non-empty cell indices."
        ) from exc

    keep_idx = [
        idx
        for idx, members in enumerate(subregion_members)
        if np.asarray(members).size >= int(min_cells)
    ]
    if len(keep_idx) != len(subregion_members):
        if not keep_idx:
            raise RuntimeError("No valid subregions remain after applying min_cells.")
        keep_arr = np.asarray(keep_idx, dtype=np.int32)
        if centers_um is not None:
            centers_um = centers_um[keep_arr]
        subregion_members = [subregion_members[int(idx)] for idx in keep_idx]
        subregion_basic_niche_ids = [
            subregion_basic_niche_ids[int(idx)] for idx in keep_idx
        ]
        region_geometries = [region_geometries[int(idx)] for idx in keep_idx]
        try:
            _validate_mutually_exclusive_memberships(
                features.shape[0], subregion_members
            )
        except RuntimeError as exc:
            raise ValueError(
                "Subregion memberships must remain mutually exclusive after min_cells filtering."
            ) from exc
    if centers_um is None:
        centers_um = np.vstack(
            [
                np.asarray(coords_um[members], dtype=np.float32).mean(axis=0)
                for members in subregion_members
            ]
        ).astype(np.float32)

    requested_n_clusters = int(n_clusters)
    candidate_ks = sanitize_candidate_n_clusters(
        candidate_n_clusters,
        fallback_n_clusters=requested_n_clusters,
        n_subregions=int(centers_um.shape[0]),
    )
    required_n_clusters = (
        max(candidate_ks) if bool(auto_n_clusters) else requested_n_clusters
    )
    if centers_um.shape[0] < required_n_clusters:
        raise ValueError(
            f"n_clusters={required_n_clusters} exceeds the number of constructed subregions={centers_um.shape[0]}."
        )
    requested_min_subregions_per_cluster = int(min_subregions_per_cluster)
    effective_min_for_requested_k = effective_min_cluster_size(
        int(centers_um.shape[0]),
        int(required_n_clusters),
        requested_min_subregions_per_cluster,
    )
    if effective_min_for_requested_k < requested_min_subregions_per_cluster:
        _progress(
            f"requested min_subregions_per_cluster={requested_min_subregions_per_cluster} exceeds the "
            f"available average for K={required_n_clusters}; using effective minimum {effective_min_for_requested_k}"
        )

    joint_refinement_summary: dict[str, object] = {
        "enabled": bool(construction_method == "joint_refinement"),
        "applied": False,
    }
    if construction_method == "joint_refinement":
        if not build_generated_subregions:
            raise ValueError(
                "subregion_construction_method='joint_refinement' requires generated subregions."
            )
        if clustering_method not in {"pooled_subregion_latent", HETEROGENEITY_DESCRIPTOR_MODE}:
            raise ValueError(
                "subregion_construction_method='joint_refinement' requires "
                "an embedding-based subregion_clustering_method."
            )
        _progress(
            "running constrained joint segmentation-clustering refinement "
            f"(iters={joint_refinement_iters}, knn={joint_refinement_knn}, "
            f"max_move_fraction={joint_refinement_max_move_fraction:g})"
        )
        initial_embeddings, _initial_diagnostics = (
            _build_subregion_latent_embeddings_from_members(
                features,
                subregion_members,
                mode=subregion_latent_embedding_mode,
                shrinkage_tau=subregion_latent_shrinkage_tau,
                heterogeneity_weight=subregion_latent_heterogeneity_weight,
                sample_ids=sample_ids,
                sample_prior_weight=subregion_latent_sample_prior_weight,
                codebook_size=subregion_latent_codebook_size,
                codebook_sample_size=subregion_latent_codebook_sample_size,
                random_state=int(seed),
                return_diagnostics=True,
            )
        )
        preliminary_k = int(requested_n_clusters)
        preliminary_auto_k_selection: dict[str, object] | None = None
        if bool(auto_n_clusters):
            preliminary_auto_k_selection = (
                comprehensive_select_k_from_latent_embeddings(
                    initial_embeddings,
                    candidate_n_clusters=candidate_ks,
                    fallback_n_clusters=int(requested_n_clusters),
                    seeds=tuple(
                        int(seed) + offset
                        for offset in range(
                            max(
                                3,
                                min(
                                    8, _env_int("SPATIAL_OT_AUTO_K_STABILITY_SEEDS", 5)
                                ),
                            )
                        )
                    ),
                    n_init=max(1, int(auto_k_pilot_n_init)),
                    max_silhouette_subregions=int(auto_k_max_score_subregions),
                    gap_references=int(auto_k_gap_references),
                    bootstrap_repeats=max(
                        0, _env_int("SPATIAL_OT_AUTO_K_BOOTSTRAP_REPEATS", 3)
                    ),
                    bootstrap_fraction=float(
                        _env_float("SPATIAL_OT_AUTO_K_BOOTSTRAP_FRACTION", 0.8)
                    ),
                    min_cluster_size=int(effective_min_for_requested_k),
                    random_state=int(seed),
                )
            )
            preliminary_k = int(preliminary_auto_k_selection["selected_k"])
        preliminary_auto_k_summary = None
        if preliminary_auto_k_selection is not None:
            preliminary_auto_k_summary = {
                "selected_k": int(
                    preliminary_auto_k_selection.get("selected_k", preliminary_k)
                ),
                "candidate_n_clusters": [
                    int(k)
                    for k in preliminary_auto_k_selection.get(
                        "candidate_n_clusters", candidate_ks
                    )
                ],
                "criterion_votes": {
                    str(key): int(value)
                    for key, value in dict(
                        preliminary_auto_k_selection.get("criterion_votes", {})
                    ).items()
                },
            }
        preliminary_fit = fit_kmeans_on_latent_embeddings(
            initial_embeddings,
            n_clusters=int(preliminary_k),
            n_init=max(int(n_init), 1),
            min_cluster_size=int(
                effective_min_cluster_size(
                    len(subregion_members),
                    preliminary_k,
                    int(min_subregions_per_cluster),
                )
            ),
            random_state=int(seed),
        )
        initial_region_count = int(len(subregion_members))
        initial_cell_memberships = int(
            sum(np.asarray(member, dtype=np.int32).size for member in subregion_members)
        )
        refined_centers, refined_members, refined_basic_ids, refinement_history = (
            refine_subregions_by_cluster_coherence(
                coords_um=coords_um,
                features=features,
                members=subregion_members,
                subregion_cluster_labels=np.asarray(
                    preliminary_fit["labels"], dtype=np.int32
                ),
                min_cells=int(min_cells),
                max_subregions=int(max_subregions),
                target_scale_um=float(
                    basic_niche_size_um
                    if basic_niche_size_um is not None
                    else stride_um
                ),
                n_iters=int(joint_refinement_iters),
                n_neighbors=int(joint_refinement_knn),
                cluster_weight=float(joint_refinement_cluster_weight),
                spatial_weight=float(joint_refinement_spatial_weight),
                cut_weight=float(joint_refinement_cut_weight),
                max_move_fraction=float(joint_refinement_max_move_fraction),
                feature_dims=int(joint_refinement_feature_dims),
                seed=int(seed),
            )
        )
        refined_cell_memberships = int(
            sum(np.asarray(member, dtype=np.int32).size for member in refined_members)
        )
        centers_um = refined_centers.astype(np.float32)
        subregion_members = [
            np.asarray(member, dtype=np.int32) for member in refined_members
        ]
        subregion_basic_niche_ids = [
            np.asarray(ids, dtype=np.int32) for ids in refined_basic_ids
        ]
        region_geometries = _region_geometries_from_observed_points(subregion_members)
        _validate_mutually_exclusive_memberships(
            features.shape[0], subregion_members, require_full_coverage=True
        )
        accepted_moves = float(
            sum(float(row.get("accepted_moves", 0.0)) for row in refinement_history)
        )
        joint_refinement_summary = {
            "enabled": True,
            "applied": True,
            "initial_region_count": initial_region_count,
            "final_region_count": int(len(subregion_members)),
            "initial_cell_memberships": initial_cell_memberships,
            "final_cell_memberships": refined_cell_memberships,
            "preliminary_k": int(preliminary_k),
            "preliminary_auto_k_selection": preliminary_auto_k_summary,
            "history": refinement_history,
            "accepted_boundary_moves": accepted_moves,
            "moved_cell_fraction": float(accepted_moves / max(features.shape[0], 1)),
            "cluster_weight": float(joint_refinement_cluster_weight),
            "spatial_weight": float(joint_refinement_spatial_weight),
            "cut_weight": float(joint_refinement_cut_weight),
            "max_move_fraction_per_iter": float(joint_refinement_max_move_fraction),
            "feature_dims": int(joint_refinement_feature_dims),
            "requires_connected_output": True,
            "requires_min_cells": True,
            "uses_soft_area_target": bool(max_subregion_area_um2 is not None),
        }
        _progress(
            "joint refinement finished: "
            f"{initial_region_count} -> {len(subregion_members)} subregions, "
            f"accepted_moves={accepted_moves:g}"
        )
        candidate_ks = sanitize_candidate_n_clusters(
            candidate_n_clusters,
            fallback_n_clusters=requested_n_clusters,
            n_subregions=int(centers_um.shape[0]),
        )
        required_n_clusters = (
            max(candidate_ks) if bool(auto_n_clusters) else requested_n_clusters
        )
        if centers_um.shape[0] < required_n_clusters:
            raise ValueError(
                f"n_clusters={required_n_clusters} exceeds the number of refined subregions={centers_um.shape[0]}."
            )
        effective_min_for_requested_k = effective_min_cluster_size(
            int(centers_um.shape[0]),
            int(required_n_clusters),
            requested_min_subregions_per_cluster,
        )

    fast_pooled_latent_enabled = _env_bool("SPATIAL_OT_FAST_POOLED_LATENT_ONLY", True)
    if (
        clustering_method == "pooled_subregion_latent"
        and fast_pooled_latent_enabled
        and len(subregion_members) > _fast_pooled_latent_only_threshold()
    ):
        return _fit_pooled_latent_only_result(
            features=features,
            sample_ids=sample_ids,
            centers_um=centers_um,
            subregion_members=subregion_members,
            n_clusters=requested_n_clusters,
            atoms_per_cluster=atoms_per_cluster,
            n_init=n_init,
            auto_n_clusters=auto_n_clusters,
            candidate_ks=candidate_ks,
            auto_k_max_score_subregions=auto_k_max_score_subregions,
            auto_k_gap_references=auto_k_gap_references,
            min_subregions_per_cluster=effective_min_for_requested_k,
            requested_min_subregions_per_cluster=requested_min_subregions_per_cluster,
            subregion_latent_embedding_mode=subregion_latent_embedding_mode,
            subregion_latent_shrinkage_tau=subregion_latent_shrinkage_tau,
            subregion_latent_heterogeneity_weight=subregion_latent_heterogeneity_weight,
            subregion_latent_sample_prior_weight=subregion_latent_sample_prior_weight,
            subregion_latent_codebook_size=subregion_latent_codebook_size,
            subregion_latent_codebook_sample_size=subregion_latent_codebook_sample_size,
            ot_eps=ot_eps,
            seed=seed,
            basic_niche_size_um=basic_niche_size_um,
            used_basic_niches=used_basic_niches,
            joint_refinement_summary=joint_refinement_summary,
        )

    _progress(
        f"constructed {len(subregion_members)} subregions; fitting shape normalizers and compressed measures"
    )
    for rid, (region, members) in enumerate(
        zip(region_geometries, subregion_members, strict=False)
    ):
        region.members = np.asarray(members, dtype=np.int32)
        if not region.region_id:
            region.region_id = f"region_{rid:04d}"

    reference_points, reference_weights = make_reference_points_unit_disk(
        geometry_samples
    )
    measures = _build_subregion_measures(
        features=features,
        coords_um=coords_um,
        centers_um=centers_um,
        region_geometries=region_geometries,
        geometry_reference_points=reference_points,
        geometry_reference_weights=reference_weights,
        geometry_eps=geometry_eps,
        geometry_samples=geometry_samples,
        compressed_support_size=compressed_support_size,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        seed=seed,
        allow_convex_hull_fallback=allow_convex_hull_fallback,
        compute_device=resolved_compute_device,
    )

    _progress("estimating overlap graph and cost scales")
    optimization_measures = _make_optimization_measures(measures)
    summaries = np.vstack([_measure_summary(m) for m in optimization_measures])
    transport_measures = None
    if clustering_method in HETEROGENEITY_DESCRIPTOR_ALIASES:
        _progress(
            "building internal heterogeneity motif embeddings "
            "(soft state composition, diversity, canonical spatial-state field, pair graph)"
        )
        subregion_latent_embeddings, subregion_latent_embedding_metadata = (
            build_internal_heterogeneity_embeddings(
                measures,
                codebook_size=int(subregion_latent_codebook_size),
                codebook_sample_size=int(subregion_latent_codebook_sample_size),
                pair_distance_bins=heterogeneity_pair_distance_bins,
                pair_graph_mode=heterogeneity_pair_graph_mode,
                pair_graph_k=int(heterogeneity_pair_graph_k),
                pair_graph_radius=heterogeneity_pair_graph_radius,
                pair_bin_normalization=heterogeneity_pair_bin_normalization,
                block_weights=heterogeneity_block_weights,
                random_state=int(seed),
                mode=str(requested_clustering_method),
            )
        )
        subregion_latent_embedding_mode = HETEROGENEITY_DESCRIPTOR_MODE
        subregion_latent_diagnostics = {
            "shrinkage_alpha": np.ones(len(subregion_members), dtype=np.float32),
            "raw_to_shrunk_distance": np.zeros(len(subregion_members), dtype=np.float32),
            "sample_ids": _subregion_sample_ids_from_members(
                sample_ids, subregion_members
            ),
            "sample_aware_shrinkage": False,
        }
    elif clustering_method in TRANSPORT_HETEROGENEITY_MODES:
        _progress(
            "building attributed metric-space subregion measures for true "
            f"{clustering_method} transport distances"
        )
        transport_measures, transport_metadata = build_subregion_fgw_measures(
            measures,
            feature_mode=heterogeneity_transport_feature_mode,
            structure_scale=heterogeneity_fgw_structure_scale,
            structure_clip=heterogeneity_fgw_structure_clip,
            codebook_size=int(subregion_latent_codebook_size),
            codebook_sample_size=int(subregion_latent_codebook_sample_size),
            random_state=int(seed),
            sample_ids=_subregion_sample_ids_from_members(sample_ids, subregion_members),
        )
        subregion_latent_embeddings = np.zeros((len(measures), 0), dtype=np.float32)
        subregion_latent_embedding_metadata = {
            "mode": str(clustering_method),
            "requested_mode": str(requested_clustering_method),
            "implemented": True,
            "uses_ot_costs": True,
            "label_assignment_source": "precomputed_transport_distance_matrix",
            "measure_builder": transport_metadata,
            "distance_note": (
                "Subregion labels are assigned from an all-pairs transport distance "
                "matrix over measured attributed metric spaces, not from Euclidean "
                "descriptor vectors."
            ),
        }
        subregion_latent_embedding_mode = str(clustering_method)
        subregion_latent_diagnostics = {
            "shrinkage_alpha": np.ones(len(subregion_members), dtype=np.float32),
            "raw_to_shrunk_distance": np.zeros(len(subregion_members), dtype=np.float32),
            "sample_ids": _subregion_sample_ids_from_members(
                sample_ids, subregion_members
            ),
            "sample_aware_shrinkage": False,
        }
    else:
        subregion_latent_embeddings, subregion_latent_diagnostics = (
            _build_subregion_latent_embeddings_from_members(
                features,
                subregion_members,
                mode=subregion_latent_embedding_mode,
                shrinkage_tau=subregion_latent_shrinkage_tau,
                heterogeneity_weight=subregion_latent_heterogeneity_weight,
                sample_ids=sample_ids,
                sample_prior_weight=subregion_latent_sample_prior_weight,
                codebook_size=subregion_latent_codebook_size,
                codebook_sample_size=subregion_latent_codebook_sample_size,
                random_state=int(seed),
                return_diagnostics=True,
            )
        )
        subregion_latent_embedding_metadata = _subregion_latent_embedding_metadata(
            mode=subregion_latent_embedding_mode,
            shrinkage_tau=subregion_latent_shrinkage_tau,
            heterogeneity_weight=subregion_latent_heterogeneity_weight,
            sample_prior_weight=subregion_latent_sample_prior_weight,
            codebook_size=subregion_latent_codebook_size,
            codebook_sample_size=subregion_latent_codebook_sample_size,
            feature_dim=int(features.shape[1]),
            embedding_dim=int(subregion_latent_embeddings.shape[1]),
            sample_aware_shrinkage=bool(
                subregion_latent_diagnostics.get("sample_aware_shrinkage", False)
            ),
        )
    subregion_latent_embedding_metadata["joint_refinement"] = dict(
        joint_refinement_summary
    )
    overlap_edge_i, overlap_edge_j, overlap_edge_weight = (
        _build_overlap_consistency_graph(
            measures=measures,
            summaries=summaries,
            min_jaccard=max(float(overlap_jaccard_min), 0.0),
            contrast_scale=max(float(overlap_contrast_scale), 1e-6),
        )
    )
    cost_scale_x, cost_scale_y = _estimate_cost_scales(
        measures,
        max_points=5000,
        random_state=seed,
        compute_device=resolved_compute_device,
    )
    auto_k_selection: dict[str, object] | None = None
    selected_n_clusters = requested_n_clusters
    reused_pilot_fit = False
    pilot_restart_summaries: list[dict[str, object]] | None = None
    pilot_best_bundle: dict[str, object] | None = None
    if bool(auto_n_clusters):
        _progress(
            "automatic K selection enabled; candidate K="
            + ",".join(str(k) for k in candidate_ks)
        )
        if clustering_method in {"pooled_subregion_latent", HETEROGENEITY_DESCRIPTOR_MODE}:
            stability_seed_count = max(
                3, min(8, _env_int("SPATIAL_OT_AUTO_K_STABILITY_SEEDS", 5))
            )
            auto_k_selection = comprehensive_select_k_from_latent_embeddings(
                subregion_latent_embeddings,
                candidate_n_clusters=candidate_ks,
                fallback_n_clusters=requested_n_clusters,
                seeds=tuple(
                    int(seed) + offset for offset in range(stability_seed_count)
                ),
                n_init=max(1, int(auto_k_pilot_n_init)),
                max_silhouette_subregions=int(auto_k_max_score_subregions),
                gap_references=int(auto_k_gap_references),
                bootstrap_repeats=max(
                    0, _env_int("SPATIAL_OT_AUTO_K_BOOTSTRAP_REPEATS", 3)
                ),
                bootstrap_fraction=float(
                    _env_float("SPATIAL_OT_AUTO_K_BOOTSTRAP_FRACTION", 0.8)
                ),
                min_cluster_size=int(min_subregions_per_cluster),
                random_state=int(seed),
            )
        elif clustering_method in TRANSPORT_HETEROGENEITY_MODES:
            raise ValueError(
                "auto_n_clusters is not yet implemented for true transport "
                f"mode '{clustering_method}'. Run fixed K values and compare "
                "precomputed-distance stability while the all-pairs FGW/OT path is "
                "validation-grade."
            )
        else:
            pilot_n_clusters = int(max(candidate_ks))
            pilot_n_init = int(auto_k_pilot_n_init)
            pilot_max_iter = int(auto_k_pilot_max_iter)
            _progress(f"auto-K OT pilot K={pilot_n_clusters}")
            _pilot_restart_results, pilot_restart_summaries, pilot_best_bundle = (
                _fit_restart_bundles(
                    measures=optimization_measures,
                    summaries=summaries,
                    n_clusters=pilot_n_clusters,
                    atoms_per_cluster=atoms_per_cluster,
                    lambda_x=lambda_x,
                    lambda_y=lambda_y,
                    ot_eps=ot_eps,
                    rho=rho,
                    align_iters=align_iters,
                    allow_reflection=allow_reflection,
                    allow_scale=allow_scale,
                    cost_scale_x=cost_scale_x,
                    cost_scale_y=cost_scale_y,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    scale_penalty=scale_penalty,
                    shift_penalty=shift_penalty,
                    max_iter=pilot_max_iter,
                    tol=tol,
                    min_subregions_per_cluster=min_subregions_per_cluster,
                    seed=seed,
                    overlap_edge_i=overlap_edge_i,
                    overlap_edge_j=overlap_edge_j,
                    overlap_edge_weight=overlap_edge_weight,
                    overlap_consistency_weight=overlap_consistency_weight,
                    n_init=pilot_n_init,
                    compute_device=compute_device,
                    resolved_compute_device=resolved_compute_device,
                    progress_label="auto-K pilot",
                )
            )
            auto_k_selection = select_k_from_ot_landmark_costs(
                np.asarray(pilot_best_bundle["transport_costs"], dtype=np.float32),
                candidate_n_clusters=candidate_ks,
                fallback_n_clusters=requested_n_clusters,
                max_score_subregions=int(auto_k_max_score_subregions),
                mds_components=int(auto_k_mds_components),
                gap_references=int(auto_k_gap_references),
                min_cluster_size=int(min_subregions_per_cluster),
                random_state=int(seed),
            )
            reused_pilot_fit = (
                int(auto_k_selection["selected_k"]) == pilot_n_clusters
                and pilot_n_init == int(n_init)
                and pilot_max_iter == int(max_iter)
            )
        selected_n_clusters = int(auto_k_selection["selected_k"])
        auto_k_selection.update(
            {
                "requested_n_clusters": int(requested_n_clusters),
                "selection_clustering_method": str(clustering_method),
                "pilot_n_clusters": int(max(candidate_ks))
                if clustering_method == "ot_dictionary"
                else None,
                "pilot_n_init": int(auto_k_pilot_n_init)
                if clustering_method == "ot_dictionary"
                else None,
                "pilot_max_iter": int(auto_k_pilot_max_iter)
                if clustering_method == "ot_dictionary"
                else None,
                "final_refit_n_init": int(n_init),
                "final_refit_max_iter": int(max_iter),
                "requested_min_subregions_per_cluster": int(
                    requested_min_subregions_per_cluster
                ),
                "reused_pilot_fit_for_final": bool(reused_pilot_fit),
                "pilot_restart_summaries": pilot_restart_summaries,
            }
        )
        _progress(
            f"automatic K selection chose K={selected_n_clusters} "
            f"from votes={auto_k_selection.get('criterion_votes', {})}"
        )

    n_clusters = int(selected_n_clusters)
    selected_restart_id = 0
    if clustering_method in {
        "pooled_subregion_latent",
        HETEROGENEITY_DESCRIPTOR_MODE,
        HETEROGENEITY_FUSED_OT_MODE,
        HETEROGENEITY_FGW_MODE,
    }:
        if clustering_method == HETEROGENEITY_DESCRIPTOR_MODE:
            _progress(
                "clustering internal heterogeneity descriptor motif embeddings "
                f"(K={n_clusters}; uses canonical within-subregion arrangement, "
                "not raw tissue position or sample labels)"
            )
            label_assignment_source = "internal_heterogeneity_descriptor_motif_embeddings"
        elif clustering_method in TRANSPORT_HETEROGENEITY_MODES:
            _progress(
                f"clustering subregions by precomputed {clustering_method} "
                f"transport distances (K={n_clusters}; fixed-K validation path)"
            )
            label_assignment_source = f"precomputed_{clustering_method}_distance_matrix"
        else:
            _progress(
                "clustering pooled raw-member feature-distribution subregion latent embeddings "
                f"(K={n_clusters}; no spatial coordinates or OT costs in label assignment)"
            )
            label_assignment_source = "pooled_subregion_latent_embeddings"
        if clustering_method in TRANSPORT_HETEROGENEITY_MODES:
            if transport_measures is None:
                raise RuntimeError("transport measures were not built before clustering.")
            transport_distance_matrix, transport_distance_metadata = (
                pairwise_transport_distance_matrix(
                    transport_measures,
                    mode=str(clustering_method),
                    max_subregions=int(heterogeneity_transport_max_subregions),
                    fused_ot_feature_weight=heterogeneity_fused_ot_feature_weight,
                    fused_ot_coordinate_weight=heterogeneity_fused_ot_coordinate_weight,
                    feature_cost_kind=heterogeneity_transport_feature_cost,
                    fgw_alpha=heterogeneity_fgw_alpha,
                    fgw_solver=heterogeneity_fgw_solver,
                    fgw_epsilon=heterogeneity_fgw_epsilon,
                    fgw_loss_fun=heterogeneity_fgw_loss_fun,
                    fgw_max_iter=heterogeneity_fgw_max_iter,
                    fgw_tol=heterogeneity_fgw_tol,
                    fgw_partial=bool(heterogeneity_fgw_partial),
                    fgw_partial_mass=heterogeneity_fgw_partial_mass,
                    fgw_partial_reg=heterogeneity_fgw_partial_reg,
                )
            )
            subregion_latent_embeddings = transport_distance_matrix.astype(np.float32)
            subregion_latent_embedding_metadata["transport_distance"] = (
                transport_distance_metadata
            )
            latent_fit = _cluster_precomputed_transport_distances(
                transport_distance_matrix,
                n_clusters=n_clusters,
                min_subregions_per_cluster=int(min_subregions_per_cluster),
            )
        else:
            latent_fit = fit_kmeans_on_latent_embeddings(
                subregion_latent_embeddings,
                n_clusters=n_clusters,
                n_init=max(int(n_init), 1),
                min_cluster_size=int(min_subregions_per_cluster),
                random_state=int(seed),
            )
        labels = np.asarray(latent_fit["labels"], dtype=np.int32)
        final_argmin_labels_best = np.asarray(
            latent_fit["argmin_labels"], dtype=np.int32
        )
        final_costs = np.asarray(latent_fit["costs"], dtype=np.float32)
        final_transport_costs = final_costs.astype(np.float32, copy=True)
        final_overlap_penalties = np.zeros_like(final_costs, dtype=np.float32)
        forced_label_mask_best = np.asarray(latent_fit["forced_label_mask"], dtype=bool)
        candidate_effective_eps_matrix_best = np.full(
            final_costs.shape, float(ot_eps), dtype=np.float32
        )
        candidate_used_fallback_matrix_best = np.zeros(final_costs.shape, dtype=bool)
        fixed_atom_bundle = _fit_fixed_label_atom_dictionary(
            measures=optimization_measures,
            labels=labels,
            n_clusters=n_clusters,
            atoms_per_cluster=atoms_per_cluster,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            ot_eps=ot_eps,
            rho=rho,
            align_iters=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            compute_device=resolved_compute_device,
        )
        atom_coords = np.asarray(fixed_atom_bundle["atom_coords"], dtype=np.float32)
        atom_features = np.asarray(fixed_atom_bundle["atom_features"], dtype=np.float32)
        betas = np.asarray(fixed_atom_bundle["betas"], dtype=np.float32)
        objective_history = list(fixed_atom_bundle["objective_history"])
        best_compute_device = resolved_compute_device
        plans = list(fixed_atom_bundle["plans"])
        transforms = list(fixed_atom_bundle["transforms"])
        thetas = list(fixed_atom_bundle["thetas"])
        assigned_effective_eps_best = np.asarray(
            fixed_atom_bundle["assigned_effective_eps"], dtype=np.float32
        )
        assigned_used_fallback_best = np.asarray(
            fixed_atom_bundle["assigned_used_fallback"], dtype=bool
        )
        restart_summaries = [
            {
                "run": 0,
                "seed": int(seed),
                "objective": float(latent_fit["inertia"]),
                "n_iter": int(len(objective_history)),
                "mean_assigned_cost": float(
                    np.mean(final_costs[np.arange(labels.shape[0]), labels])
                ),
                "mean_assigned_transport_cost": float(
                    np.mean(fixed_atom_bundle["assigned_costs"])
                ),
                "mean_assigned_overlap_penalty": 0.0,
                "device": str(resolved_compute_device),
                "assigned_ot_fallback_fraction": float(
                    np.mean(assigned_used_fallback_best.astype(np.float32))
                ),
                "candidate_ot_fallback_fraction": 0.0,
                "initial_min_size_forced_label_count": int(
                    forced_label_mask_best.sum()
                ),
                "final_min_size_forced_label_count": int(forced_label_mask_best.sum()),
                "min_subregions_per_cluster": int(
                    effective_min_cluster_size(
                        len(labels), n_clusters, int(min_subregions_per_cluster)
                    )
                ),
                "runtime_memory": dict(fixed_atom_bundle.get("runtime_memory", {})),
                "label_assignment_source": label_assignment_source,
            }
        ]
    else:
        if (
            reused_pilot_fit
            and pilot_best_bundle is not None
            and pilot_restart_summaries is not None
        ):
            restart_summaries = pilot_restart_summaries
            best_bundle = pilot_best_bundle
        else:
            _final_restart_results, restart_summaries, best_bundle = (
                _fit_restart_bundles(
                    measures=optimization_measures,
                    summaries=summaries,
                    n_clusters=n_clusters,
                    atoms_per_cluster=atoms_per_cluster,
                    lambda_x=lambda_x,
                    lambda_y=lambda_y,
                    ot_eps=ot_eps,
                    rho=rho,
                    align_iters=align_iters,
                    allow_reflection=allow_reflection,
                    allow_scale=allow_scale,
                    cost_scale_x=cost_scale_x,
                    cost_scale_y=cost_scale_y,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    scale_penalty=scale_penalty,
                    shift_penalty=shift_penalty,
                    max_iter=max_iter,
                    tol=tol,
                    min_subregions_per_cluster=min_subregions_per_cluster,
                    seed=seed,
                    overlap_edge_i=overlap_edge_i,
                    overlap_edge_j=overlap_edge_j,
                    overlap_edge_weight=overlap_edge_weight,
                    overlap_consistency_weight=overlap_consistency_weight,
                    n_init=n_init,
                    compute_device=compute_device,
                    resolved_compute_device=resolved_compute_device,
                    progress_label="final fit",
                )
            )
        selected_restart_id = int(best_bundle["run"])
        _progress(
            f"selected restart {selected_restart_id}; computing final diagnostics"
        )
        atom_coords = np.asarray(best_bundle["atom_coords"], dtype=np.float32)
        atom_features = np.asarray(best_bundle["atom_features"], dtype=np.float32)
        betas = np.asarray(best_bundle["betas"], dtype=np.float32)
        objective_history = list(best_bundle["objective_history"])
        best_compute_device = _resolve_compute_device(str(best_bundle["device"]))
        (
            final_transport_costs,
            candidate_effective_eps_matrix_best,
            candidate_used_fallback_matrix_best,
        ) = _compute_assignment_costs(
            measures=measures,
            atom_coords=atom_coords,
            atom_features=atom_features,
            betas=betas,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            eps=ot_eps,
            rho=rho,
            align_iters=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
            compute_device=best_compute_device,
            return_diagnostics=True,
        )
        (
            final_transport_costs,
            candidate_effective_eps_matrix_best,
            candidate_used_fallback_matrix_best,
        ) = _stabilize_mixed_candidate_assignment_costs(
            measures=measures,
            atom_coords=atom_coords,
            atom_features=atom_features,
            betas=betas,
            transport_costs=final_transport_costs,
            candidate_effective_eps_matrix=candidate_effective_eps_matrix_best,
            candidate_used_fallback_matrix=candidate_used_fallback_matrix_best,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            ot_eps=ot_eps,
            rho=rho,
            align_iters=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
            compute_device=best_compute_device,
        )
        final_costs, final_overlap_penalties = (
            _apply_overlap_consistency_regularization(
                final_transport_costs,
                edge_i=overlap_edge_i,
                edge_j=overlap_edge_j,
                edge_weight=overlap_edge_weight,
                overlap_consistency_weight=overlap_consistency_weight,
            )
        )
        final_argmin_labels_best = final_costs.argmin(axis=1).astype(np.int32)
        labels, forced_label_mask_best = _ensure_minimum_cluster_size(
            final_argmin_labels_best,
            final_costs,
            n_clusters=n_clusters,
            min_subregions_per_cluster=min_subregions_per_cluster,
        )
        (
            plans,
            transforms,
            thetas,
            _assigned_transport_costs_best,
            assigned_effective_eps_best,
            assigned_used_fallback_best,
        ) = _compute_assigned_artifacts(
            measures=measures,
            labels=labels,
            atom_coords=atom_coords,
            atom_features=atom_features,
            betas=betas,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            eps=ot_eps,
            rho=rho,
            align_iters=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
            compute_device=best_compute_device,
        )
    assigned_overlap_penalties_best = final_overlap_penalties[
        np.arange(labels.shape[0], dtype=np.int64),
        labels.astype(np.int64),
    ].astype(np.float32)
    (
        assigned_geometry_transport_costs_best,
        assigned_feature_transport_costs_best,
        assigned_transform_penalties_best,
    ) = _compute_assigned_cost_breakdowns(
        measures=measures,
        labels=labels,
        plans=plans,
        transforms=transforms,
        atom_coords=atom_coords,
        atom_features=atom_features,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        cost_scale_x=cost_scale_x,
        cost_scale_y=cost_scale_y,
        scale_penalty=scale_penalty,
        shift_penalty=shift_penalty,
    )
    (
        assigned_rotation_deg_best,
        assigned_reflection_best,
        assigned_scale_best,
        assigned_translation_norm_best,
    ) = _transform_diagnostic_arrays(transforms)

    thetas_assigned = np.vstack(
        [np.asarray(theta, dtype=np.float32) for theta in thetas]
    ).astype(np.float32)
    subregion_cluster_probs = _softmax_over_negative_costs(
        final_costs, temperature=max(float(np.std(final_costs)), 1e-3)
    )
    cluster_supports = np.concatenate([atom_coords, atom_features], axis=2).astype(
        np.float32
    )

    _progress("projecting cell-level cluster probabilities")
    cell_cluster_labels, cell_cluster_probs, cell_feature_probs, cell_context_probs = (
        _project_cells_from_subregions(
            features=features,
            coords_um=coords_um,
            measures=measures,
            subregion_labels=labels,
            atom_coords=atom_coords,
            atom_features=atom_features,
            prototype_weights=betas,
            assigned_transforms=transforms,
            subregion_cluster_costs=final_costs,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            assignment_temperature=max(cost_scale_y, 1e-3),
            context_weight=0.5,
            compute_device=resolved_compute_device,
        )
    )
    if compute_spot_latent:
        _progress("computing occurrence-level spot latent charts")
        spot_latent = compute_spot_level_latent_charts(
            features=features,
            coords_um=coords_um,
            measures=measures,
            subregion_labels=labels,
            subregion_cluster_probs=subregion_cluster_probs,
            atom_coords=atom_coords,
            atom_features=atom_features,
            prototype_weights=betas,
            assigned_transforms=transforms,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            assignment_temperature=max(cost_scale_y, 1e-3),
            compute_device=resolved_compute_device,
        )
    else:
        _progress("spot latent charts disabled")
        spot_latent = empty_spot_level_latent_charts(
            n_cells=int(features.shape[0]),
            atoms_per_cluster=int(atom_coords.shape[1]),
            n_clusters=int(atom_coords.shape[0]),
        )

    return MultilevelOTResult(
        basic_niche_size_um=float(basic_niche_size_um)
        if used_basic_niches and basic_niche_size_um is not None
        else None,
        basic_niche_centers_um=basic_niche_centers_um.astype(np.float32),
        basic_niche_members=[
            np.asarray(members, dtype=np.int32) for members in basic_niche_members
        ],
        subregion_basic_niche_ids=[
            np.asarray(niche_ids, dtype=np.int32)
            for niche_ids in subregion_basic_niche_ids
        ],
        subregion_centers_um=centers_um.astype(np.float32),
        subregion_members=[m.members for m in measures],
        subregion_argmin_labels=final_argmin_labels_best.astype(np.int32),
        subregion_forced_label_mask=forced_label_mask_best.astype(bool),
        subregion_geometry_point_counts=np.asarray(
            [m.geometry_point_count for m in measures], dtype=np.int32
        ),
        subregion_geometry_sources=[
            m.normalizer_diagnostics.geometry_source for m in measures
        ],
        subregion_geometry_used_fallback=np.asarray(
            [m.normalizer_diagnostics.used_fallback for m in measures], dtype=bool
        ),
        subregion_normalizer_radius_p95=np.asarray(
            [
                m.normalizer_diagnostics.mapped_radius_p95
                if m.normalizer_diagnostics.mapped_radius_p95 is not None
                else np.nan
                for m in measures
            ],
            dtype=np.float32,
        ),
        subregion_normalizer_radius_max=np.asarray(
            [
                m.normalizer_diagnostics.mapped_radius_max
                if m.normalizer_diagnostics.mapped_radius_max is not None
                else np.nan
                for m in measures
            ],
            dtype=np.float32,
        ),
        subregion_normalizer_interpolation_residual=np.asarray(
            [
                m.normalizer_diagnostics.interpolation_residual
                if m.normalizer_diagnostics.interpolation_residual is not None
                else np.nan
                for m in measures
            ],
            dtype=np.float32,
        ),
        subregion_cluster_labels=labels,
        subregion_cluster_probs=subregion_cluster_probs.astype(np.float32),
        subregion_cluster_costs=final_costs.astype(np.float32),
        subregion_cluster_transport_costs=final_transport_costs.astype(np.float32),
        subregion_cluster_overlap_penalties=final_overlap_penalties.astype(np.float32),
        subregion_atom_weights=thetas_assigned.astype(np.float32),
        subregion_measure_summaries=summaries.astype(np.float32),
        subregion_latent_embeddings=subregion_latent_embeddings.astype(np.float32),
        subregion_latent_embedding_mode=subregion_latent_embedding_mode,
        subregion_latent_embedding_metadata=subregion_latent_embedding_metadata,
        subregion_latent_shrinkage_alpha=np.asarray(
            subregion_latent_diagnostics["shrinkage_alpha"], dtype=np.float32
        ),
        subregion_latent_raw_to_shrunk_distance=np.asarray(
            subregion_latent_diagnostics["raw_to_shrunk_distance"], dtype=np.float32
        ),
        subregion_sample_ids=np.asarray(
            subregion_latent_diagnostics["sample_ids"], dtype=object
        ),
        subregion_clustering_method=str(clustering_method),
        subregion_clustering_uses_spatial=bool(
            clustering_method != "pooled_subregion_latent"
        ),
        subregion_assigned_effective_eps=assigned_effective_eps_best.astype(np.float32),
        subregion_assigned_used_ot_fallback=assigned_used_fallback_best.astype(bool),
        subregion_candidate_effective_eps_matrix=candidate_effective_eps_matrix_best.astype(
            np.float32
        ),
        subregion_candidate_used_ot_fallback_matrix=candidate_used_fallback_matrix_best.astype(
            bool
        ),
        subregion_assigned_geometry_transport_costs=assigned_geometry_transport_costs_best.astype(
            np.float32
        ),
        subregion_assigned_feature_transport_costs=assigned_feature_transport_costs_best.astype(
            np.float32
        ),
        subregion_assigned_transform_penalties=assigned_transform_penalties_best.astype(
            np.float32
        ),
        subregion_assigned_overlap_consistency_penalties=assigned_overlap_penalties_best.astype(
            np.float32
        ),
        subregion_assigned_transform_rotation_deg=assigned_rotation_deg_best.astype(
            np.float32
        ),
        subregion_assigned_transform_reflection=assigned_reflection_best.astype(bool),
        subregion_assigned_transform_scale=assigned_scale_best.astype(np.float32),
        subregion_assigned_transform_translation_norm=assigned_translation_norm_best.astype(
            np.float32
        ),
        cluster_supports=cluster_supports.astype(np.float32),
        cluster_atom_coords=atom_coords.astype(np.float32),
        cluster_atom_features=atom_features.astype(np.float32),
        cluster_prototype_weights=betas.astype(np.float32),
        cell_feature_cluster_probs=cell_feature_probs.astype(np.float32),
        cell_context_cluster_probs=cell_context_probs.astype(np.float32),
        cell_cluster_probs=cell_cluster_probs.astype(np.float32),
        cell_cluster_labels=cell_cluster_labels.astype(np.int32),
        spot_latent_cell_indices=spot_latent["spot_latent_cell_indices"].astype(
            np.int32
        ),
        spot_latent_subregion_ids=spot_latent["spot_latent_subregion_ids"].astype(
            np.int32
        ),
        spot_latent_cluster_labels=spot_latent["spot_latent_cluster_labels"].astype(
            np.int32
        ),
        spot_latent_coords=spot_latent["spot_latent_coords"].astype(np.float32),
        spot_latent_within_coords=spot_latent["spot_latent_within_coords"].astype(
            np.float32
        ),
        spot_latent_cluster_anchors=spot_latent["spot_latent_cluster_anchors"].astype(
            np.float32
        ),
        spot_latent_atom_embedding=spot_latent["spot_latent_atom_embedding"].astype(
            np.float32
        ),
        spot_latent_aligned_coords=spot_latent["spot_latent_aligned_coords"].astype(
            np.float32
        ),
        spot_latent_cluster_probs=spot_latent["spot_latent_cluster_probs"].astype(
            np.float32
        ),
        spot_latent_atom_confidence=spot_latent["spot_latent_atom_confidence"].astype(
            np.float32
        ),
        spot_latent_posterior_entropy=spot_latent[
            "spot_latent_posterior_entropy"
        ].astype(np.float32),
        spot_latent_normalized_posterior_entropy=spot_latent[
            "spot_latent_normalized_posterior_entropy"
        ].astype(np.float32),
        spot_latent_atom_argmax=spot_latent["spot_latent_atom_argmax"].astype(np.int32),
        spot_latent_temperature_used=spot_latent["spot_latent_temperature_used"].astype(
            np.float32
        ),
        spot_latent_temperature_cost_gap=spot_latent[
            "spot_latent_temperature_cost_gap"
        ].astype(np.float32),
        spot_latent_temperature_fixed=spot_latent[
            "spot_latent_temperature_fixed"
        ].astype(np.float32),
        spot_latent_weights=spot_latent["spot_latent_weights"].astype(np.float32),
        spot_latent_atom_posteriors=spot_latent["spot_latent_atom_posteriors"].astype(
            np.float32
        ),
        spot_latent_posterior_entropy_cost_gap=spot_latent[
            "spot_latent_posterior_entropy_cost_gap"
        ].astype(np.float32),
        spot_latent_normalized_posterior_entropy_cost_gap=spot_latent[
            "spot_latent_normalized_posterior_entropy_cost_gap"
        ].astype(np.float32),
        spot_latent_posterior_entropy_fixed=spot_latent[
            "spot_latent_posterior_entropy_fixed"
        ].astype(np.float32),
        spot_latent_normalized_posterior_entropy_fixed=spot_latent[
            "spot_latent_normalized_posterior_entropy_fixed"
        ].astype(np.float32),
        spot_latent_cluster_anchor_distance=spot_latent[
            "spot_latent_cluster_anchor_distance"
        ].astype(np.float32),
        spot_latent_cluster_anchor_ot_fallback_matrix=spot_latent[
            "spot_latent_cluster_anchor_ot_fallback_matrix"
        ].astype(bool),
        spot_latent_cluster_anchor_solver_status_matrix=spot_latent[
            "spot_latent_cluster_anchor_solver_status_matrix"
        ].astype(np.int8),
        spot_latent_cluster_anchor_ot_fallback_fraction=float(
            spot_latent["spot_latent_cluster_anchor_ot_fallback_fraction"].item()
        ),
        spot_latent_atom_mds_stress=spot_latent["spot_latent_atom_mds_stress"].astype(
            np.float32
        ),
        spot_latent_atom_mds_positive_eigenvalue_mass_2d=spot_latent[
            "spot_latent_atom_mds_positive_eigenvalue_mass_2d"
        ].astype(np.float32),
        spot_latent_atom_mds_negative_eigenvalue_mass_fraction=spot_latent[
            "spot_latent_atom_mds_negative_eigenvalue_mass_fraction"
        ].astype(np.float32),
        cell_spot_latent_unweighted_coords=spot_latent[
            "cell_spot_latent_unweighted_coords"
        ].astype(np.float32),
        cell_spot_latent_confidence_weighted_coords=spot_latent[
            "cell_spot_latent_confidence_weighted_coords"
        ].astype(np.float32),
        cell_spot_latent_coords=spot_latent["cell_spot_latent_coords"].astype(
            np.float32
        ),
        cell_spot_latent_cluster_labels=spot_latent[
            "cell_spot_latent_cluster_labels"
        ].astype(np.int32),
        cell_spot_latent_weights=spot_latent["cell_spot_latent_weights"].astype(
            np.float32
        ),
        cell_spot_latent_posterior_entropy=spot_latent[
            "cell_spot_latent_posterior_entropy"
        ].astype(np.float32),
        spot_latent_mode=str(spot_latent["spot_latent_mode"].item()),
        spot_latent_chart_learning_mode=str(
            spot_latent["spot_latent_chart_learning_mode"].item()
        ),
        spot_latent_projection_mode=str(
            spot_latent["spot_latent_projection_mode"].item()
        ),
        spot_latent_validation_role=str(
            spot_latent["spot_latent_validation_role"].item()
        ),
        spot_latent_global_within_scale=float(
            spot_latent["spot_latent_global_within_scale"].item()
        ),
        spot_latent_assignment_temperature=float(
            spot_latent["spot_latent_assignment_temperature"].item()
        ),
        spot_latent_temperature_mode=str(
            spot_latent["spot_latent_temperature_mode"].item()
        ),
        spot_latent_cluster_anchor_distance_method=str(
            spot_latent["spot_latent_cluster_anchor_distance_method"].item()
        ),
        spot_latent_cluster_anchor_distance_requested_method=str(
            spot_latent["spot_latent_cluster_anchor_distance_requested_method"].item()
        ),
        spot_latent_cluster_anchor_distance_effective_method=str(
            spot_latent["spot_latent_cluster_anchor_distance_effective_method"].item()
        ),
        spot_latent_cluster_mds_stress=float(
            spot_latent["spot_latent_cluster_mds_stress"].item()
        ),
        spot_latent_cluster_mds_positive_eigenvalue_mass_2d=float(
            spot_latent["spot_latent_cluster_mds_positive_eigenvalue_mass_2d"].item()
        ),
        spot_latent_cluster_mds_negative_eigenvalue_mass_fraction=float(
            spot_latent[
                "spot_latent_cluster_mds_negative_eigenvalue_mass_fraction"
            ].item()
        ),
        cost_scale_x=float(cost_scale_x),
        cost_scale_y=float(cost_scale_y),
        objective_history=objective_history,
        selected_restart=int(selected_restart_id),
        restart_summaries=restart_summaries,
        min_subregions_per_cluster=int(requested_min_subregions_per_cluster),
        effective_min_subregions_per_cluster=int(
            effective_min_cluster_size(
                len(labels), n_clusters, int(min_subregions_per_cluster)
            )
        ),
        auto_k_selection=auto_k_selection,
    )
