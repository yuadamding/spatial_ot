from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import ot
from sklearn.cluster import MiniBatchKMeans

from .types import SubregionMeasure

HETEROGENEITY_DESCRIPTOR_MODE = "heterogeneity_descriptor_niche"
LEGACY_HETEROGENEITY_OT_ALIAS = "heterogeneity_ot_niche"
HETEROGENEITY_FUSED_OT_MODE = "heterogeneity_fused_ot_niche"
HETEROGENEITY_FGW_MODE = "heterogeneity_fgw_niche"
HETEROGENEITY_DESCRIPTOR_ALIASES = {
    HETEROGENEITY_DESCRIPTOR_MODE,
    LEGACY_HETEROGENEITY_OT_ALIAS,
}
TRANSPORT_HETEROGENEITY_MODES = {
    HETEROGENEITY_FUSED_OT_MODE,
    HETEROGENEITY_FGW_MODE,
}
DEFAULT_BLOCK_WEIGHTS = {
    "composition": 0.20,
    "diversity": 0.15,
    "spatial_field": 0.35,
    "pair_cooccurrence": 0.30,
}
VALID_PAIR_GRAPH_MODES = {"all_pairs", "knn", "radius"}


@dataclass(frozen=True)
class SubregionFGWMeasure:
    coords: np.ndarray
    features: np.ndarray
    weights: np.ndarray
    structure: np.ndarray
    sample_id: str | None = None
    subregion_id: int | None = None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _normalize_block_weights(
    block_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    incoming = dict(DEFAULT_BLOCK_WEIGHTS)
    if block_weights:
        for key, value in block_weights.items():
            if key in incoming:
                incoming[key] = float(value)
    weights = {key: max(float(value), 0.0) for key, value in incoming.items()}
    total = float(sum(weights.values()))
    if total <= 1e-12:
        return dict(DEFAULT_BLOCK_WEIGHTS)
    return {key: float(value / total) for key, value in weights.items()}


def _env_block_weights() -> dict[str, float]:
    return _normalize_block_weights(
        {
        "composition": max(
            0.0,
            _env_float(
                "SPATIAL_OT_HETEROGENEITY_COMPOSITION_WEIGHT",
                DEFAULT_BLOCK_WEIGHTS["composition"],
            ),
        ),
        "diversity": max(
            0.0,
            _env_float(
                "SPATIAL_OT_HETEROGENEITY_DIVERSITY_WEIGHT",
                DEFAULT_BLOCK_WEIGHTS["diversity"],
            ),
        ),
        "spatial_field": max(
            0.0,
            _env_float(
                "SPATIAL_OT_HETEROGENEITY_FIELD_WEIGHT",
                DEFAULT_BLOCK_WEIGHTS["spatial_field"],
            ),
        ),
        "pair_cooccurrence": max(
            0.0,
            _env_float(
                "SPATIAL_OT_HETEROGENEITY_PAIR_WEIGHT",
                DEFAULT_BLOCK_WEIGHTS["pair_cooccurrence"],
            ),
        ),
        }
    )


def _resolve_pair_graph_mode(mode: str | None) -> str:
    resolved = str(mode or "all_pairs").strip().lower().replace("-", "_")
    if resolved not in VALID_PAIR_GRAPH_MODES:
        raise ValueError(
            "heterogeneity pair_graph_mode must be one of "
            f"{sorted(VALID_PAIR_GRAPH_MODES)}, got '{mode}'."
        )
    return resolved


def _resolve_pair_bins(
    value: tuple[float, ...] | list[float] | np.ndarray | str | None,
) -> np.ndarray:
    if value is None:
        raw = os.environ.get(
            "SPATIAL_OT_HETEROGENEITY_PAIR_BINS",
            "0.25,0.5,1.0,2.0",
        )
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        parsed = [float(part) for part in parts]
    elif isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        parsed = [float(part) for part in parts]
    else:
        parsed = [float(part) for part in value]
    bins = np.asarray(parsed, dtype=np.float32)
    if bins.size == 0:
        bins = np.asarray([0.25, 0.5, 1.0, 2.0], dtype=np.float32)
    bins = np.sort(np.unique(bins[bins > 0])).astype(np.float32)
    if bins.size == 0:
        bins = np.asarray([1.0], dtype=np.float32)
    return bins


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    denom = np.maximum(x.sum(axis=1, keepdims=True), 1e-12)
    return (x / denom).astype(np.float32)


def _normalize_mass(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = np.where(np.isfinite(x) & (x > 0), x, 0.0)
    total = float(np.sum(x))
    if total <= 1e-12:
        if x.size == 0:
            raise ValueError("mass vector must contain at least one entry.")
        return np.full(x.size, 1.0 / float(x.size), dtype=np.float64)
    return (x / total).astype(np.float64)


def _numeric_summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
        "max": float(np.max(x)),
    }


def _pairwise_sqeuclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return np.maximum(
        np.sum(a * a, axis=1, keepdims=True)
        + np.sum(b * b, axis=1, keepdims=True).T
        - 2.0 * (a @ b.T),
        0.0,
    )


def _normalize_cost_matrix(cost: np.ndarray) -> np.ndarray:
    x = np.asarray(cost, dtype=np.float64)
    finite = x[np.isfinite(x)]
    positive = finite[finite > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    return np.asarray(x / max(scale, 1e-12), dtype=np.float64)


def hellinger_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    p = p / np.maximum(np.sum(p, axis=1, keepdims=True), 1e-12)
    q = q / np.maximum(np.sum(q, axis=1, keepdims=True), 1e-12)
    diff = np.sqrt(np.maximum(p[:, None, :], 0.0)) - np.sqrt(
        np.maximum(q[None, :, :], 0.0)
    )
    return 0.5 * np.sum(diff * diff, axis=2)


def feature_cost(
    left: SubregionFGWMeasure,
    right: SubregionFGWMeasure,
    *,
    feature_cost_kind: str = "hellinger_codebook",
    normalize: bool = True,
) -> np.ndarray:
    requested = str(feature_cost_kind or "hellinger_codebook").strip().lower()
    if requested in {"hellinger", "hellinger_codebook", "codebook_hellinger"}:
        cost = hellinger_cost(left.features, right.features)
    elif requested in {"sqeuclidean", "squared_euclidean", "euclidean"}:
        cost = _pairwise_sqeuclidean(left.features, right.features)
    else:
        raise ValueError(
            "feature_cost_kind must be 'hellinger_codebook' or 'sqeuclidean', "
            f"got '{feature_cost_kind}'."
        )
    return (
        _normalize_cost_matrix(cost)
        if bool(normalize)
        else np.asarray(cost, dtype=np.float64)
    )


def structure_cost(
    coords: np.ndarray,
    *,
    scale: float,
    clip: float | None = 3.0,
) -> np.ndarray:
    u = np.asarray(coords, dtype=np.float64)
    diff = u[:, None, :] - u[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2)) / max(float(scale), 1e-12)
    if clip is not None and float(clip) > 0:
        dist = np.minimum(dist, float(clip))
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


def _global_structure_scale(
    measures: list[SubregionMeasure],
    *,
    structure_scale: str | float = "global_median",
) -> float:
    if isinstance(structure_scale, (int, float)):
        return max(float(structure_scale), 1e-8)
    requested = str(structure_scale or "global_median").strip().lower()
    if requested in {"fixed", "unit", "canonical_unit"}:
        return 1.0
    values: list[np.ndarray] = []
    for measure in measures:
        coords = np.asarray(measure.canonical_coords, dtype=np.float64)
        if coords.shape[0] < 2:
            continue
        rows, cols = np.triu_indices(coords.shape[0], k=1)
        d = np.sqrt(np.sum((coords[rows] - coords[cols]) ** 2, axis=1))
        d = d[np.isfinite(d) & (d > 1e-12)]
        if d.size:
            values.append(d)
    if not values:
        return 1.0
    all_dist = np.concatenate(values)
    if requested in {"global_q75", "q75", "p75", "global_75th"}:
        return max(float(np.quantile(all_dist, 0.75)), 1e-8)
    if requested not in {"global_median", "median"}:
        raise ValueError(
            "heterogeneity FGW structure_scale must be global_median, global_q75, "
            f"canonical_unit, or a positive number, got '{structure_scale}'."
        )
    return max(float(np.median(all_dist)), 1e-8)


def _softmax_negative_sqdist(
    x: np.ndarray,
    centers: np.ndarray,
    temperature: float,
) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    c64 = np.asarray(centers, dtype=np.float64)
    d2 = (
        np.sum(x64 * x64, axis=1, keepdims=True)
        + np.sum(c64 * c64, axis=1, keepdims=True).T
        - 2.0 * (x64 @ c64.T)
    )
    logits = -np.maximum(d2, 0.0) / max(float(temperature), 1e-8)
    logits -= np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits).astype(np.float32)
    probs /= np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-8)
    return probs.astype(np.float32)


def _fit_state_codebook(
    features: np.ndarray,
    *,
    codebook_size: int,
    sample_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("features must be a 2D matrix.")
    if x.shape[0] == 0:
        raise ValueError("features must contain at least one row.")
    n_codes = min(max(int(codebook_size), 2), int(x.shape[0]))
    rng = np.random.default_rng(int(random_state))
    fit_n = min(max(int(sample_size), n_codes), int(x.shape[0]))
    fit_idx = np.sort(rng.choice(int(x.shape[0]), size=fit_n, replace=False))
    sample = x[fit_idx].astype(np.float64, copy=False)
    center = np.mean(sample, axis=0, dtype=np.float64)
    scale = np.std(sample, axis=0, dtype=np.float64)
    finite_scale = scale[np.isfinite(scale) & (scale > 1e-8)]
    scale_floor = float(np.median(finite_scale) * 1e-3) if finite_scale.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale > max(scale_floor, 1e-8)), scale, 1.0)
    sample_z = ((sample - center[None, :]) / scale[None, :]).astype(np.float32)
    model = MiniBatchKMeans(
        n_clusters=int(n_codes),
        n_init=3,
        batch_size=min(max(int(n_codes) * 256, 2048), max(int(fit_n), 2048)),
        random_state=int(random_state),
        max_iter=40,
        max_no_improvement=5,
        reassignment_ratio=0.0,
    )
    model.fit(sample_z)
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    sample_probs = _softmax_negative_sqdist(sample_z, centers, temperature=1.0)
    sample_d2 = -np.log(np.maximum(sample_probs.max(axis=1), 1e-8))
    positive = sample_d2[np.isfinite(sample_d2) & (sample_d2 > 1e-8)]
    temperature = float(np.median(positive)) if positive.size else 1.0
    return centers, center.astype(np.float32), scale.astype(np.float32), max(temperature, 1e-4)


def _canonical_grid_bins(coords: np.ndarray, *, grid_size: int, radius: float) -> np.ndarray:
    u = np.asarray(coords, dtype=np.float32)
    scaled = (np.clip(u, -float(radius), float(radius)) + float(radius)) / max(
        2.0 * float(radius), 1e-8
    )
    bins_xy = np.floor(scaled * int(grid_size)).astype(np.int32)
    bins_xy = np.clip(bins_xy, 0, int(grid_size) - 1)
    return (bins_xy[:, 0] * int(grid_size) + bins_xy[:, 1]).astype(np.int32)


def _weighted_state_diversity(composition: np.ndarray) -> np.ndarray:
    p = np.asarray(composition, dtype=np.float64)
    p = p / max(float(p.sum()), 1e-12)
    k = int(p.shape[0])
    entropy = -float(np.sum(p * np.log(np.maximum(p, 1e-12))))
    normalized_entropy = entropy / max(float(np.log(max(k, 2))), 1e-8)
    simpson = 1.0 - float(np.sum(p * p))
    effective_states = float(np.exp(entropy) / max(k, 1))
    top_dominance = float(np.max(p)) if p.size else 0.0
    rare_mass = float(np.sum(p[p < (1.0 / max(k, 1)) * 0.25]))
    return np.asarray(
        [normalized_entropy, simpson, effective_states, top_dominance, rare_mass],
        dtype=np.float32,
    )


def _spatial_state_field(
    coords: np.ndarray,
    probs: np.ndarray,
    weights: np.ndarray,
    *,
    grid_size: int,
    radius: float,
) -> np.ndarray:
    n_codes = int(probs.shape[1])
    n_bins = int(grid_size) * int(grid_size)
    field = np.zeros((n_bins, n_codes), dtype=np.float64)
    bins = _canonical_grid_bins(coords, grid_size=int(grid_size), radius=float(radius))
    weighted_probs = np.asarray(probs, dtype=np.float64) * np.asarray(
        weights, dtype=np.float64
    )[:, None]
    for code in range(n_codes):
        field[:, code] = np.bincount(
            bins, weights=weighted_probs[:, code], minlength=n_bins
        )
    total = max(float(np.sum(field)), 1e-12)
    return (field / total).reshape(-1).astype(np.float32)


def _pair_edges(
    coords: np.ndarray,
    *,
    graph_mode: str,
    graph_k: int,
    graph_radius: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(coords, dtype=np.float32)
    n = int(u.shape[0])
    if n < 2:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
        )
    rows, cols = np.triu_indices(n, k=1)
    all_dist = np.sqrt(np.sum((u[rows] - u[cols]) ** 2, axis=1)).astype(np.float32)
    mode = _resolve_pair_graph_mode(graph_mode)
    if mode == "all_pairs":
        return rows.astype(np.int64), cols.astype(np.int64), all_dist
    dist_matrix = np.sqrt(
        np.sum((u[:, None, :] - u[None, :, :]) ** 2, axis=2)
    ).astype(np.float32)
    np.fill_diagonal(dist_matrix, np.inf)
    if mode == "radius":
        radius = (
            float(graph_radius)
            if graph_radius is not None and float(graph_radius) > 0
            else float(np.inf)
        )
        mask = all_dist <= radius
        return rows[mask].astype(np.int64), cols[mask].astype(np.int64), all_dist[mask]
    k = min(max(int(graph_k), 1), n - 1)
    edge_set: set[tuple[int, int]] = set()
    for row_idx in range(n):
        neighbors = np.argpartition(dist_matrix[row_idx], kth=k - 1)[:k]
        for col_idx in neighbors.tolist():
            if row_idx == col_idx:
                continue
            left, right = sorted((int(row_idx), int(col_idx)))
            edge_set.add((left, right))
    if not edge_set:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
        )
    edge_arr = np.asarray(sorted(edge_set), dtype=np.int64)
    edge_rows = edge_arr[:, 0]
    edge_cols = edge_arr[:, 1]
    edge_dist = np.sqrt(np.sum((u[edge_rows] - u[edge_cols]) ** 2, axis=1)).astype(
        np.float32
    )
    return edge_rows, edge_cols, edge_dist


def _pair_cooccurrence(
    coords: np.ndarray,
    probs: np.ndarray,
    weights: np.ndarray,
    *,
    distance_bins: np.ndarray,
    normalization: str = "observed_over_expected",
    graph_mode: str = "all_pairs",
    graph_k: int = 8,
    graph_radius: float | None = None,
    bin_normalization: str = "per_bin",
) -> np.ndarray:
    u = np.asarray(coords, dtype=np.float32)
    q = np.asarray(probs, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    n = int(u.shape[0])
    k = int(q.shape[1])
    bins = np.asarray(distance_bins, dtype=np.float32)
    if n < 2:
        return np.zeros((int(bins.shape[0]), k, k), dtype=np.float32).reshape(-1)
    rows, cols, d = _pair_edges(
        u,
        graph_mode=graph_mode,
        graph_k=int(graph_k),
        graph_radius=graph_radius,
    )
    if rows.size == 0:
        return np.zeros((int(bins.shape[0]), k, k), dtype=np.float32).reshape(-1)
    pair_w = (w[rows] * w[cols]).astype(np.float64)
    out = np.zeros((int(bins.shape[0]), k, k), dtype=np.float64)
    lower = 0.0
    for bin_idx, upper in enumerate(bins.tolist()):
        mask = (d >= lower) & (d < float(upper))
        lower = float(upper)
        if not np.any(mask):
            continue
        left = q[rows[mask]].astype(np.float64)
        right = q[cols[mask]].astype(np.float64)
        weights_bin = pair_w[mask]
        mat = left.T @ (right * weights_bin[:, None])
        out[bin_idx] = mat + mat.T
    requested = str(normalization or "observed_over_expected").strip().lower()
    if requested in {"observed_over_expected", "enrichment", "oe"}:
        composition = np.sum(q.astype(np.float64) * w.astype(np.float64)[:, None], axis=0)
        composition = composition / max(float(np.sum(composition)), 1e-12)
        expected = composition[:, None] * composition[None, :]
        out = np.log1p(out / np.maximum(expected[None, :, :], 1e-8))
    bin_requested = str(bin_normalization or "per_bin").strip().lower()
    if bin_requested in {"per_bin", "bin", "distance_bin"}:
        bin_sums = np.sum(out, axis=(1, 2), keepdims=True)
        normalized = np.zeros_like(out)
        out = np.divide(
            out,
            np.maximum(bin_sums, 1e-12),
            out=normalized,
            where=bin_sums > 0,
        )
    total = max(float(np.sum(out)), 1e-12)
    return (out / total).reshape(-1).astype(np.float32)


def _standardize_descriptor_block(
    block: np.ndarray,
    *,
    weight: float,
) -> tuple[np.ndarray, dict[str, object]]:
    x = np.asarray(block, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("heterogeneity descriptor blocks must be 2D.")
    if x.shape[1] == 0:
        return x, {
            "dimension": 0,
            "weight": float(weight),
            "mean_unweighted_l2": 0.0,
            "mean_weighted_l2": 0.0,
        }
    center = np.mean(x, axis=0, dtype=np.float64)
    scale = np.std(x, axis=0, dtype=np.float64)
    finite_scale = scale[np.isfinite(scale) & (scale > 1e-8)]
    scale_floor = float(np.median(finite_scale) * 1e-3) if finite_scale.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale > max(scale_floor, 1e-8)), scale, 1.0)
    z = ((x.astype(np.float64, copy=False) - center[None, :]) / scale[None, :]).astype(
        np.float32
    )
    weighted = z * (float(weight) / float(np.sqrt(max(int(x.shape[1]), 1))))
    stats = {
        "dimension": int(x.shape[1]),
        "weight": float(weight),
        "standardization": "subregion_block_mean_std_then_weight_over_sqrt_dimension",
        "mean_unweighted_l2": (
            float(np.mean(np.linalg.norm(z, axis=1))) if z.size else 0.0
        ),
        "mean_weighted_l2": (
            float(np.mean(np.linalg.norm(weighted, axis=1))) if weighted.size else 0.0
        ),
    }
    return weighted.astype(np.float32), stats


def build_subregion_fgw_measures(
    measures: list[SubregionMeasure],
    *,
    feature_mode: str = "soft_codebook",
    structure_metric: str = "canonical_euclidean",
    structure_scale: str | float = "global_median",
    structure_clip: float | None = 3.0,
    codebook_size: int = 32,
    codebook_sample_size: int = 50000,
    random_state: int = 1337,
    sample_ids: list[str] | np.ndarray | None = None,
) -> tuple[list[SubregionFGWMeasure], dict[str, object]]:
    """Build measured attributed metric spaces for fused OT / FGW clustering."""

    if not measures:
        return [], {
            "mode": "subregion_fgw_measures",
            "n_subregions": 0,
            "uses_ot_costs": True,
        }
    requested_feature_mode = str(feature_mode or "soft_codebook").strip().lower()
    if requested_feature_mode not in {
        "soft_codebook",
        "whitened_features",
        "whitened_features_plus_soft_codebook",
    }:
        raise ValueError(
            "feature_mode must be soft_codebook, whitened_features, or "
            f"whitened_features_plus_soft_codebook; got '{feature_mode}'."
        )
    requested_structure_metric = str(structure_metric or "canonical_euclidean").strip().lower()
    if requested_structure_metric != "canonical_euclidean":
        raise ValueError("Only canonical_euclidean structure_metric is currently implemented.")
    all_features = np.vstack([np.asarray(m.features, dtype=np.float32) for m in measures])
    max_codes = _env_int("SPATIAL_OT_HETEROGENEITY_MAX_CODEBOOK_SIZE", 16)
    n_codes = min(max(int(codebook_size), 2), max(int(max_codes), 2), int(all_features.shape[0]))
    centers, center, scale, temperature = _fit_state_codebook(
        all_features,
        codebook_size=n_codes,
        sample_size=int(codebook_sample_size),
        random_state=int(random_state),
    )
    resolved_structure_scale = _global_structure_scale(
        measures,
        structure_scale=structure_scale,
    )
    sample_values = (
        [None] * len(measures)
        if sample_ids is None
        else [str(value) for value in np.asarray(sample_ids).reshape(-1).tolist()]
    )
    if len(sample_values) != len(measures):
        sample_values = [None] * len(measures)
    transport_measures: list[SubregionFGWMeasure] = []
    code_entropy_values: list[np.ndarray] = []
    support_sizes: list[int] = []
    for idx, measure in enumerate(measures):
        raw_features = np.asarray(measure.features, dtype=np.float32)
        coords = np.asarray(measure.canonical_coords, dtype=np.float32)
        weights = _normalize_mass(measure.weights).astype(np.float64)
        z = (
            (raw_features.astype(np.float64) - center[None, :]) / scale[None, :]
        ).astype(np.float32)
        probs = _softmax_negative_sqdist(z, centers, temperature=temperature)
        entropy = -np.sum(
            probs.astype(np.float64) * np.log(np.maximum(probs, 1e-12)),
            axis=1,
        ) / max(float(np.log(max(n_codes, 2))), 1e-8)
        code_entropy_values.append(entropy.astype(np.float32))
        if requested_feature_mode == "soft_codebook":
            node_features = probs.astype(np.float64)
        elif requested_feature_mode == "whitened_features":
            node_features = z.astype(np.float64)
        else:
            node_features = np.hstack([z.astype(np.float64), probs.astype(np.float64)])
        structure = structure_cost(
            coords,
            scale=float(resolved_structure_scale),
            clip=structure_clip,
        )
        transport_measures.append(
            SubregionFGWMeasure(
                coords=coords.astype(np.float64),
                features=np.asarray(node_features, dtype=np.float64),
                weights=weights,
                structure=structure,
                sample_id=sample_values[idx],
                subregion_id=int(measure.subregion_id),
            )
        )
        support_sizes.append(int(coords.shape[0]))
    metadata: dict[str, object] = {
        "mode": "subregion_fgw_measures",
        "implemented": True,
        "n_subregions": int(len(transport_measures)),
        "uses_ot_costs": True,
        "feature_mode": requested_feature_mode,
        "feature_cost_default": "hellinger_codebook"
        if requested_feature_mode == "soft_codebook"
        else "sqeuclidean",
        "cell_state_codebook_size": int(n_codes),
        "cell_state_codebook_temperature": float(temperature),
        "cell_state_codebook_assignment_entropy_summary": _numeric_summary(
            np.concatenate(code_entropy_values) if code_entropy_values else []
        ),
        "codebook_feature_standardization": "mean_std_whitening_fit_on_compressed_support_sample",
        "structure_metric": requested_structure_metric,
        "structure_scale": str(structure_scale),
        "structure_scale_value": float(resolved_structure_scale),
        "structure_clip": None if structure_clip is None else float(structure_clip),
        "support_size_summary": _numeric_summary(support_sizes),
        "uses_raw_spatial_coordinates": False,
        "uses_subregion_centers": False,
        "uses_internal_canonical_coordinates": True,
    }
    return transport_measures, metadata


def fused_ot_distance(
    left: SubregionFGWMeasure,
    right: SubregionFGWMeasure,
    *,
    feature_weight: float = 0.5,
    coordinate_weight: float = 0.5,
    feature_cost_kind: str = "hellinger_codebook",
    solver: str = "emd",
    epsilon: float = 0.05,
    return_coupling: bool = False,
) -> tuple[float, np.ndarray | None, dict[str, object]]:
    p = _normalize_mass(left.weights)
    q = _normalize_mass(right.weights)
    m_feature = feature_cost(left, right, feature_cost_kind=feature_cost_kind)
    m_coord = _normalize_cost_matrix(_pairwise_sqeuclidean(left.coords, right.coords))
    fw = max(float(feature_weight), 0.0)
    cw = max(float(coordinate_weight), 0.0)
    total = max(fw + cw, 1e-12)
    cost = (fw / total) * m_feature + (cw / total) * m_coord
    requested_solver = str(solver or "emd").strip().lower()
    if requested_solver in {"sinkhorn", "entropic"}:
        coupling = ot.sinkhorn(p, q, cost, reg=max(float(epsilon), 1e-8))
        solver_name = "ot.sinkhorn"
    elif requested_solver in {"emd", "exact", "network_simplex"}:
        coupling = ot.emd(p, q, cost)
        solver_name = "ot.emd"
    else:
        raise ValueError("fused_ot solver must be 'emd' or 'sinkhorn'.")
    distance = float(np.sum(coupling * cost))
    meta = {
        "mode": HETEROGENEITY_FUSED_OT_MODE,
        "uses_ot_costs": True,
        "solver": solver_name,
        "feature_weight": float(fw / total),
        "coordinate_weight": float(cw / total),
        "feature_cost": str(feature_cost_kind),
        "n_source": int(p.size),
        "n_target": int(q.size),
        "distance": distance,
    }
    return distance, coupling if return_coupling else None, meta


def fgw_distance(
    left: SubregionFGWMeasure,
    right: SubregionFGWMeasure,
    *,
    alpha: float = 0.5,
    feature_cost_kind: str = "hellinger_codebook",
    solver: str = "conditional_gradient",
    epsilon: float = 0.05,
    loss_fun: str = "square_loss",
    max_iter: int = 500,
    tol: float = 1e-7,
    partial: bool = False,
    partial_mass: float = 0.85,
    partial_reg: float = 0.05,
    return_coupling: bool = False,
) -> tuple[float, np.ndarray | None, dict[str, object]]:
    p = _normalize_mass(left.weights)
    q = _normalize_mass(right.weights)
    c1 = np.asarray(left.structure, dtype=np.float64)
    c2 = np.asarray(right.structure, dtype=np.float64)
    m = feature_cost(left, right, feature_cost_kind=feature_cost_kind)
    a = float(np.clip(float(alpha), 0.0, 1.0))
    requested_solver = str(solver or "conditional_gradient").strip().lower()
    if a <= 1e-12 and not bool(partial):
        if return_coupling:
            coupling = ot.emd(p, q, m)
            distance = float(np.sum(coupling * m))
        else:
            coupling = None
            distance = ot.emd2(p, q, m)
        solver_name = "ot.emd_feature_only_fgw_limit"
    elif bool(partial):
        distance = ot.gromov.entropic_partial_fused_gromov_wasserstein2(
            m,
            c1,
            c2,
            p=p,
            q=q,
            reg=max(float(partial_reg), 1e-8),
            m=min(max(float(partial_mass), 1e-8), 1.0),
            loss_fun=str(loss_fun),
            alpha=a,
            numItermax=int(max_iter),
            tol=float(tol),
            symmetric=True,
            log=False,
        )
        coupling = None
        solver_name = "ot.gromov.entropic_partial_fused_gromov_wasserstein2"
    elif requested_solver in {"entropic", "pgd", "ppa"}:
        distance = ot.gromov.entropic_fused_gromov_wasserstein2(
            m,
            c1,
            c2,
            p=p,
            q=q,
            loss_fun=str(loss_fun),
            epsilon=max(float(epsilon), 1e-8),
            symmetric=True,
            alpha=a,
            max_iter=int(max_iter),
            tol=float(tol),
            solver="PPA" if requested_solver == "ppa" else "PGD",
            log=False,
        )
        coupling = None
        solver_name = "ot.gromov.entropic_fused_gromov_wasserstein2"
    elif requested_solver in {"conditional_gradient", "cg", "exact", "non_entropic"}:
        distance = ot.gromov.fused_gromov_wasserstein2(
            m,
            c1,
            c2,
            p=p,
            q=q,
            loss_fun=str(loss_fun),
            symmetric=True,
            alpha=a,
            max_iter=int(max_iter),
            tol_rel=float(tol),
            tol_abs=float(tol),
            log=False,
        )
        coupling = None
        solver_name = "ot.gromov.fused_gromov_wasserstein2"
        if return_coupling:
            coupling = ot.gromov.fused_gromov_wasserstein(
                m,
                c1,
                c2,
                p=p,
                q=q,
                loss_fun=str(loss_fun),
                symmetric=True,
                alpha=a,
                max_iter=int(max_iter),
                tol_rel=float(tol),
                tol_abs=float(tol),
                log=False,
            )
    else:
        raise ValueError(
            "fgw solver must be conditional_gradient, entropic, ppa, or partial."
        )
    distance_float = float(distance)
    meta = {
        "mode": HETEROGENEITY_FGW_MODE,
        "uses_ot_costs": True,
        "solver": solver_name,
        "alpha": a,
        "feature_cost": str(feature_cost_kind),
        "loss_fun": str(loss_fun),
        "partial": bool(partial),
        "n_source": int(p.size),
        "n_target": int(q.size),
        "distance": distance_float,
    }
    if coupling is not None:
        meta["source_marginal_error"] = float(np.max(np.abs(coupling.sum(axis=1) - p)))
        meta["target_marginal_error"] = float(np.max(np.abs(coupling.sum(axis=0) - q)))
    return distance_float, coupling, meta


def pairwise_transport_distance_matrix(
    measures: list[SubregionFGWMeasure],
    *,
    mode: str,
    max_subregions: int = 800,
    fused_ot_feature_weight: float = 0.5,
    fused_ot_coordinate_weight: float = 0.5,
    feature_cost_kind: str = "hellinger_codebook",
    fgw_alpha: float = 0.5,
    fgw_solver: str = "conditional_gradient",
    fgw_epsilon: float = 0.05,
    fgw_loss_fun: str = "square_loss",
    fgw_max_iter: int = 500,
    fgw_tol: float = 1e-7,
    fgw_partial: bool = False,
    fgw_partial_mass: float = 0.85,
    fgw_partial_reg: float = 0.05,
) -> tuple[np.ndarray, dict[str, object]]:
    requested = str(mode).strip().lower()
    if requested not in TRANSPORT_HETEROGENEITY_MODES:
        raise ValueError(
            f"mode must be one of {sorted(TRANSPORT_HETEROGENEITY_MODES)}, got '{mode}'."
        )
    n = int(len(measures))
    cap = int(max_subregions)
    if cap > 0 and n > cap:
        raise ValueError(
            f"{requested} requires an all-pairs transport distance matrix; "
            f"n_subregions={n} exceeds heterogeneity_transport_max_subregions={cap}. "
            "Use descriptor clustering, reduce/landmark subregions, or raise the cap intentionally."
        )
    distances = np.zeros((n, n), dtype=np.float32)
    solved: list[float] = []
    for left_idx in range(n):
        for right_idx in range(left_idx + 1, n):
            if requested == HETEROGENEITY_FUSED_OT_MODE:
                value, _coupling, _meta = fused_ot_distance(
                    measures[left_idx],
                    measures[right_idx],
                    feature_weight=fused_ot_feature_weight,
                    coordinate_weight=fused_ot_coordinate_weight,
                    feature_cost_kind=feature_cost_kind,
                    solver="emd",
                    return_coupling=False,
                )
            else:
                value, _coupling, _meta = fgw_distance(
                    measures[left_idx],
                    measures[right_idx],
                    alpha=fgw_alpha,
                    feature_cost_kind=feature_cost_kind,
                    solver=fgw_solver,
                    epsilon=fgw_epsilon,
                    loss_fun=fgw_loss_fun,
                    max_iter=fgw_max_iter,
                    tol=fgw_tol,
                    partial=fgw_partial,
                    partial_mass=fgw_partial_mass,
                    partial_reg=fgw_partial_reg,
                    return_coupling=False,
                )
            distances[left_idx, right_idx] = distances[right_idx, left_idx] = float(value)
            solved.append(float(value))
    metadata = {
        "mode": requested,
        "implemented": True,
        "uses_ot_costs": True,
        "distance_matrix_shape": [int(n), int(n)],
        "distance_summary": _numeric_summary(solved),
        "feature_cost": str(feature_cost_kind),
        "max_subregions": int(max_subregions),
    }
    if requested == HETEROGENEITY_FUSED_OT_MODE:
        total = max(
            max(float(fused_ot_feature_weight), 0.0)
            + max(float(fused_ot_coordinate_weight), 0.0),
            1e-12,
        )
        metadata.update(
            {
                "solver": "ot.emd",
                "feature_weight": float(max(float(fused_ot_feature_weight), 0.0) / total),
                "coordinate_weight": float(max(float(fused_ot_coordinate_weight), 0.0) / total),
            }
        )
    else:
        metadata.update(
            {
                "solver": str(fgw_solver),
                "alpha": float(np.clip(float(fgw_alpha), 0.0, 1.0)),
                "epsilon": float(fgw_epsilon),
                "loss_fun": str(fgw_loss_fun),
                "max_iter": int(fgw_max_iter),
                "tol": float(fgw_tol),
                "partial": bool(fgw_partial),
                "partial_mass": float(fgw_partial_mass),
                "partial_reg": float(fgw_partial_reg),
            }
        )
    return distances, metadata


def build_internal_heterogeneity_embeddings(
    measures: list[SubregionMeasure],
    *,
    codebook_size: int = 32,
    codebook_sample_size: int = 50000,
    grid_size: int | None = None,
    grid_radius: float | None = None,
    pair_distance_bins: tuple[float, ...] | list[float] | str | None = None,
    pair_graph_mode: str | None = None,
    pair_graph_k: int | None = None,
    pair_graph_radius: float | None = None,
    pair_bin_normalization: str | None = None,
    block_weights: dict[str, float] | None = None,
    random_state: int = 1337,
    mode: str = HETEROGENEITY_DESCRIPTOR_MODE,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build subregion embeddings from internal spatial-cell-state heterogeneity.

    The returned rows are descriptor embeddings for recurring internal niche
    motifs. They deliberately use canonical within-subregion coordinates and
    soft cell-state codebook posteriors rather than raw tissue position.
    """

    requested_mode = str(mode or HETEROGENEITY_DESCRIPTOR_MODE).strip().lower()
    if requested_mode not in HETEROGENEITY_DESCRIPTOR_ALIASES:
        raise ValueError(
            "Current heterogeneity embedding builder supports "
            f"'{HETEROGENEITY_DESCRIPTOR_MODE}' and legacy alias "
            f"'{LEGACY_HETEROGENEITY_OT_ALIAS}'."
        )
    if not measures:
        return np.zeros((0, 0), dtype=np.float32), {
            "mode": HETEROGENEITY_DESCRIPTOR_MODE,
            "requested_mode": requested_mode,
            "implemented": True,
            "n_subregions": 0,
        }
    grid = max(int(grid_size or _env_int("SPATIAL_OT_HETEROGENEITY_GRID_SIZE", 6)), 2)
    radius = float(grid_radius or _env_float("SPATIAL_OT_HETEROGENEITY_GRID_RADIUS", 1.5))
    bins = _resolve_pair_bins(pair_distance_bins)
    pair_normalization = os.environ.get(
        "SPATIAL_OT_HETEROGENEITY_PAIR_NORMALIZATION",
        "observed_over_expected",
    ).strip().lower()
    graph_mode = _resolve_pair_graph_mode(
        pair_graph_mode
        or os.environ.get("SPATIAL_OT_HETEROGENEITY_PAIR_GRAPH_MODE", "all_pairs")
    )
    graph_k = max(
        int(
            pair_graph_k
            if pair_graph_k is not None
            else _env_int("SPATIAL_OT_HETEROGENEITY_PAIR_GRAPH_K", 8)
        ),
        1,
    )
    graph_radius = (
        float(pair_graph_radius)
        if pair_graph_radius is not None
        else _env_float("SPATIAL_OT_HETEROGENEITY_PAIR_GRAPH_RADIUS", 0.0)
    )
    if graph_radius <= 0:
        graph_radius = None
    bin_normalization = str(
        pair_bin_normalization
        or os.environ.get(
            "SPATIAL_OT_HETEROGENEITY_PAIR_BIN_NORMALIZATION",
            "per_bin",
        )
    ).strip().lower()

    all_features = np.vstack([np.asarray(m.features, dtype=np.float32) for m in measures])
    max_codes = _env_int("SPATIAL_OT_HETEROGENEITY_MAX_CODEBOOK_SIZE", 16)
    n_codes = min(max(int(codebook_size), 2), max(int(max_codes), 2), int(all_features.shape[0]))
    centers, center, scale, temperature = _fit_state_codebook(
        all_features,
        codebook_size=n_codes,
        sample_size=int(codebook_sample_size),
        random_state=int(random_state),
    )

    resolved_block_weights = (
        _normalize_block_weights(block_weights)
        if block_weights is not None
        else _env_block_weights()
    )
    composition_blocks: list[np.ndarray] = []
    diversity_blocks: list[np.ndarray] = []
    field_blocks: list[np.ndarray] = []
    pair_blocks: list[np.ndarray] = []
    code_entropy_values: list[np.ndarray] = []
    subregion_effective_codes: list[float] = []
    for measure in measures:
        features = np.asarray(measure.features, dtype=np.float32)
        coords = np.asarray(measure.canonical_coords, dtype=np.float32)
        weights = np.asarray(measure.weights, dtype=np.float32)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        z = ((features.astype(np.float64) - center[None, :]) / scale[None, :]).astype(
            np.float32
        )
        probs = _softmax_negative_sqdist(z, centers, temperature=temperature)
        entropy = -np.sum(
            probs.astype(np.float64) * np.log(np.maximum(probs, 1e-12)),
            axis=1,
        ) / max(float(np.log(max(n_codes, 2))), 1e-8)
        code_entropy_values.append(entropy.astype(np.float32))
        composition = np.sum(probs * weights[:, None], axis=0).astype(np.float32)
        composition = _normalize_rows(composition[None, :])[0]
        subregion_entropy = -float(
            np.sum(
                composition.astype(np.float64)
                * np.log(np.maximum(composition.astype(np.float64), 1e-12))
            )
        )
        subregion_effective_codes.append(
            float(np.exp(subregion_entropy) / max(int(n_codes), 1))
        )
        diversity = _weighted_state_diversity(composition)
        field = _spatial_state_field(
            coords,
            probs,
            weights,
            grid_size=grid,
            radius=radius,
        )
        pair = _pair_cooccurrence(
            coords,
            probs,
            weights,
            distance_bins=bins,
            normalization=pair_normalization,
            graph_mode=graph_mode,
            graph_k=graph_k,
            graph_radius=graph_radius,
            bin_normalization=bin_normalization,
        )
        composition_blocks.append(composition.astype(np.float32))
        diversity_blocks.append(diversity.astype(np.float32))
        field_blocks.append(field.astype(np.float32))
        pair_blocks.append(pair.astype(np.float32))

    raw_blocks = {
        "composition": np.vstack(composition_blocks).astype(np.float32),
        "diversity": np.vstack(diversity_blocks).astype(np.float32),
        "spatial_field": np.vstack(field_blocks).astype(np.float32),
        "pair_cooccurrence": np.vstack(pair_blocks).astype(np.float32),
    }
    weighted_blocks: list[np.ndarray] = []
    block_diagnostics: dict[str, dict[str, object]] = {}
    block_slices: dict[str, list[int]] = {}
    cursor = 0
    for block_name in ("composition", "diversity", "spatial_field", "pair_cooccurrence"):
        weighted, stats = _standardize_descriptor_block(
            raw_blocks[block_name],
            weight=float(resolved_block_weights[block_name]),
        )
        weighted_blocks.append(weighted)
        block_diagnostics[block_name] = stats
        block_slices[block_name] = [int(cursor), int(cursor + weighted.shape[1])]
        cursor += int(weighted.shape[1])
    embeddings = np.hstack(weighted_blocks).astype(np.float32)
    total_energy = np.maximum(np.sum(embeddings * embeddings, axis=1), 1e-12)
    for block_name, weighted in zip(
        ("composition", "diversity", "spatial_field", "pair_cooccurrence"),
        weighted_blocks,
        strict=True,
    ):
        block_energy = np.sum(weighted * weighted, axis=1)
        block_diagnostics[block_name]["mean_squared_distance_contribution_fraction"] = float(
            np.mean(block_energy / total_energy)
        )

    metadata: dict[str, object] = {
        "mode": HETEROGENEITY_DESCRIPTOR_MODE,
        "requested_mode": requested_mode,
        "legacy_alias_requested": bool(requested_mode == LEGACY_HETEROGENEITY_OT_ALIAS),
        "implemented": True,
        "description": (
            "Internal heterogeneity motif embedding over compressed subregion measures: "
            "soft cell-state composition, diversity, canonical spatial-state density field, "
            "and within-subregion state-pair co-occurrence graph. Raw tissue coordinates, "
            "subregion centers, and sample labels are excluded from clustering."
        ),
        "validation_role": "primary_spatial_niche_target_descriptor; fused_ot_or_fgw_distance_should_be_used_for_publication_validation",
        "n_subregions": int(len(measures)),
        "feature_dim": int(all_features.shape[1]),
        "embedding_dim": int(embeddings.shape[1]),
        "cell_state_codebook_size": int(n_codes),
        "cell_state_codebook_temperature": float(temperature),
        "cell_state_codebook_assignment_entropy_summary": _numeric_summary(
            np.concatenate(code_entropy_values) if code_entropy_values else []
        ),
        "subregion_effective_code_fraction_summary": _numeric_summary(
            subregion_effective_codes
        ),
        "codebook_feature_standardization": "mean_std_whitening_fit_on_compressed_support_sample",
        "canonical_grid_size": int(grid),
        "canonical_grid_radius": float(radius),
        "pair_distance_bins_canonical": [float(x) for x in bins.tolist()],
        "pair_cooccurrence_normalization": str(pair_normalization),
        "pair_graph_mode": str(graph_mode),
        "pair_graph_k": int(graph_k),
        "pair_graph_radius_canonical": float(graph_radius)
        if graph_radius is not None
        else None,
        "pair_bin_normalization": str(bin_normalization),
        "blocks": [
            "soft_cell_state_composition",
            "state_diversity_multimodality",
            "canonical_spatial_state_density_field",
            "within_subregion_state_pair_cooccurrence",
        ],
        "block_weights": dict(resolved_block_weights),
        "block_slices": block_slices,
        "block_diagnostics": block_diagnostics,
        "block_scaling": "each block is mean/std standardized across subregions, multiplied by block_weight/sqrt(block_dimension), then concatenated",
        "uses_raw_spatial_coordinates": False,
        "uses_subregion_centers": False,
        "uses_internal_canonical_coordinates": True,
        "uses_pairwise_internal_spatial_graph": True,
        "uses_ot_costs": False,
        "reserved_ot_modes": ["heterogeneity_fused_ot_niche", "heterogeneity_fgw_niche"],
        "distance_note": (
            "This mode clusters a block-normalized Euclidean descriptor of the "
            "heterogeneity object. True fused OT/FGW over canonical coordinate plus "
            "cell-state measures is reserved for heterogeneity_fused_ot_niche / "
            "heterogeneity_fgw_niche."
        ),
    }
    return embeddings, metadata
