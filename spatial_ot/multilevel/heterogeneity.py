from __future__ import annotations

import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .types import SubregionMeasure

HETEROGENEITY_DESCRIPTOR_MODE = "heterogeneity_descriptor_niche"
LEGACY_HETEROGENEITY_OT_ALIAS = "heterogeneity_ot_niche"
HETEROGENEITY_DESCRIPTOR_ALIASES = {
    HETEROGENEITY_DESCRIPTOR_MODE,
    LEGACY_HETEROGENEITY_OT_ALIAS,
}
DEFAULT_BLOCK_WEIGHTS = {
    "composition": 0.20,
    "diversity": 0.15,
    "spatial_field": 0.35,
    "pair_cooccurrence": 0.30,
}


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


def _env_block_weights() -> dict[str, float]:
    weights = {
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
    total = float(sum(weights.values()))
    if total <= 1e-12:
        return dict(DEFAULT_BLOCK_WEIGHTS)
    return {key: float(value / total) for key, value in weights.items()}


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    denom = np.maximum(x.sum(axis=1, keepdims=True), 1e-12)
    return (x / denom).astype(np.float32)


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


def _pair_cooccurrence(
    coords: np.ndarray,
    probs: np.ndarray,
    weights: np.ndarray,
    *,
    distance_bins: np.ndarray,
    normalization: str = "observed_over_expected",
) -> np.ndarray:
    u = np.asarray(coords, dtype=np.float32)
    q = np.asarray(probs, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    n = int(u.shape[0])
    k = int(q.shape[1])
    bins = np.asarray(distance_bins, dtype=np.float32)
    if n < 2:
        return np.zeros((int(bins.shape[0]), k, k), dtype=np.float32).reshape(-1)
    rows, cols = np.triu_indices(n, k=1)
    d = np.sqrt(np.sum((u[rows] - u[cols]) ** 2, axis=1))
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


def build_internal_heterogeneity_embeddings(
    measures: list[SubregionMeasure],
    *,
    codebook_size: int = 32,
    codebook_sample_size: int = 50000,
    grid_size: int | None = None,
    grid_radius: float | None = None,
    pair_distance_bins: tuple[float, ...] | None = None,
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
    if pair_distance_bins is None:
        raw_bins = os.environ.get(
            "SPATIAL_OT_HETEROGENEITY_PAIR_BINS",
            "0.25,0.5,1.0,2.0",
        )
        pair_distance_bins = tuple(
            float(part.strip()) for part in raw_bins.split(",") if part.strip()
        )
    bins = np.asarray(pair_distance_bins, dtype=np.float32)
    if bins.size == 0:
        bins = np.asarray([0.25, 0.5, 1.0, 2.0], dtype=np.float32)
    bins = np.sort(np.unique(bins[bins > 0])).astype(np.float32)
    if bins.size == 0:
        bins = np.asarray([1.0], dtype=np.float32)
    pair_normalization = os.environ.get(
        "SPATIAL_OT_HETEROGENEITY_PAIR_NORMALIZATION",
        "observed_over_expected",
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

    block_weights = _env_block_weights()
    composition_blocks: list[np.ndarray] = []
    diversity_blocks: list[np.ndarray] = []
    field_blocks: list[np.ndarray] = []
    pair_blocks: list[np.ndarray] = []
    for measure in measures:
        features = np.asarray(measure.features, dtype=np.float32)
        coords = np.asarray(measure.canonical_coords, dtype=np.float32)
        weights = np.asarray(measure.weights, dtype=np.float32)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        z = ((features.astype(np.float64) - center[None, :]) / scale[None, :]).astype(
            np.float32
        )
        probs = _softmax_negative_sqdist(z, centers, temperature=temperature)
        composition = np.sum(probs * weights[:, None], axis=0).astype(np.float32)
        composition = _normalize_rows(composition[None, :])[0]
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
            weight=float(block_weights[block_name]),
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
        "codebook_feature_standardization": "mean_std_whitening_fit_on_compressed_support_sample",
        "canonical_grid_size": int(grid),
        "canonical_grid_radius": float(radius),
        "pair_distance_bins_canonical": [float(x) for x in bins.tolist()],
        "pair_cooccurrence_normalization": str(pair_normalization),
        "blocks": [
            "soft_cell_state_composition",
            "state_diversity_multimodality",
            "canonical_spatial_state_density_field",
            "within_subregion_state_pair_cooccurrence",
        ],
        "block_weights": dict(block_weights),
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
