from __future__ import annotations

import os

import numpy as np
import ot
import torch

from .numerics import pairwise_sqdist_array
from .transforms import apply_similarity
from .types import SubregionMeasure

DEFAULT_LOCAL_POSTERIOR_RADII: tuple[float, ...] = (0.25, 0.5, 1.0)
SPOT_LATENT_MODE_ATOM_BARYCENTRIC_MDS = "atom_barycentric_mds"
SPOT_LATENT_MODE_DIAGNOSTIC_FISHER = "diagnostic_fisher_current"
DEFAULT_SPOT_LATENT_MODE = SPOT_LATENT_MODE_ATOM_BARYCENTRIC_MDS
ANCHOR_DISTANCE_MODES = {
    "balanced_ot",
    "sinkhorn_ot",
    "expected_cross_cost",
    "feature_only_ot",
    "geometry_only_ot",
}

ANCHOR_SOLVER_STATUS_DIAGONAL = 0
ANCHOR_SOLVER_STATUS_EXPECTED_CROSS_COST = 1
ANCHOR_SOLVER_STATUS_OT_SUCCESS = 2
ANCHOR_SOLVER_STATUS_EXPECTED_CROSS_COST_FALLBACK = 3


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _normalize_spot_latent_mode(mode: str | None) -> str:
    requested = str(mode or DEFAULT_SPOT_LATENT_MODE).strip().lower()
    aliases = {
        "atom": SPOT_LATENT_MODE_ATOM_BARYCENTRIC_MDS,
        "atom_barycentric": SPOT_LATENT_MODE_ATOM_BARYCENTRIC_MDS,
        "atom_barycentric_mds": SPOT_LATENT_MODE_ATOM_BARYCENTRIC_MDS,
        "ot_atom_barycentric_mds": SPOT_LATENT_MODE_ATOM_BARYCENTRIC_MDS,
        "fisher": SPOT_LATENT_MODE_DIAGNOSTIC_FISHER,
        "diagnostic_fisher": SPOT_LATENT_MODE_DIAGNOSTIC_FISHER,
        "diagnostic_fisher_current": SPOT_LATENT_MODE_DIAGNOSTIC_FISHER,
    }
    if requested not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"Unknown spot latent mode '{mode}'. Expected one of: {valid}."
        )
    return aliases[requested]


def spot_latent_mode_metadata(mode: str | None = None) -> dict[str, object]:
    normalized = _normalize_spot_latent_mode(mode)
    if normalized == SPOT_LATENT_MODE_DIAGNOSTIC_FISHER:
        return {
            "mode": normalized,
            "latent_projection_mode": "global_fisher_discriminative_chart_over_ot_atom_posteriors_aligned_coords_and_local_context",
            "coordinate_scope": "global_fisher_discriminative_with_cluster_local_residual",
            "chart_learning_mode": "supervised_by_fitted_ot_subregion_labels",
            "validation_role": "diagnostic_visualization_not_independent_evidence",
            "unsupervised_baseline_required_for_validation": True,
            "label_permutation_control_recommended": True,
            "latent_refinement": "local_pca_residual_plus_weighted_fisher_discriminant",
            "includes_aligned_coordinates_in_chart_features": True,
            "uses_forced_cluster_local_radius": True,
        }
    return {
        "mode": SPOT_LATENT_MODE_ATOM_BARYCENTRIC_MDS,
        "latent_projection_mode": "balanced_ot_atom_barycentric_mds_over_cluster_atom_posteriors",
        "coordinate_scope": "cluster_atom_measure_mds_anchors_plus_atom_posterior_barycentric_within_cluster_residual",
        "chart_learning_mode": "model_grounded_atom_distance_mds_without_fisher_labels",
        "validation_role": "diagnostic_visualization_not_independent_evidence",
        "unsupervised_baseline_required_for_validation": True,
        "label_permutation_control_recommended": False,
        "latent_refinement": "atom_posterior_barycenter_without_local_pca_radius_equalization",
        "includes_aligned_coordinates_in_chart_features": False,
        "uses_forced_cluster_local_radius": False,
    }


def empty_spot_level_latent_charts(
    *,
    n_cells: int,
    atoms_per_cluster: int,
    n_clusters: int = 0,
    mode: str | None = None,
) -> dict[str, np.ndarray]:
    metadata = spot_latent_mode_metadata(mode)
    n_clusters = int(n_clusters)
    return {
        "spot_latent_cell_indices": np.zeros(0, dtype=np.int32),
        "spot_latent_subregion_ids": np.zeros(0, dtype=np.int32),
        "spot_latent_cluster_labels": np.zeros(0, dtype=np.int32),
        "spot_latent_coords": np.zeros((0, 2), dtype=np.float32),
        "spot_latent_within_coords": np.zeros((0, 2), dtype=np.float32),
        "spot_latent_cluster_anchors": np.zeros((n_clusters, 2), dtype=np.float32),
        "spot_latent_atom_embedding": np.zeros(
            (n_clusters, int(atoms_per_cluster), 2), dtype=np.float32
        ),
        "spot_latent_aligned_coords": np.zeros((0, 2), dtype=np.float32),
        "spot_latent_cluster_probs": np.zeros(0, dtype=np.float32),
        "spot_latent_atom_confidence": np.zeros(0, dtype=np.float32),
        "spot_latent_posterior_entropy": np.zeros(0, dtype=np.float32),
        "spot_latent_normalized_posterior_entropy": np.zeros(0, dtype=np.float32),
        "spot_latent_atom_argmax": np.zeros(0, dtype=np.int32),
        "spot_latent_temperature_used": np.zeros(0, dtype=np.float32),
        "spot_latent_temperature_cost_gap": np.zeros(0, dtype=np.float32),
        "spot_latent_temperature_fixed": np.zeros(0, dtype=np.float32),
        "spot_latent_weights": np.zeros(0, dtype=np.float32),
        "spot_latent_atom_posteriors": np.zeros(
            (0, int(atoms_per_cluster)), dtype=np.float32
        ),
        "spot_latent_posterior_entropy_cost_gap": np.zeros(0, dtype=np.float32),
        "spot_latent_normalized_posterior_entropy_cost_gap": np.zeros(
            0, dtype=np.float32
        ),
        "spot_latent_posterior_entropy_fixed": np.zeros(0, dtype=np.float32),
        "spot_latent_normalized_posterior_entropy_fixed": np.zeros(0, dtype=np.float32),
        "spot_latent_cluster_anchor_distance": np.zeros(
            (n_clusters, n_clusters), dtype=np.float32
        ),
        "spot_latent_cluster_anchor_ot_fallback_matrix": np.zeros(
            (n_clusters, n_clusters), dtype=bool
        ),
        "spot_latent_cluster_anchor_solver_status_matrix": np.zeros(
            (n_clusters, n_clusters), dtype=np.int8
        ),
        "spot_latent_cluster_anchor_ot_fallback_fraction": np.array(
            0.0, dtype=np.float32
        ),
        "spot_latent_atom_mds_stress": np.full(n_clusters, np.nan, dtype=np.float32),
        "spot_latent_atom_mds_positive_eigenvalue_mass_2d": np.full(
            n_clusters, np.nan, dtype=np.float32
        ),
        "spot_latent_atom_mds_negative_eigenvalue_mass_fraction": np.full(
            n_clusters, np.nan, dtype=np.float32
        ),
        "cell_spot_latent_unweighted_coords": np.full(
            (int(n_cells), 2), np.nan, dtype=np.float32
        ),
        "cell_spot_latent_confidence_weighted_coords": np.full(
            (int(n_cells), 2), np.nan, dtype=np.float32
        ),
        "cell_spot_latent_coords": np.full((int(n_cells), 2), np.nan, dtype=np.float32),
        "cell_spot_latent_cluster_labels": np.full(int(n_cells), -1, dtype=np.int32),
        "cell_spot_latent_weights": np.zeros(int(n_cells), dtype=np.float32),
        "cell_spot_latent_posterior_entropy": np.full(
            int(n_cells), np.nan, dtype=np.float32
        ),
        "spot_latent_mode": np.array(str(metadata["mode"])),
        "spot_latent_chart_learning_mode": np.array(
            str(metadata["chart_learning_mode"])
        ),
        "spot_latent_projection_mode": np.array(
            str(metadata["latent_projection_mode"])
        ),
        "spot_latent_validation_role": np.array(str(metadata["validation_role"])),
        "spot_latent_global_within_scale": np.array(np.nan, dtype=np.float32),
        "spot_latent_assignment_temperature": np.array(np.nan, dtype=np.float32),
        "spot_latent_temperature_mode": np.array("auto_entropy"),
        "spot_latent_cluster_anchor_distance_method": np.array("balanced_ot"),
        "spot_latent_cluster_anchor_distance_requested_method": np.array("balanced_ot"),
        "spot_latent_cluster_anchor_distance_effective_method": np.array("balanced_ot"),
        "spot_latent_cluster_mds_stress": np.array(np.nan, dtype=np.float32),
        "spot_latent_cluster_mds_positive_eigenvalue_mass_2d": np.array(
            np.nan, dtype=np.float32
        ),
        "spot_latent_cluster_mds_negative_eigenvalue_mass_fraction": np.array(
            np.nan, dtype=np.float32
        ),
    }


def weighted_atom_posteriors(
    total_cost: np.ndarray,
    prototype_weights: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    temp = max(float(temperature), 1e-5)
    weights = np.clip(np.asarray(prototype_weights, dtype=np.float32), 1e-8, None)
    logits = (
        -np.asarray(total_cost, dtype=np.float32) / temp
        + np.log(weights).astype(np.float32)[None, :]
    )
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(logits, -80.0, 80.0)).astype(np.float32)
    return (
        exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-8)
    ).astype(np.float32)


def _cost_gap_temperature(total_cost: np.ndarray, base_temperature: float) -> float:
    base = max(float(base_temperature), 1e-5)
    cost = np.asarray(total_cost, dtype=np.float64)
    finite = cost[np.isfinite(cost)]
    if cost.ndim != 2 or cost.shape[1] <= 1 or finite.size == 0:
        return base
    sorted_cost = np.sort(cost, axis=1)
    gaps = sorted_cost[:, 1] - sorted_cost[:, 0]
    positive_gaps = gaps[np.isfinite(gaps) & (gaps > 1e-8)]
    if positive_gaps.size:
        scale = float(np.median(positive_gaps))
    else:
        p10, p90 = np.percentile(finite, [10.0, 90.0])
        scale = float(0.1 * max(p90 - p10, 0.0))
    if not np.isfinite(scale) or scale <= 0:
        scale = base
    p05, p95 = np.percentile(finite, [5.0, 95.0])
    spread = max(float(p95 - p05), 1e-4)
    return float(np.clip(scale, 1e-4, max(spread, 1e-4)))


def _resolve_posterior_temperature(
    total_cost: np.ndarray,
    prototype_weights: np.ndarray,
    base_temperature: float,
    mode: str,
) -> float:
    requested = str(mode or "auto").strip().lower()
    base = max(float(base_temperature), 1e-5)
    if requested in {"fixed", "manual", "none"}:
        return base
    cost = np.asarray(total_cost, dtype=np.float64)
    finite = cost[np.isfinite(cost)]
    if cost.ndim != 2 or cost.shape[1] <= 1 or finite.size == 0:
        return base
    if requested in {"auto_cost_gap", "cost_gap", "auto"}:
        return _cost_gap_temperature(cost, base)
    if requested not in {
        "auto_entropy",
        "entropy",
        "entropy_target",
        "auto_cost_gap_entropy",
    }:
        return _cost_gap_temperature(cost, base)

    target = float(
        np.clip(
            _env_float("SPATIAL_OT_SPOT_LATENT_TARGET_NORMALIZED_ENTROPY", 0.45),
            0.05,
            0.95,
        )
    )
    p05, p95 = np.percentile(finite, [5.0, 95.0])
    spread = max(float(p95 - p05), 1e-4)
    cost_gap_temp = _cost_gap_temperature(cost, base)
    lo = 1e-5
    hi = max(float(spread) * 4.0, float(base) * 10.0, float(cost_gap_temp) * 10.0, 1e-3)
    weights = np.asarray(prototype_weights, dtype=np.float32)
    for _ in range(36):
        mid = float(np.sqrt(lo * hi))
        posterior = weighted_atom_posteriors(cost, weights, temperature=mid)
        _entropy, normalized = _posterior_entropy(posterior)
        median_entropy = float(np.nanmedian(normalized)) if normalized.size else target
        if not np.isfinite(median_entropy):
            break
        if median_entropy < target:
            lo = mid
        else:
            hi = mid
    return float(np.clip(hi, 1e-5, hi))


def _posterior_entropy(posterior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(posterior, dtype=np.float32)
    if q.ndim != 2 or q.shape[0] == 0:
        return np.zeros(q.shape[0] if q.ndim >= 1 else 0, dtype=np.float32), np.zeros(
            q.shape[0] if q.ndim >= 1 else 0,
            dtype=np.float32,
        )
    clipped = np.clip(q, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1).astype(np.float32)
    max_entropy = float(np.log(q.shape[1])) if q.shape[1] > 1 else 1.0
    normalized = (entropy / max(max_entropy, 1e-8)).astype(np.float32)
    return entropy, normalized


def _pairwise_sqdist_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 2 or y_arr.ndim != 2 or x_arr.shape[1] != y_arr.shape[1]:
        return np.zeros((x_arr.shape[0], y_arr.shape[0]), dtype=np.float32)
    diff = x_arr[:, None, :] - y_arr[None, :, :]
    return np.maximum(np.sum(diff * diff, axis=2), 0.0).astype(np.float32)


def _atom_metric_sqdist(
    coords_a: np.ndarray,
    features_a: np.ndarray,
    coords_b: np.ndarray,
    features_b: np.ndarray,
    *,
    lambda_x: float,
    lambda_y: float,
    cost_scale_x: float,
    cost_scale_y: float,
    include_geometry: bool = True,
    include_features: bool = True,
) -> np.ndarray:
    n_a = int(np.asarray(coords_a).shape[0])
    n_b = int(np.asarray(coords_b).shape[0])
    cost = np.zeros((n_a, n_b), dtype=np.float32)
    if include_geometry and float(lambda_x) > 0:
        cost += (
            float(lambda_x)
            * _pairwise_sqdist_np(coords_a, coords_b)
            / max(float(cost_scale_x), 1e-5)
        ).astype(np.float32)
    if include_features and float(lambda_y) > 0:
        cost += (
            float(lambda_y)
            * _pairwise_sqdist_np(features_a, features_b)
            / max(float(cost_scale_y), 1e-5)
        ).astype(np.float32)
    return np.maximum(cost, 0.0).astype(np.float32)


def _normalised_weights(weights: np.ndarray, n: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != int(n) or not np.any(w > 0):
        w = np.ones(int(n), dtype=np.float64)
    w = np.maximum(w, 1e-12)
    return (w / max(float(w.sum()), 1e-12)).astype(np.float64)


def _empty_mds_diagnostics() -> dict[str, float]:
    return {
        "stress": float("nan"),
        "positive_eigenvalue_mass_2d": float("nan"),
        "negative_eigenvalue_mass_fraction": float("nan"),
    }


def _mds_stress(
    sqdist: np.ndarray, coords: np.ndarray, weights: np.ndarray | None = None
) -> float:
    d2 = np.asarray(sqdist, dtype=np.float64)
    z = np.asarray(coords, dtype=np.float64)
    if (
        d2.ndim != 2
        or d2.shape[0] != d2.shape[1]
        or z.ndim != 2
        or z.shape[0] != d2.shape[0]
        or d2.shape[0] < 2
    ):
        return float("nan")
    d = np.sqrt(np.maximum(0.5 * (d2 + d2.T), 0.0))
    dz = np.sqrt(np.maximum(_pairwise_sqdist_np(z, z).astype(np.float64), 0.0))
    mask = np.triu(np.ones(d.shape, dtype=bool), k=1)
    if weights is None:
        pair_w = np.ones_like(d, dtype=np.float64)
    else:
        w = _normalised_weights(weights, d.shape[0])
        pair_w = w[:, None] * w[None, :]
    denom = float(np.sum(pair_w[mask] * d[mask] * d[mask]))
    if denom <= 1e-12:
        return 0.0
    raw = float(np.sum(pair_w[mask] * (d[mask] - dz[mask]) ** 2))
    return float(np.sqrt(max(raw, 0.0) / denom))


def _classical_mds_from_sqdist_with_diagnostics(
    sqdist: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    n_components: int = 2,
) -> tuple[np.ndarray, dict[str, float]]:
    d2 = np.asarray(sqdist, dtype=np.float64)
    if d2.ndim != 2 or d2.shape[0] != d2.shape[1]:
        return np.zeros(
            (0, int(n_components)), dtype=np.float32
        ), _empty_mds_diagnostics()
    n = int(d2.shape[0])
    n_components = int(n_components)
    if n == 0:
        return np.zeros((0, n_components), dtype=np.float32), _empty_mds_diagnostics()
    if n == 1:
        return np.zeros((1, n_components), dtype=np.float32), {
            "stress": 0.0,
            "positive_eigenvalue_mass_2d": 1.0,
            "negative_eigenvalue_mass_fraction": 0.0,
        }
    d2 = np.maximum(0.5 * (d2 + d2.T), 0.0)
    if weights is None:
        row_mean = d2.mean(axis=1)
        col_mean = d2.mean(axis=0)
        total_mean = float(d2.mean())
    else:
        w = _normalised_weights(weights, n)
        row_mean = d2 @ w
        col_mean = w @ d2
        total_mean = float(w @ d2 @ w)
    gram = -0.5 * (d2 - row_mean[:, None] - col_mean[None, :] + total_mean)
    gram = 0.5 * (gram + gram.T)
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    positive = np.maximum(eigvals[order[:n_components]], 0.0)
    basis = eigvecs[:, order[:n_components]]
    coords = basis * np.sqrt(positive)[None, :]
    if coords.shape[1] < n_components:
        coords = np.pad(
            coords, ((0, 0), (0, n_components - coords.shape[1])), constant_values=0.0
        )
    for dim in range(coords.shape[1]):
        anchor = int(np.argmax(np.abs(coords[:, dim])))
        if coords[anchor, dim] < 0:
            coords[:, dim] *= -1.0
    if weights is None:
        center = coords.mean(axis=0)
    else:
        center = np.average(coords, axis=0, weights=_normalised_weights(weights, n))
    coords = coords - center[None, :]
    coords[~np.isfinite(coords)] = 0.0
    ordered_eigvals = eigvals[order]
    positive_mass = float(np.sum(np.maximum(ordered_eigvals, 0.0)))
    positive_mass_2d = float(np.sum(np.maximum(ordered_eigvals[:n_components], 0.0)))
    absolute_mass = float(np.sum(np.abs(ordered_eigvals)))
    negative_mass = float(np.sum(np.abs(ordered_eigvals[ordered_eigvals < 0.0])))
    coords_out = coords[:, :n_components].astype(np.float32)
    diagnostics = {
        "stress": _mds_stress(d2, coords_out, weights=weights),
        "positive_eigenvalue_mass_2d": (
            float(positive_mass_2d / max(positive_mass, 1e-12))
            if positive_mass > 0.0
            else 0.0
        ),
        "negative_eigenvalue_mass_fraction": (
            float(negative_mass / max(absolute_mass, 1e-12))
            if absolute_mass > 0.0
            else 0.0
        ),
    }
    return coords_out, diagnostics


def _classical_mds_from_sqdist(
    sqdist: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    n_components: int = 2,
) -> np.ndarray:
    coords, _diagnostics = _classical_mds_from_sqdist_with_diagnostics(
        sqdist,
        weights=weights,
        n_components=n_components,
    )
    return coords


def _cluster_atom_measure_sqdist(
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    prototype_weights: np.ndarray,
    *,
    lambda_x: float,
    lambda_y: float,
    cost_scale_x: float,
    cost_scale_y: float,
    distance_mode: str = "balanced_ot",
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    requested = str(distance_mode or "balanced_ot").strip().lower()
    if requested not in ANCHOR_DISTANCE_MODES:
        requested = "balanced_ot"
    include_geometry = requested != "feature_only_ot"
    include_features = requested != "geometry_only_ot"
    use_sinkhorn = requested == "sinkhorn_ot"
    use_expected = requested == "expected_cross_cost"
    n_clusters = int(atom_coords.shape[0])
    distances = np.zeros((n_clusters, n_clusters), dtype=np.float32)
    fallback_matrix = np.zeros((n_clusters, n_clusters), dtype=bool)
    solver_status_matrix = np.full(
        (n_clusters, n_clusters),
        ANCHOR_SOLVER_STATUS_DIAGONAL,
        dtype=np.int8,
    )
    for left in range(n_clusters):
        weights_left = _normalised_weights(
            prototype_weights[left], atom_coords.shape[1]
        )
        for right in range(left + 1, n_clusters):
            weights_right = _normalised_weights(
                prototype_weights[right], atom_coords.shape[1]
            )
            cross_cost = _atom_metric_sqdist(
                atom_coords[left],
                atom_features[left],
                atom_coords[right],
                atom_features[right],
                lambda_x=lambda_x,
                lambda_y=lambda_y,
                cost_scale_x=cost_scale_x,
                cost_scale_y=cost_scale_y,
                include_geometry=include_geometry,
                include_features=include_features,
            )
            if use_expected:
                value = float(
                    weights_left @ cross_cost.astype(np.float64) @ weights_right
                )
                solver_status = ANCHOR_SOLVER_STATUS_EXPECTED_CROSS_COST
            else:
                try:
                    if use_sinkhorn:
                        eps = max(
                            _env_float("SPATIAL_OT_SPOT_LATENT_ANCHOR_OT_EPS", 0.03),
                            1e-6,
                        )
                        value = float(
                            ot.sinkhorn2(
                                weights_left,
                                weights_right,
                                cross_cost.astype(np.float64),
                                reg=eps,
                                numItermax=2000,
                                stopThr=1e-9,
                            )
                        )
                    else:
                        value = float(
                            ot.emd2(
                                weights_left,
                                weights_right,
                                cross_cost.astype(np.float64),
                            )
                        )
                    solver_status = ANCHOR_SOLVER_STATUS_OT_SUCCESS
                except Exception:
                    value = float(
                        weights_left @ cross_cost.astype(np.float64) @ weights_right
                    )
                    solver_status = ANCHOR_SOLVER_STATUS_EXPECTED_CROSS_COST_FALLBACK
                    fallback_matrix[left, right] = fallback_matrix[right, left] = True
            distances[left, right] = distances[right, left] = max(value, 0.0)
            solver_status_matrix[left, right] = solver_status_matrix[right, left] = (
                solver_status
            )
    if not return_diagnostics:
        return distances
    pair_mask = np.triu(np.ones((n_clusters, n_clusters), dtype=bool), k=1)
    fallback_fraction = (
        float(np.mean(fallback_matrix[pair_mask])) if np.any(pair_mask) else 0.0
    )
    effective_method = requested
    if fallback_fraction > 0.0:
        effective_method = f"{requested}_with_expected_cross_cost_fallback"
    diagnostics = {
        "requested_method": str(requested),
        "effective_method": str(effective_method),
        "fallback_matrix": fallback_matrix,
        "fallback_fraction": fallback_fraction,
        "solver_status_matrix": solver_status_matrix,
        "solver_status_codebook": {
            "diagonal": ANCHOR_SOLVER_STATUS_DIAGONAL,
            "expected_cross_cost": ANCHOR_SOLVER_STATUS_EXPECTED_CROSS_COST,
            "ot_success": ANCHOR_SOLVER_STATUS_OT_SUCCESS,
            "expected_cross_cost_fallback": ANCHOR_SOLVER_STATUS_EXPECTED_CROSS_COST_FALLBACK,
        },
    }
    return distances, diagnostics


def _atom_barycentric_mds_chart(
    *,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    prototype_weights: np.ndarray,
    atom_posteriors: np.ndarray,
    cluster_labels: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    cost_scale_x: float,
    cost_scale_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, object]]:
    labels = np.asarray(cluster_labels, dtype=np.int32)
    posteriors = np.asarray(atom_posteriors, dtype=np.float32)
    n_clusters = int(atom_coords.shape[0])
    atoms_per_cluster = int(atom_coords.shape[1]) if atom_coords.ndim == 3 else 0
    atom_embeddings = np.zeros((n_clusters, atoms_per_cluster, 2), dtype=np.float32)
    atom_mds_stress = np.full(n_clusters, np.nan, dtype=np.float32)
    atom_mds_positive_mass = np.full(n_clusters, np.nan, dtype=np.float32)
    atom_mds_negative_mass = np.full(n_clusters, np.nan, dtype=np.float32)
    within = np.zeros((labels.shape[0], 2), dtype=np.float32)
    if n_clusters == 0 or atoms_per_cluster == 0 or labels.shape[0] == 0:
        empty_status = np.zeros((n_clusters, n_clusters), dtype=np.int8)
        return (
            np.zeros((labels.shape[0], 2), dtype=np.float32),
            within,
            np.zeros((n_clusters, 2), dtype=np.float32),
            atom_embeddings,
            1.0,
            {
                "cluster_anchor_distance": np.zeros(
                    (n_clusters, n_clusters), dtype=np.float32
                ),
                "cluster_anchor_distance_method": "balanced_ot",
                "cluster_anchor_distance_requested_method": "balanced_ot",
                "cluster_anchor_distance_effective_method": "balanced_ot",
                "cluster_anchor_ot_fallback_matrix": np.zeros(
                    (n_clusters, n_clusters), dtype=bool
                ),
                "cluster_anchor_solver_status_matrix": empty_status,
                "cluster_anchor_ot_fallback_fraction": 0.0,
                "cluster_anchor_solver_status_codebook": {
                    "diagonal": ANCHOR_SOLVER_STATUS_DIAGONAL,
                    "expected_cross_cost": ANCHOR_SOLVER_STATUS_EXPECTED_CROSS_COST,
                    "ot_success": ANCHOR_SOLVER_STATUS_OT_SUCCESS,
                    "expected_cross_cost_fallback": ANCHOR_SOLVER_STATUS_EXPECTED_CROSS_COST_FALLBACK,
                },
                "cluster_mds": _empty_mds_diagnostics(),
                "atom_mds_stress": atom_mds_stress,
                "atom_mds_positive_eigenvalue_mass_2d": atom_mds_positive_mass,
                "atom_mds_negative_eigenvalue_mass_fraction": atom_mds_negative_mass,
            },
        )
    for cluster_id in range(n_clusters):
        atom_sqdist = _atom_metric_sqdist(
            atom_coords[cluster_id],
            atom_features[cluster_id],
            atom_coords[cluster_id],
            atom_features[cluster_id],
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
        )
        embedding, atom_diag = _classical_mds_from_sqdist_with_diagnostics(
            atom_sqdist,
            weights=prototype_weights[cluster_id],
            n_components=2,
        )
        atom_embeddings[cluster_id] = embedding.astype(np.float32)
        atom_mds_stress[cluster_id] = float(atom_diag["stress"])
        atom_mds_positive_mass[cluster_id] = float(
            atom_diag["positive_eigenvalue_mass_2d"]
        )
        atom_mds_negative_mass[cluster_id] = float(
            atom_diag["negative_eigenvalue_mass_fraction"]
        )
        idx = np.flatnonzero(labels == cluster_id)
        if idx.size:
            within[idx] = (
                posteriors[idx].astype(np.float32) @ embedding.astype(np.float32)
            ).astype(np.float32)

    anchor_distance_method = (
        os.environ.get("SPATIAL_OT_SPOT_LATENT_ANCHOR_DISTANCE", "balanced_ot")
        .strip()
        .lower()
    )
    cluster_sqdist, anchor_diagnostics = _cluster_atom_measure_sqdist(
        atom_coords,
        atom_features,
        prototype_weights,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        cost_scale_x=cost_scale_x,
        cost_scale_y=cost_scale_y,
        distance_mode=anchor_distance_method,
        return_diagnostics=True,
    )
    anchors, cluster_mds_diag = _classical_mds_from_sqdist_with_diagnostics(
        cluster_sqdist,
        weights=np.maximum(
            np.asarray(prototype_weights, dtype=np.float32).sum(axis=1), 1e-8
        ),
        n_components=2,
    )
    latent = np.zeros_like(within, dtype=np.float32)
    valid = (labels >= 0) & (labels < n_clusters)
    latent[valid] = anchors[labels[valid]] + within[valid]
    latent[~np.all(np.isfinite(latent), axis=1)] = 0.0
    within[~np.all(np.isfinite(within), axis=1)] = 0.0
    diagnostics = {
        "cluster_anchor_distance": cluster_sqdist.astype(np.float32),
        "cluster_anchor_distance_method": str(anchor_diagnostics["requested_method"]),
        "cluster_anchor_distance_requested_method": str(
            anchor_diagnostics["requested_method"]
        ),
        "cluster_anchor_distance_effective_method": str(
            anchor_diagnostics["effective_method"]
        ),
        "cluster_anchor_ot_fallback_matrix": np.asarray(
            anchor_diagnostics["fallback_matrix"], dtype=bool
        ),
        "cluster_anchor_solver_status_matrix": np.asarray(
            anchor_diagnostics["solver_status_matrix"], dtype=np.int8
        ),
        "cluster_anchor_ot_fallback_fraction": float(
            anchor_diagnostics["fallback_fraction"]
        ),
        "cluster_anchor_solver_status_codebook": dict(
            anchor_diagnostics["solver_status_codebook"]
        ),
        "cluster_mds": cluster_mds_diag,
        "atom_mds_stress": atom_mds_stress.astype(np.float32),
        "atom_mds_positive_eigenvalue_mass_2d": atom_mds_positive_mass.astype(
            np.float32
        ),
        "atom_mds_negative_eigenvalue_mass_fraction": atom_mds_negative_mass.astype(
            np.float32
        ),
    }
    return (
        latent.astype(np.float32),
        within.astype(np.float32),
        anchors.astype(np.float32),
        atom_embeddings,
        1.0,
        diagnostics,
    )


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
        local_mean = (neighbors.astype(np.float32) @ posterior) / counts.astype(
            np.float32
        )
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
    return (
        (x_arr - mean.astype(np.float32)[None, :]) / std.astype(np.float32)[None, :]
    ).astype(np.float32)


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
    global_center = np.average(
        x_arr.astype(np.float64), axis=0, weights=np.maximum(w, 1e-8)
    )
    centroids = np.tile(global_center[None, :], (int(n_clusters), 1)).astype(np.float32)
    masses = np.zeros(int(n_clusters), dtype=np.float64)
    for cluster_id in range(int(n_clusters)):
        idx = np.flatnonzero(label_arr == cluster_id)
        if idx.size == 0:
            continue
        local_w = np.maximum(w[idx], 1e-8)
        masses[cluster_id] = float(local_w.sum())
        centroids[cluster_id] = np.average(
            x_arr[idx].astype(np.float64), axis=0, weights=local_w
        ).astype(np.float32)
    return centroids, masses.astype(np.float32)


def _weighted_pca_scores(
    x: np.ndarray, weights: np.ndarray, *, n_components: int = 2
) -> np.ndarray:
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
        scores = np.column_stack(
            [scores, np.zeros((scores.shape[0], int(n_components) - scores.shape[1]))]
        )
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
        whitening = (
            eigvecs_w[:, keep]
            @ np.diag(1.0 / np.sqrt(eigvals_w[keep]))
            @ eigvecs_w[:, keep].T
        )
        fisher = whitening @ sb @ whitening
        eigvals_f, eigvecs_f = np.linalg.eigh((fisher + fisher.T) * 0.5)
        order = np.argsort(eigvals_f)[::-1]
        basis = whitening @ eigvecs_f[:, order[: max(1, int(n_components))]]
        scores = (x_arr - global_mean[None, :]) @ basis
    except np.linalg.LinAlgError:
        return _weighted_pca_scores(x_arr, w, n_components=int(n_components))

    if scores.shape[1] < int(n_components):
        scores = np.column_stack(
            [scores, np.zeros((scores.shape[0], int(n_components) - scores.shape[1]))]
        )
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
    if (
        latent.ndim != 2
        or latent.shape[1] != 2
        or labels.ndim != 1
        or labels.shape[0] != latent.shape[0]
    ):
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
        int(np.sum(labels[valid] == cluster_id)) for cluster_id in range(n_clusters)
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
        radius = np.sqrt(
            np.sum((latent[idx].astype(np.float64) - center[None, :]) ** 2, axis=1)
        )
        within.append(float(np.average(radius, weights=local_w)))
    center_arr = (
        np.vstack(centers).astype(np.float32)
        if centers
        else np.zeros((0, 2), dtype=np.float32)
    )
    finite_centers = np.all(np.isfinite(center_arr), axis=1)
    if int(np.sum(finite_centers)) >= 2:
        distances = np.sqrt(
            np.maximum(
                (
                    (
                        center_arr[finite_centers, None, :]
                        - center_arr[finite_centers][None, :, :]
                    )
                    ** 2
                ).sum(axis=2),
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
        "median_between_cluster_center_distance": float(np.median(between))
        if between.size
        else None,
        "min_between_cluster_center_distance": float(np.min(between))
        if between.size
        else None,
        "separation_ratio_median_between_over_mean_within": (
            float(np.median(between) / max(float(mean_within), 1e-8))
            if between.size and mean_within is not None
            else None
        ),
        "minimum_between_cluster_distance_forced": False,
    }
    if subregion_ids is not None:
        subregion_arr = np.asarray(subregion_ids)
        out["n_subregions"] = (
            int(np.unique(subregion_arr).size)
            if subregion_arr.shape[0] == labels.shape[0]
            else None
        )
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
    spot_latent_mode: str | None = None,
) -> dict[str, np.ndarray]:
    compute_device = compute_device or torch.device("cpu")
    mode = _normalize_spot_latent_mode(
        os.environ.get(
            "SPATIAL_OT_SPOT_LATENT_MODE", spot_latent_mode or DEFAULT_SPOT_LATENT_MODE
        )
    )
    temperature_mode = os.environ.get(
        "SPATIAL_OT_SPOT_LATENT_TEMPERATURE_MODE", "auto_entropy"
    )
    metadata = spot_latent_mode_metadata(mode)
    needs_fisher_chart_features = mode == SPOT_LATENT_MODE_DIAGNOSTIC_FISHER
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
        return empty_spot_level_latent_charts(
            n_cells=n_cells,
            atoms_per_cluster=atoms_per_cluster,
            n_clusters=n_clusters,
            mode=mode,
        )

    cell_indices = np.zeros(n_occurrences, dtype=np.int32)
    subregion_ids = np.zeros(n_occurrences, dtype=np.int32)
    cluster_labels = np.zeros(n_occurrences, dtype=np.int32)
    aligned_coords = np.zeros((n_occurrences, 2), dtype=np.float32)
    cluster_probs = np.zeros(n_occurrences, dtype=np.float32)
    atom_confidence = np.zeros(n_occurrences, dtype=np.float32)
    posterior_entropy = np.zeros(n_occurrences, dtype=np.float32)
    normalized_posterior_entropy = np.zeros(n_occurrences, dtype=np.float32)
    atom_argmax = np.zeros(n_occurrences, dtype=np.int32)
    temperature_used = np.zeros(n_occurrences, dtype=np.float32)
    temperature_cost_gap = np.zeros(n_occurrences, dtype=np.float32)
    temperature_fixed = np.zeros(n_occurrences, dtype=np.float32)
    weights = np.zeros(n_occurrences, dtype=np.float32)
    atom_posteriors = np.zeros((n_occurrences, atoms_per_cluster), dtype=np.float32)
    posterior_entropy_cost_gap = np.zeros(n_occurrences, dtype=np.float32)
    normalized_posterior_entropy_cost_gap = np.zeros(n_occurrences, dtype=np.float32)
    posterior_entropy_fixed = np.zeros(n_occurrences, dtype=np.float32)
    normalized_posterior_entropy_fixed = np.zeros(n_occurrences, dtype=np.float32)
    cluster_anchor_distance = np.zeros((n_clusters, n_clusters), dtype=np.float32)
    cluster_anchor_ot_fallback_matrix = np.zeros((n_clusters, n_clusters), dtype=bool)
    cluster_anchor_solver_status_matrix = np.zeros(
        (n_clusters, n_clusters), dtype=np.int8
    )
    cluster_anchor_ot_fallback_fraction = 0.0
    atom_mds_stress = np.full(n_clusters, np.nan, dtype=np.float32)
    atom_mds_positive_mass = np.full(n_clusters, np.nan, dtype=np.float32)
    atom_mds_negative_mass = np.full(n_clusters, np.nan, dtype=np.float32)
    cluster_anchor_distance_method = "balanced_ot"
    cluster_anchor_distance_requested_method = "balanced_ot"
    cluster_anchor_distance_effective_method = "balanced_ot"
    cluster_mds_stress = float("nan")
    cluster_mds_positive_mass = float("nan")
    cluster_mds_negative_mass = float("nan")
    chart_dim = 2 + atoms_per_cluster * (1 + len(local_posterior_radii))
    chart_features = (
        np.zeros((n_occurrences, chart_dim), dtype=np.float32)
        if needs_fisher_chart_features
        else np.zeros((n_occurrences, 0), dtype=np.float32)
    )

    offset = 0
    for r, measure in enumerate(measures):
        members = np.asarray(measure.members, dtype=np.int32)
        n_members = int(members.size)
        if n_members == 0:
            continue
        k = int(subregion_labels[r])
        canonical = measure.normalizer.transform(coords_um[members]).astype(np.float32)
        aligned = apply_similarity(canonical, assigned_transforms[r]).astype(np.float32)
        cx = pairwise_sqdist_array(
            aligned, atom_coords[k], device=compute_device
        ) / max(float(cost_scale_x), 1e-5)
        cy = pairwise_sqdist_array(
            features[members], atom_features[k], device=compute_device
        ) / max(float(cost_scale_y), 1e-5)
        total_cost = float(lambda_x) * cx + float(lambda_y) * cy
        fixed_temperature = max(float(assignment_temperature), 1e-5)
        cost_gap_temperature = _cost_gap_temperature(total_cost, fixed_temperature)
        effective_temperature = _resolve_posterior_temperature(
            total_cost,
            prototype_weights[k],
            fixed_temperature,
            temperature_mode,
        )
        posterior = weighted_atom_posteriors(
            total_cost,
            prototype_weights[k],
            temperature=effective_temperature,
        )
        entropy, norm_entropy = _posterior_entropy(posterior)
        cost_gap_posterior = weighted_atom_posteriors(
            total_cost,
            prototype_weights[k],
            temperature=cost_gap_temperature,
        )
        entropy_cost_gap, norm_entropy_cost_gap = _posterior_entropy(cost_gap_posterior)
        fixed_posterior = weighted_atom_posteriors(
            total_cost,
            prototype_weights[k],
            temperature=fixed_temperature,
        )
        entropy_fixed, norm_entropy_fixed = _posterior_entropy(fixed_posterior)
        assigned_cluster_prob = np.full(
            n_members, float(subregion_cluster_probs[r, k]), dtype=np.float32
        )
        confidence = posterior.max(axis=1).astype(np.float32)
        occurrence_weight = (assigned_cluster_prob * confidence).astype(np.float32)

        stop = offset + n_members
        cell_indices[offset:stop] = members
        subregion_ids[offset:stop] = int(r)
        cluster_labels[offset:stop] = int(k)
        aligned_coords[offset:stop] = aligned
        cluster_probs[offset:stop] = assigned_cluster_prob
        atom_confidence[offset:stop] = confidence
        posterior_entropy[offset:stop] = entropy
        normalized_posterior_entropy[offset:stop] = norm_entropy
        atom_argmax[offset:stop] = posterior.argmax(axis=1).astype(np.int32)
        temperature_used[offset:stop] = float(effective_temperature)
        temperature_cost_gap[offset:stop] = float(cost_gap_temperature)
        temperature_fixed[offset:stop] = float(fixed_temperature)
        weights[offset:stop] = occurrence_weight
        atom_posteriors[offset:stop] = posterior
        posterior_entropy_cost_gap[offset:stop] = entropy_cost_gap
        normalized_posterior_entropy_cost_gap[offset:stop] = norm_entropy_cost_gap
        posterior_entropy_fixed[offset:stop] = entropy_fixed
        normalized_posterior_entropy_fixed[offset:stop] = norm_entropy_fixed
        if needs_fisher_chart_features:
            local_features = _local_posterior_features(
                aligned,
                posterior,
                radii=local_posterior_radii,
            )
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
        posterior_entropy = posterior_entropy[:offset]
        normalized_posterior_entropy = normalized_posterior_entropy[:offset]
        atom_argmax = atom_argmax[:offset]
        temperature_used = temperature_used[:offset]
        temperature_cost_gap = temperature_cost_gap[:offset]
        temperature_fixed = temperature_fixed[:offset]
        weights = weights[:offset]
        atom_posteriors = atom_posteriors[:offset]
        posterior_entropy_cost_gap = posterior_entropy_cost_gap[:offset]
        normalized_posterior_entropy_cost_gap = normalized_posterior_entropy_cost_gap[
            :offset
        ]
        posterior_entropy_fixed = posterior_entropy_fixed[:offset]
        normalized_posterior_entropy_fixed = normalized_posterior_entropy_fixed[:offset]
        chart_features = chart_features[:offset]

    if mode == SPOT_LATENT_MODE_DIAGNOSTIC_FISHER:
        within_latent_coords = _cluster_local_pca_chart(
            chart_features,
            cluster_labels,
            n_clusters=n_clusters,
        )
        latent_coords = _global_discriminative_latent_chart(
            chart_features,
            cluster_labels,
            np.maximum(weights, 1e-6),
            within_latent_coords,
            n_clusters=n_clusters,
        )
        cluster_anchors, _ = _cluster_centroids(
            latent_coords,
            cluster_labels,
            np.maximum(weights, 1e-6),
            n_clusters=n_clusters,
        )
        atom_embedding = np.zeros((n_clusters, atoms_per_cluster, 2), dtype=np.float32)
        global_within_scale = 0.85
    else:
        (
            latent_coords,
            within_latent_coords,
            cluster_anchors,
            atom_embedding,
            global_within_scale,
            mds_diagnostics,
        ) = _atom_barycentric_mds_chart(
            atom_coords=atom_coords,
            atom_features=atom_features,
            prototype_weights=prototype_weights,
            atom_posteriors=atom_posteriors,
            cluster_labels=cluster_labels,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
        )
        cluster_anchor_distance = np.asarray(
            mds_diagnostics["cluster_anchor_distance"],
            dtype=np.float32,
        )
        atom_mds_stress = np.asarray(
            mds_diagnostics["atom_mds_stress"], dtype=np.float32
        )
        atom_mds_positive_mass = np.asarray(
            mds_diagnostics["atom_mds_positive_eigenvalue_mass_2d"],
            dtype=np.float32,
        )
        atom_mds_negative_mass = np.asarray(
            mds_diagnostics["atom_mds_negative_eigenvalue_mass_fraction"],
            dtype=np.float32,
        )
        cluster_anchor_distance_method = str(
            mds_diagnostics["cluster_anchor_distance_method"]
        )
        cluster_anchor_distance_requested_method = str(
            mds_diagnostics["cluster_anchor_distance_requested_method"]
        )
        cluster_anchor_distance_effective_method = str(
            mds_diagnostics["cluster_anchor_distance_effective_method"]
        )
        cluster_anchor_ot_fallback_matrix = np.asarray(
            mds_diagnostics["cluster_anchor_ot_fallback_matrix"],
            dtype=bool,
        )
        cluster_anchor_solver_status_matrix = np.asarray(
            mds_diagnostics["cluster_anchor_solver_status_matrix"],
            dtype=np.int8,
        )
        cluster_anchor_ot_fallback_fraction = float(
            mds_diagnostics["cluster_anchor_ot_fallback_fraction"]
        )
        cluster_mds = mds_diagnostics["cluster_mds"]
        cluster_mds_stress = float(cluster_mds["stress"])
        cluster_mds_positive_mass = float(cluster_mds["positive_eigenvalue_mass_2d"])
        cluster_mds_negative_mass = float(
            cluster_mds["negative_eigenvalue_mass_fraction"]
        )

    cell_latent_num = np.zeros((n_cells, n_clusters, 2), dtype=np.float32)
    cell_latent_den = np.zeros((n_cells, n_clusters), dtype=np.float32)
    cell_latent_num_unweighted = np.zeros((n_cells, n_clusters, 2), dtype=np.float32)
    cell_latent_den_unweighted = np.zeros((n_cells, n_clusters), dtype=np.float32)
    cell_entropy_num = np.zeros((n_cells, n_clusters), dtype=np.float32)
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
        np.add.at(
            cell_latent_num_unweighted[:, cluster_id, 0],
            members,
            latent_coords[mask, 0],
        )
        np.add.at(
            cell_latent_num_unweighted[:, cluster_id, 1],
            members,
            latent_coords[mask, 1],
        )
        np.add.at(cell_latent_den_unweighted[:, cluster_id], members, 1.0)
        np.add.at(
            cell_entropy_num,
            (members, np.full(members.shape[0], cluster_id, dtype=np.int64)),
            occurrence_weight * posterior_entropy[mask],
        )

    cell_spot_labels = cell_latent_den.argmax(axis=1).astype(np.int32)
    row_idx = np.arange(n_cells, dtype=np.int64)
    cell_spot_weights = cell_latent_den[
        row_idx, cell_spot_labels.astype(np.int64)
    ].astype(np.float32)
    cell_spot_coords = np.full((n_cells, 2), np.nan, dtype=np.float32)
    cell_spot_unweighted_coords = np.full((n_cells, 2), np.nan, dtype=np.float32)
    cell_spot_confidence_weighted_coords = np.full(
        (n_cells, 2), np.nan, dtype=np.float32
    )
    cell_spot_entropy = np.full(n_cells, np.nan, dtype=np.float32)
    covered = cell_spot_weights > 0
    if np.any(covered):
        covered_idx = np.flatnonzero(covered)
        covered_labels = cell_spot_labels[covered_idx].astype(np.int64)
        cell_spot_coords[covered_idx] = (
            cell_latent_num[covered_idx, covered_labels]
            / np.maximum(cell_spot_weights[covered_idx, None], 1e-8)
        ).astype(np.float32)
        cell_spot_confidence_weighted_coords[covered_idx] = cell_spot_coords[
            covered_idx
        ]
        unweighted_den = cell_latent_den_unweighted[covered_idx, covered_labels]
        cell_spot_unweighted_coords[covered_idx] = (
            cell_latent_num_unweighted[covered_idx, covered_labels]
            / np.maximum(unweighted_den[:, None], 1e-8)
        ).astype(np.float32)
        cell_spot_coords[covered_idx] = cell_spot_unweighted_coords[covered_idx]
        cell_spot_entropy[covered_idx] = (
            cell_entropy_num[covered_idx, covered_labels]
            / np.maximum(cell_spot_weights[covered_idx], 1e-8)
        ).astype(np.float32)
    cell_spot_labels[~covered] = -1

    return {
        "spot_latent_cell_indices": cell_indices.astype(np.int32),
        "spot_latent_subregion_ids": subregion_ids.astype(np.int32),
        "spot_latent_cluster_labels": cluster_labels.astype(np.int32),
        "spot_latent_coords": latent_coords.astype(np.float32),
        "spot_latent_within_coords": within_latent_coords.astype(np.float32),
        "spot_latent_cluster_anchors": cluster_anchors.astype(np.float32),
        "spot_latent_atom_embedding": atom_embedding.astype(np.float32),
        "spot_latent_aligned_coords": aligned_coords.astype(np.float32),
        "spot_latent_cluster_probs": cluster_probs.astype(np.float32),
        "spot_latent_atom_confidence": atom_confidence.astype(np.float32),
        "spot_latent_posterior_entropy": posterior_entropy.astype(np.float32),
        "spot_latent_normalized_posterior_entropy": normalized_posterior_entropy.astype(
            np.float32
        ),
        "spot_latent_atom_argmax": atom_argmax.astype(np.int32),
        "spot_latent_temperature_used": temperature_used.astype(np.float32),
        "spot_latent_temperature_cost_gap": temperature_cost_gap.astype(np.float32),
        "spot_latent_temperature_fixed": temperature_fixed.astype(np.float32),
        "spot_latent_weights": weights.astype(np.float32),
        "spot_latent_atom_posteriors": atom_posteriors.astype(np.float32),
        "spot_latent_posterior_entropy_cost_gap": posterior_entropy_cost_gap.astype(
            np.float32
        ),
        "spot_latent_normalized_posterior_entropy_cost_gap": normalized_posterior_entropy_cost_gap.astype(
            np.float32
        ),
        "spot_latent_posterior_entropy_fixed": posterior_entropy_fixed.astype(
            np.float32
        ),
        "spot_latent_normalized_posterior_entropy_fixed": normalized_posterior_entropy_fixed.astype(
            np.float32
        ),
        "spot_latent_cluster_anchor_distance": cluster_anchor_distance.astype(
            np.float32
        ),
        "spot_latent_cluster_anchor_ot_fallback_matrix": cluster_anchor_ot_fallback_matrix.astype(
            bool
        ),
        "spot_latent_cluster_anchor_solver_status_matrix": cluster_anchor_solver_status_matrix.astype(
            np.int8
        ),
        "spot_latent_cluster_anchor_ot_fallback_fraction": np.array(
            float(cluster_anchor_ot_fallback_fraction),
            dtype=np.float32,
        ),
        "spot_latent_atom_mds_stress": atom_mds_stress.astype(np.float32),
        "spot_latent_atom_mds_positive_eigenvalue_mass_2d": atom_mds_positive_mass.astype(
            np.float32
        ),
        "spot_latent_atom_mds_negative_eigenvalue_mass_fraction": atom_mds_negative_mass.astype(
            np.float32
        ),
        "cell_spot_latent_unweighted_coords": cell_spot_unweighted_coords.astype(
            np.float32
        ),
        "cell_spot_latent_confidence_weighted_coords": cell_spot_confidence_weighted_coords.astype(
            np.float32
        ),
        "cell_spot_latent_coords": cell_spot_coords.astype(np.float32),
        "cell_spot_latent_cluster_labels": cell_spot_labels.astype(np.int32),
        "cell_spot_latent_weights": cell_spot_weights.astype(np.float32),
        "cell_spot_latent_posterior_entropy": cell_spot_entropy.astype(np.float32),
        "spot_latent_mode": np.array(str(metadata["mode"])),
        "spot_latent_chart_learning_mode": np.array(
            str(metadata["chart_learning_mode"])
        ),
        "spot_latent_projection_mode": np.array(
            str(metadata["latent_projection_mode"])
        ),
        "spot_latent_validation_role": np.array(str(metadata["validation_role"])),
        "spot_latent_global_within_scale": np.array(
            float(global_within_scale), dtype=np.float32
        ),
        "spot_latent_assignment_temperature": np.array(
            float(np.median(temperature_used))
            if temperature_used.size
            else float(assignment_temperature),
            dtype=np.float32,
        ),
        "spot_latent_temperature_mode": np.array(str(temperature_mode)),
        "spot_latent_cluster_anchor_distance_method": np.array(
            str(cluster_anchor_distance_method)
        ),
        "spot_latent_cluster_anchor_distance_requested_method": np.array(
            str(cluster_anchor_distance_requested_method)
        ),
        "spot_latent_cluster_anchor_distance_effective_method": np.array(
            str(cluster_anchor_distance_effective_method)
        ),
        "spot_latent_cluster_mds_stress": np.array(
            float(cluster_mds_stress), dtype=np.float32
        ),
        "spot_latent_cluster_mds_positive_eigenvalue_mass_2d": np.array(
            float(cluster_mds_positive_mass),
            dtype=np.float32,
        ),
        "spot_latent_cluster_mds_negative_eigenvalue_mass_fraction": np.array(
            float(cluster_mds_negative_mass),
            dtype=np.float32,
        ),
    }
