from __future__ import annotations

import numpy as np

from .types import MultilevelOTResult

LEAKAGE_BALANCED_ACCURACY_WARNING = 0.35
LEAKAGE_PERMUTATION_P95_MARGIN_WARNING = 0.02
LEAKAGE_PERMUTATION_MEAN_EXCESS_WARNING = 0.05


def assigned_transport_cost_decomposition(result: MultilevelOTResult) -> dict[str, float]:
    geometry = np.asarray(result.subregion_assigned_geometry_transport_costs, dtype=np.float64)
    feature = np.asarray(result.subregion_assigned_feature_transport_costs, dtype=np.float64)
    transform = np.asarray(result.subregion_assigned_transform_penalties, dtype=np.float64)
    overlap = np.asarray(result.subregion_assigned_overlap_consistency_penalties, dtype=np.float64)
    transport_plus_transform = geometry + feature + transform
    assigned_transport_objective = np.asarray(
        result.subregion_cluster_transport_costs[
            np.arange(result.subregion_cluster_labels.shape[0], dtype=np.int64),
            result.subregion_cluster_labels.astype(np.int64),
        ],
        dtype=np.float64,
    )
    assigned_total_objective = np.asarray(
        result.subregion_cluster_costs[
            np.arange(result.subregion_cluster_labels.shape[0], dtype=np.int64),
            result.subregion_cluster_labels.astype(np.int64),
        ],
        dtype=np.float64,
    )
    transport_sum = float(np.sum(transport_plus_transform))
    transport_objective_sum = float(np.sum(assigned_transport_objective))
    total_objective_sum = float(np.sum(assigned_total_objective))
    transport_denom = max(transport_sum, 1e-12)
    transport_objective_denom = max(transport_objective_sum, 1e-12)
    total_objective_denom = max(total_objective_sum, 1e-12)
    return {
        "mean_geometry_transport_cost": float(np.mean(geometry)) if geometry.size else 0.0,
        "mean_feature_transport_cost": float(np.mean(feature)) if feature.size else 0.0,
        "mean_transform_penalty": float(np.mean(transform)) if transform.size else 0.0,
        "mean_overlap_consistency_penalty": float(np.mean(overlap)) if overlap.size else 0.0,
        "mean_transport_plus_transform_cost": float(np.mean(transport_plus_transform)) if transport_plus_transform.size else 0.0,
        "mean_regularized_objective": float(np.mean(assigned_transport_objective)) if assigned_transport_objective.size else 0.0,
        "mean_transport_assignment_objective": float(np.mean(assigned_transport_objective)) if assigned_transport_objective.size else 0.0,
        "mean_total_assignment_cost": float(np.mean(assigned_total_objective)) if assigned_total_objective.size else 0.0,
        "mean_ot_regularization_gap": float(np.mean(assigned_transport_objective - transport_plus_transform)) if assigned_transport_objective.size else 0.0,
        "geometry_transport_fraction": float(np.sum(geometry) / transport_denom),
        "feature_transport_fraction": float(np.sum(feature) / transport_denom),
        "transform_penalty_fraction": float(np.sum(transform) / transport_denom),
        "overlap_consistency_fraction_of_total": float(np.sum(overlap) / total_objective_denom),
        "ot_regularization_gap_fraction_of_transport_objective": float(
            np.sum(assigned_transport_objective - transport_plus_transform) / transport_objective_denom
        ),
    }


def cost_reliability_metrics(result: MultilevelOTResult) -> dict[str, object]:
    effective_eps = np.asarray(result.subregion_candidate_effective_eps_matrix, dtype=np.float64)
    used_fallback = np.asarray(result.subregion_candidate_used_ot_fallback_matrix, dtype=bool)
    sorted_costs = np.sort(np.asarray(result.subregion_cluster_costs, dtype=np.float64), axis=1)
    margins = (
        sorted_costs[:, 1] - sorted_costs[:, 0]
        if sorted_costs.shape[1] >= 2
        else np.full(sorted_costs.shape[0], np.nan, dtype=np.float64)
    )
    finite_margins = margins[np.isfinite(margins)]
    mixed_eps = np.mean((np.max(effective_eps, axis=1) - np.min(effective_eps, axis=1)) > 1e-8) if effective_eps.size else 0.0
    mixed_fallback = np.mean(np.any(used_fallback != used_fallback[:, :1], axis=1)) if used_fallback.size else 0.0
    return {
        "effective_eps_matrix_available": True,
        "fallback_fraction_all_costs": float(np.mean(used_fallback.astype(np.float32))) if used_fallback.size else 0.0,
        "fallback_fraction_assigned": float(np.mean(result.subregion_assigned_used_ot_fallback.astype(np.float32))),
        "mixed_candidate_effective_eps_fraction": float(mixed_eps),
        "mixed_candidate_fallback_fraction": float(mixed_fallback),
        "effective_eps_min": float(np.min(effective_eps)) if effective_eps.size else None,
        "effective_eps_max": float(np.max(effective_eps)) if effective_eps.size else None,
        "assignment_margin_mean": float(np.mean(finite_margins)) if finite_margins.size else None,
        "assignment_margin_median": float(np.median(finite_margins)) if finite_margins.size else None,
        "assignment_margin_p10": float(np.quantile(finite_margins, 0.10)) if finite_margins.size else None,
        "fallback_fraction_by_cluster": {
            f"C{int(cid)}": float(np.mean(used_fallback[:, int(cid)].astype(np.float32)))
            for cid in range(used_fallback.shape[1])
        }
        if used_fallback.ndim == 2
        else {},
    }


def transform_diagnostics(result: MultilevelOTResult) -> dict[str, float | None]:
    rotation = np.asarray(result.subregion_assigned_transform_rotation_deg, dtype=np.float64)
    reflection = np.asarray(result.subregion_assigned_transform_reflection, dtype=bool)
    scale = np.asarray(result.subregion_assigned_transform_scale, dtype=np.float64)
    translation_norm = np.asarray(result.subregion_assigned_transform_translation_norm, dtype=np.float64)
    if rotation.size == 0:
        return {
            "mean_abs_rotation_deg": None,
            "p95_abs_rotation_deg": None,
            "reflection_fraction": None,
            "scale_mean": None,
            "scale_deviation_mean": None,
            "scale_deviation_p95": None,
            "translation_norm_mean": None,
            "translation_norm_p95": None,
        }
    abs_rotation = np.abs(rotation)
    scale_dev = np.abs(scale - 1.0)
    return {
        "mean_abs_rotation_deg": float(np.mean(abs_rotation)),
        "p95_abs_rotation_deg": float(np.quantile(abs_rotation, 0.95)),
        "reflection_fraction": float(np.mean(reflection.astype(np.float32))),
        "scale_mean": float(np.mean(scale)),
        "scale_deviation_mean": float(np.mean(scale_dev)),
        "scale_deviation_p95": float(np.quantile(scale_dev, 0.95)),
        "translation_norm_mean": float(np.mean(translation_norm)),
        "translation_norm_p95": float(np.quantile(translation_norm, 0.95)),
    }


def _qc_warning(code: str, severity: str, message: str, *, value: float | int | str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "code": str(code),
        "severity": str(severity),
        "message": str(message),
    }
    if value is not None:
        payload["value"] = value
    return payload


def _finite_metric(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _leakage_qc_warning(
    *,
    diagnostics: dict | None,
    code: str,
    descriptor_label: str,
) -> dict[str, object] | None:
    if not diagnostics:
        return None
    observed = _finite_metric(diagnostics.get("balanced_accuracy"))
    if observed is None:
        return None
    permutation = diagnostics.get("permutation")
    if not isinstance(permutation, dict):
        permutation = {}
    perm_p95 = _finite_metric(permutation.get("perm_p95"))
    excess = _finite_metric(permutation.get("excess"))
    above_null = perm_p95 is not None and observed > perm_p95 + LEAKAGE_PERMUTATION_P95_MARGIN_WARNING
    high_absolute = observed >= LEAKAGE_BALANCED_ACCURACY_WARNING
    high_excess = excess is not None and excess >= LEAKAGE_PERMUTATION_MEAN_EXCESS_WARNING
    if not (above_null or high_absolute or high_excess):
        return None
    return _qc_warning(
        code,
        "warning",
        f"Subregion cluster labels are predictable from {descriptor_label}; treat biological niche interpretation as leakage-sensitive until null/ablation checks pass.",
        value=float(observed),
    )


def build_qc_warnings(
    *,
    feature_embedding_warning: str | None,
    fallback_fraction: float,
    assigned_ot_fallback_fraction: float,
    assigned_effective_eps_values: list[float],
    requested_ot_eps: float,
    coverage_fraction: float,
    mean_assignment_margin: float | None,
    assigned_transport_cost_decomposition: dict[str, float],
    cost_reliability: dict[str, object],
    transform_diagnostics: dict[str, float | None],
    forced_label_fraction: float,
    deep_summary: dict,
    shape_leakage_diagnostics: dict | None = None,
    density_leakage_diagnostics: dict | None = None,
    subregion_construction: dict | None = None,
    auto_k_enabled: bool = False,
) -> list[dict[str, object]]:
    warnings_out: list[dict[str, object]] = [
        _qc_warning(
            "cell_projection_is_approximate_assigned_subregion",
            "info",
            "Auxiliary cell-level projection scores are approximate; primary cell labels are inherited from fitted mutually exclusive subregions.",
        )
    ]
    if subregion_construction and not bool(subregion_construction.get("radius_used_for_membership", True)):
        warnings_out.append(
            _qc_warning(
                "subregion_radius_not_membership_radius",
                "info",
                "Generated subregion membership is a data-driven mutually exclusive partition; radius_um is not a fixed neighborhood membership radius.",
            )
        )
    if bool(auto_k_enabled):
        warnings_out.append(
            _qc_warning(
                "auto_k_is_exploratory",
                "info",
                "Automatic K selection is an exploratory shortlist/final-refit convenience and should be confirmed with stability and ablation analysis.",
            )
        )
    if feature_embedding_warning == "umap_exploratory":
        warnings_out.append(
            _qc_warning(
                "umap_feature_space_exploratory",
                "warning",
                "UMAP was used as the OT feature space, so the run should be treated as exploratory.",
            )
        )
    elif feature_embedding_warning == "visualization_embedding_like":
        warnings_out.append(
            _qc_warning(
                "visualization_like_feature_space",
                "warning",
                "A visualization-like embedding was used as the OT feature space, so the run should be treated as exploratory.",
            )
        )
    if fallback_fraction > 0.0:
        warnings_out.append(
            _qc_warning(
                "observed_hull_geometry_fallback_active",
                "warning",
                "Observed-coordinate convex-hull fallback was used for at least one subregion, so boundary-shape invariance is not fully supported.",
                value=float(fallback_fraction),
            )
        )
    if assigned_ot_fallback_fraction > 0.0:
        warnings_out.append(
            _qc_warning(
                "assigned_ot_regularization_fallback_active",
                "warning",
                "At least one assigned OT solve needed a larger effective epsilon than requested.",
                value=float(assigned_ot_fallback_fraction),
            )
        )
    if assigned_effective_eps_values:
        max_effective_eps = float(max(assigned_effective_eps_values))
        if max_effective_eps > float(requested_ot_eps) * 1.5:
            warnings_out.append(
                _qc_warning(
                    "effective_eps_exceeds_requested",
                    "warning",
                    "The effective OT epsilon increased substantially above the requested value during fallback.",
                    value=max_effective_eps,
                )
            )
    if coverage_fraction < 1.0:
        warnings_out.append(
            _qc_warning(
                "incomplete_cell_subregion_coverage",
                "warning",
                "Some cells were not covered by any retained subregion.",
                value=float(coverage_fraction),
            )
        )
    if mean_assignment_margin is not None and np.isfinite(mean_assignment_margin) and mean_assignment_margin < 0.05:
        warnings_out.append(
            _qc_warning(
                "low_mean_assignment_margin",
                "warning",
                "Mean subregion assignment margin is low, so niche assignments may be unstable or weakly separated.",
                value=float(mean_assignment_margin),
            )
        )
    if float(assigned_transport_cost_decomposition.get("geometry_transport_fraction", 0.0)) >= 0.75:
        warnings_out.append(
            _qc_warning(
                "geometry_dominates_assigned_cost",
                "warning",
                "Most of the assigned transport cost comes from geometry rather than feature matching.",
                value=float(assigned_transport_cost_decomposition["geometry_transport_fraction"]),
            )
        )
    shape_warning = _leakage_qc_warning(
        diagnostics=shape_leakage_diagnostics,
        code="shape_descriptors_predict_subregion_clusters",
        descriptor_label="shape descriptors",
    )
    if shape_warning is not None:
        warnings_out.append(shape_warning)
    density_warning = _leakage_qc_warning(
        diagnostics=density_leakage_diagnostics,
        code="density_descriptors_predict_subregion_clusters",
        descriptor_label="subregion size/density descriptors",
    )
    if density_warning is not None:
        warnings_out.append(density_warning)
    if forced_label_fraction > 0.0:
        warnings_out.append(
            _qc_warning(
                "forced_subregion_cluster_size_assignment",
                "warning",
                "At least one subregion label was forced to satisfy the requested minimum subregion count per subregion cluster.",
                value=float(forced_label_fraction),
            )
        )
    mixed_candidate_fallback_fraction = float(cost_reliability.get("mixed_candidate_fallback_fraction", 0.0) or 0.0)
    mixed_candidate_effective_eps_fraction = float(cost_reliability.get("mixed_candidate_effective_eps_fraction", 0.0) or 0.0)
    if mixed_candidate_effective_eps_fraction > 0.0:
        warnings_out.append(
            _qc_warning(
                "candidate_costs_use_mixed_effective_eps",
                "warning",
                "Some subregions still compare candidate clusters under different effective OT eps values after stabilization attempts.",
                value=mixed_candidate_effective_eps_fraction,
            )
        )
    if mixed_candidate_fallback_fraction > 0.0:
        warnings_out.append(
            _qc_warning(
                "candidate_costs_use_mixed_fallback",
                "warning",
                "Some subregions compared candidate clusters under different OT fallback states or effective eps values.",
                value=mixed_candidate_fallback_fraction,
            )
        )
    reflection_fraction = transform_diagnostics.get("reflection_fraction")
    if reflection_fraction is not None and float(reflection_fraction) > 0.0:
        warnings_out.append(
            _qc_warning(
                "reflection_used_in_assigned_transforms",
                "info",
                "At least one assigned subregion-to-cluster alignment used a reflected transform.",
                value=float(reflection_fraction),
            )
        )
    scale_deviation_p95 = transform_diagnostics.get("scale_deviation_p95")
    if scale_deviation_p95 is not None and float(scale_deviation_p95) > 0.25:
        warnings_out.append(
            _qc_warning(
                "large_transform_scale_drift",
                "warning",
                "Assigned transforms show substantial scale drift from 1.0 for at least part of the run.",
                value=float(scale_deviation_p95),
            )
        )
    if bool(deep_summary.get("enabled")) and deep_summary.get("output_embedding") == "joint":
        warnings_out.append(
            _qc_warning(
                "joint_embedding_used_for_ot",
                "info",
                "The OT feature view used the deep joint embedding under explicit opt-in.",
            )
        )
    return warnings_out


def probability_diagnostics(probs: np.ndarray, *, prefix: str) -> dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return {
            f"{prefix}_assignment_entropy_mean": 0.0,
            f"{prefix}_assignment_entropy_p95": 0.0,
            f"{prefix}_assignment_confidence_mean": 0.0,
            f"{prefix}_assignment_confidence_p05": 0.0,
        }
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, None)), axis=1)
    confidence = np.max(probs, axis=1)
    return {
        f"{prefix}_assignment_entropy_mean": float(np.mean(entropy)),
        f"{prefix}_assignment_entropy_p95": float(np.quantile(entropy, 0.95)),
        f"{prefix}_assignment_confidence_mean": float(np.mean(confidence)),
        f"{prefix}_assignment_confidence_p05": float(np.quantile(confidence, 0.05)),
    }


def cell_subregion_coverage(n_cells: int, subregion_members: list[np.ndarray]) -> dict[str, object]:
    if n_cells <= 0:
        return {
            "covered_cell_count": 0,
            "uncovered_cell_count": 0,
            "cell_subregion_coverage_fraction": 0.0,
            "cell_subregion_duplicate_count": 0,
            "cell_subregion_duplicate_fraction": 0.0,
            "cell_subregion_max_memberships": 0,
            "cell_subregion_partition_complete": False,
            "subregion_membership_mode": "empty",
        }
    membership_counts = np.zeros(n_cells, dtype=np.int32)
    for members in subregion_members:
        member_arr = np.asarray(members, dtype=np.int64)
        if member_arr.size:
            np.add.at(membership_counts, member_arr, 1)
    covered = membership_counts > 0
    covered_count = int(covered.sum())
    duplicate_count = int(np.sum(membership_counts > 1))
    uncovered_count = int(n_cells - covered_count)
    max_memberships = int(membership_counts.max(initial=0))
    partition_complete = bool(uncovered_count == 0 and duplicate_count == 0)
    if partition_complete:
        membership_mode = "mutually_exclusive_complete"
    elif duplicate_count == 0:
        membership_mode = "mutually_exclusive_partial"
    else:
        membership_mode = "overlapping"
    return {
        "covered_cell_count": covered_count,
        "uncovered_cell_count": uncovered_count,
        "cell_subregion_coverage_fraction": float(covered_count / max(n_cells, 1)),
        "cell_subregion_duplicate_count": duplicate_count,
        "cell_subregion_duplicate_fraction": float(duplicate_count / max(n_cells, 1)),
        "cell_subregion_max_memberships": max_memberships,
        "cell_subregion_partition_complete": partition_complete,
        "subregion_membership_mode": membership_mode,
    }
