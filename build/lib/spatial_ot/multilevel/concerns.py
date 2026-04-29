from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _read_summary(run_dir: str | Path) -> dict[str, object]:
    path = Path(run_dir) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing multilevel OT summary: {path}")
    return json.loads(path.read_text())


def _warning_codes(summary: dict[str, object]) -> set[str]:
    return {
        str(item.get("code"))
        for item in summary.get("qc_warnings", [])
        if isinstance(item, dict) and item.get("code") is not None
    }


def _is_coordinate_only_baseline(summary: dict[str, object] | None) -> bool:
    if summary is None:
        return False
    construction = summary.get("subregion_construction")
    if not isinstance(construction, dict):
        return False
    return bool(construction.get("coordinate_only_baseline", False))


def _selected_k(summary: dict[str, object]) -> int | None:
    auto_k = summary.get("auto_k_selection")
    if isinstance(auto_k, dict) and auto_k.get("selected_k") is not None:
        return int(auto_k["selected_k"])
    if summary.get("n_clusters") is not None:
        return int(summary["n_clusters"])
    return None


def _shell_value(value: object) -> str:
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _env_token(key: str, value: object | None) -> str | None:
    if value is None:
        return None
    return f"{key}={_shell_value(value)}"


def _metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "NA"
    return str(value)


def _deep_model_path(summary: dict[str, object], run_dir: Path) -> Path | None:
    deep_summary = summary.get("deep_features")
    model_path = None
    if isinstance(deep_summary, dict) and deep_summary.get("model_path"):
        model_path = Path(str(deep_summary["model_path"]))
    if model_path is None:
        candidate = Path(str(summary.get("output_dir", run_dir))) / "deep_feature_model.pt"
        model_path = candidate if candidate.exists() else None
    return model_path if model_path is not None and model_path.exists() else None


def _common_run_env(summary: dict[str, object], run_dir: Path, *, omit: set[str] | None = None) -> list[str]:
    omit = omit or set()
    construction = summary.get("subregion_construction")
    construction = construction if isinstance(construction, dict) else {}
    deep_summary = summary.get("deep_features")
    deep_summary = deep_summary if isinstance(deep_summary, dict) else {}
    deep_segmentation = construction.get("deep_segmentation")
    deep_segmentation = deep_segmentation if isinstance(deep_segmentation, dict) else {}
    latent_metadata = summary.get("subregion_latent_embedding_metadata")
    latent_metadata = latent_metadata if isinstance(latent_metadata, dict) else {}
    min_cluster_size = summary.get("effective_min_subregions_per_cluster", summary.get("min_subregions_per_cluster"))
    env_items = [
        ("SAMPLE_OBS_KEY", summary.get("sample_obs_key")),
        ("BASIC_NICHE_SIZE_UM", construction.get("basic_niche_size_um")),
        ("RADIUS_UM", construction.get("radius_um")),
        ("STRIDE_UM", construction.get("stride_um")),
        ("MIN_CELLS", construction.get("min_cells")),
        ("MAX_SUBREGIONS", construction.get("max_subregions")),
        ("MIN_SUBREGIONS_PER_CLUSTER", min_cluster_size),
        ("DEEP_FEATURE_METHOD", deep_summary.get("method") or "none"),
        ("DEEP_OUTPUT_EMBEDDING", deep_summary.get("output_embedding")),
        ("DEEP_OUTPUT_OBSM_KEY", deep_summary.get("output_feature_obsm_key")),
        ("DEEP_SEGMENTATION_KNN", deep_segmentation.get("knn")),
        ("DEEP_SEGMENTATION_FEATURE_DIMS", deep_segmentation.get("feature_dims")),
        ("DEEP_SEGMENTATION_FEATURE_WEIGHT", deep_segmentation.get("feature_weight")),
        ("DEEP_SEGMENTATION_SPATIAL_WEIGHT", deep_segmentation.get("spatial_weight")),
        ("DEEP_SEGMENTATION_REFINEMENT_ITERS", deep_segmentation.get("refinement_iters")),
        ("MAX_SPOT_LATENT_PLOT_OCCURRENCES", 0),
        ("COMPUTE_SPOT_LATENT", 0),
        ("PLOT_SAMPLE_SPOT_LATENT", 0),
        ("SUBREGION_CLUSTERING_METHOD", summary.get("subregion_clustering_method")),
        ("SUBREGION_LATENT_EMBEDDING_MODE", summary.get("subregion_latent_embedding_mode")),
        ("SUBREGION_LATENT_SHRINKAGE_TAU", summary.get("subregion_latent_shrinkage_tau")),
        ("SUBREGION_LATENT_HETEROGENEITY_WEIGHT", latent_metadata.get("heterogeneity_block_weight")),
        ("SUBREGION_LATENT_SAMPLE_PRIOR_WEIGHT", latent_metadata.get("sample_prior_weight")),
        ("SUBREGION_LATENT_CODEBOOK_SIZE", summary.get("subregion_latent_codebook_size")),
        ("SUBREGION_LATENT_CODEBOOK_SAMPLE_SIZE", summary.get("subregion_latent_codebook_sample_size")),
    ]
    tokens = [_env_token(key, value) for key, value in env_items if key not in omit]
    deep_model = _deep_model_path(summary, run_dir)
    if deep_model is not None:
        tokens.append(_env_token("DEEP_PRETRAINED_MODEL", deep_model.as_posix()))
    return [token for token in tokens if token is not None]


def _run_script_for_construction(summary: dict[str, object]) -> str:
    construction = summary.get("subregion_construction")
    if isinstance(construction, dict) and construction.get("construction_method") == "deep_segmentation":
        return "scripts/run_deep_segmentation_cohort_gpu.sh"
    return "scripts/run_prepared_cohort_gpu.sh"


def _baseline_comparison(
    primary: dict[str, object],
    baseline: dict[str, object] | None,
) -> dict[str, object] | None:
    if baseline is None:
        return None
    return {
        "baseline_run_dir": str(baseline.get("output_dir")),
        "baseline_is_coordinate_only": _is_coordinate_only_baseline(baseline),
        "primary_n_subregions": primary.get("n_subregions"),
        "baseline_n_subregions": baseline.get("n_subregions"),
        "primary_n_clusters": primary.get("n_clusters"),
        "baseline_n_clusters": baseline.get("n_clusters"),
        "primary_coverage_fraction": primary.get("cell_subregion_coverage_fraction"),
        "baseline_coverage_fraction": baseline.get("cell_subregion_coverage_fraction"),
        "primary_warning_codes": sorted(_warning_codes(primary)),
        "baseline_warning_codes": sorted(_warning_codes(baseline)),
    }


def _suggest_coordinate_only_command(summary: dict[str, object], run_dir: Path) -> str:
    selected_k = _selected_k(summary) or int(summary.get("n_clusters", 8))
    baseline_dir = f"{run_dir.as_posix()}_coordinate_only_baseline"
    tokens = [
        _env_token("OUTPUT_DIR", baseline_dir),
        _env_token("AUTO_N_CLUSTERS", 0),
        _env_token("N_CLUSTERS", selected_k),
        *_common_run_env(summary, run_dir),
        _env_token("SUBREGION_CONSTRUCTION_METHOD", "data_driven"),
        _env_token("SUBREGION_FEATURE_WEIGHT", 0),
    ]
    return " ".join(token for token in tokens if token is not None) + " bash scripts/run_prepared_cohort_gpu.sh"


def _suggest_stability_commands(summary: dict[str, object], run_dir: Path) -> list[str]:
    selected_k = _selected_k(summary)
    if selected_k is None:
        return []
    candidates = [max(2, selected_k - 1), selected_k, selected_k + 1]
    commands = []
    for k in candidates:
        tokens = [
            _env_token("OUTPUT_DIR", f"{run_dir.as_posix()}_fixed_k{k}_seed1338"),
            _env_token("AUTO_N_CLUSTERS", 0),
            _env_token("N_CLUSTERS", k),
            _env_token("SEED", 1338),
            *_common_run_env(summary, run_dir),
            _env_token(
                "SUBREGION_CONSTRUCTION_METHOD",
                (
                    summary.get("subregion_construction", {}).get("construction_method")
                    if isinstance(summary.get("subregion_construction"), dict)
                    else None
                ),
            ),
        ]
        commands.append(" ".join(token for token in tokens if token is not None) + f" bash {_run_script_for_construction(summary)}")
    return commands


def _suggest_leakage_ablation_commands(summary: dict[str, object], run_dir: Path, kind: str) -> list[str]:
    selected_k = _selected_k(summary)
    construction = summary.get("subregion_construction")
    construction = construction if isinstance(construction, dict) else {}
    base_scale = construction.get("basic_niche_size_um")
    if selected_k is None or base_scale is None:
        return []
    try:
        base_scale_f = float(base_scale)
    except (TypeError, ValueError):
        return []
    scale_factor = 1.25 if kind == "shape" else 0.75
    scaled = max(5.0, base_scale_f * scale_factor)
    output = f"{run_dir.as_posix()}_{kind}_leakage_scale{scaled:g}_fixed_k{selected_k}"
    tokens = [
        _env_token("OUTPUT_DIR", output),
        _env_token("AUTO_N_CLUSTERS", 0),
        _env_token("N_CLUSTERS", selected_k),
        _env_token("SEED", int(summary.get("seed", 1337)) + 1),
        *_common_run_env(summary, run_dir, omit={"BASIC_NICHE_SIZE_UM"}),
        _env_token("BASIC_NICHE_SIZE_UM", scaled),
        _env_token(
            "SUBREGION_CONSTRUCTION_METHOD",
            (
                construction.get("construction_method")
                if kind == "shape"
                else "data_driven"
            ),
        ),
        _env_token("SUBREGION_FEATURE_WEIGHT", 0 if kind == "density" else construction.get("partition_feature_weight")),
    ]
    return [" ".join(token for token in tokens if token is not None) + f" bash {_run_script_for_construction(summary)}"]


def _suffix_value(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _suggest_fixed_parameter_commands(
    summary: dict[str, object],
    run_dir: Path,
    *,
    suite_name: str,
    env_name: str,
    values: Iterable[object],
    extra_env: dict[str, object] | None = None,
) -> list[str]:
    selected_k = _selected_k(summary)
    if selected_k is None:
        return []
    construction = summary.get("subregion_construction")
    construction = construction if isinstance(construction, dict) else {}
    extra_env = extra_env or {}
    commands = []
    omit = {env_name, *extra_env.keys()}
    for value in values:
        tokens = [
            _env_token("OUTPUT_DIR", f"{run_dir.as_posix()}_{suite_name}_{_suffix_value(value)}_fixed_k{selected_k}"),
            _env_token("AUTO_N_CLUSTERS", 0),
            _env_token("N_CLUSTERS", selected_k),
            _env_token("SEED", int(summary.get("seed", 1337)) + 1),
            *_common_run_env(summary, run_dir, omit=omit),
            _env_token(env_name, value),
            *[_env_token(key, env_value) for key, env_value in extra_env.items()],
            _env_token("SUBREGION_CONSTRUCTION_METHOD", construction.get("construction_method")),
        ]
        commands.append(" ".join(token for token in tokens if token is not None) + f" bash {_run_script_for_construction(summary)}")
    return commands


def _validation_suite_commands(summary: dict[str, object], run_dir: Path) -> dict[str, object]:
    latent_mode = str(summary.get("subregion_latent_embedding_mode") or "mean_std_shrunk")
    selected_k = _selected_k(summary)
    fixed_k_commands = _suggest_stability_commands(summary, run_dir)
    suite = {
        "selected_k": selected_k,
        "purpose": (
            "Executable validation plan for pooled feature-distribution subregion clusters. "
            "Run these as sensitivity/ablation checks before making primary biological niche claims."
        ),
        "fixed_k_stability": fixed_k_commands,
        "shrinkage_tau_sensitivity": _suggest_fixed_parameter_commands(
            summary,
            run_dir,
            suite_name="shrinkage_tau",
            env_name="SUBREGION_LATENT_SHRINKAGE_TAU",
            values=[0, 10, 25, 50, 100],
        ),
        "sample_prior_weight_sensitivity": _suggest_fixed_parameter_commands(
            summary,
            run_dir,
            suite_name="sample_prior",
            env_name="SUBREGION_LATENT_SAMPLE_PRIOR_WEIGHT",
            values=[0, 0.25, 0.5, 0.75, 1.0],
        ),
        "heterogeneity_weight_sensitivity": _suggest_fixed_parameter_commands(
            summary,
            run_dir,
            suite_name="heterogeneity",
            env_name="SUBREGION_LATENT_HETEROGENEITY_WEIGHT",
            values=[0, 0.25, 0.5, 1.0, 2.0],
        ),
        "codebook_size_sensitivity": _suggest_fixed_parameter_commands(
            summary,
            run_dir,
            suite_name="codebook_size",
            env_name="SUBREGION_LATENT_CODEBOOK_SIZE",
            values=[16, 32, 64, 128],
            extra_env={
                "SUBREGION_LATENT_EMBEDDING_MODE": latent_mode
                if "codebook" in latent_mode
                else "mean_std_codebook"
            },
        ),
        "shape_leakage_ablation": _suggest_leakage_ablation_commands(summary, run_dir, "shape"),
        "density_leakage_ablation": _suggest_leakage_ablation_commands(summary, run_dir, "density"),
        "spatial_niche_validation": [
            f"../.venv/bin/python -m spatial_ot spatial-niche-validation --run-dir {_shell_value(run_dir.as_posix())}"
        ],
        "refresh_concern_report": [
            f"../.venv/bin/python -m spatial_ot validate-run-concerns --run-dir {_shell_value(run_dir.as_posix())} --no-strict"
        ],
    }
    suite["command_count"] = int(sum(len(value) for value in suite.values() if isinstance(value, list)))
    return suite


def _leakage_ablation_evidence(
    summaries: list[dict[str, object]],
    kind: str,
) -> list[dict[str, object]]:
    warning_code = f"{kind}_descriptors_predict_subregion_clusters"
    rows = []
    for summary in summaries:
        construction = summary.get("subregion_construction")
        construction = construction if isinstance(construction, dict) else {}
        diagnostics = summary.get(f"{kind}_leakage_diagnostics")
        rows.append(
            {
                "run_dir": str(summary.get("output_dir")),
                "warning_present": warning_code in _warning_codes(summary),
                "n_subregions": summary.get("n_subregions"),
                "n_clusters": summary.get("n_clusters"),
                "coverage_fraction": summary.get("cell_subregion_coverage_fraction"),
                "subregion_cluster_count_min": summary.get("subregion_cluster_count_min"),
                "construction_method": construction.get("construction_method"),
                "coordinate_only_baseline": construction.get("coordinate_only_baseline"),
                "basic_niche_size_um": construction.get("basic_niche_size_um"),
                "diagnostics": diagnostics if isinstance(diagnostics, dict) else None,
            }
        )
    return rows


def _leakage_status(
    summary: dict[str, object],
    kind: str,
    *,
    leakage_ablation_summaries: list[dict[str, object]],
) -> tuple[str, bool, dict[str, object]]:
    diagnostics = summary.get(f"{kind}_leakage_diagnostics")
    thresholds = summary.get("leakage_qc_thresholds")
    codes = _warning_codes(summary)
    warning_code = f"{kind}_descriptors_predict_subregion_clusters"
    if not isinstance(diagnostics, dict):
        return "not_run", False, {"diagnostics": None, "thresholds": thresholds, "ablation_runs": []}
    ablation_runs = _leakage_ablation_evidence(leakage_ablation_summaries, kind)
    if warning_code not in codes:
        return (
            "passed_current_thresholds",
            False,
            {"diagnostics": diagnostics, "thresholds": thresholds, "ablation_runs": ablation_runs},
        )
    if not ablation_runs:
        return (
            "needs_leakage_ablation",
            True,
            {"diagnostics": diagnostics, "thresholds": thresholds, "ablation_runs": ablation_runs},
        )
    warning_flags = [bool(item["warning_present"]) for item in ablation_runs]
    if any(warning_flags):
        return (
            "leakage_persists_after_ablation",
            True,
            {"diagnostics": diagnostics, "thresholds": thresholds, "ablation_runs": ablation_runs},
        )
    return (
        "ablation_runs_passed_current_thresholds",
        False,
        {"diagnostics": diagnostics, "thresholds": thresholds, "ablation_runs": ablation_runs},
    )


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ot_cost_comparability_status(summary: dict[str, object]) -> tuple[str, bool, dict[str, object]]:
    reliability = summary.get("cost_reliability")
    if not isinstance(reliability, dict):
        fallback_fraction = _float_or_zero(summary.get("assigned_ot_fallback_fraction"))
        return (
            "assigned_costs_only",
            fallback_fraction > 0,
            {
                "cost_reliability": None,
                "assigned_ot_fallback_fraction": fallback_fraction,
                "note": "Full candidate effective-epsilon diagnostics were not available in this summary.",
            },
        )
    mixed_eps = _float_or_zero(reliability.get("mixed_candidate_effective_eps_fraction"))
    mixed_fallback = _float_or_zero(reliability.get("mixed_candidate_fallback_fraction"))
    fallback_all = _float_or_zero(reliability.get("fallback_fraction_all_costs"))
    fallback_assigned = _float_or_zero(reliability.get("fallback_fraction_assigned"))
    blocking = mixed_eps > 0 or mixed_fallback > 0
    if blocking:
        status = "mixed_candidate_costs_remaining"
    elif fallback_all > 0 or fallback_assigned > 0:
        status = "common_epsilon_fallback_used"
    else:
        status = "passed_common_epsilon_checks"
    return status, blocking, {"cost_reliability": reliability}


def _spot_latent_visualization_status(summary: dict[str, object]) -> tuple[str, dict[str, object]]:
    spot_latent = summary.get("spot_level_latent")
    if not isinstance(spot_latent, dict) or not bool(spot_latent.get("implemented")):
        return "not_computed", {"spot_level_latent": spot_latent}
    spot_latent_evidence = dict(spot_latent)
    if "chart_learning_mode" not in spot_latent_evidence and "fisher" in str(
        spot_latent_evidence.get("projection", "")
    ).lower():
        spot_latent_evidence["chart_learning_mode"] = "supervised_by_fitted_ot_subregion_labels"
    spot_latent_evidence.setdefault("validation_role", "diagnostic_visualization_not_independent_evidence")
    chart_mode = str(spot_latent_evidence.get("chart_learning_mode", "")).lower()
    projection = str(spot_latent_evidence.get("projection", "")).lower()
    if "supervised" not in chart_mode and "fisher" not in projection:
        return (
            "diagnostic_only_model_grounded_atom_barycentric",
            {
                "spot_level_latent": spot_latent_evidence,
                "required_interpretation": (
                    "Treat OT atom-barycentric spot latent maps as downstream diagnostic visualization. They are "
                    "less label-supervised than the Fisher diagnostic chart, but still are not independent biological "
                    "validation without stability, leakage, and held-out-sample checks."
                ),
            },
        )
    return (
        "diagnostic_only_supervised_by_fitted_ot_labels",
        {
            "spot_level_latent": spot_latent_evidence,
            "required_interpretation": (
                "Treat Fisher/discriminative spot latent separation as label-conditional visualization, not "
                "independent evidence of biological niche separation."
            ),
        },
    )


def _summary_median(summary: dict[str, object], key: str) -> float | None:
    value = summary.get(key)
    if not isinstance(value, dict):
        return None
    median = value.get("median")
    try:
        out = float(median)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _summary_max(summary: dict[str, object], key: str) -> float | None:
    value = summary.get(key)
    if not isinstance(value, dict):
        return None
    max_value = value.get("max")
    try:
        out = float(max_value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _float_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _within_niche_latent_claim_status(summary: dict[str, object]) -> tuple[str, bool, dict[str, object]]:
    spot = summary.get("spot_level_latent")
    if not isinstance(spot, dict) or not bool(spot.get("implemented")):
        return "spot_latent_not_computed", True, {"spot_level_latent": spot}
    blockers: list[str] = []
    feature_source = summary.get("feature_source")
    feature_source = feature_source if isinstance(feature_source, dict) else {}
    feature_warning = summary.get("feature_embedding_warning") or feature_source.get("feature_embedding_warning")
    feature_kind = feature_source.get("feature_space_kind")
    if feature_warning == "umap_exploratory" or feature_kind == "umap_embedding":
        blockers.append("umap_feature_space")
    anchor_method = str(spot.get("cluster_anchor_distance_method", "")).lower()
    if anchor_method not in {"balanced_ot", "sinkhorn_ot"}:
        blockers.append("non_ot_cluster_anchor_distance")
    anchor_effective = str(spot.get("cluster_anchor_distance_effective_method", anchor_method)).lower()
    anchor_fallback_fraction = _float_or_none(spot.get("cluster_anchor_ot_fallback_fraction")) or 0.0
    if anchor_fallback_fraction > 0.0 or "fallback" in anchor_effective:
        blockers.append("anchor_ot_fallback_used")
    stress = spot.get("cluster_anchor_mds_stress")
    try:
        stress_value = float(stress)
    except (TypeError, ValueError):
        stress_value = float("nan")
    if not (stress_value == stress_value):
        blockers.append("missing_cluster_anchor_mds_stress")
    elif stress_value > 0.20:
        blockers.append("high_cluster_anchor_mds_stress")
    positive_mass = _float_or_none(spot.get("cluster_anchor_mds_positive_eigenvalue_mass_2d"))
    if positive_mass is not None and positive_mass < 0.70:
        blockers.append("low_cluster_anchor_mds_positive_eigenvalue_mass")
    negative_mass = _float_or_none(spot.get("cluster_anchor_mds_negative_eigenvalue_mass_fraction"))
    if negative_mass is not None and negative_mass > 0.10:
        blockers.append("non_euclidean_cluster_anchor_distance_geometry")
    atom_stress_max = _summary_max(spot, "atom_mds_stress_summary")
    if atom_stress_max is not None and atom_stress_max > 0.25:
        blockers.append("high_atom_mds_stress")
    entropy_median = _summary_median(spot, "normalized_posterior_entropy_summary")
    if entropy_median is None:
        blockers.append("missing_posterior_entropy_summary")
    elif entropy_median < 0.25 or entropy_median > 0.65:
        blockers.append("posterior_entropy_outside_target_range")
    temperature_ratio = _float_or_none(spot.get("temperature_used_q95_q05_ratio"))
    if temperature_ratio is not None and temperature_ratio > 20.0:
        blockers.append("temperature_calibration_inconsistent_cost_scale")
    if not bool(spot.get("held_out_sample_projection_available", False)):
        blockers.append("missing_held_out_sample_projection")
    if not bool(spot.get("unsupervised_baseline_available", False)):
        blockers.append("missing_unsupervised_latent_baseline")
    if not bool(spot.get("marker_axis_interpretation_available", False)):
        blockers.append("missing_marker_or_cell_type_axis_interpretation")
    status = "passed_within_niche_latent_claim_checks" if not blockers else "blocked_for_within_niche_latent_claim"
    return (
        status,
        bool(blockers),
        {
            "spot_level_latent": spot,
            "blockers": blockers,
            "feature_embedding_warning": feature_warning,
            "feature_space_kind": feature_kind,
            "recommended_entropy_median_range": [0.25, 0.65],
            "cluster_anchor_mds_stress_threshold": 0.20,
            "cluster_anchor_mds_strong_threshold": 0.10,
            "atom_mds_stress_threshold": 0.25,
            "cluster_anchor_positive_eigenvalue_mass_2d_warning": 0.70,
            "cluster_anchor_negative_eigenvalue_mass_fraction_warning": 0.10,
            "anchor_ot_fallback_fraction": anchor_fallback_fraction,
            "cluster_anchor_distance_effective_method": anchor_effective,
            "temperature_used_q95_q05_ratio": temperature_ratio,
        },
    )


def build_concern_resolution_report(
    run_dir: str | Path,
    *,
    coordinate_baseline_run_dir: str | Path | None = None,
    stability_run_dirs: Iterable[str | Path] = (),
    leakage_ablation_run_dirs: Iterable[str | Path] = (),
) -> dict[str, object]:
    run_path = Path(run_dir)
    primary = _read_summary(run_path)
    baseline = _read_summary(coordinate_baseline_run_dir) if coordinate_baseline_run_dir is not None else None
    warning_codes = _warning_codes(primary)
    primary_is_coordinate_only = _is_coordinate_only_baseline(primary)
    baseline_ok = _is_coordinate_only_baseline(baseline)
    baseline_comparison = _baseline_comparison(primary, baseline)
    stability_dirs = [Path(item) for item in stability_run_dirs]
    stability_summaries = [_read_summary(path) for path in stability_dirs]
    leakage_ablation_dirs = [Path(item) for item in leakage_ablation_run_dirs]
    leakage_ablation_summaries = [_read_summary(path) for path in leakage_ablation_dirs]
    selected_k = _selected_k(primary)

    concerns: list[dict[str, object]] = []
    coordinate_boundary_ok = primary_is_coordinate_only or baseline_ok
    concerns.append(
        {
            "code": "feature_boundary_circularity",
            "status": (
                "not_applicable_primary_is_coordinate_only"
                if primary_is_coordinate_only
                else "addressed_by_coordinate_only_baseline"
                if baseline_ok
                else "needs_coordinate_only_baseline"
            ),
            "blocking_for_primary_claim": not coordinate_boundary_ok,
            "evidence": {
                "warning_present": "feature_aware_boundary_circularity_risk" in warning_codes,
                "subregion_construction": primary.get("subregion_construction"),
                "baseline_comparison": baseline_comparison,
            },
            "required_fix": (
                "Run a coordinate-only boundary baseline using the same OT feature view and selected K, then compare "
                "cluster enrichments, sample composition, subregion statistics, and biological conclusions."
            ),
            "suggested_commands": [_suggest_coordinate_only_command(primary, run_path)] if not coordinate_boundary_ok else [],
        }
    )
    concerns.append(
        {
            "code": "coordinate_only_boundary_baseline",
            "status": (
                "not_required_primary_is_coordinate_only"
                if primary_is_coordinate_only
                else "available"
                if baseline_ok
                else "missing"
            ),
            "blocking_for_primary_claim": not coordinate_boundary_ok,
            "evidence": baseline_comparison,
            "required_fix": "Treat deep/feature-aware segmentation as exploratory until a coordinate-only boundary baseline is available.",
        }
    )

    leakage_statuses: dict[str, str] = {}
    for kind in ("shape", "density"):
        status, blocking, evidence = _leakage_status(
            primary,
            kind,
            leakage_ablation_summaries=leakage_ablation_summaries,
        )
        leakage_statuses[kind] = status
        concerns.append(
            {
                "code": f"{kind}_leakage",
                "status": status,
                "blocking_for_primary_claim": blocking,
                "evidence": evidence,
                "required_fix": (
                    f"Run {kind}-aware ablations/nulls and require the biological conclusion to be stable when "
                    f"{kind} descriptors are controlled, permuted, or used as a negative-control predictor."
                ),
                "suggested_commands": (
                    _suggest_leakage_ablation_commands(primary, run_path, kind)
                    if status == "needs_leakage_ablation"
                    else []
                ),
            }
        )
    latent_mode = str(primary.get("subregion_latent_embedding_mode") or "")
    if latent_mode == "mean_std_skew_count":
        density_ok = leakage_statuses.get("density") in {
            "passed_current_thresholds",
            "ablation_runs_passed_current_thresholds",
        }
        concerns.append(
            {
                "code": "density_aware_subregion_latent_mode",
                "status": "accepted_density_controls_passed" if density_ok else "needs_density_leakage_controls",
                "blocking_for_primary_claim": not density_ok,
                "evidence": {
                    "subregion_latent_embedding_mode": latent_mode,
                    "density_leakage_status": leakage_statuses.get("density"),
                    "warning_present": "density_aware_subregion_latent_mode" in warning_codes,
                },
                "required_fix": (
                    "The mean_std_skew_count latent directly includes cell-count-derived features. "
                    "Use it for diagnostics unless density leakage is insignificant or controlled by ablations."
                ),
            }
        )

    cost_status, cost_blocking, cost_evidence = _ot_cost_comparability_status(primary)
    concerns.append(
        {
            "code": "ot_cost_comparability",
            "status": cost_status,
            "blocking_for_primary_claim": cost_blocking,
            "evidence": cost_evidence,
            "required_fix": (
                "Use strict common-effective-epsilon candidate cost comparison. Mixed candidate effective epsilon "
                "or fallback states after stabilization should block primary claims."
            ),
        }
    )

    spot_status, spot_evidence = _spot_latent_visualization_status(primary)
    latent_claim_status, latent_claim_blocking, latent_claim_evidence = _within_niche_latent_claim_status(primary)
    spot_concern_code = (
        "spot_latent_supervised_visualization"
        if "supervised" in spot_status
        else "spot_latent_visualization_validation"
    )
    concerns.append(
        {
            "code": spot_concern_code,
            "status": spot_status,
            "blocking_for_primary_claim": False,
            "blocking_for_within_niche_latent_claim": latent_claim_blocking,
            "evidence": spot_evidence,
            "required_fix": (
                "Pair spot-latent maps with unsupervised baselines, posterior-entropy/atom-argmax diagnostics, "
                "and held-out-sample controls before using visualization as support for biological interpretation."
            ),
        }
    )
    concerns.append(
        {
            "code": "within_niche_latent_heterogeneity_claim",
            "status": latent_claim_status,
            "blocking_for_primary_claim": False,
            "blocking_for_within_niche_latent_claim": latent_claim_blocking,
            "evidence": latent_claim_evidence,
            "required_fix": (
                "For within-niche latent biology, require non-UMAP features, OT-based cluster anchors, acceptable MDS "
                "stress, calibrated posterior entropy, an unsupervised latent baseline, held-out-sample projection, "
                "and marker/cell-type interpretation of latent axes."
            ),
        }
    )

    fixed_k_runs = [
        {
            "run_dir": str(path),
            "n_clusters": summary.get("n_clusters"),
            "seed": summary.get("seed"),
            "auto_n_clusters": summary.get("auto_n_clusters"),
            "qc_warning_count": summary.get("qc_warning_count"),
        }
        for path, summary in zip(stability_dirs, stability_summaries, strict=False)
    ]
    has_fixed_selected_k = any(
        item.get("n_clusters") == selected_k and item.get("auto_n_clusters") is False
        for item in fixed_k_runs
    )
    concerns.append(
        {
            "code": "auto_k_exploratory",
            "status": "fixed_k_stability_runs_available" if has_fixed_selected_k else "needs_fixed_k_stability",
            "blocking_for_primary_claim": not has_fixed_selected_k,
            "evidence": {
                "selected_k": selected_k,
                "criterion_votes": (
                    primary.get("auto_k_selection", {}).get("criterion_votes")
                    if isinstance(primary.get("auto_k_selection"), dict)
                    else None
                ),
                "fixed_k_runs": fixed_k_runs,
            },
            "required_fix": (
                "Use auto-K as a shortlist only. Refit fixed K values around the selected K across seeds, then compare "
                "co-assignment stability, cluster-size repairs, leakage diagnostics, and biological enrichments."
            ),
            "suggested_commands": _suggest_stability_commands(primary, run_path) if not has_fixed_selected_k else [],
        }
    )

    blocking = [item["code"] for item in concerns if item.get("blocking_for_primary_claim")]
    latent_blocking = [item["code"] for item in concerns if item.get("blocking_for_within_niche_latent_claim")]
    report = {
        "run_dir": str(run_path),
        "summary_json": str(run_path / "summary.json"),
        "overall_status": "needs_validation" if blocking else "validation_concerns_addressed",
        "strict_validation_passed": not blocking,
        "primary_claim_status": "blocked_by_validation_concerns" if blocking else "ready_after_current_validation",
        "claim_validation": {
            "subregion_niche_clustering": {
                "blocking_concerns": blocking,
                "status": "blocked" if blocking else "ready_after_current_validation",
            },
            "within_niche_latent_heterogeneity": {
                "blocking_concerns": latent_blocking,
                "status": "blocked" if latent_blocking else "ready_after_current_validation",
            },
        },
        "blocking_concerns": blocking,
        "within_niche_latent_blocking_concerns": latent_blocking,
        "n_cells": primary.get("n_cells"),
        "n_subregions": primary.get("n_subregions"),
        "n_clusters": primary.get("n_clusters"),
        "qc_warning_codes": sorted(warning_codes),
        "validation_suite": _validation_suite_commands(primary, run_path),
        "concerns": concerns,
    }
    return report


def write_concern_resolution_report(
    run_dir: str | Path,
    *,
    coordinate_baseline_run_dir: str | Path | None = None,
    stability_run_dirs: Iterable[str | Path] = (),
    leakage_ablation_run_dirs: Iterable[str | Path] = (),
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
) -> dict[str, object]:
    report = build_concern_resolution_report(
        run_dir,
        coordinate_baseline_run_dir=coordinate_baseline_run_dir,
        stability_run_dirs=stability_run_dirs,
        leakage_ablation_run_dirs=leakage_ablation_run_dirs,
    )
    run_path = Path(run_dir)
    json_path = Path(output_json) if output_json is not None else run_path / "concern_resolution_report.json"
    md_path = Path(output_md) if output_md is not None else run_path / "concern_resolution_report.md"
    json_path.write_text(json.dumps(report, indent=2))
    lines = [
        "# Concern Resolution Report",
        "",
        f"- Run: `{report['run_dir']}`",
        f"- Overall status: `{report['overall_status']}`",
        f"- Blocking concerns: `{', '.join(report['blocking_concerns']) or 'none'}`",
        "",
        "## Validation Suite",
        "",
        (
            "These commands are generated from the finished run metadata. Use them as fixed-K sensitivity "
            "and ablation checks before primary biological claims."
        ),
    ]
    suite = report.get("validation_suite")
    if isinstance(suite, dict):
        for name, commands in suite.items():
            if not isinstance(commands, list) or not commands:
                continue
            lines.extend(["", f"### {name}", ""])
            for command in commands:
                lines.append(f"- `{command}`")
    lines.extend(["", "## Concerns"])
    for item in report["concerns"]:
        lines.extend(
            [
                "",
                f"### {item['code']}",
                "",
                f"- Status: `{item['status']}`",
                f"- Blocking for primary claim: `{bool(item['blocking_for_primary_claim'])}`",
                f"- Required fix: {item['required_fix']}",
            ]
        )
        evidence = item.get("evidence")
        if isinstance(evidence, dict):
            cost_reliability = evidence.get("cost_reliability")
            if isinstance(cost_reliability, dict):
                lines.append(
                    "- Cost reliability: "
                    f"fallback_all={_metric(cost_reliability.get('fallback_fraction_all_costs'))}, "
                    f"fallback_assigned={_metric(cost_reliability.get('fallback_fraction_assigned'))}, "
                    f"mixed_eps={_metric(cost_reliability.get('mixed_candidate_effective_eps_fraction'))}, "
                    f"mixed_fallback={_metric(cost_reliability.get('mixed_candidate_fallback_fraction'))}"
                )
            spot_latent = evidence.get("spot_level_latent")
            if isinstance(spot_latent, dict):
                lines.append(
                    "- Spot latent role: "
                    f"implemented={bool(spot_latent.get('implemented'))}, "
                    f"chart_learning_mode={spot_latent.get('chart_learning_mode', 'unspecified')}, "
                    f"validation_role={spot_latent.get('validation_role', 'diagnostic')}"
                )
            baseline_comparison = evidence.get("baseline_comparison")
            if isinstance(baseline_comparison, dict):
                lines.append(
                    "- Baseline comparison: "
                    f"primary subregions={baseline_comparison.get('primary_n_subregions')}, "
                    f"baseline subregions={baseline_comparison.get('baseline_n_subregions')}, "
                    f"primary coverage={_metric(baseline_comparison.get('primary_coverage_fraction'))}, "
                    f"baseline coverage={_metric(baseline_comparison.get('baseline_coverage_fraction'))}"
                )
            diagnostics = evidence.get("diagnostics")
            if isinstance(diagnostics, dict):
                permutation = diagnostics.get("permutation")
                permutation = permutation if isinstance(permutation, dict) else {}
                lines.append(
                    "- Primary diagnostics: "
                    f"balanced_accuracy={_metric(diagnostics.get('balanced_accuracy'))}, "
                    f"spatial_block_accuracy={_metric(diagnostics.get('spatial_block_accuracy'))}, "
                    f"permutation_excess={_metric(permutation.get('excess'))}"
                )
            ablation_runs = evidence.get("ablation_runs")
            if isinstance(ablation_runs, list) and ablation_runs:
                lines.append("- Ablation evidence:")
                for run in ablation_runs:
                    if not isinstance(run, dict):
                        continue
                    diagnostics = run.get("diagnostics")
                    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
                    lines.append(
                        "  - "
                        f"`{run.get('run_dir')}`: "
                        f"warning_present={bool(run.get('warning_present'))}, "
                        f"method={run.get('construction_method')}, "
                        f"coordinate_only={bool(run.get('coordinate_only_baseline'))}, "
                        f"scale_um={_metric(run.get('basic_niche_size_um'))}, "
                        f"balanced_accuracy={_metric(diagnostics.get('balanced_accuracy'))}"
                    )
            fixed_k_runs = evidence.get("fixed_k_runs")
            if isinstance(fixed_k_runs, list) and fixed_k_runs:
                lines.append("- Fixed-K evidence:")
                for run in fixed_k_runs:
                    if not isinstance(run, dict):
                        continue
                    lines.append(
                        "  - "
                        f"`{run.get('run_dir')}`: "
                        f"K={run.get('n_clusters')}, seed={run.get('seed')}, "
                        f"auto_k={bool(run.get('auto_n_clusters'))}, "
                        f"qc_warning_count={run.get('qc_warning_count')}"
                    )
        commands = item.get("suggested_commands") or []
        if commands:
            lines.append("- Suggested command(s):")
            for command in commands:
                lines.append(f"  - `{command}`")
    md_path.write_text("\n".join(lines) + "\n")
    report["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, indent=2))
    return report
