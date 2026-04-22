from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .config import DeepFeatureConfig, MultilevelExperimentConfig, MultilevelOTConfig, MultilevelPathConfig, validate_multilevel_config


@dataclass(frozen=True)
class SearchCandidate:
    name: str
    stage: str
    overrides: dict[str, object]


def _clamp01(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _summary_value(summary: dict, *path: str) -> object | None:
    current: object = summary
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def score_multilevel_summary(summary: dict[str, object]) -> dict[str, object]:
    silhouette_native = _summary_value(summary, "subregion_embedding_compactness", "silhouette_native")
    weight_silhouette = _summary_value(summary, "subregion_weight_silhouette")
    assignment_margin = _summary_value(summary, "mean_assignment_margin")
    same_label_edge_fraction = _summary_value(summary, "boundary_separation", "same_label_edge_fraction")
    high_overlap_same_label_fraction = _summary_value(summary, "boundary_separation", "high_overlap_same_label_fraction")
    cell_adjacency_same_label_fraction = _summary_value(summary, "boundary_separation", "cell_adjacency_same_label_fraction")
    compactness_ratio = _summary_value(summary, "subregion_embedding_compactness", "compactness_ratio")
    isolated_subregion_fraction = _summary_value(summary, "boundary_separation", "isolated_subregion_fraction")
    fallback_fraction_all_costs = _summary_value(summary, "cost_reliability", "fallback_fraction_all_costs")
    mixed_candidate_fallback_fraction = _summary_value(summary, "cost_reliability", "mixed_candidate_fallback_fraction")
    convex_hull_fallback_fraction = _summary_value(summary, "convex_hull_fallback_fraction")
    forced_label_fraction = _summary_value(summary, "forced_label_fraction")
    geometry_transport_fraction = _summary_value(summary, "assigned_transport_cost_decomposition", "geometry_transport_fraction")
    qc_warning_count = _summary_value(summary, "qc_warning_count")
    scale_deviation_p95 = _summary_value(summary, "transform_diagnostics", "scale_deviation_p95")

    positive_terms = {
        "silhouette_native": 2.0 * _clamp01((float(silhouette_native) + 1.0) / 2.0 if silhouette_native is not None else None),
        "subregion_weight_silhouette": 1.5 * _clamp01((float(weight_silhouette) + 1.0) / 2.0 if weight_silhouette is not None else None),
        "assignment_margin": 1.5 * _clamp01(float(assignment_margin) / 0.25 if assignment_margin is not None else None),
        "same_label_edge_fraction": 1.0 * _clamp01(float(same_label_edge_fraction) if same_label_edge_fraction is not None else None),
        "high_overlap_same_label_fraction": 1.0 * _clamp01(
            float(high_overlap_same_label_fraction) if high_overlap_same_label_fraction is not None else None
        ),
        "cell_adjacency_same_label_fraction": 0.75 * _clamp01(
            float(cell_adjacency_same_label_fraction) if cell_adjacency_same_label_fraction is not None else None
        ),
    }
    geometry_penalty = 0.0
    if geometry_transport_fraction is not None and math.isfinite(float(geometry_transport_fraction)):
        geometry_penalty = max(float(geometry_transport_fraction) - 0.6, 0.0) / 0.4
    negative_terms = {
        "compactness_ratio": 0.75 * _clamp01(float(compactness_ratio) if compactness_ratio is not None else None),
        "isolated_subregion_fraction": 0.5 * _clamp01(
            float(isolated_subregion_fraction) if isolated_subregion_fraction is not None else None
        ),
        "fallback_fraction_all_costs": 1.5 * _clamp01(
            float(fallback_fraction_all_costs) if fallback_fraction_all_costs is not None else None
        ),
        "mixed_candidate_fallback_fraction": 1.0 * _clamp01(
            float(mixed_candidate_fallback_fraction) if mixed_candidate_fallback_fraction is not None else None
        ),
        "convex_hull_fallback_fraction": 2.0 * _clamp01(
            float(convex_hull_fallback_fraction) if convex_hull_fallback_fraction is not None else None
        ),
        "forced_label_fraction": 2.0 * _clamp01(
            min(float(forced_label_fraction) * 10.0, 1.0) if forced_label_fraction is not None else None
        ),
        "geometry_transport_fraction": 0.75 * _clamp01(geometry_penalty),
        "scale_deviation_p95": 0.5 * _clamp01(
            float(scale_deviation_p95) / 0.25 if scale_deviation_p95 is not None else None
        ),
        "qc_warning_count": 0.1 * _clamp01(
            min(float(qc_warning_count) / 10.0, 1.0) if qc_warning_count is not None else None
        ),
    }
    total_score = float(sum(positive_terms.values()) - sum(negative_terms.values()))
    return {
        "total_score": total_score,
        "positive_terms": positive_terms,
        "negative_terms": negative_terms,
        "raw_metrics": {
            "silhouette_native": silhouette_native,
            "subregion_weight_silhouette": weight_silhouette,
            "mean_assignment_margin": assignment_margin,
            "same_label_edge_fraction": same_label_edge_fraction,
            "high_overlap_same_label_fraction": high_overlap_same_label_fraction,
            "cell_adjacency_same_label_fraction": cell_adjacency_same_label_fraction,
            "compactness_ratio": compactness_ratio,
            "isolated_subregion_fraction": isolated_subregion_fraction,
            "fallback_fraction_all_costs": fallback_fraction_all_costs,
            "mixed_candidate_fallback_fraction": mixed_candidate_fallback_fraction,
            "convex_hull_fallback_fraction": convex_hull_fallback_fraction,
            "forced_label_fraction": forced_label_fraction,
            "geometry_transport_fraction": geometry_transport_fraction,
            "scale_deviation_p95": scale_deviation_p95,
            "qc_warning_count": qc_warning_count,
        },
    }


def _dedupe_candidates(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    seen: set[tuple[tuple[str, object], ...]] = set()
    deduped: list[SearchCandidate] = []
    for candidate in candidates:
        key = tuple(sorted(candidate.overrides.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def build_default_search_candidates(config: MultilevelExperimentConfig) -> list[SearchCandidate]:
    base = config.ot
    base_radius = float(base.radius_um)
    base_stride = float(base.stride_um)
    base_clusters = int(base.n_clusters)
    base_lambda_x = float(base.lambda_x)
    base_ot_eps = float(base.ot_eps)
    base_rho = float(base.rho)
    base_overlap = float(base.overlap_consistency_weight)

    radius_low = max(50.0, base_radius - 20.0)
    radius_high = base_radius + 20.0
    stride_tighter = min(base_radius, max(40.0, round(base_stride * 0.75, 3)))
    cluster_low = max(4, base_clusters - 2)
    cluster_high = base_clusters + 2
    lambda_low = max(0.2, round(base_lambda_x - 0.15, 3))
    lambda_high = min(1.0, round(base_lambda_x + 0.15, 3))
    overlap_low = max(0.0, round(base_overlap * 0.4, 3))
    overlap_high = round(base_overlap * 2.0, 3)
    ot_eps_low = max(0.015, round(base_ot_eps * 0.67, 4))
    ot_eps_high = min(0.08, round(base_ot_eps * 1.67, 4))
    rho_low = max(0.2, round(base_rho * 0.7, 4))
    rho_high = min(1.25, round(base_rho * 1.6, 4))

    candidates = [
        SearchCandidate("baseline", "coarse", {}),
        SearchCandidate("cluster_low", "coarse", {"n_clusters": cluster_low}),
        SearchCandidate("cluster_high", "coarse", {"n_clusters": cluster_high}),
        SearchCandidate("radius_low", "coarse", {"radius_um": radius_low, "stride_um": min(radius_low, stride_tighter)}),
        SearchCandidate("radius_high", "coarse", {"radius_um": radius_high, "stride_um": min(radius_high, base_stride)}),
        SearchCandidate("tighter_overlap_grid", "coarse", {"stride_um": min(base_radius, stride_tighter)}),
        SearchCandidate("lambda_x_low", "coarse", {"lambda_x": lambda_low}),
        SearchCandidate("lambda_x_high", "coarse", {"lambda_x": lambda_high}),
        SearchCandidate("ot_eps_low_rho_low", "coarse", {"ot_eps": ot_eps_low, "rho": rho_low}),
        SearchCandidate("ot_eps_high_rho_high", "coarse", {"ot_eps": ot_eps_high, "rho": rho_high}),
        SearchCandidate("overlap_low", "coarse", {"overlap_consistency_weight": overlap_low}),
        SearchCandidate("overlap_high", "coarse", {"overlap_consistency_weight": overlap_high}),
        SearchCandidate("cluster_high_radius_low", "coarse", {"n_clusters": cluster_high, "radius_um": radius_low, "stride_um": min(radius_low, stride_tighter)}),
        SearchCandidate("cluster_low_radius_high", "coarse", {"n_clusters": cluster_low, "radius_um": radius_high, "stride_um": min(radius_high, base_stride)}),
        SearchCandidate(
            "boundary_focused",
            "coarse",
            {
                "radius_um": radius_low,
                "stride_um": min(radius_low, stride_tighter),
                "lambda_x": lambda_low,
                "overlap_consistency_weight": overlap_high,
            },
        ),
        SearchCandidate(
            "geometry_relaxed",
            "coarse",
            {
                "radius_um": radius_high,
                "stride_um": min(radius_high, base_stride),
                "lambda_x": lambda_high,
                "overlap_consistency_weight": overlap_low,
            },
        ),
    ]
    return _dedupe_candidates(candidates)


def build_refine_candidates(config: MultilevelExperimentConfig, top_rows: list[dict[str, object]]) -> list[SearchCandidate]:
    candidates: list[SearchCandidate] = []
    for idx, row in enumerate(top_rows[:3]):
        params = dict(row.get("effective_params", {}))
        n_clusters = int(params.get("n_clusters", config.ot.n_clusters))
        radius_um = float(params.get("radius_um", config.ot.radius_um))
        stride_um = float(params.get("stride_um", config.ot.stride_um))
        overlap_consistency_weight = float(params.get("overlap_consistency_weight", config.ot.overlap_consistency_weight))
        lambda_x = float(params.get("lambda_x", config.ot.lambda_x))

        candidates.extend(
            [
                SearchCandidate(
                    f"refine_top{idx+1}_confirm",
                    "refine",
                    params,
                ),
                SearchCandidate(
                    f"refine_top{idx+1}_clusters_down",
                    "refine",
                    {**params, "n_clusters": max(4, n_clusters - 1)},
                ),
                SearchCandidate(
                    f"refine_top{idx+1}_clusters_up",
                    "refine",
                    {**params, "n_clusters": n_clusters + 1},
                ),
                SearchCandidate(
                    f"refine_top{idx+1}_radius_down",
                    "refine",
                    {**params, "radius_um": max(50.0, radius_um - 10.0), "stride_um": min(max(50.0, radius_um - 10.0), stride_um)},
                ),
                SearchCandidate(
                    f"refine_top{idx+1}_radius_up",
                    "refine",
                    {**params, "radius_um": radius_um + 10.0, "stride_um": min(radius_um + 10.0, stride_um)},
                ),
                SearchCandidate(
                    f"refine_top{idx+1}_overlap_up",
                    "refine",
                    {**params, "overlap_consistency_weight": round(overlap_consistency_weight + 0.02, 3)},
                ),
                SearchCandidate(
                    f"refine_top{idx+1}_overlap_down",
                    "refine",
                    {**params, "overlap_consistency_weight": max(0.0, round(overlap_consistency_weight - 0.02, 3))},
                ),
                SearchCandidate(
                    f"refine_top{idx+1}_lambda_x_mid",
                    "refine",
                    {**params, "lambda_x": max(0.2, min(1.0, round(lambda_x + (0.05 if lambda_x <= config.ot.lambda_x else -0.05), 3)))},
                ),
            ]
        )
    return _dedupe_candidates(candidates)


def _bool_flag(name: str, value: bool) -> list[str]:
    return [f"--{name}" if value else f"--no-{name}"]


def _candidate_config(base_config: MultilevelExperimentConfig, candidate: SearchCandidate, *, final_stage: bool) -> MultilevelExperimentConfig:
    payload = base_config.as_dict()
    payload["ot"].update(candidate.overrides)
    config = MultilevelExperimentConfig(
        paths=MultilevelPathConfig(**payload["paths"]),
        ot=MultilevelOTConfig(**payload["ot"]),
        deep=DeepFeatureConfig(**payload["deep"]),
    )
    if not final_stage:
        config.ot.n_init = max(1, min(2, int(config.ot.n_init)))
        config.ot.max_iter = max(4, min(6, int(config.ot.max_iter)))
        config.ot.align_iters = max(2, min(3, int(config.ot.align_iters)))
    return validate_multilevel_config(config)


def _candidate_command(config: MultilevelExperimentConfig) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "spatial_ot",
        "multilevel-ot",
        "--input-h5ad",
        str(config.paths.input_h5ad),
        "--output-dir",
        str(config.paths.output_dir),
        "--feature-obsm-key",
        str(config.paths.feature_obsm_key),
        "--spatial-x-key",
        str(config.paths.spatial_x_key),
        "--spatial-y-key",
        str(config.paths.spatial_y_key),
        "--spatial-scale",
        str(config.paths.spatial_scale),
        "--n-clusters",
        str(config.ot.n_clusters),
        "--atoms-per-cluster",
        str(config.ot.atoms_per_cluster),
        "--radius-um",
        str(config.ot.radius_um),
        "--stride-um",
        str(config.ot.stride_um),
        "--basic-niche-size-um",
        str(config.ot.basic_niche_size_um) if config.ot.basic_niche_size_um is not None else "0",
        "--min-cells",
        str(config.ot.min_cells),
        "--max-subregions",
        str(config.ot.max_subregions),
        "--lambda-x",
        str(config.ot.lambda_x),
        "--lambda-y",
        str(config.ot.lambda_y),
        "--geometry-eps",
        str(config.ot.geometry_eps),
        "--ot-eps",
        str(config.ot.ot_eps),
        "--rho",
        str(config.ot.rho),
        "--geometry-samples",
        str(config.ot.geometry_samples),
        "--compressed-support-size",
        str(config.ot.compressed_support_size),
        "--align-iters",
        str(config.ot.align_iters),
        "--min-scale",
        str(config.ot.min_scale),
        "--max-scale",
        str(config.ot.max_scale),
        "--scale-penalty",
        str(config.ot.scale_penalty),
        "--shift-penalty",
        str(config.ot.shift_penalty),
        "--n-init",
        str(config.ot.n_init),
        "--overlap-consistency-weight",
        str(config.ot.overlap_consistency_weight),
        "--overlap-jaccard-min",
        str(config.ot.overlap_jaccard_min),
        "--overlap-contrast-scale",
        str(config.ot.overlap_contrast_scale),
        "--max-iter",
        str(config.ot.max_iter),
        "--tol",
        str(config.ot.tol),
        "--seed",
        str(config.ot.seed),
        "--compute-device",
        str(config.ot.compute_device),
    ]
    cmd.extend(_bool_flag("allow-reflection", bool(config.ot.allow_reflection)))
    cmd.extend(_bool_flag("allow-scale", bool(config.ot.allow_scale)))
    cmd.extend(_bool_flag("allow-observed-hull-geometry", bool(config.ot.allow_convex_hull_fallback)))
    cmd.extend(_bool_flag("allow-umap-as-feature", bool(config.paths.allow_umap_as_feature)))
    if config.paths.region_obs_key:
        cmd.extend(["--region-obs-key", str(config.paths.region_obs_key)])
    if str(config.deep.method) != "none":
        cmd.extend(
            [
                "--deep-feature-method",
                str(config.deep.method),
                "--deep-latent-dim",
                str(config.deep.latent_dim),
                "--deep-hidden-dim",
                str(config.deep.hidden_dim),
                "--deep-layers",
                str(config.deep.layers),
                "--deep-neighbor-k",
                str(config.deep.neighbor_k),
                "--deep-graph-layers",
                str(config.deep.graph_layers),
                "--deep-graph-max-neighbors",
                str(config.deep.graph_max_neighbors),
                "--deep-full-batch-max-cells",
                str(config.deep.full_batch_max_cells),
                "--deep-epochs",
                str(config.deep.epochs),
                "--deep-batch-size",
                str(config.deep.batch_size),
                "--deep-lr",
                str(config.deep.learning_rate),
                "--deep-weight-decay",
                str(config.deep.weight_decay),
                "--deep-validation",
                str(config.deep.validation),
                "--deep-validation-context-mode",
                str(config.deep.validation_context_mode),
                "--deep-device",
                str(config.deep.device),
                "--deep-reconstruction-weight",
                str(config.deep.reconstruction_weight),
                "--deep-context-weight",
                str(config.deep.context_weight),
                "--deep-contrastive-weight",
                str(config.deep.contrastive_weight),
                "--deep-variance-weight",
                str(config.deep.variance_weight),
                "--deep-decorrelation-weight",
                str(config.deep.decorrelation_weight),
                "--deep-output-embedding",
                str(config.deep.output_embedding),
                "--deep-output-obsm-key",
                str(config.deep.output_obsm_key),
            ]
        )
        if config.deep.radius_um is not None:
            cmd.extend(["--deep-radius-um", str(config.deep.radius_um)])
        if config.deep.short_radius_um is not None:
            cmd.extend(["--deep-short-radius-um", str(config.deep.short_radius_um)])
        if config.deep.mid_radius_um is not None:
            cmd.extend(["--deep-mid-radius-um", str(config.deep.mid_radius_um)])
        if config.deep.batch_key is not None:
            cmd.extend(["--deep-batch-key", str(config.deep.batch_key)])
        if config.deep.count_layer is not None:
            cmd.extend(
                [
                    "--deep-count-layer",
                    str(config.deep.count_layer),
                    "--deep-count-decoder-rank",
                    str(config.deep.count_decoder_rank),
                    "--deep-count-chunk-size",
                    str(config.deep.count_chunk_size),
                    "--deep-count-loss-weight",
                    str(config.deep.count_loss_weight),
                ]
            )
        cmd.extend(_bool_flag("deep-allow-joint-ot-embedding", bool(config.deep.allow_joint_ot_embedding)))
        cmd.extend(_bool_flag("deep-save-model", bool(config.deep.save_model)))
    return cmd


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _prune_candidate_artifacts(run_dir: Path) -> None:
    keep = {"summary.json", "search_candidate.json", "search_score.json", "stdout.log", "stderr.log"}
    for child in run_dir.iterdir():
        if child.name in keep:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _rank_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    completed = [row for row in rows if row.get("status") == "completed" and row.get("score") is not None]
    return sorted(completed, key=lambda row: float(row["score"]), reverse=True)


def run_multilevel_optimal_search(
    *,
    config: MultilevelExperimentConfig,
    search_output_dir: str | Path,
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    plot_spatial_x_key: str = "cell_x",
    plot_spatial_y_key: str = "cell_y",
    default_sample_id: str = "pooled_cohort",
    time_budget_hours: float = 20.0,
    keep_top_k: int = 3,
    plot_best_sample_maps: bool = True,
) -> dict[str, object]:
    search_root = Path(search_output_dir)
    search_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    deadline = time.monotonic() + max(float(time_budget_hours), 0.01) * 3600.0
    rows: list[dict[str, object]] = []

    def run_candidate(candidate: SearchCandidate, *, final_stage: bool) -> dict[str, object]:
        candidate_config = _candidate_config(config, candidate, final_stage=final_stage)
        run_dir = search_root / "candidates" / f"{candidate.stage}_{candidate.name}"
        candidate_config.paths.output_dir = str(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "search_candidate.json",
            {
                "name": candidate.name,
                "stage": candidate.stage,
                "final_stage": bool(final_stage),
                "overrides": candidate.overrides,
                "effective_params": asdict(candidate_config.ot),
            },
        )
        command = _candidate_command(candidate_config)
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        started = time.time()
        with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                check=False,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
        runtime_seconds = float(time.time() - started)
        summary_path = run_dir / "summary.json"
        row: dict[str, object] = {
            "name": candidate.name,
            "stage": candidate.stage,
            "status": "completed" if completed.returncode == 0 and summary_path.exists() else "failed",
            "returncode": int(completed.returncode),
            "runtime_seconds": runtime_seconds,
            "run_dir": str(run_dir),
            "effective_params": asdict(candidate_config.ot),
            "command": command,
        }
        if row["status"] == "completed":
            summary = json.loads(summary_path.read_text())
            scoring = score_multilevel_summary(summary)
            row["score"] = float(scoring["total_score"])
            row["summary_path"] = str(summary_path)
            row["scoring"] = scoring
            _write_json(
                run_dir / "search_score.json",
                {
                    "score": row["score"],
                    "scoring": scoring,
                    "runtime_seconds": runtime_seconds,
                },
            )
        else:
            row["score"] = None
        return row

    coarse_candidates = build_default_search_candidates(config)
    for candidate in coarse_candidates:
        if time.monotonic() >= deadline:
            break
        row = run_candidate(candidate, final_stage=False)
        rows.append(row)
        ranked = _rank_results(rows)
        allowed_dirs = {Path(item["run_dir"]) for item in ranked[: max(int(keep_top_k), 1)]}
        for existing in rows:
            if existing.get("status") != "completed":
                continue
            existing_dir = Path(str(existing["run_dir"]))
            if existing_dir not in allowed_dirs:
                _prune_candidate_artifacts(existing_dir)
                existing["artifacts_pruned"] = True
            else:
                existing["artifacts_pruned"] = False
        _write_json(search_root / "search_results.json", {"rows": rows, "ranked": ranked})

    ranked_after_coarse = _rank_results(rows)
    refine_candidates = build_refine_candidates(config, ranked_after_coarse)
    for candidate in refine_candidates:
        if time.monotonic() >= deadline:
            break
        row = run_candidate(candidate, final_stage=True)
        rows.append(row)
        ranked = _rank_results(rows)
        allowed_dirs = {Path(item["run_dir"]) for item in ranked[: max(int(keep_top_k), 1)]}
        for existing in rows:
            if existing.get("status") != "completed":
                continue
            existing_dir = Path(str(existing["run_dir"]))
            if existing_dir not in allowed_dirs:
                _prune_candidate_artifacts(existing_dir)
                existing["artifacts_pruned"] = True
            else:
                existing["artifacts_pruned"] = False
        _write_json(search_root / "search_results.json", {"rows": rows, "ranked": ranked})

    ranked = _rank_results(rows)
    best_row = ranked[0] if ranked else None
    best_run_dir = None
    if best_row is not None:
        best_run_dir = Path(str(best_row["run_dir"]))
        if plot_best_sample_maps:
            sample_plot_dir = best_run_dir / "sample_niche_plots"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spatial_ot",
                    "plot-sample-niches",
                    "--run-dir",
                    str(best_run_dir),
                    "--output-dir",
                    str(sample_plot_dir),
                    "--sample-obs-key",
                    str(sample_obs_key),
                    "--source-file-obs-key",
                    str(source_file_obs_key),
                    "--plot-spatial-x-key",
                    str(plot_spatial_x_key),
                    "--plot-spatial-y-key",
                    str(plot_spatial_y_key),
                    "--default-sample-id",
                    str(default_sample_id),
                ],
                cwd=repo_root,
                check=False,
            )

    summary = {
        "search_output_dir": str(search_root),
        "time_budget_hours": float(time_budget_hours),
        "completed_candidates": int(sum(row.get("status") == "completed" for row in rows)),
        "failed_candidates": int(sum(row.get("status") == "failed" for row in rows)),
        "best_candidate": {
            "name": best_row["name"],
            "stage": best_row["stage"],
            "score": best_row["score"],
            "run_dir": best_row["run_dir"],
            "effective_params": best_row["effective_params"],
        }
        if best_row is not None
        else None,
        "ranked_candidates": [
            {
                "name": row["name"],
                "stage": row["stage"],
                "score": row["score"],
                "run_dir": row["run_dir"],
                "effective_params": row["effective_params"],
            }
            for row in ranked
        ],
        "results_path": str(search_root / "search_results.json"),
        "best_run_dir": str(best_run_dir) if best_run_dir is not None else None,
    }
    _write_json(search_root / "search_summary.json", summary)
    _write_json(search_root / "search_results.json", {"rows": rows, "ranked": ranked, "summary": summary})
    return summary


__all__ = [
    "SearchCandidate",
    "build_default_search_candidates",
    "build_refine_candidates",
    "run_multilevel_optimal_search",
    "score_multilevel_summary",
]
