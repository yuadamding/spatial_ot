from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from spatial_ot.multilevel.concerns import (
    build_concern_resolution_report,
    write_concern_resolution_report,
)


def _write_summary(path: Path, *, coordinate_only: bool, warnings: list[str], auto_k: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    construction_method = "data_driven"
    mode = "generated_data_driven_coordinate_partition" if coordinate_only else "generated_deep_graph_segmentation"
    summary = {
        "output_dir": str(path),
        "n_cells": 100,
        "n_subregions": 10,
        "n_clusters": 4,
        "seed": 11,
        "auto_n_clusters": auto_k,
        "auto_k_selection": {"selected_k": 4, "criterion_votes": {"silhouette": 4}} if auto_k else None,
        "cell_subregion_coverage_fraction": 1.0,
        "subregion_construction": {
            "mode": mode,
            "construction_method": construction_method,
            "coordinate_only_baseline": coordinate_only,
            "feature_boundary_circularity_risk": not coordinate_only,
        },
        "shape_leakage_diagnostics": {
            "balanced_accuracy": 0.2,
            "permutation": {"observed": 0.2, "perm_mean": 0.1, "perm_p95": 0.12, "excess": 0.1},
        },
        "density_leakage_diagnostics": {
            "balanced_accuracy": 0.08,
            "permutation": {"observed": 0.08, "perm_mean": 0.07, "perm_p95": 0.09, "excess": 0.01},
        },
        "leakage_qc_thresholds": {
            "balanced_accuracy_warning": 0.1,
            "permutation_p95_margin_warning": 0.02,
            "permutation_mean_excess_warning": 0.03,
        },
        "cost_reliability": {
            "fallback_fraction_all_costs": 0.0,
            "fallback_fraction_assigned": 0.0,
            "mixed_candidate_effective_eps_fraction": 0.0,
            "mixed_candidate_fallback_fraction": 0.0,
        },
        "spot_level_latent": {
            "implemented": True,
            "chart_learning_mode": "supervised_by_fitted_ot_subregion_labels",
            "validation_role": "diagnostic_visualization_not_independent_evidence",
        },
        "qc_warnings": [{"code": code, "severity": "warning"} for code in warnings],
    }
    (path / "summary.json").write_text(json.dumps(summary))


def test_concern_report_requires_baseline_and_stability_for_feature_boundaries(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_summary(
        run_dir,
        coordinate_only=False,
        warnings=[
            "feature_aware_boundary_circularity_risk",
            "shape_descriptors_predict_subregion_clusters",
        ],
    )

    report = build_concern_resolution_report(run_dir)

    assert report["overall_status"] == "needs_validation"
    assert "feature_boundary_circularity" in report["blocking_concerns"]
    assert "shape_leakage" in report["blocking_concerns"]
    assert "auto_k_exploratory" in report["blocking_concerns"]
    assert report["strict_validation_passed"] is False
    feature = next(item for item in report["concerns"] if item["code"] == "feature_boundary_circularity")
    assert feature["status"] == "needs_coordinate_only_baseline"
    assert feature["suggested_commands"]
    cost = next(item for item in report["concerns"] if item["code"] == "ot_cost_comparability")
    assert cost["status"] == "passed_common_epsilon_checks"
    spot = next(item for item in report["concerns"] if item["code"] == "spot_latent_supervised_visualization")
    assert spot["status"] == "diagnostic_only_supervised_by_fitted_ot_labels"
    latent_claim = next(item for item in report["concerns"] if item["code"] == "within_niche_latent_heterogeneity_claim")
    assert latent_claim["blocking_for_primary_claim"] is False
    assert latent_claim["blocking_for_within_niche_latent_claim"] is True
    assert "within_niche_latent_heterogeneity_claim" in report["within_niche_latent_blocking_concerns"]


def test_concern_report_accepts_coordinate_baseline_and_writes_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    baseline_dir = tmp_path / "baseline"
    stability_dir = tmp_path / "fixed_k"
    _write_summary(run_dir, coordinate_only=False, warnings=[])
    _write_summary(baseline_dir, coordinate_only=True, warnings=[], auto_k=False)
    _write_summary(stability_dir, coordinate_only=False, warnings=[], auto_k=False)

    report = write_concern_resolution_report(
        run_dir,
        coordinate_baseline_run_dir=baseline_dir,
        stability_run_dirs=[stability_dir],
    )

    assert report["outputs"]["json"].endswith("concern_resolution_report.json")
    assert Path(report["outputs"]["json"]).exists()
    assert Path(report["outputs"]["markdown"]).exists()
    feature = next(item for item in report["concerns"] if item["code"] == "feature_boundary_circularity")
    assert feature["status"] == "addressed_by_coordinate_only_baseline"
    baseline = next(item for item in report["concerns"] if item["code"] == "coordinate_only_boundary_baseline")
    assert baseline["status"] == "available"
    auto_k = next(item for item in report["concerns"] if item["code"] == "auto_k_exploratory")
    assert auto_k["status"] == "fixed_k_stability_runs_available"


def test_concern_report_tracks_leakage_ablation_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ablation_dir = tmp_path / "shape_ablation"
    _write_summary(
        run_dir,
        coordinate_only=False,
        warnings=[
            "shape_descriptors_predict_subregion_clusters",
            "density_descriptors_predict_subregion_clusters",
        ],
    )
    _write_summary(ablation_dir, coordinate_only=True, warnings=[])

    report = build_concern_resolution_report(run_dir, leakage_ablation_run_dirs=[ablation_dir])

    shape = next(item for item in report["concerns"] if item["code"] == "shape_leakage")
    density = next(item for item in report["concerns"] if item["code"] == "density_leakage")
    assert shape["status"] == "ablation_runs_passed_current_thresholds"
    assert shape["blocking_for_primary_claim"] is False
    assert density["status"] == "ablation_runs_passed_current_thresholds"
    assert density["evidence"]["ablation_runs"][0]["run_dir"] == str(ablation_dir)


def test_concern_report_blocks_mixed_candidate_costs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_summary(run_dir, coordinate_only=True, warnings=[], auto_k=False)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["cost_reliability"]["mixed_candidate_effective_eps_fraction"] = 0.1
    summary_path.write_text(json.dumps(summary))

    report = build_concern_resolution_report(run_dir, coordinate_baseline_run_dir=run_dir, stability_run_dirs=[run_dir])

    assert "ot_cost_comparability" in report["blocking_concerns"]
    cost = next(item for item in report["concerns"] if item["code"] == "ot_cost_comparability")
    assert cost["status"] == "mixed_candidate_costs_remaining"


def test_concern_report_allows_clustering_but_blocks_unvalidated_latent_claim(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_summary(run_dir, coordinate_only=True, warnings=[], auto_k=False)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["spot_level_latent"] = {
        "implemented": True,
        "projection": "balanced_ot_atom_barycentric_mds_over_cluster_atom_posteriors",
        "chart_learning_mode": "model_grounded_atom_distance_mds_without_fisher_labels",
        "validation_role": "diagnostic_visualization_not_independent_evidence",
        "cluster_anchor_distance_method": "balanced_ot",
        "cluster_anchor_mds_stress": 0.08,
        "normalized_posterior_entropy_summary": {"median": 0.45},
    }
    summary_path.write_text(json.dumps(summary))

    report = build_concern_resolution_report(run_dir, coordinate_baseline_run_dir=run_dir, stability_run_dirs=[run_dir])

    assert report["claim_validation"]["subregion_niche_clustering"]["status"] == "ready_after_current_validation"
    assert report["claim_validation"]["within_niche_latent_heterogeneity"]["status"] == "blocked"
    latent_claim = next(item for item in report["concerns"] if item["code"] == "within_niche_latent_heterogeneity_claim")
    assert latent_claim["status"] == "blocked_for_within_niche_latent_claim"
    assert "missing_held_out_sample_projection" in latent_claim["evidence"]["blockers"]


def test_concern_report_blocks_latent_claim_on_anchor_ot_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_summary(run_dir, coordinate_only=True, warnings=[], auto_k=False)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["spot_level_latent"] = {
        "implemented": True,
        "projection": "balanced_ot_atom_barycentric_mds_over_cluster_atom_posteriors",
        "chart_learning_mode": "model_grounded_atom_distance_mds_without_fisher_labels",
        "validation_role": "diagnostic_visualization_not_independent_evidence",
        "cluster_anchor_distance_method": "balanced_ot",
        "cluster_anchor_distance_effective_method": "balanced_ot_with_expected_cross_cost_fallback",
        "cluster_anchor_ot_fallback_fraction": 0.25,
        "cluster_anchor_mds_stress": 0.08,
        "cluster_anchor_mds_positive_eigenvalue_mass_2d": 0.9,
        "cluster_anchor_mds_negative_eigenvalue_mass_fraction": 0.0,
        "atom_mds_stress_summary": {"max": 0.1},
        "normalized_posterior_entropy_summary": {"median": 0.45},
        "temperature_used_q95_q05_ratio": 1.5,
    }
    summary_path.write_text(json.dumps(summary))

    report = build_concern_resolution_report(run_dir, coordinate_baseline_run_dir=run_dir, stability_run_dirs=[run_dir])

    latent_claim = next(item for item in report["concerns"] if item["code"] == "within_niche_latent_heterogeneity_claim")
    assert "anchor_ot_fallback_used" in latent_claim["evidence"]["blockers"]
    assert latent_claim["evidence"]["anchor_ot_fallback_fraction"] == 0.25


def test_validate_run_concerns_strict_exits_nonzero_for_blockers(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_summary(run_dir, coordinate_only=False, warnings=["shape_descriptors_predict_subregion_clusters"])

    completed = subprocess.run(
        [sys.executable, "-m", "spatial_ot", "validate-run-concerns", "--run-dir", str(run_dir), "--strict"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "blocking_concerns" in completed.stdout
