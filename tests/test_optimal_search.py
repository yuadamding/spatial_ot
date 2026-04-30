from __future__ import annotations

import subprocess
from pathlib import Path

from spatial_ot.config import MultilevelExperimentConfig
from spatial_ot.optimal_search import (
    SearchCandidate,
    _candidate_command,
    build_default_search_candidates,
    build_refine_candidates,
    run_multilevel_optimal_search,
    score_multilevel_summary,
)


def test_score_multilevel_summary_prefers_compact_low_warning_runs() -> None:
    good_summary = {
        "subregion_embedding_compactness": {
            "silhouette_native": 0.42,
            "compactness_ratio": 0.28,
        },
        "subregion_weight_silhouette": 0.38,
        "mean_assignment_margin": 0.16,
        "boundary_separation": {
            "same_label_edge_fraction": 0.82,
            "high_overlap_same_label_fraction": 0.9,
            "cell_adjacency_same_label_fraction": 0.79,
            "isolated_subregion_fraction": 0.02,
        },
        "cost_reliability": {
            "fallback_fraction_all_costs": 0.0,
            "mixed_candidate_fallback_fraction": 0.0,
        },
        "convex_hull_fallback_fraction": 0.0,
        "forced_label_fraction": 0.0,
        "assigned_transport_cost_decomposition": {
            "geometry_transport_fraction": 0.42,
        },
        "transform_diagnostics": {
            "scale_deviation_p95": 0.08,
        },
        "qc_warning_count": 1,
    }
    bad_summary = {
        "subregion_embedding_compactness": {
            "silhouette_native": -0.1,
            "compactness_ratio": 1.2,
        },
        "subregion_weight_silhouette": -0.05,
        "mean_assignment_margin": 0.01,
        "boundary_separation": {
            "same_label_edge_fraction": 0.45,
            "high_overlap_same_label_fraction": 0.4,
            "cell_adjacency_same_label_fraction": 0.46,
            "isolated_subregion_fraction": 0.2,
        },
        "cost_reliability": {
            "fallback_fraction_all_costs": 0.25,
            "mixed_candidate_fallback_fraction": 0.2,
        },
        "convex_hull_fallback_fraction": 1.0,
        "forced_label_fraction": 0.12,
        "assigned_transport_cost_decomposition": {
            "geometry_transport_fraction": 0.92,
        },
        "transform_diagnostics": {
            "scale_deviation_p95": 0.4,
        },
        "qc_warning_count": 6,
    }

    good_score = score_multilevel_summary(good_summary)
    bad_score = score_multilevel_summary(bad_summary)

    assert good_score["total_score"] > bad_score["total_score"]
    assert good_score["rankable"] is True
    assert bad_score["rankable"] is False
    assert "convex_hull_fallback" in bad_score["rank_blockers"]
    assert "forced_subregion_cluster_size_repair" in bad_score["rank_blockers"]


def test_default_search_candidates_include_baseline_and_are_unique() -> None:
    config = MultilevelExperimentConfig()
    candidates = build_default_search_candidates(config)

    assert any(candidate.name == "baseline" for candidate in candidates)
    assert any(
        candidate.name == "coordinate_only_boundaries" for candidate in candidates
    )
    assert any(candidate.name == "feature_boundaries_025" for candidate in candidates)
    assert len({candidate.name for candidate in candidates}) == len(candidates)
    assert len(
        {tuple(sorted(candidate.overrides.items())) for candidate in candidates}
    ) == len(candidates)


def test_refine_candidates_are_derived_from_top_ranked_rows() -> None:
    config = MultilevelExperimentConfig()
    rows = [
        {
            "effective_params": {
                "n_clusters": 8,
                "radius_um": 80.0,
                "stride_um": 80.0,
                "lambda_x": 0.35,
                "overlap_consistency_weight": 0.05,
            }
        }
    ]

    candidates = build_refine_candidates(config, rows)

    assert candidates
    assert any(candidate.stage == "refine" for candidate in candidates)


def test_candidate_command_preserves_pretrained_deep_model_and_regularization_flags(
    tmp_path: Path,
) -> None:
    config = MultilevelExperimentConfig()
    config.paths.input_h5ad = str(tmp_path / "input.h5ad")
    config.paths.output_dir = str(tmp_path / "out")
    config.paths.feature_obsm_key = "X_pca"
    config.deep.method = "autoencoder"
    config.deep.output_embedding = "intrinsic"
    config.deep.pretrained_model = str(tmp_path / "deep_feature_model.pt")
    config.deep.independence_weight = 0.23
    config.deep.early_stopping_patience = 4
    config.deep.min_delta = 0.002
    config.deep.restore_best = False
    config.ot.max_subregion_area_um2 = 123.5

    command = _candidate_command(config)

    assert "--pretrained-deep-model" in command
    assert str(tmp_path / "deep_feature_model.pt") in command
    assert command[command.index("--deep-independence-weight") + 1] == "0.23"
    assert command[command.index("--deep-early-stopping-patience") + 1] == "4"
    assert command[command.index("--deep-min-delta") + 1] == "0.002"
    assert "--no-deep-restore-best" in command
    assert "--compute-spot-latent" in command
    assert command[command.index("--min-subregions-per-cluster") + 1] == str(
        config.ot.min_subregions_per_cluster
    )
    assert (
        command[command.index("--subregion-construction-method") + 1]
        == config.ot.subregion_construction_method
    )
    assert command[command.index("--subregion-feature-weight") + 1] == str(
        config.ot.subregion_feature_weight
    )
    assert command[command.index("--subregion-feature-dims") + 1] == str(
        config.ot.subregion_feature_dims
    )
    assert command[command.index("--deep-segmentation-knn") + 1] == str(
        config.ot.deep_segmentation_knn
    )
    assert command[command.index("--joint-refinement-iters") + 1] == str(
        config.ot.joint_refinement_iters
    )
    assert command[command.index("--joint-refinement-max-move-fraction") + 1] == str(
        config.ot.joint_refinement_max_move_fraction
    )
    assert command[command.index("--joint-refinement-acceptance-margin") + 1] == str(
        config.ot.joint_refinement_acceptance_margin
    )
    assert command[
        command.index("--heterogeneity-transport-max-subregions") + 1
    ] == str(config.ot.heterogeneity_transport_max_subregions)
    assert command[command.index("--heterogeneity-transport-feature-mode") + 1] == str(
        config.ot.heterogeneity_transport_feature_mode
    )
    assert command[command.index("--heterogeneity-fgw-alpha") + 1] == str(
        config.ot.heterogeneity_fgw_alpha
    )
    assert command[command.index("--heterogeneity-fused-ot-solver") + 1] == str(
        config.ot.heterogeneity_fused_ot_solver
    )
    assert command[command.index("--heterogeneity-fgw-solver") + 1] == str(
        config.ot.heterogeneity_fgw_solver
    )
    assert command[command.index("--max-subregion-area-um2") + 1] == "123.5"


def test_optimal_search_marks_timed_out_candidates(monkeypatch, tmp_path: Path) -> None:
    config = MultilevelExperimentConfig()
    config.paths.input_h5ad = str(tmp_path / "dummy_input.h5ad")
    config.paths.output_dir = str(tmp_path / "search_out")
    config.paths.feature_obsm_key = "X_pca"

    monkeypatch.setattr(
        "spatial_ot.optimal_search.build_default_search_candidates",
        lambda _config: [SearchCandidate("baseline", "coarse", {})],
    )
    monkeypatch.setattr(
        "spatial_ot.optimal_search.build_refine_candidates",
        lambda _config, _rows: [],
    )

    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=kwargs.get("args", args[0] if args else []), timeout=1.0
        )

    monkeypatch.setattr("spatial_ot.optimal_search.subprocess.run", _fake_run)

    summary = run_multilevel_optimal_search(
        config=config,
        search_output_dir=tmp_path / "search_root",
        time_budget_hours=0.001,
        keep_top_k=1,
        plot_best_sample_maps=False,
    )

    assert summary["timed_out_candidates"] == 1
    assert summary["completed_candidates"] == 0
    assert summary["best_candidate"] is None
