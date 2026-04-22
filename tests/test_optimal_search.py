from __future__ import annotations

from spatial_ot.config import MultilevelExperimentConfig
from spatial_ot.optimal_search import build_default_search_candidates, build_refine_candidates, score_multilevel_summary


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


def test_default_search_candidates_include_baseline_and_are_unique() -> None:
    config = MultilevelExperimentConfig()
    candidates = build_default_search_candidates(config)

    assert any(candidate.name == "baseline" for candidate in candidates)
    assert len({candidate.name for candidate in candidates}) == len(candidates)
    assert len({tuple(sorted(candidate.overrides.items())) for candidate in candidates}) == len(candidates)


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
