from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
import torch

from spatial_ot.multilevel.core import (
    _cell_cluster_feature_costs,
    _compute_assignment_costs_rk_gpu,
    _compute_assigned_artifacts_r_gpu,
    _build_subregion_latent_embeddings_from_members,
    _ensure_minimum_cluster_size,
    _gpu_assignment_subregion_batch_size,
    _stabilize_mixed_candidate_assignment_costs,
)
from spatial_ot.multilevel.embedding import (
    compute_subregion_embedding,
    subregion_graph_metrics,
)
from spatial_ot.multilevel.diagnostics import build_qc_warnings, cell_subregion_coverage
from spatial_ot.multilevel.geometry import (
    _region_geometries_from_observed_points,
    _validate_mutually_exclusive_memberships,
    build_basic_niches,
    build_deep_graph_segmentation_subregions,
    build_partition_subregions_from_grid_tiles,
    refine_subregions_by_cluster_coherence,
)
from spatial_ot.multilevel.heterogeneity import (
    HETEROGENEITY_FGW_MODE,
    HETEROGENEITY_FUSED_OT_MODE,
    SubregionFGWMeasure,
    build_internal_heterogeneity_embeddings,
    build_subregion_fgw_measures,
    feature_cost,
    fgw_distance,
    fit_transport_cost_scales,
    fused_ot_distance,
    pairwise_transport_distance_matrix,
)
from spatial_ot.multilevel.io import _cluster_count_dict
from spatial_ot.multilevel.model_selection import (
    comprehensive_select_k_from_latent_embeddings,
    prepare_latent_clustering_embedding,
    select_k_from_ot_landmark_costs,
)
from spatial_ot.multilevel.spot_latent import (
    _cluster_atom_measure_sqdist,
    _global_discriminative_latent_chart,
    _posterior_entropy,
    _resolve_posterior_temperature,
    spot_latent_separation_diagnostics,
    weighted_atom_posteriors,
)
from spatial_ot.multilevel import (
    RegionGeometry,
    ShapeNormalizer,
    ShapeNormalizerDiagnostics,
    SubregionMeasure,
    _build_subregion_measures,
    _compute_assignment_costs,
    _initialize_cluster_atoms,
    _project_cells_from_subregions,
    _shape_descriptor_frame,
    _shape_leakage_balanced_accuracy,
    _standardize_features,
    _ensure_nonempty_clusters,
    weighted_similarity_fit,
    sample_geometry_points,
    build_subregions,
    build_composite_subregions_from_basic_niches,
    fit_multilevel_ot,
    fit_ot_shape_normalizer,
    make_reference_points_unit_disk,
)


def _cell_membership_counts(
    n_cells: int, subregion_members: list[np.ndarray]
) -> np.ndarray:
    counts = np.zeros(int(n_cells), dtype=np.int32)
    for members in subregion_members:
        np.add.at(counts, np.asarray(members, dtype=np.int64), 1)
    return counts


def test_auto_k_selector_uses_ot_landmark_scores() -> None:
    rng = np.random.default_rng(17)
    centers = np.array(
        [
            [0.0, 0.0, 4.0, 4.0, 4.0],
            [4.0, 0.0, 0.0, 4.0, 4.0],
            [4.0, 4.0, 0.0, 0.0, 4.0],
        ],
        dtype=np.float32,
    )
    costs = np.vstack(
        [
            centers[0] + rng.normal(scale=0.08, size=(24, centers.shape[1])),
            centers[1] + rng.normal(scale=0.08, size=(24, centers.shape[1])),
            centers[2] + rng.normal(scale=0.08, size=(24, centers.shape[1])),
        ]
    ).astype(np.float32)

    selection = select_k_from_ot_landmark_costs(
        costs,
        candidate_n_clusters=(2, 3, 4, 5),
        fallback_n_clusters=4,
        max_score_subregions=0,
        gap_references=3,
        random_state=17,
    )

    assert selection["selected_k"] == 3
    assert selection["criterion_votes"]
    assert len(selection["scores"]) == 4
    assert selection["distance_source"] == "pilot_ot_landmark_transport_cost_profiles"
    assert all(row["passes_min_cluster_size"] for row in selection["scores"])
    assert all(
        row["cluster_size_min"] >= row["effective_min_cluster_size"]
        for row in selection["scores"]
    )


def test_comprehensive_k_selector_scores_pooled_latent_without_spatial_inputs() -> None:
    rng = np.random.default_rng(19)
    latent = np.vstack(
        [
            rng.normal(loc=-4.0, scale=0.18, size=(24, 6)),
            rng.normal(loc=0.0, scale=0.18, size=(24, 6)),
            rng.normal(loc=4.0, scale=0.18, size=(24, 6)),
        ]
    ).astype(np.float32)

    selection = comprehensive_select_k_from_latent_embeddings(
        latent,
        candidate_n_clusters=(2, 3, 4),
        fallback_n_clusters=3,
        seeds=(19, 20, 21),
        n_init=3,
        min_cluster_size=8,
        gap_references=2,
        bootstrap_repeats=2,
        bootstrap_fraction=0.75,
        max_silhouette_subregions=0,
        random_state=19,
    )

    assert selection["uses_spatial"] is False
    assert (
        selection["distance_source"]
        == "pooled_raw_member_feature_distribution_subregion_latent_embeddings"
    )
    assert selection["selected_k"] == 3
    assert selection["criterion_votes"]
    assert len(selection["scores"]) == 3
    assert all(row["passes_min_cluster_size"] for row in selection["scores"])
    assert all(
        row["cluster_size_scope"] == "all_subregions" for row in selection["scores"]
    )
    assert any(row["bootstrap_ari_mean"] is not None for row in selection["scores"])


def test_latent_clustering_embedding_reduces_high_dimensional_noise() -> None:
    rng = np.random.default_rng(29)
    latent = rng.normal(size=(40, 12)).astype(np.float32)
    embedding, metadata = prepare_latent_clustering_embedding(
        latent,
        max_components=4,
        sample_size=30,
        random_state=29,
    )

    assert embedding.shape == (40, 4)
    assert metadata["reduction"] == "sampled_pca_whiten"
    assert metadata["embedding_dim_raw"] == 12
    assert metadata["embedding_dim_used"] == 4
    assert metadata["uses_spatial_coordinates"] is False
    assert metadata["uses_ot_costs"] is False


def test_fit_multilevel_ot_auto_k_refits_with_selected_cluster_count() -> None:
    rng = np.random.default_rng(23)
    coords = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.5, size=(40, 2)),
            rng.normal(loc=[8.0, 8.0], scale=0.5, size=(40, 2)),
        ]
    ).astype(np.float32)
    features = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0, 0.0], scale=0.1, size=(40, 3)),
            rng.normal(loc=[4.0, 4.0, 4.0], scale=0.1, size=(40, 3)),
        ]
    ).astype(np.float32)

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords,
        n_clusters=3,
        atoms_per_cluster=2,
        radius_um=2.0,
        stride_um=1.0,
        min_cells=5,
        max_subregions=8,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=32,
        compressed_support_size=8,
        align_iters=1,
        allow_reflection=False,
        allow_scale=False,
        allow_convex_hull_fallback=False,
        max_iter=2,
        tol=1e-4,
        basic_niche_size_um=None,
        compute_spot_latent=False,
        auto_n_clusters=True,
        candidate_n_clusters=(2, 3),
        auto_k_gap_references=2,
        auto_k_max_score_subregions=0,
        auto_k_pilot_n_init=1,
        auto_k_pilot_max_iter=1,
        seed=23,
        compute_device="cpu",
    )

    assert result.auto_k_selection is not None
    assert result.auto_k_selection["selected_k"] in {2, 3}
    assert result.cluster_supports.shape[0] == result.auto_k_selection["selected_k"]
    assert result.cell_cluster_probs.shape[1] == result.auto_k_selection["selected_k"]


def test_spot_latent_chart_does_not_force_separation_without_signal() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
    features = np.zeros((labels.shape[0], 5), dtype=np.float32)
    local = np.zeros((labels.shape[0], 2), dtype=np.float32)
    weights = np.ones(labels.shape[0], dtype=np.float32)

    latent = _global_discriminative_latent_chart(
        features,
        labels,
        weights,
        local,
        n_clusters=2,
    )
    diagnostics = spot_latent_separation_diagnostics(latent, labels, weights)

    assert np.allclose(latent, 0.0)
    assert diagnostics["minimum_between_cluster_distance_forced"] is False
    assert diagnostics["min_between_cluster_center_distance"] is None


def test_spot_latent_cluster_anchor_distance_uses_balanced_ot_matching() -> None:
    atom_coords = np.asarray(
        [
            [[0.0, 0.0], [10.0, 0.0]],
            [[10.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    atom_features = np.zeros((2, 2, 1), dtype=np.float32)
    weights = np.full((2, 2), 0.5, dtype=np.float32)

    balanced = _cluster_atom_measure_sqdist(
        atom_coords,
        atom_features,
        weights,
        lambda_x=1.0,
        lambda_y=0.0,
        cost_scale_x=1.0,
        cost_scale_y=1.0,
        distance_mode="balanced_ot",
    )
    expected = _cluster_atom_measure_sqdist(
        atom_coords,
        atom_features,
        weights,
        lambda_x=1.0,
        lambda_y=0.0,
        cost_scale_x=1.0,
        cost_scale_y=1.0,
        distance_mode="expected_cross_cost",
    )

    assert balanced[0, 1] < 1e-6
    assert expected[0, 1] > 40.0


def test_spot_latent_anchor_distance_reports_balanced_ot_fallback(monkeypatch) -> None:
    atom_coords = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[2.0, 0.0], [3.0, 0.0]],
        ],
        dtype=np.float32,
    )
    atom_features = np.zeros((2, 2, 1), dtype=np.float32)
    weights = np.full((2, 2), 0.5, dtype=np.float32)

    def _raise_emd2(*args, **kwargs):
        raise RuntimeError("forced OT failure")

    import spatial_ot.multilevel.spot_latent as spot_latent

    monkeypatch.setattr(spot_latent.ot, "emd2", _raise_emd2)

    distances, diagnostics = _cluster_atom_measure_sqdist(
        atom_coords,
        atom_features,
        weights,
        lambda_x=1.0,
        lambda_y=0.0,
        cost_scale_x=1.0,
        cost_scale_y=1.0,
        distance_mode="balanced_ot",
        return_diagnostics=True,
    )

    assert distances.shape == (2, 2)
    assert diagnostics["requested_method"] == "balanced_ot"
    assert (
        diagnostics["effective_method"]
        == "balanced_ot_with_expected_cross_cost_fallback"
    )
    assert diagnostics["fallback_fraction"] == 1.0
    assert diagnostics["fallback_matrix"][0, 1]
    assert diagnostics["solver_status_matrix"][0, 1] == 3


def test_spot_latent_auto_entropy_temperature_hits_target_range() -> None:
    total_cost = np.asarray(
        [
            [0.0, 0.3, 1.2, 2.4],
            [0.1, 0.5, 1.0, 1.8],
            [0.0, 0.7, 0.8, 2.0],
        ],
        dtype=np.float32,
    )
    prototype_weights = np.full(4, 0.25, dtype=np.float32)

    temperature = _resolve_posterior_temperature(
        total_cost,
        prototype_weights,
        base_temperature=0.1,
        mode="auto_entropy",
    )
    posterior = weighted_atom_posteriors(
        total_cost, prototype_weights, temperature=temperature
    )
    _entropy, normalized = _posterior_entropy(posterior)

    assert temperature > 0.0
    assert 0.25 <= float(np.median(normalized)) <= 0.65


def test_build_subregions_respects_min_cells() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],
            [11.0, 11.0],
        ],
        dtype=np.float32,
    )
    centers, members = build_subregions(
        coords_um=coords,
        radius_um=2.0,
        stride_um=10.0,
        min_cells=3,
        max_subregions=10,
    )
    assert centers.shape[0] >= 2
    assert all(len(m) >= 3 for m in members)
    counts = _cell_membership_counts(coords.shape[0], members)
    assert np.all(counts == 1)


def test_feature_informed_subregion_construction_keeps_boundaries_data_driven(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SPATIAL_OT_SUBREGION_FEATURE_WEIGHT", "10.0")
    monkeypatch.setenv("SPATIAL_OT_SUBREGION_FEATURE_DIMS", "2")
    monkeypatch.setenv("SPATIAL_OT_SUBREGION_BOUNDARY_REFINEMENT_ITERS", "5")
    monkeypatch.setenv("SPATIAL_OT_SUBREGION_BOUNDARY_KNN", "30")
    rng = np.random.default_rng(211)
    coords = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.35, size=(30, 2)),
            rng.normal(loc=[0.2, 0.1], scale=0.35, size=(30, 2)),
        ]
    ).astype(np.float32)
    features = np.vstack(
        [
            rng.normal(loc=[-3.0, 0.0], scale=0.1, size=(30, 2)),
            rng.normal(loc=[3.0, 0.0], scale=0.1, size=(30, 2)),
        ]
    ).astype(np.float32)
    truth = np.repeat(np.array([0, 1], dtype=np.int32), 30)

    _, members, _, _, _ = build_composite_subregions_from_basic_niches(
        coords_um=coords,
        radius_um=0.5,
        stride_um=0.5,
        min_cells=10,
        max_subregions=2,
        basic_niche_size_um=0.5,
        partition_features=features,
        seed=211,
    )

    counts = _cell_membership_counts(coords.shape[0], members)
    assert np.all(counts == 1)
    purities = []
    for member in members:
        labels, label_counts = np.unique(truth[member], return_counts=True)
        purities.append(float(label_counts.max() / label_counts.sum()))
    assert min(purities) >= 0.9


def test_deep_graph_segmentation_subregions_cut_learned_affinity_boundaries() -> None:
    rng = np.random.default_rng(312)
    y = np.linspace(-2.0, 2.0, 40, dtype=np.float32)
    left = np.column_stack(
        [
            rng.normal(loc=-0.35, scale=0.08, size=y.shape[0]),
            y + rng.normal(scale=0.03, size=y.shape[0]),
        ]
    )
    right = np.column_stack(
        [
            rng.normal(loc=0.35, scale=0.08, size=y.shape[0]),
            y + rng.normal(scale=0.03, size=y.shape[0]),
        ]
    )
    coords = np.vstack([left, right]).astype(np.float32)
    deep_embedding = np.vstack(
        [
            rng.normal(loc=[-4.0, 0.0], scale=0.05, size=(left.shape[0], 2)),
            rng.normal(loc=[4.0, 0.0], scale=0.05, size=(right.shape[0], 2)),
        ]
    ).astype(np.float32)
    truth = np.repeat(np.array([0, 1], dtype=np.int32), [left.shape[0], right.shape[0]])

    centers, members, basic_centers, basic_members, basic_ids = (
        build_deep_graph_segmentation_subregions(
            coords_um=coords,
            segmentation_features=deep_embedding,
            target_scale_um=1.0,
            min_cells=20,
            max_subregions=2,
            segmentation_knn=8,
            segmentation_feature_dims=2,
            segmentation_feature_weight=5.0,
            segmentation_spatial_weight=0.01,
            seed=312,
        )
    )

    assert centers.shape[0] == len(members) == 2
    assert basic_centers.shape[0] == len(basic_members)
    assert len(basic_ids) == len(members)
    counts = _cell_membership_counts(coords.shape[0], members)
    assert np.all(counts == 1)
    assert all(member.size >= 20 for member in members)
    purities = []
    for member in members:
        _, label_counts = np.unique(truth[member], return_counts=True)
        purities.append(float(label_counts.max() / label_counts.sum()))
    assert min(purities) >= 0.95


def test_deep_graph_segmentation_keeps_fine_min_size_subregions() -> None:
    rng = np.random.default_rng(314)
    coords = rng.uniform(0.0, 6.0, size=(720, 2)).astype(np.float32)
    stripes = np.floor(coords[:, 0] * 2.0).astype(np.float32)
    deep_embedding = np.column_stack(
        [
            stripes + rng.normal(scale=0.05, size=coords.shape[0]),
            coords[:, 1] + rng.normal(scale=0.05, size=coords.shape[0]),
        ]
    ).astype(np.float32)

    centers, members, _basic_centers, _basic_members, _basic_ids = (
        build_deep_graph_segmentation_subregions(
            coords_um=coords,
            segmentation_features=deep_embedding,
            target_scale_um=1.0,
            min_cells=10,
            max_subregions=30,
            segmentation_knn=8,
            segmentation_feature_dims=2,
            segmentation_feature_weight=1.0,
            segmentation_spatial_weight=0.05,
            seed=314,
        )
    )

    counts = _cell_membership_counts(coords.shape[0], members)
    assert np.all(counts == 1)
    assert centers.shape[0] == len(members)
    assert len(members) >= 20
    assert all(member.size >= 10 for member in members)


def test_fit_multilevel_ot_uses_deep_graph_segmentation_mode() -> None:
    rng = np.random.default_rng(313)
    coords = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.2, size=(36, 2)),
            rng.normal(loc=[2.0, 0.0], scale=0.2, size=(36, 2)),
        ]
    ).astype(np.float32)
    features = np.vstack(
        [
            rng.normal(loc=[-2.0, 0.0], scale=0.1, size=(36, 2)),
            rng.normal(loc=[2.0, 0.0], scale=0.1, size=(36, 2)),
        ]
    ).astype(np.float32)

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords,
        n_clusters=2,
        atoms_per_cluster=2,
        radius_um=1.0,
        stride_um=1.0,
        min_cells=12,
        max_subregions=4,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=32,
        compressed_support_size=8,
        align_iters=1,
        allow_reflection=False,
        allow_scale=False,
        allow_convex_hull_fallback=False,
        max_iter=1,
        tol=1e-4,
        basic_niche_size_um=1.0,
        subregion_construction_method="deep_segmentation",
        deep_segmentation_knn=8,
        deep_segmentation_feature_dims=2,
        deep_segmentation_feature_weight=5.0,
        deep_segmentation_spatial_weight=0.01,
        compute_spot_latent=False,
        seed=313,
        compute_device="cpu",
    )

    counts = _cell_membership_counts(coords.shape[0], result.subregion_members)
    assert np.all(counts == 1)
    assert all(len(member) >= 12 for member in result.subregion_members)
    assert result.subregion_geometry_sources == ["observed_point_cloud"] * len(
        result.subregion_members
    )


def test_joint_refinement_preserves_partition_and_records_metadata() -> None:
    rng = np.random.default_rng(315)
    left = np.column_stack(
        [
            rng.normal(loc=-0.4, scale=0.12, size=50),
            rng.normal(loc=0.0, scale=0.35, size=50),
        ]
    )
    right = np.column_stack(
        [
            rng.normal(loc=0.4, scale=0.12, size=50),
            rng.normal(loc=0.0, scale=0.35, size=50),
        ]
    )
    coords = np.vstack([left, right]).astype(np.float32)
    features = np.vstack(
        [
            rng.normal(loc=[-2.0, 0.0], scale=0.15, size=(50, 2)),
            rng.normal(loc=[2.0, 0.0], scale=0.15, size=(50, 2)),
        ]
    ).astype(np.float32)

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords,
        n_clusters=2,
        atoms_per_cluster=2,
        radius_um=1.0,
        stride_um=1.0,
        min_cells=15,
        max_subregions=4,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=32,
        compressed_support_size=8,
        align_iters=1,
        allow_reflection=False,
        allow_scale=False,
        allow_convex_hull_fallback=False,
        max_iter=1,
        tol=1e-4,
        basic_niche_size_um=1.0,
        subregion_construction_method="joint_refinement",
        deep_segmentation_knn=8,
        deep_segmentation_feature_dims=2,
        deep_segmentation_feature_weight=5.0,
        deep_segmentation_spatial_weight=0.01,
        joint_refinement_iters=1,
        joint_refinement_knn=8,
        joint_refinement_feature_dims=2,
        joint_refinement_max_move_fraction=0.10,
        compute_spot_latent=False,
        seed=315,
        compute_device="cpu",
    )

    counts = _cell_membership_counts(coords.shape[0], result.subregion_members)
    assert np.all(counts == 1)
    assert all(len(member) >= 15 for member in result.subregion_members)
    metadata = result.subregion_latent_embedding_metadata["joint_refinement"]
    assert metadata["enabled"] is True
    assert metadata["applied"] is True
    assert metadata["preliminary_embedding_source"] == "heterogeneity_descriptor_niche"
    assert metadata["acceptance_margin"] == pytest.approx(1e-3)
    assert metadata["transport_inside_boundary_loop"] is False
    assert metadata["connectivity_checked_during_moves"] is True
    assert metadata["requires_connected_output"] is True
    assert metadata["requires_min_cells"] is True
    assert result.joint_refinement_initial_cell_subregion_labels.shape == (
        coords.shape[0],
    )
    assert result.joint_refinement_refined_cell_subregion_labels.shape == (
        coords.shape[0],
    )
    assert result.joint_refinement_initial_subregion_cluster_labels.shape[0] == metadata[
        "initial_region_count"
    ]
    assert result.joint_refinement_energy
    assert "total_energy_before" in result.joint_refinement_energy[0]


def test_cluster_coherence_refinement_keeps_mutually_exclusive_connected_min_cell_regions() -> (
    None
):
    rng = np.random.default_rng(316)
    coords = np.vstack(
        [
            rng.normal(loc=[-1.0, 0.0], scale=0.18, size=(24, 2)),
            rng.normal(loc=[0.0, 0.0], scale=0.18, size=(24, 2)),
            rng.normal(loc=[1.0, 0.0], scale=0.18, size=(24, 2)),
        ]
    ).astype(np.float32)
    features = np.vstack(
        [
            rng.normal(loc=[-2.0, 0.0], scale=0.1, size=(24, 2)),
            rng.normal(loc=[0.0, 0.0], scale=0.1, size=(24, 2)),
            rng.normal(loc=[2.0, 0.0], scale=0.1, size=(24, 2)),
        ]
    ).astype(np.float32)
    members = [
        np.arange(0, 24, dtype=np.int32),
        np.arange(24, 48, dtype=np.int32),
        np.arange(48, 72, dtype=np.int32),
    ]

    move_log: list[dict[str, object]] = []
    centers, refined_members, _ids, history = refine_subregions_by_cluster_coherence(
        coords,
        features,
        members,
        np.asarray([0, 1, 0], dtype=np.int32),
        min_cells=12,
        max_subregions=4,
        target_scale_um=1.0,
        n_iters=1,
        n_neighbors=8,
        max_move_fraction=0.05,
        acceptance_margin=0.0,
        feature_dims=2,
        seed=316,
        move_log=move_log,
    )

    assert centers.shape[0] == len(refined_members)
    counts = _cell_membership_counts(coords.shape[0], refined_members)
    assert np.all(counts == 1)
    assert all(len(member) >= 12 for member in refined_members)
    _validate_mutually_exclusive_memberships(
        coords.shape[0], refined_members, require_full_coverage=True
    )
    assert history
    assert "total_energy_delta" in history[0]
    assert move_log
    assert any(bool(row["accepted"]) for row in move_log)


def test_cluster_coherence_refinement_margin_blocks_tiny_or_negative_moves() -> None:
    rng = np.random.default_rng(316)
    coords = np.vstack(
        [
            rng.normal(loc=[-1.0, 0.0], scale=0.18, size=(24, 2)),
            rng.normal(loc=[0.0, 0.0], scale=0.18, size=(24, 2)),
            rng.normal(loc=[1.0, 0.0], scale=0.18, size=(24, 2)),
        ]
    ).astype(np.float32)
    features = np.vstack(
        [
            rng.normal(loc=[-2.0, 0.0], scale=0.1, size=(24, 2)),
            rng.normal(loc=[0.0, 0.0], scale=0.1, size=(24, 2)),
            rng.normal(loc=[2.0, 0.0], scale=0.1, size=(24, 2)),
        ]
    ).astype(np.float32)
    members = [
        np.arange(0, 24, dtype=np.int32),
        np.arange(24, 48, dtype=np.int32),
        np.arange(48, 72, dtype=np.int32),
    ]
    move_log: list[dict[str, object]] = []

    _centers, _refined_members, _ids, history = refine_subregions_by_cluster_coherence(
        coords,
        features,
        members,
        np.asarray([0, 1, 0], dtype=np.int32),
        min_cells=12,
        max_subregions=4,
        target_scale_um=1.0,
        n_iters=1,
        n_neighbors=8,
        max_move_fraction=0.05,
        acceptance_margin=1e6,
        feature_dims=2,
        seed=316,
        move_log=move_log,
    )

    assert history[0]["accepted_moves"] == 0
    assert history[0]["acceptance_margin"] == pytest.approx(1e6)
    assert move_log
    assert all(not bool(row["accepted"]) for row in move_log)


def test_batched_gpu_helpers_mark_nonconverged_finite_solves_as_fallback(
    monkeypatch,
) -> None:
    def fake_sinkhorn(a, beta, cost, eps, rho, *, num_iter, tol):
        del eps, rho, num_iter, tol
        gamma = a.unsqueeze(-1) * beta.unsqueeze(-2)
        objective = (gamma * cost).sum(dim=(-1, -2))
        return gamma, objective, False, 1.0

    monkeypatch.setattr(
        "spatial_ot.multilevel.core.sinkhorn_semirelaxed_unbalanced_log_torch",
        fake_sinkhorn,
    )
    normalizer = ShapeNormalizer(center=np.zeros((1, 2)), scale=1.0, interpolator=None)
    diagnostics = ShapeNormalizerDiagnostics(
        geometry_source="polygon",
        used_fallback=False,
        ot_cost=None,
        sinkhorn_converged=None,
        mapped_radius_p95=1.0,
        mapped_radius_max=1.0,
        interpolation_residual=0.0,
    )
    measures = [
        SubregionMeasure(
            subregion_id=0,
            center_um=np.zeros(2, dtype=np.float32),
            members=np.array([0, 1], dtype=np.int32),
            canonical_coords=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            features=np.array([[0.0], [1.0]], dtype=np.float32),
            weights=np.array([0.5, 0.5], dtype=np.float32),
            geometry_point_count=3,
            compressed_point_count=2,
            normalizer=normalizer,
            normalizer_diagnostics=diagnostics,
        )
    ]
    atom_coords = np.array(
        [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]], dtype=np.float32
    )
    atom_features = np.array([[[0.0], [1.0]], [[1.0], [0.0]]], dtype=np.float32)
    betas = np.full((2, 2), 0.5, dtype=np.float32)

    costs, effective_eps, used_fallback = _compute_assignment_costs_rk_gpu(
        measures=measures,
        atom_coords=atom_coords,
        atom_features=atom_features,
        betas=betas,
        lambda_x=0.5,
        lambda_y=1.0,
        eps=0.03,
        rho=0.5,
        align_iters=1,
        allow_reflection=False,
        allow_scale=False,
        cost_scale_x=1.0,
        cost_scale_y=1.0,
        min_scale=0.75,
        max_scale=1.33,
        scale_penalty=0.05,
        shift_penalty=0.05,
        compute_device=torch.device("cpu"),
        return_diagnostics=True,
    )
    assert costs.shape == (1, 2)
    assert effective_eps.shape == (1, 2)
    assert np.all(used_fallback)

    _, _, _, _, assigned_eps, assigned_fallback = _compute_assigned_artifacts_r_gpu(
        measures=measures,
        labels=np.array([0], dtype=np.int32),
        atom_coords=atom_coords,
        atom_features=atom_features,
        betas=betas,
        lambda_x=0.5,
        lambda_y=1.0,
        eps=0.03,
        rho=0.5,
        align_iters=1,
        allow_reflection=False,
        allow_scale=False,
        cost_scale_x=1.0,
        cost_scale_y=1.0,
        min_scale=0.75,
        max_scale=1.33,
        scale_penalty=0.05,
        shift_penalty=0.05,
        compute_device=torch.device("cpu"),
    )
    assert assigned_eps.shape == (1,)
    assert np.all(assigned_fallback)


def test_gpu_assignment_subregion_batch_size_can_be_env_capped(monkeypatch) -> None:
    monkeypatch.setenv("SPATIAL_OT_GPU_ASSIGNMENT_SUBREGION_BATCH_SIZE", "2")
    batch_size = _gpu_assignment_subregion_batch_size(
        n_subregions=5,
        n_clusters=8,
        measure_size=48,
        support_size=96,
        feature_dim=512,
        device=torch.device("cpu"),
    )
    assert batch_size == 2


def test_gpu_assignment_subregion_batch_size_accounts_for_measure_size(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SPATIAL_OT_GPU_ASSIGNMENT_SUBREGION_BATCH_SIZE", raising=False)
    monkeypatch.setenv("SPATIAL_OT_CUDA_TARGET_VRAM_GB", "1")
    small = _gpu_assignment_subregion_batch_size(
        n_subregions=5000,
        n_clusters=8,
        measure_size=16,
        support_size=48,
        feature_dim=512,
        device=torch.device("cpu"),
    )
    large = _gpu_assignment_subregion_batch_size(
        n_subregions=5000,
        n_clusters=8,
        measure_size=96,
        support_size=48,
        feature_dim=512,
        device=torch.device("cpu"),
    )
    assert 1 <= large < small < 5000


def test_cell_cluster_feature_costs_match_manual_softmin_scores() -> None:
    features = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    support_features = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    prototype_weights = np.array(
        [
            [0.6, 0.4],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    temperature = 0.3

    observed = _cell_cluster_feature_costs(
        features=features,
        support_features=support_features,
        prototype_weights=prototype_weights,
        temperature=temperature,
        compute_device=torch.device("cpu"),
    )

    expected = np.zeros_like(observed)
    for k in range(support_features.shape[0]):
        dist = ((features[:, None, :] - support_features[k][None, :, :]) ** 2).sum(
            axis=-1
        )
        scores = np.exp(-dist / temperature) * prototype_weights[k][None, :]
        expected[:, k] = -temperature * np.log(np.maximum(scores.sum(axis=1), 1e-8))

    assert np.allclose(observed, expected, atol=1e-6)


def test_subregion_embedding_fallback_handles_one_dimensional_weights(
    monkeypatch,
) -> None:
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "umap.umap_":
            raise ImportError("forced missing umap")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    embedding, method = compute_subregion_embedding(
        np.asarray([[0.1], [0.5], [0.9]], dtype=np.float32),
        seed=0,
    )

    assert embedding.shape == (3, 2)
    assert method == "1D"
    assert np.all(np.isfinite(embedding))


def test_composite_subregions_are_unions_of_basic_niches() -> None:
    xs, ys = np.meshgrid(np.arange(0.0, 600.0, 20.0), np.arange(0.0, 600.0, 20.0))
    coords = np.column_stack([xs.reshape(-1), ys.reshape(-1)]).astype(np.float32)
    (
        subregion_centers,
        subregion_members,
        basic_centers,
        basic_members,
        subregion_basic_niche_ids,
    ) = build_composite_subregions_from_basic_niches(
        coords_um=coords,
        radius_um=240.0,
        stride_um=200.0,
        min_cells=5,
        max_subregions=32,
        basic_niche_size_um=200.0,
    )
    assert basic_centers.shape[0] >= 4
    assert len(subregion_members) == len(subregion_basic_niche_ids)
    assert len(subregion_members) == subregion_centers.shape[0]
    for members, niche_ids in zip(
        subregion_members, subregion_basic_niche_ids, strict=False
    ):
        expected = np.unique(
            np.concatenate(
                [basic_members[int(niche_id)] for niche_id in niche_ids.tolist()]
            )
        )
        assert np.array_equal(np.sort(members), np.sort(expected))
        assert len(niche_ids) >= 1
    counts = _cell_membership_counts(coords.shape[0], subregion_members)
    assert np.all(counts == 1)


def test_composite_subregions_treat_max_area_as_soft_qc_target() -> None:
    xs, ys = np.meshgrid(np.arange(0.0, 120.0, 5.0), np.arange(0.0, 120.0, 5.0))
    coords = np.column_stack([xs.reshape(-1), ys.reshape(-1)]).astype(np.float32)
    max_area = 500.0

    _centers, members, _basic_centers, _basic_members, _basic_ids = (
        build_composite_subregions_from_basic_niches(
            coords_um=coords,
            radius_um=120.0,
            stride_um=120.0,
            min_cells=10,
            max_subregions=128,
            basic_niche_size_um=120.0,
            max_area_um2=max_area,
        )
    )

    shape_df = _shape_descriptor_frame(members, coords)
    assert float(shape_df["shape_area"].max()) > max_area
    assert all(member.size >= 10 for member in members)
    counts = _cell_membership_counts(coords.shape[0], members)
    assert np.all(counts == 1)


def test_qc_warns_when_min_cell_constraint_is_not_satisfied() -> None:
    warnings = build_qc_warnings(
        feature_embedding_warning=None,
        fallback_fraction=0.0,
        assigned_ot_fallback_fraction=0.0,
        assigned_effective_eps_values=[],
        requested_ot_eps=0.03,
        coverage_fraction=1.0,
        mean_assignment_margin=0.5,
        assigned_transport_cost_decomposition={"geometry_transport_fraction": 0.0},
        cost_reliability={},
        transform_diagnostics={},
        forced_label_fraction=0.0,
        deep_summary={},
        realized_subregion_statistics={
            "minimum_cell_constraint": 25,
            "minimum_cell_constraint_satisfied": False,
            "maximum_area_qc_target_um2": 500.0,
            "n_cells": {"min": 1.0},
        },
    )

    min_cell_warnings = [
        item
        for item in warnings
        if item["code"] == "minimum_cell_constraint_not_satisfied"
    ]
    assert len(min_cell_warnings) == 1
    assert min_cell_warnings[0]["severity"] == "warning"
    assert "minimum subregion size constraint" in min_cell_warnings[0]["message"]


def test_qc_reports_soft_area_target_violations_without_hard_failure() -> None:
    warnings = build_qc_warnings(
        feature_embedding_warning=None,
        fallback_fraction=0.0,
        assigned_ot_fallback_fraction=0.0,
        assigned_effective_eps_values=[],
        requested_ot_eps=0.03,
        coverage_fraction=1.0,
        mean_assignment_margin=0.5,
        assigned_transport_cost_decomposition={"geometry_transport_fraction": 0.0},
        cost_reliability={},
        transform_diagnostics={},
        forced_label_fraction=0.0,
        deep_summary={},
        realized_subregion_statistics={
            "minimum_cell_constraint": 25,
            "minimum_cell_constraint_satisfied": True,
            "maximum_area_qc_target_um2": 500.0,
            "maximum_area_qc_target_violation_count": 3,
        },
    )

    area_warnings = [
        item
        for item in warnings
        if item["code"] == "soft_max_subregion_area_target_exceeded"
    ]
    assert len(area_warnings) == 1
    assert area_warnings[0]["severity"] == "info"
    assert "soft QC target" in area_warnings[0]["message"]


def test_basic_niche_subregions_still_respect_min_cells() -> None:
    dense_xs, dense_ys = np.meshgrid(
        np.arange(0.0, 120.0, 20.0), np.arange(0.0, 120.0, 20.0)
    )
    sparse = np.array([[260.0, 0.0], [280.0, 0.0]], dtype=np.float32)
    coords = np.vstack(
        [np.column_stack([dense_xs.reshape(-1), dense_ys.reshape(-1)]), sparse]
    ).astype(np.float32)

    (
        subregion_centers,
        subregion_members,
        basic_centers,
        basic_members,
        subregion_basic_niche_ids,
    ) = build_composite_subregions_from_basic_niches(
        coords_um=coords,
        radius_um=100.0,
        stride_um=200.0,
        min_cells=5,
        max_subregions=16,
        basic_niche_size_um=200.0,
    )
    assert basic_centers.shape[0] >= 2
    assert any(len(members) < 5 for members in basic_members)
    assert all(len(members) >= 5 for members in subregion_members)
    assert any(len(niche_ids) > 1 for niche_ids in subregion_basic_niche_ids)
    counts = _cell_membership_counts(coords.shape[0], subregion_members)
    assert np.all(counts == 1)


def test_basic_niches_cover_all_cells_without_circle_gaps() -> None:
    xs, ys = np.meshgrid(np.arange(0.0, 150.0, 10.0), np.arange(0.0, 150.0, 10.0))
    coords = np.column_stack([xs.reshape(-1), ys.reshape(-1)]).astype(np.float32)

    (
        subregion_centers,
        subregion_members,
        basic_centers,
        basic_members,
        subregion_basic_niche_ids,
    ) = build_composite_subregions_from_basic_niches(
        coords_um=coords,
        radius_um=100.0,
        stride_um=100.0,
        min_cells=1,
        max_subregions=32,
        basic_niche_size_um=50.0,
    )

    basic_covered = np.zeros(coords.shape[0], dtype=bool)
    for members in basic_members:
        basic_covered[members] = True
    subregion_covered = np.zeros(coords.shape[0], dtype=bool)
    for members in subregion_members:
        subregion_covered[members] = True

    assert basic_centers.shape[0] > 0
    assert len(subregion_basic_niche_ids) == len(subregion_members)
    assert np.all(basic_covered)
    assert np.all(subregion_covered)
    counts = _cell_membership_counts(coords.shape[0], subregion_members)
    assert np.all(counts == 1)


def test_basic_niche_cap_merges_instead_of_dropping_cells() -> None:
    xs, ys = np.meshgrid(np.arange(0.0, 100.0, 10.0), np.arange(0.0, 100.0, 10.0))
    coords = np.column_stack([xs.reshape(-1), ys.reshape(-1)]).astype(np.float32)

    centers, members = build_basic_niches(
        coords_um=coords,
        niche_size_um=10.0,
        min_cells=1,
        max_subregions=8,
    )

    counts = _cell_membership_counts(coords.shape[0], members)
    assert centers.shape[0] <= 8
    assert np.all(counts == 1)


def test_fit_rejects_overlapping_explicit_subregions() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.0, 0.2],
            [3.0, 3.0],
            [3.2, 3.0],
            [3.0, 3.2],
        ],
        dtype=np.float32,
    )
    features = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit_multilevel_ot(
            features=features,
            coords_um=coords,
            subregion_members=[
                np.array([0, 1, 2], dtype=np.int32),
                np.array([2, 3, 4], dtype=np.int32),
            ],
            n_clusters=2,
            atoms_per_cluster=1,
            radius_um=1.0,
            stride_um=1.0,
            min_cells=3,
            max_subregions=0,
            lambda_x=0.5,
            lambda_y=1.0,
            geometry_eps=0.03,
            ot_eps=0.03,
            rho=0.5,
            geometry_samples=32,
            compressed_support_size=2,
            align_iters=1,
            allow_convex_hull_fallback=True,
            max_iter=1,
            tol=1e-4,
            n_init=1,
            basic_niche_size_um=None,
            seed=12,
            compute_device="cpu",
        )


def test_fit_rejects_duplicate_indices_inside_explicit_subregion() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.0, 0.2],
            [3.0, 3.0],
            [3.2, 3.0],
            [3.0, 3.2],
        ],
        dtype=np.float32,
    )
    features = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit_multilevel_ot(
            features=features,
            coords_um=coords,
            subregion_members=[
                np.array([0, 1, 1], dtype=np.int32),
                np.array([3, 4, 5], dtype=np.int32),
            ],
            n_clusters=2,
            atoms_per_cluster=1,
            radius_um=1.0,
            stride_um=1.0,
            min_cells=3,
            max_subregions=0,
            lambda_x=0.5,
            lambda_y=1.0,
            geometry_eps=0.03,
            ot_eps=0.03,
            rho=0.5,
            geometry_samples=32,
            compressed_support_size=2,
            align_iters=1,
            allow_convex_hull_fallback=True,
            max_iter=1,
            tol=1e-4,
            n_init=1,
            basic_niche_size_um=None,
            seed=12,
            compute_device="cpu",
        )


def test_membership_validator_rejects_non_integer_indices() -> None:
    with pytest.raises(RuntimeError, match="integer cell indices"):
        _validate_mutually_exclusive_memberships(
            4,
            [
                np.array([0, 1.5], dtype=np.float32),
                np.array([2, 3], dtype=np.int32),
            ],
        )


def test_membership_validator_reports_complete_partition_counts() -> None:
    counts = _validate_mutually_exclusive_memberships(
        5,
        [
            np.array([0, 2], dtype=np.int32),
            np.array([1, 3, 4], dtype=np.int32),
        ],
        require_full_coverage=True,
    )

    assert np.array_equal(counts, np.ones(5, dtype=np.int32))


def test_cell_subregion_coverage_counts_repeated_indices() -> None:
    summary = cell_subregion_coverage(
        3,
        [
            np.array([0, 0, 1], dtype=np.int32),
            np.array([2], dtype=np.int32),
        ],
    )

    assert summary["covered_cell_count"] == 3
    assert summary["cell_subregion_duplicate_count"] == 1
    assert summary["cell_subregion_max_memberships"] == 2
    assert summary["cell_subregion_partition_complete"] is False
    assert summary["subregion_membership_mode"] == "overlapping"


def test_subregion_graph_metrics_use_primary_subregion_cell_labels() -> None:
    class Result:
        pass

    coords = np.vstack(
        [
            np.column_stack(
                [np.arange(7, dtype=np.float32), np.zeros(7, dtype=np.float32)]
            ),
            np.column_stack(
                [100.0 + np.arange(7, dtype=np.float32), np.zeros(7, dtype=np.float32)]
            ),
        ]
    ).astype(np.float32)
    result = Result()
    result.subregion_members = [
        np.arange(0, 7, dtype=np.int32),
        np.arange(7, 14, dtype=np.int32),
    ]
    result.subregion_centers_um = np.array([[3.0, 0.0], [103.0, 0.0]], dtype=np.float32)
    result.subregion_cluster_labels = np.array([0, 1], dtype=np.int32)
    result.subregion_cluster_probs = np.array(
        [[0.95, 0.05], [0.05, 0.95]], dtype=np.float32
    )
    result.cell_cluster_labels = np.tile(np.array([0, 1], dtype=np.int32), 7)

    metrics = subregion_graph_metrics(
        n_cells=coords.shape[0],
        result=result,
        radius_um=10.0,
        stride_um=10.0,
        coords_um=coords,
    )

    assert metrics["cell_adjacency_same_label_fraction"] == pytest.approx(1.0)


def test_multilevel_ot_recovers_two_subregion_families() -> None:
    rng = np.random.default_rng(1337)
    group_centers = np.array(
        [
            [0.0, 0.0],
            [0.0, 30.0],
            [100.0, 0.0],
            [100.0, 30.0],
        ],
        dtype=np.float32,
    )
    true_group = np.array([0, 0, 1, 1], dtype=np.int32)

    feature_atoms = {
        0: np.array([[0.0, 0.0, 0.0], [1.0, 0.2, -0.1]], dtype=np.float32),
        1: np.array([[5.0, 5.0, 5.0], [6.0, 4.8, 5.2]], dtype=np.float32),
    }

    coords = []
    features = []
    cell_truth = []
    for group_idx, center in enumerate(group_centers):
        atoms = feature_atoms[int(true_group[group_idx])]
        mix = (
            np.array([0.75, 0.25], dtype=np.float32)
            if true_group[group_idx] == 0
            else np.array([0.2, 0.8], dtype=np.float32)
        )
        n_cells = 36
        atom_ids = rng.choice(2, size=n_cells, p=mix)
        feat = atoms[atom_ids] + rng.normal(scale=0.08, size=(n_cells, 3)).astype(
            np.float32
        )
        xy = center + rng.normal(scale=1.0, size=(n_cells, 2)).astype(np.float32)
        coords.append(xy)
        features.append(feat)
        cell_truth.extend([int(true_group[group_idx])] * n_cells)

    coords = np.vstack(coords).astype(np.float32)
    features = np.vstack(features).astype(np.float32)
    cell_truth = np.asarray(cell_truth, dtype=np.int32)

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords,
        n_clusters=2,
        atoms_per_cluster=2,
        radius_um=15.0,
        stride_um=30.0,
        min_cells=20,
        max_subregions=16,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=96,
        compressed_support_size=24,
        align_iters=3,
        allow_reflection=True,
        allow_scale=True,
        allow_convex_hull_fallback=True,
        max_iter=8,
        tol=1e-4,
        basic_niche_size_um=None,
        subregion_clustering_method="ot_dictionary",
        min_subregions_per_cluster=2,
        seed=1337,
        compute_device="cpu",
    )

    subregion_truth = []
    for members in result.subregion_members:
        labels, counts = np.unique(cell_truth[members], return_counts=True)
        subregion_truth.append(int(labels[np.argmax(counts)]))
    subregion_truth = np.asarray(subregion_truth, dtype=np.int32)

    ari = adjusted_rand_score(subregion_truth, result.subregion_cluster_labels)
    assert ari > 0.8
    assert result.cell_cluster_probs.shape[1] == 2
    assert np.unique(result.cell_cluster_labels).size == 2
    natural_assignment = ~result.subregion_forced_label_mask.astype(bool)
    assert np.array_equal(
        result.subregion_cluster_labels[natural_assignment],
        result.subregion_cluster_costs.argmin(axis=1)[natural_assignment],
    )
    assert result.cluster_atom_coords.shape == (2, 2, 2)
    assert result.cluster_atom_features.shape == (2, 2, 3)
    assert result.subregion_assigned_effective_eps.shape[0] == len(
        result.subregion_members
    )
    assert np.all(result.subregion_assigned_effective_eps > 0)
    assert (
        result.subregion_candidate_effective_eps_matrix.shape
        == result.subregion_cluster_costs.shape
    )
    assert (
        result.subregion_candidate_used_ot_fallback_matrix.shape
        == result.subregion_cluster_costs.shape
    )
    assert (
        result.subregion_cluster_transport_costs.shape
        == result.subregion_cluster_costs.shape
    )
    assert (
        result.subregion_cluster_overlap_penalties.shape
        == result.subregion_cluster_costs.shape
    )
    assert result.subregion_measure_summaries.shape[0] == len(result.subregion_members)
    assert result.subregion_assigned_geometry_transport_costs.shape[0] == len(
        result.subregion_members
    )
    assert result.subregion_assigned_feature_transport_costs.shape[0] == len(
        result.subregion_members
    )
    assert result.subregion_assigned_transform_penalties.shape[0] == len(
        result.subregion_members
    )
    assert result.subregion_assigned_overlap_consistency_penalties.shape[0] == len(
        result.subregion_members
    )
    assert result.subregion_assigned_transform_rotation_deg.shape[0] == len(
        result.subregion_members
    )
    assert result.subregion_assigned_transform_reflection.shape[0] == len(
        result.subregion_members
    )
    assert result.subregion_assigned_transform_scale.shape[0] == len(
        result.subregion_members
    )
    assert result.subregion_assigned_transform_translation_norm.shape[0] == len(
        result.subregion_members
    )
    expected_spot_occurrences = sum(
        len(members) for members in result.subregion_members
    )
    assert expected_spot_occurrences == features.shape[0]
    counts = _cell_membership_counts(features.shape[0], result.subregion_members)
    assert np.all(counts == 1)
    assert result.spot_latent_cell_indices.shape == (expected_spot_occurrences,)
    assert result.spot_latent_subregion_ids.shape == (expected_spot_occurrences,)
    assert result.spot_latent_cluster_labels.shape == (expected_spot_occurrences,)
    assert result.spot_latent_coords.shape == (expected_spot_occurrences, 2)
    assert result.spot_latent_within_coords.shape == (expected_spot_occurrences, 2)
    assert result.spot_latent_cluster_anchors.shape == (2, 2)
    assert result.spot_latent_atom_embedding.shape == (2, 2, 2)
    assert result.spot_latent_aligned_coords.shape == (expected_spot_occurrences, 2)
    assert result.spot_latent_atom_posteriors.shape == (expected_spot_occurrences, 2)
    assert np.allclose(result.spot_latent_atom_posteriors.sum(axis=1), 1.0, atol=1e-5)
    assert result.spot_latent_posterior_entropy.shape == (expected_spot_occurrences,)
    assert result.spot_latent_normalized_posterior_entropy.shape == (
        expected_spot_occurrences,
    )
    assert result.spot_latent_atom_argmax.shape == (expected_spot_occurrences,)
    assert result.spot_latent_temperature_used.shape == (expected_spot_occurrences,)
    assert result.spot_latent_temperature_cost_gap.shape == (expected_spot_occurrences,)
    assert result.spot_latent_temperature_fixed.shape == (expected_spot_occurrences,)
    assert np.all(result.spot_latent_posterior_entropy >= -1e-6)
    assert np.all(result.spot_latent_temperature_used > 0.0)
    assert result.spot_latent_mode == "atom_barycentric_mds"
    assert (
        result.spot_latent_chart_learning_mode
        == "model_grounded_atom_distance_mds_without_fisher_labels"
    )
    assert (
        result.spot_latent_projection_mode
        == "balanced_ot_atom_barycentric_mds_over_cluster_atom_posteriors"
    )
    assert result.spot_latent_temperature_mode == "auto_entropy"
    assert result.spot_latent_cluster_anchor_distance_method == "balanced_ot"
    assert result.spot_latent_cluster_anchor_distance_requested_method == "balanced_ot"
    assert result.spot_latent_cluster_anchor_distance_effective_method == "balanced_ot"
    assert result.spot_latent_cluster_anchor_distance.shape == (2, 2)
    assert result.spot_latent_cluster_anchor_ot_fallback_matrix.shape == (2, 2)
    assert result.spot_latent_cluster_anchor_solver_status_matrix.shape == (2, 2)
    assert result.spot_latent_cluster_anchor_ot_fallback_fraction == 0.0
    assert result.spot_latent_atom_mds_stress.shape == (2,)
    assert np.isfinite(result.spot_latent_cluster_mds_stress)
    assert np.all(np.isfinite(result.spot_latent_atom_mds_stress))
    assert np.all(result.spot_latent_weights >= 0.0)
    assert np.array_equal(
        result.spot_latent_cluster_labels,
        result.subregion_cluster_labels[result.spot_latent_subregion_ids],
    )
    latent_centers = []
    within_radius = []
    for cluster_id in np.unique(result.spot_latent_cluster_labels).tolist():
        mask = result.spot_latent_cluster_labels == int(cluster_id)
        coords = result.spot_latent_coords[mask]
        center = coords.mean(axis=0)
        latent_centers.append(center)
        within_radius.append(
            float(np.mean(np.linalg.norm(coords - center[None, :], axis=1)))
        )
    latent_centers_arr = np.asarray(latent_centers, dtype=np.float32)
    if latent_centers_arr.shape[0] >= 2:
        between = float(np.linalg.norm(latent_centers_arr[0] - latent_centers_arr[1]))
        assert between > float(np.mean(within_radius))
    assert result.cell_spot_latent_coords.shape == (features.shape[0], 2)
    assert result.cell_spot_latent_unweighted_coords.shape == (features.shape[0], 2)
    assert result.cell_spot_latent_confidence_weighted_coords.shape == (
        features.shape[0],
        2,
    )
    assert result.cell_spot_latent_cluster_labels.shape == (features.shape[0],)
    assert result.cell_spot_latent_weights.shape == (features.shape[0],)
    assert result.cell_spot_latent_posterior_entropy.shape == (features.shape[0],)
    covered_latent = result.cell_spot_latent_cluster_labels >= 0
    assert np.any(covered_latent)
    assert np.all(np.isfinite(result.cell_spot_latent_coords[covered_latent]))
    reconstructed_transport_cost = (
        result.subregion_assigned_geometry_transport_costs
        + result.subregion_assigned_feature_transport_costs
        + result.subregion_assigned_transform_penalties
    )
    assigned_cost = result.subregion_cluster_costs[
        np.arange(result.subregion_cluster_labels.shape[0]),
        result.subregion_cluster_labels,
    ]
    assigned_transport_cost = result.subregion_cluster_transport_costs[
        np.arange(result.subregion_cluster_labels.shape[0]),
        result.subregion_cluster_labels,
    ]
    assert np.all(reconstructed_transport_cost > 0.0)
    assert np.allclose(
        result.subregion_cluster_transport_costs
        + result.subregion_cluster_overlap_penalties,
        result.subregion_cluster_costs,
        atol=1e-5,
    )
    assert np.all(reconstructed_transport_cost <= assigned_transport_cost + 1e-4)
    assert np.all(assigned_transport_cost <= assigned_cost + 1e-4)


def test_default_subregion_clustering_pools_feature_latents_without_spatial_labels() -> (
    None
):
    rng = np.random.default_rng(2026)
    coords_parts = []
    feature_parts = []
    subregion_members = []
    start = 0
    spatial_centers = [
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([100.0, 0.0]),
        np.array([102.0, 0.0]),
    ]
    feature_centers = [
        np.array([-3.0, 0.0]),
        np.array([3.0, 0.0]),
        np.array([-3.0, 0.0]),
        np.array([3.0, 0.0]),
    ]
    for spatial_center, feature_center in zip(
        spatial_centers, feature_centers, strict=True
    ):
        coords_parts.append(spatial_center + rng.normal(scale=0.15, size=(24, 2)))
        feature_parts.append(feature_center + rng.normal(scale=0.05, size=(24, 2)))
        subregion_members.append(np.arange(start, start + 24, dtype=np.int32))
        start += 24
    coords = np.vstack(coords_parts).astype(np.float32)
    features = np.vstack(feature_parts).astype(np.float32)
    centers = np.vstack(
        [coords[members].mean(axis=0) for members in subregion_members]
    ).astype(np.float32)

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords,
        subregion_members=subregion_members,
        subregion_centers_um=centers,
        n_clusters=2,
        atoms_per_cluster=2,
        radius_um=10.0,
        stride_um=10.0,
        min_cells=12,
        max_subregions=4,
        lambda_x=100.0,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=32,
        compressed_support_size=8,
        align_iters=1,
        allow_reflection=False,
        allow_scale=False,
        allow_convex_hull_fallback=True,
        max_iter=1,
        tol=1e-4,
        basic_niche_size_um=None,
        min_subregions_per_cluster=2,
        compute_spot_latent=False,
        subregion_clustering_method="pooled_subregion_latent",
        seed=2026,
        compute_device="cpu",
    )

    labels = result.subregion_cluster_labels
    assert result.subregion_clustering_method == "pooled_subregion_latent"
    assert result.subregion_clustering_uses_spatial is False
    assert result.subregion_latent_embeddings.shape[0] == 4
    assert labels[0] == labels[2]
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]


def test_internal_heterogeneity_embedding_separates_arrangement_not_composition() -> (
    None
):
    rng = np.random.default_rng(2028)

    def make_measure(subregion_id: int, arrangement: str) -> SubregionMeasure:
        if arrangement == "left_right":
            coords_a = np.column_stack(
                [
                    rng.normal(-0.6, 0.04, size=16),
                    rng.normal(0.0, 0.20, size=16),
                ]
            )
            coords_b = np.column_stack(
                [
                    rng.normal(0.6, 0.04, size=16),
                    rng.normal(0.0, 0.20, size=16),
                ]
            )
        else:
            coords_a = np.column_stack(
                [
                    rng.normal(0.0, 0.20, size=16),
                    rng.normal(-0.6, 0.04, size=16),
                ]
            )
            coords_b = np.column_stack(
                [
                    rng.normal(0.0, 0.20, size=16),
                    rng.normal(0.6, 0.04, size=16),
                ]
            )
        coords = np.vstack([coords_a, coords_b]).astype(np.float32)
        features = np.vstack(
            [
                np.tile(np.array([1.0, 0.0], dtype=np.float32), (16, 1)),
                np.tile(np.array([0.0, 1.0], dtype=np.float32), (16, 1)),
            ]
        )
        return SubregionMeasure(
            subregion_id=int(subregion_id),
            center_um=np.zeros(2, dtype=np.float32),
            members=np.arange(32, dtype=np.int32),
            canonical_coords=coords,
            features=features,
            weights=np.full(32, 1.0 / 32.0, dtype=np.float32),
            geometry_point_count=32,
            compressed_point_count=32,
            normalizer=ShapeNormalizer(
                center=np.zeros((1, 2), dtype=np.float32),
                scale=1.0,
                interpolator=None,
            ),
            normalizer_diagnostics=ShapeNormalizerDiagnostics(
                geometry_source="test",
                used_fallback=False,
                ot_cost=None,
                sinkhorn_converged=None,
                mapped_radius_p95=None,
                mapped_radius_max=None,
                interpolation_residual=None,
            ),
        )

    measures = [
        make_measure(0, "left_right"),
        make_measure(1, "bottom_top"),
        make_measure(2, "left_right"),
        make_measure(3, "bottom_top"),
    ]
    embeddings, metadata = build_internal_heterogeneity_embeddings(
        measures,
        codebook_size=2,
        codebook_sample_size=64,
        grid_size=4,
        random_state=2028,
    )
    assert metadata["mode"] == "heterogeneity_descriptor_niche"
    assert metadata["requested_mode"] == "heterogeneity_descriptor_niche"
    assert metadata["uses_ot_costs"] is False
    assert metadata["uses_internal_canonical_coordinates"] is True
    assert metadata["uses_pairwise_internal_spatial_graph"] is True
    assert metadata["pair_cooccurrence_normalization"] == "observed_over_expected"
    assert metadata["reserved_ot_modes"] == [
        "heterogeneity_fused_ot_niche",
        "heterogeneity_fgw_niche",
    ]
    assert set(metadata["block_weights"]) == {
        "composition",
        "diversity",
        "spatial_field",
        "pair_cooccurrence",
    }
    assert set(metadata["block_slices"]) == set(metadata["block_weights"])
    assert set(metadata["block_diagnostics"]) == set(metadata["block_weights"])
    assert embeddings.shape[0] == 4

    dist = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=2)
    assert dist[0, 2] < dist[0, 1]
    assert dist[1, 3] < dist[1, 0]


def test_heterogeneity_codebook_size_defaults_to_requested_value(monkeypatch) -> None:
    monkeypatch.delenv("SPATIAL_OT_HETEROGENEITY_MAX_CODEBOOK_SIZE", raising=False)
    rng = np.random.default_rng(2030)

    measures = []
    for subregion_id in range(4):
        measures.append(
            SubregionMeasure(
                subregion_id=int(subregion_id),
                center_um=np.zeros(2, dtype=np.float32),
                members=np.arange(10, dtype=np.int32),
                canonical_coords=rng.normal(size=(10, 2)).astype(np.float32),
                features=rng.normal(size=(10, 5)).astype(np.float32),
                weights=np.full(10, 0.1, dtype=np.float32),
                geometry_point_count=10,
                compressed_point_count=10,
                normalizer=ShapeNormalizer(
                    center=np.zeros((1, 2), dtype=np.float32),
                    scale=1.0,
                    interpolator=None,
                ),
                normalizer_diagnostics=ShapeNormalizerDiagnostics(
                    geometry_source="test",
                    used_fallback=False,
                    ot_cost=None,
                    sinkhorn_converged=None,
                    mapped_radius_p95=None,
                    mapped_radius_max=None,
                    interpolation_residual=None,
                ),
            )
        )

    _, descriptor_metadata = build_internal_heterogeneity_embeddings(
        measures,
        codebook_size=18,
        codebook_sample_size=40,
        random_state=2030,
    )
    _, transport_metadata = build_subregion_fgw_measures(
        measures,
        codebook_size=18,
        codebook_sample_size=40,
        random_state=2030,
    )

    assert descriptor_metadata["cell_state_codebook_size"] == 18
    assert descriptor_metadata["cell_state_codebook_size_requested"] == 18
    assert descriptor_metadata["cell_state_codebook_size_cap"] == 18
    assert transport_metadata["cell_state_codebook_size"] == 18
    assert transport_metadata["cell_state_codebook_size_requested"] == 18
    assert transport_metadata["cell_state_codebook_size_cap"] == 18


def test_internal_heterogeneity_composition_only_loses_arrangement() -> None:
    rng = np.random.default_rng(2029)

    def make_measure(subregion_id: int, arrangement: str) -> SubregionMeasure:
        if arrangement == "left_right":
            coords_a = np.column_stack(
                [
                    rng.normal(-0.6, 0.04, size=12),
                    rng.normal(0.0, 0.20, size=12),
                ]
            )
            coords_b = np.column_stack(
                [
                    rng.normal(0.6, 0.04, size=12),
                    rng.normal(0.0, 0.20, size=12),
                ]
            )
        else:
            coords_a = np.column_stack(
                [
                    rng.normal(0.0, 0.20, size=12),
                    rng.normal(-0.6, 0.04, size=12),
                ]
            )
            coords_b = np.column_stack(
                [
                    rng.normal(0.0, 0.20, size=12),
                    rng.normal(0.6, 0.04, size=12),
                ]
            )
        coords = np.vstack([coords_a, coords_b]).astype(np.float32)
        features = np.vstack(
            [
                np.tile(np.array([1.0, 0.0], dtype=np.float32), (12, 1)),
                np.tile(np.array([0.0, 1.0], dtype=np.float32), (12, 1)),
            ]
        )
        return SubregionMeasure(
            subregion_id=int(subregion_id),
            center_um=np.zeros(2, dtype=np.float32),
            members=np.arange(24, dtype=np.int32),
            canonical_coords=coords,
            features=features,
            weights=np.full(24, 1.0 / 24.0, dtype=np.float32),
            geometry_point_count=24,
            compressed_point_count=24,
            normalizer=ShapeNormalizer(
                center=np.zeros((1, 2), dtype=np.float32),
                scale=1.0,
                interpolator=None,
            ),
            normalizer_diagnostics=ShapeNormalizerDiagnostics(
                geometry_source="test",
                used_fallback=False,
                ot_cost=None,
                sinkhorn_converged=None,
                mapped_radius_p95=None,
                mapped_radius_max=None,
                interpolation_residual=None,
            ),
        )

    embeddings, metadata = build_internal_heterogeneity_embeddings(
        [
            make_measure(0, "left_right"),
            make_measure(1, "bottom_top"),
            make_measure(2, "left_right"),
            make_measure(3, "bottom_top"),
        ],
        codebook_size=2,
        codebook_sample_size=64,
        grid_size=4,
        block_weights={
            "composition": 1.0,
            "diversity": 0.0,
            "spatial_field": 0.0,
            "pair_cooccurrence": 0.0,
        },
        random_state=2029,
        mode="heterogeneity_ot_niche",
    )

    assert metadata["mode"] == "heterogeneity_descriptor_niche"
    assert metadata["requested_mode"] == "heterogeneity_ot_niche"
    assert metadata["legacy_alias_requested"] is True
    assert metadata["block_weights"]["composition"] == pytest.approx(1.0)
    assert metadata["block_weights"]["spatial_field"] == pytest.approx(0.0)
    assert metadata["block_weights"]["pair_cooccurrence"] == pytest.approx(0.0)
    pairwise_dist = np.linalg.norm(
        embeddings[:, None, :] - embeddings[None, :, :],
        axis=2,
    )
    assert np.max(pairwise_dist) < 1e-6


def test_internal_heterogeneity_pair_block_detects_contact_motif() -> None:
    def make_measure(subregion_id: int, motif: str) -> SubregionMeasure:
        ys = np.linspace(-0.35, 0.35, 8, dtype=np.float32)
        if motif == "intermixed":
            coords_a = np.column_stack([np.full(8, -0.08, dtype=np.float32), ys])
            coords_b = np.column_stack([np.full(8, 0.08, dtype=np.float32), ys])
        else:
            coords_a = np.column_stack([np.full(8, -0.55, dtype=np.float32), ys])
            coords_b = np.column_stack([np.full(8, 0.55, dtype=np.float32), ys])
        coords = np.vstack([coords_a, coords_b]).astype(np.float32)
        features = np.vstack(
            [
                np.tile(np.array([1.0, 0.0], dtype=np.float32), (8, 1)),
                np.tile(np.array([0.0, 1.0], dtype=np.float32), (8, 1)),
            ]
        )
        return SubregionMeasure(
            subregion_id=int(subregion_id),
            center_um=np.zeros(2, dtype=np.float32),
            members=np.arange(16, dtype=np.int32),
            canonical_coords=coords,
            features=features,
            weights=np.full(16, 1.0 / 16.0, dtype=np.float32),
            geometry_point_count=16,
            compressed_point_count=16,
            normalizer=ShapeNormalizer(
                center=np.zeros((1, 2), dtype=np.float32),
                scale=1.0,
                interpolator=None,
            ),
            normalizer_diagnostics=ShapeNormalizerDiagnostics(
                geometry_source="test",
                used_fallback=False,
                ot_cost=None,
                sinkhorn_converged=None,
                mapped_radius_p95=None,
                mapped_radius_max=None,
                interpolation_residual=None,
            ),
        )

    embeddings, metadata = build_internal_heterogeneity_embeddings(
        [
            make_measure(0, "intermixed"),
            make_measure(1, "segregated"),
            make_measure(2, "intermixed"),
            make_measure(3, "segregated"),
        ],
        codebook_size=2,
        codebook_sample_size=64,
        grid_size=4,
        pair_distance_bins=(0.25, 0.5, 1.0, 2.0),
        pair_graph_mode="radius",
        pair_graph_radius=0.25,
        block_weights={
            "composition": 0.0,
            "diversity": 0.0,
            "spatial_field": 0.0,
            "pair_cooccurrence": 1.0,
        },
        random_state=2030,
    )

    assert metadata["block_weights"]["pair_cooccurrence"] == pytest.approx(1.0)
    assert metadata["pair_graph_mode"] == "radius"
    assert metadata["pair_graph_radius_canonical"] == pytest.approx(0.25)
    assert metadata["pair_cooccurrence_normalization"] == "observed_over_expected"
    assert metadata["pair_bin_normalization"] == "per_bin"
    assert metadata["cell_state_codebook_assignment_entropy_summary"]["count"] == 64
    dist = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=2)
    assert dist[0, 2] < 1e-6
    assert dist[1, 3] < 1e-6
    assert dist[0, 1] > 1e-3
    assert dist[0, 2] < dist[0, 1]
    assert dist[1, 3] < dist[1, 0]


def _fgw_measure(
    coords: np.ndarray,
    features: np.ndarray,
    weights: np.ndarray | None = None,
) -> SubregionFGWMeasure:
    coords64 = np.asarray(coords, dtype=np.float64)
    features64 = np.asarray(features, dtype=np.float64)
    if weights is None:
        weights64 = np.full(coords64.shape[0], 1.0 / coords64.shape[0])
    else:
        weights64 = np.asarray(weights, dtype=np.float64)
        weights64 = weights64 / weights64.sum()
    diff = coords64[:, None, :] - coords64[None, :, :]
    structure = np.sqrt(np.sum(diff * diff, axis=2))
    return SubregionFGWMeasure(
        coords=coords64,
        features=features64,
        weights=weights64,
        structure=structure,
    )


def test_fgw_feature_only_limit_ignores_internal_geometry() -> None:
    features = np.eye(3, dtype=np.float64)
    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    triangle = np.array([[0.0, 0.0], [0.8, 0.6], [0.1, 1.4]], dtype=np.float64)
    left = _fgw_measure(line, features)
    right = _fgw_measure(triangle, features)

    distance, coupling, meta = fgw_distance(
        left,
        right,
        alpha=0.0,
        max_iter=200,
        tol=1e-9,
        return_coupling=True,
    )

    assert meta["mode"] == HETEROGENEITY_FGW_MODE
    assert meta["uses_ot_costs"] is True
    assert distance < 1e-8
    assert coupling is not None
    np.testing.assert_allclose(coupling.sum(axis=1), left.weights, atol=1e-6)
    np.testing.assert_allclose(coupling.sum(axis=0), right.weights, atol=1e-6)


def test_fgw_structure_only_limit_ignores_feature_distribution() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.2, 0.9]], dtype=np.float64)
    features = np.eye(3, dtype=np.float64)
    left = _fgw_measure(coords, features)
    right = _fgw_measure(coords, np.tile(features[[0]], (3, 1)))

    feature_only, _coupling, _meta = fgw_distance(
        left,
        right,
        alpha=0.0,
        max_iter=200,
        tol=1e-9,
    )
    structure_only, coupling, meta = fgw_distance(
        left,
        right,
        alpha=1.0,
        max_iter=200,
        tol=1e-9,
        return_coupling=True,
    )

    assert feature_only > 1e-3
    assert structure_only < 1e-8
    assert coupling is not None
    assert meta["source_marginal_error"] < 1e-6
    assert meta["target_marginal_error"] < 1e-6


def test_hellinger_feature_cost_rejects_signed_whitened_features() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    left = _fgw_measure(coords, np.array([[-1.0, 0.5], [0.2, -0.4]]))
    right = _fgw_measure(coords, np.array([[0.3, -0.1], [-0.5, 0.7]]))
    left = SubregionFGWMeasure(
        coords=left.coords,
        features=left.features,
        weights=left.weights,
        structure=left.structure,
        feature_mode="whitened_features",
        n_whitened_features=2,
    )
    right = SubregionFGWMeasure(
        coords=right.coords,
        features=right.features,
        weights=right.weights,
        structure=right.structure,
        feature_mode="whitened_features",
        n_whitened_features=2,
    )

    with pytest.raises(ValueError, match="Hellinger feature cost requires"):
        feature_cost(left, right, feature_cost_kind="hellinger_codebook")


def test_mixed_feature_mode_requires_split_transport_cost() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    base = _fgw_measure(
        coords,
        np.array([[0.0, 0.0, 1.0, 0.0], [2.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    )
    mixed = SubregionFGWMeasure(
        coords=base.coords,
        features=base.features,
        weights=base.weights,
        structure=base.structure,
        feature_mode="whitened_features_plus_soft_codebook",
        n_whitened_features=2,
        n_codebook_features=2,
    )

    with pytest.raises(ValueError, match="Mixed feature mode requires"):
        feature_cost(mixed, mixed, feature_cost_kind="sqeuclidean")


def test_split_mixed_feature_cost_uses_marker_and_codebook_components() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    base = _fgw_measure(
        coords,
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )
    shifted = _fgw_measure(
        coords,
        np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [2.0, 1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    left = SubregionFGWMeasure(
        coords=base.coords,
        features=base.features,
        weights=base.weights,
        structure=base.structure,
        feature_mode="whitened_features_plus_soft_codebook",
        n_whitened_features=2,
        n_codebook_features=2,
    )
    right = SubregionFGWMeasure(
        coords=shifted.coords,
        features=shifted.features,
        weights=shifted.weights,
        structure=shifted.structure,
        feature_mode="whitened_features_plus_soft_codebook",
        n_whitened_features=2,
        n_codebook_features=2,
    )

    marker_only = feature_cost(
        left,
        right,
        feature_cost_kind="split_marker_codebook",
        marker_feature_weight=1.0,
        codebook_feature_weight=0.0,
        marker_feature_scale=1.0,
        codebook_feature_scale=1.0,
    )
    codebook_only = feature_cost(
        left,
        right,
        feature_cost_kind="split_marker_codebook",
        marker_feature_weight=0.0,
        codebook_feature_weight=1.0,
        marker_feature_scale=1.0,
        codebook_feature_scale=1.0,
    )

    assert marker_only[0, 0] == pytest.approx(1.0)
    assert marker_only[0, 1] == pytest.approx(5.0)
    assert codebook_only[0, 0] > 0.0
    assert codebook_only[0, 1] == pytest.approx(0.0)

    scales, metadata = fit_transport_cost_scales(
        [left, right],
        feature_cost_kind="split_marker_codebook",
    )
    assert scales.marker_feature_scale is not None
    assert scales.codebook_feature_scale is not None
    assert metadata["marker_feature_scale"] == pytest.approx(
        scales.marker_feature_scale
    )
    assert metadata["codebook_feature_scale"] == pytest.approx(
        scales.codebook_feature_scale
    )


def test_balanced_fgw_records_multistart_audit_metadata() -> None:
    coords_left = np.array([[0.0, 0.0], [1.0, 0.0], [0.2, 0.9]], dtype=np.float64)
    coords_right = np.array([[0.0, 0.0], [0.8, 0.1], [0.1, 0.7]], dtype=np.float64)
    features = np.eye(3, dtype=np.float64)
    left = _fgw_measure(coords_left, features)
    right = _fgw_measure(coords_right, features[[1, 0, 2]])

    distance, coupling, meta = fgw_distance(
        left,
        right,
        alpha=0.5,
        feature_cost_kind="sqeuclidean",
        max_iter=80,
        tol=1e-8,
        n_init=3,
        init="outer_product,feature_ot,coordinate_ot",
        return_coupling=True,
    )

    assert np.isfinite(distance)
    assert coupling is not None
    assert meta["distance_family"] == "balanced_fgw"
    assert meta["fgw_n_init"] == 3
    assert set(meta["fgw_objective_by_init"]) == {
        "outer_product",
        "feature_ot",
        "coordinate_ot",
    }
    assert meta["fgw_best_init"] in meta["fgw_objective_by_init"]
    assert meta["fgw_objective_spread"] is not None
    assert meta["source_marginal_error"] < 1e-5
    assert meta["target_marginal_error"] < 1e-5


def test_pairwise_fgw_distance_matrix_summarizes_multistart_instability() -> None:
    coords_left = np.array([[0.0, 0.0], [1.0, 0.0], [0.2, 0.9]], dtype=np.float64)
    coords_right = np.array([[0.0, 0.0], [0.8, 0.1], [0.1, 0.7]], dtype=np.float64)
    features = np.eye(3, dtype=np.float64)
    measures = [
        _fgw_measure(coords_left, features),
        _fgw_measure(coords_right, features[[1, 0, 2]]),
    ]

    distances, metadata = pairwise_transport_distance_matrix(
        measures,
        mode=HETEROGENEITY_FGW_MODE,
        max_subregions=2,
        feature_cost_kind="sqeuclidean",
        fgw_alpha=0.5,
        fgw_max_iter=80,
        fgw_tol=1e-8,
        fgw_n_init=3,
        fgw_init="outer_product,feature_ot,coordinate_ot",
    )

    assert distances.shape == (2, 2)
    assert np.isfinite(distances[0, 1])
    summary = metadata["fgw_multistart_summary"]
    assert summary["n_pairs"] == 1
    assert summary["n_pairs_with_multiple_inits"] == 1
    assert sum(summary["best_init_counts"].values()) == 1
    assert summary["objective_spread_summary"]["count"] == 1


def test_fused_ot_distance_uses_cross_coordinate_cost() -> None:
    features = np.tile(np.array([[1.0, 0.0]], dtype=np.float64), (3, 1))
    left = _fgw_measure(
        np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], dtype=np.float64),
        features,
    )
    right_same = _fgw_measure(
        np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], dtype=np.float64),
        features,
    )
    right_shifted = _fgw_measure(
        np.array([[2.0, 0.0], [2.1, 0.0], [2.2, 0.0]], dtype=np.float64),
        features,
    )

    same_distance, coupling, meta = fused_ot_distance(
        left,
        right_same,
        feature_weight=0.0,
        coordinate_weight=1.0,
        return_coupling=True,
    )
    shifted_distance, _coupling, _meta = fused_ot_distance(
        left,
        right_shifted,
        feature_weight=0.0,
        coordinate_weight=1.0,
    )

    assert meta["mode"] == HETEROGENEITY_FUSED_OT_MODE
    assert same_distance < 1e-8
    assert shifted_distance > 0.1
    assert coupling is not None
    np.testing.assert_allclose(coupling.sum(axis=1), left.weights, atol=1e-6)
    np.testing.assert_allclose(coupling.sum(axis=0), right_same.weights, atol=1e-6)


def test_build_subregion_fgw_measures_and_pairwise_transport_cap() -> None:
    def make_measure(subregion_id: int, offset: float) -> SubregionMeasure:
        coords = np.array(
            [[offset, 0.0], [offset + 0.2, 0.0], [offset, 0.2]],
            dtype=np.float32,
        )
        features = np.eye(3, dtype=np.float32)
        return SubregionMeasure(
            subregion_id=int(subregion_id),
            center_um=np.zeros(2, dtype=np.float32),
            members=np.arange(3, dtype=np.int32),
            canonical_coords=coords,
            features=features,
            weights=np.full(3, 1.0 / 3.0, dtype=np.float32),
            geometry_point_count=3,
            compressed_point_count=3,
            normalizer=ShapeNormalizer(
                center=np.zeros((1, 2), dtype=np.float32),
                scale=1.0,
                interpolator=None,
            ),
            normalizer_diagnostics=ShapeNormalizerDiagnostics(
                geometry_source="test",
                used_fallback=False,
                ot_cost=None,
                sinkhorn_converged=None,
                mapped_radius_p95=None,
                mapped_radius_max=None,
                interpolation_residual=None,
            ),
        )

    transport_measures, metadata = build_subregion_fgw_measures(
        [make_measure(0, 0.0), make_measure(1, 0.5)],
        codebook_size=3,
        codebook_sample_size=6,
        structure_scale="global_median",
        random_state=2031,
    )
    assert metadata["uses_ot_costs"] is True
    assert metadata["structure_metric"] == "canonical_euclidean"
    assert len(transport_measures) == 2

    distances, distance_meta = pairwise_transport_distance_matrix(
        transport_measures,
        mode=HETEROGENEITY_FUSED_OT_MODE,
        max_subregions=2,
        fused_ot_feature_weight=0.5,
        fused_ot_coordinate_weight=0.5,
    )
    assert distances.shape == (2, 2)
    assert np.allclose(np.diag(distances), 0.0)
    assert distance_meta["uses_ot_costs"] is True
    assert distance_meta["n_pairwise_solves"] == 1
    assert distance_meta["transport_cost_scales"]["feature_scale"] > 0
    assert distance_meta["transport_cost_scales"]["coordinate_scale"] > 0

    with pytest.raises(
        ValueError, match="exceeds heterogeneity_transport_max_subregions"
    ):
        pairwise_transport_distance_matrix(
            transport_measures,
            mode=HETEROGENEITY_FGW_MODE,
            max_subregions=1,
        )


def test_transport_cost_scales_are_global_not_pair_local() -> None:
    base_coords = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float64)
    measures = [
        _fgw_measure(base_coords, np.array([[0.0], [1.0]], dtype=np.float64)),
        _fgw_measure(base_coords, np.array([[10.0], [11.0]], dtype=np.float64)),
        _fgw_measure(base_coords, np.array([[100.0], [101.0]], dtype=np.float64)),
    ]

    scales, metadata = fit_transport_cost_scales(
        measures,
        feature_cost_kind="sqeuclidean",
        max_pair_samples=10,
        random_state=99,
    )

    assert scales.feature_scale > 1.0
    assert scales.coordinate_scale > 0.0
    assert metadata["scale_source"] == "sampled_global_median_positive_cost"


def test_fit_multilevel_ot_fgw_mode_uses_precomputed_transport_matrix() -> None:
    rng = np.random.default_rng(2032)
    coords_parts = []
    feature_parts = []
    subregion_members = []
    start = 0
    motifs = ["line", "triangle", "line", "triangle"]
    offsets = [0.0, 4.0, 8.0, 12.0]
    for offset, motif in zip(offsets, motifs, strict=True):
        if motif == "line":
            local = np.column_stack(
                [
                    np.linspace(-0.7, 0.7, 9),
                    rng.normal(0.0, 0.02, size=9),
                ]
            )
        else:
            angles = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)
            local = np.column_stack([0.65 * np.cos(angles), 0.65 * np.sin(angles)])
        coords_parts.append(local + np.array([offset, 0.0]))
        feature_parts.append(np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (9, 1)))
        subregion_members.append(np.arange(start, start + 9, dtype=np.int32))
        start += 9
    coords = np.vstack(coords_parts).astype(np.float32)
    features = np.vstack(feature_parts).astype(np.float32)
    centers = np.vstack(
        [coords[members].mean(axis=0) for members in subregion_members]
    ).astype(np.float32)

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords,
        subregion_members=subregion_members,
        subregion_centers_um=centers,
        n_clusters=2,
        atoms_per_cluster=2,
        radius_um=10.0,
        stride_um=10.0,
        min_cells=6,
        max_subregions=4,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=32,
        compressed_support_size=9,
        align_iters=1,
        allow_reflection=False,
        allow_scale=False,
        allow_convex_hull_fallback=True,
        max_iter=1,
        tol=1e-4,
        basic_niche_size_um=None,
        min_subregions_per_cluster=1,
        compute_spot_latent=False,
        subregion_clustering_method=HETEROGENEITY_FGW_MODE,
        heterogeneity_transport_max_subregions=4,
        heterogeneity_transport_feature_mode="soft_codebook",
        heterogeneity_transport_feature_cost="sqeuclidean",
        heterogeneity_fgw_alpha=1.0,
        heterogeneity_fgw_max_iter=100,
        seed=2032,
        compute_device="cpu",
    )

    metadata = result.subregion_latent_embedding_metadata
    assert result.subregion_clustering_method == HETEROGENEITY_FGW_MODE
    assert result.subregion_clustering_uses_spatial is True
    assert result.subregion_latent_embeddings.shape == (4, 4)
    assert metadata["uses_ot_costs"] is True
    assert metadata["transport_distance"]["mode"] == HETEROGENEITY_FGW_MODE
    assert metadata["transport_distance"]["uses_ot_costs"] is True
    assert "transport_cost_scales" in metadata["transport_distance"]
    assert metadata["transport_clustering"]["algorithm"].startswith("average_linkage")


def test_pooled_subregion_latent_uses_uncompressed_member_feature_distribution() -> (
    None
):
    rng = np.random.default_rng(2027)
    coords_parts = []
    feature_parts = []
    subregion_members = []
    start = 0
    spatial_centers = [
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([100.0, 0.0]),
        np.array([102.0, 0.0]),
    ]
    feature_patterns = [
        np.tile(np.array([[-0.05], [0.05]], dtype=np.float32), (16, 1)),
        np.tile(np.array([[-6.0], [6.0]], dtype=np.float32), (16, 1)),
        np.tile(np.array([[-0.05], [0.05]], dtype=np.float32), (16, 1)),
        np.tile(np.array([[-6.0], [6.0]], dtype=np.float32), (16, 1)),
    ]
    for spatial_center, pattern in zip(spatial_centers, feature_patterns, strict=True):
        coords_parts.append(
            spatial_center + rng.normal(scale=0.1, size=(pattern.shape[0], 2))
        )
        feature_parts.append(pattern)
        subregion_members.append(
            np.arange(start, start + pattern.shape[0], dtype=np.int32)
        )
        start += pattern.shape[0]
    coords = np.vstack(coords_parts).astype(np.float32)
    features = np.vstack(feature_parts).astype(np.float32)
    centers = np.vstack(
        [coords[members].mean(axis=0) for members in subregion_members]
    ).astype(np.float32)

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords,
        subregion_members=subregion_members,
        subregion_centers_um=centers,
        n_clusters=2,
        atoms_per_cluster=2,
        radius_um=10.0,
        stride_um=10.0,
        min_cells=12,
        max_subregions=4,
        lambda_x=100.0,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=32,
        compressed_support_size=2,
        align_iters=1,
        allow_reflection=False,
        allow_scale=False,
        allow_convex_hull_fallback=True,
        max_iter=1,
        tol=1e-4,
        basic_niche_size_um=None,
        min_subregions_per_cluster=2,
        compute_spot_latent=False,
        subregion_clustering_method="pooled_subregion_latent",
        subregion_latent_embedding_mode="mean_std",
        seed=2027,
        compute_device="cpu",
    )

    labels = result.subregion_cluster_labels
    assert result.subregion_clustering_method == "pooled_subregion_latent"
    assert result.subregion_clustering_uses_spatial is False
    assert result.subregion_latent_embeddings.shape == (4, 2)
    standardized_features = (
        (features - features.mean(axis=0, keepdims=True))
        / features.std(axis=0, keepdims=True)
    ).astype(np.float32)
    expected_latents = []
    for members in subregion_members:
        values = standardized_features[members]
        expected_latents.append([float(values.mean()), float(values.std())])
    assert np.allclose(
        result.subregion_latent_embeddings,
        np.asarray(expected_latents, dtype=np.float32),
        atol=1e-5,
    )
    assert (
        result.subregion_latent_embeddings[1, 1]
        > result.subregion_latent_embeddings[0, 1]
    )
    assert labels[0] == labels[2]
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]


def test_subregion_latent_embedding_modes_are_distributional_and_uncertainty_aware() -> (
    None
):
    features = np.asarray(
        [
            [-2.0, 0.0],
            [-1.0, 0.1],
            [2.0, 0.2],
            [4.0, 3.0],
            [4.5, 2.8],
            [9.0, 3.2],
        ],
        dtype=np.float32,
    )
    members = [
        np.asarray([0, 1, 2], dtype=np.int32),
        np.asarray([3, 4, 5], dtype=np.int32),
    ]

    mean_std = _build_subregion_latent_embeddings_from_members(
        features, members, mode="mean_std"
    )
    shrunk = _build_subregion_latent_embeddings_from_members(
        features,
        members,
        mode="mean_std_shrunk",
        shrinkage_tau=100.0,
    )
    sample_ids = np.asarray(["s1", "s1", "s1", "s2", "s2", "s2"], dtype=object)
    sample_shrunk, diagnostics = _build_subregion_latent_embeddings_from_members(
        features,
        members,
        mode="mean_std_shrunk",
        shrinkage_tau=100.0,
        heterogeneity_weight=2.0,
        sample_ids=sample_ids,
        sample_prior_weight=1.0,
        return_diagnostics=True,
    )
    skew_count = _build_subregion_latent_embeddings_from_members(
        features,
        members,
        mode="mean_std_skew_count",
        shrinkage_tau=25.0,
    )
    quantile = _build_subregion_latent_embeddings_from_members(
        features, members, mode="mean_std_quantile"
    )
    codebook = _build_subregion_latent_embeddings_from_members(
        features,
        members,
        mode="codebook_histogram",
        codebook_size=3,
        codebook_sample_size=6,
        random_state=11,
    )
    combined = _build_subregion_latent_embeddings_from_members(
        features,
        members,
        mode="mean_std_codebook",
        codebook_size=3,
        codebook_sample_size=6,
        random_state=11,
    )

    assert mean_std.shape == (2, 4)
    assert shrunk.shape == mean_std.shape
    assert not np.allclose(shrunk, mean_std)
    assert sample_shrunk.shape == mean_std.shape
    assert bool(diagnostics["sample_aware_shrinkage"]) is True
    assert np.allclose(
        diagnostics["shrinkage_alpha"], np.asarray([3 / 103, 3 / 103], dtype=np.float32)
    )
    assert list(diagnostics["sample_ids"]) == ["s1", "s2"]
    assert float(np.max(diagnostics["raw_to_shrunk_distance"])) > 0.0
    assert np.linalg.norm(sample_shrunk[:, 2:]) > np.linalg.norm(shrunk[:, 2:])
    assert skew_count.shape == (2, 8)
    assert quantile.shape == (2, 14)
    assert codebook.shape == (2, 3)
    assert np.allclose(codebook.sum(axis=1), 1.0)
    assert combined.shape == (2, 7)


def test_generated_subregions_use_data_driven_geometry_without_hull_fallback() -> None:
    rng = np.random.default_rng(314)
    coords = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.5, size=(32, 2)),
            rng.normal(loc=[10.0, 10.0], scale=0.5, size=(32, 2)),
        ]
    ).astype(np.float32)
    features = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0, 0.0], scale=0.1, size=(32, 3)),
            rng.normal(loc=[3.0, 3.0, 3.0], scale=0.1, size=(32, 3)),
        ]
    ).astype(np.float32)

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords,
        n_clusters=2,
        atoms_per_cluster=2,
        radius_um=2.0,
        stride_um=4.0,
        min_cells=8,
        max_subregions=8,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=32,
        compressed_support_size=8,
        align_iters=1,
        allow_reflection=False,
        allow_scale=False,
        allow_convex_hull_fallback=False,
        max_iter=2,
        tol=1e-4,
        basic_niche_size_um=None,
        seed=19,
        compute_device="cpu",
    )

    assert np.all(~result.subregion_geometry_used_fallback)
    assert set(result.subregion_geometry_sources) == {"observed_point_cloud"}


def test_basic_niche_subregions_use_data_driven_geometry_without_hull_fallback() -> (
    None
):
    xs, ys = np.meshgrid(np.arange(0.0, 150.0, 10.0), np.arange(0.0, 150.0, 10.0))
    coords = np.column_stack([xs.reshape(-1), ys.reshape(-1)]).astype(np.float32)
    features = np.column_stack([coords[:, 0] / 50.0, coords[:, 1] / 50.0]).astype(
        np.float32
    )

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords,
        n_clusters=2,
        atoms_per_cluster=2,
        radius_um=60.0,
        stride_um=50.0,
        min_cells=5,
        max_subregions=32,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=32,
        compressed_support_size=8,
        align_iters=1,
        allow_reflection=False,
        allow_scale=False,
        allow_convex_hull_fallback=False,
        max_iter=2,
        tol=1e-4,
        basic_niche_size_um=50.0,
        seed=23,
        compute_device="cpu",
    )

    assert np.all(~result.subregion_geometry_used_fallback)
    assert set(result.subregion_geometry_sources) == {"observed_point_cloud"}


def test_shape_normalizer_reduces_boundary_shape_difference() -> None:
    q, w = make_reference_points_unit_disk(128)
    circle = np.stack(
        [
            np.cos(np.linspace(0, 2 * np.pi, 180, endpoint=False)),
            np.sin(np.linspace(0, 2 * np.pi, 180, endpoint=False)),
        ],
        axis=1,
    ).astype(np.float32)
    ellipse = circle.copy()
    ellipse[:, 0] *= 2.5

    phi_circle, _ = fit_ot_shape_normalizer(circle, q, w, eps_geom=0.03)
    phi_ellipse, _ = fit_ot_shape_normalizer(ellipse, q, w, eps_geom=0.03)

    mapped_circle = phi_circle.transform(circle)
    mapped_ellipse = phi_ellipse.transform(ellipse)

    def radial_stats(x: np.ndarray) -> tuple[float, float]:
        r = np.sqrt(np.sum(x**2, axis=1))
        return float(r.mean()), float(r.std())

    mean_circle, std_circle = radial_stats(mapped_circle)
    mean_ellipse, std_ellipse = radial_stats(mapped_ellipse)

    assert abs(mean_circle - mean_ellipse) < 0.15
    assert abs(std_circle - std_ellipse) < 0.15


def test_shape_normalizer_handles_degenerate_geometry_points() -> None:
    q, w = make_reference_points_unit_disk(64)

    single = np.array([[2.0, -3.0]], dtype=np.float32)
    phi_single, diag_single = fit_ot_shape_normalizer(single, q, w, eps_geom=0.03)
    mapped_single = phi_single.transform(single)
    assert phi_single.interpolator is None
    assert mapped_single.shape == (1, 2)
    assert np.allclose(mapped_single, 0.0)
    assert diag_single.ot_cost is None
    assert diag_single.interpolation_residual == 0.0

    pair = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    phi_pair, diag_pair = fit_ot_shape_normalizer(pair, q, w, eps_geom=0.03)
    mapped_pair = phi_pair.transform(pair)
    assert phi_pair.interpolator is None
    assert mapped_pair.shape == (2, 2)
    assert np.allclose(mapped_pair.mean(axis=0), 0.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(mapped_pair[0] - mapped_pair[1]), 2.0, atol=1e-5)
    assert diag_pair.ot_cost is None
    assert diag_pair.interpolation_residual == 0.0


def test_shape_leakage_diagnostic_runs() -> None:
    coords = np.vstack(
        [
            np.random.default_rng(0).normal(loc=[0.0, 0.0], scale=0.2, size=(30, 2)),
            np.random.default_rng(1).normal(
                loc=[3.0, 0.0], scale=[1.2, 0.1], size=(30, 2)
            ),
            np.random.default_rng(2).normal(loc=[0.0, 3.0], scale=0.2, size=(30, 2)),
            np.random.default_rng(3).normal(
                loc=[3.0, 3.0], scale=[1.2, 0.1], size=(30, 2)
            ),
        ]
    ).astype(np.float32)
    members = [
        np.arange(0, 30, dtype=np.int32),
        np.arange(30, 60, dtype=np.int32),
        np.arange(60, 90, dtype=np.int32),
        np.arange(90, 120, dtype=np.int32),
    ]
    labels = np.array([0, 1, 0, 1], dtype=np.int32)
    shape_df = _shape_descriptor_frame(members, coords)
    score = _shape_leakage_balanced_accuracy(shape_df, labels, seed=0)
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_cell_projection_boundary_uses_latent_geometry_not_subregion_centers() -> None:
    features = np.array(
        [
            [0.0, 0.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )
    coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    atom_features = np.array(
        [
            [[0.0, 0.0], [0.1, -0.1]],
            [[10.0, 10.0], [9.9, 10.1]],
        ],
        dtype=np.float32,
    )
    atom_coords = np.array(
        [
            [[0.0, 0.0], [0.1, 0.0]],
            [[1.0, 1.0], [1.1, 1.0]],
        ],
        dtype=np.float32,
    )
    prototype_weights = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    measures = [
        SubregionMeasure(
            subregion_id=0,
            center_um=np.array([0.5, 0.5], dtype=np.float32),
            members=np.array([0, 1], dtype=np.int32),
            canonical_coords=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            features=features.copy(),
            weights=np.array([0.5, 0.5], dtype=np.float32),
            geometry_point_count=2,
            compressed_point_count=2,
            normalizer=ShapeNormalizer(
                center=np.zeros((1, 2)), scale=1.0, interpolator=None
            ),
            normalizer_diagnostics=ShapeNormalizerDiagnostics(
                geometry_source="test",
                used_fallback=False,
                ot_cost=None,
                sinkhorn_converged=None,
                mapped_radius_p95=None,
                mapped_radius_max=None,
                interpolation_residual=None,
            ),
        ),
        SubregionMeasure(
            subregion_id=1,
            center_um=np.array([0.5, 0.5], dtype=np.float32),
            members=np.array([0, 1], dtype=np.int32),
            canonical_coords=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            features=features.copy(),
            weights=np.array([0.5, 0.5], dtype=np.float32),
            geometry_point_count=2,
            compressed_point_count=2,
            normalizer=ShapeNormalizer(
                center=np.zeros((1, 2)), scale=1.0, interpolator=None
            ),
            normalizer_diagnostics=ShapeNormalizerDiagnostics(
                geometry_source="test",
                used_fallback=False,
                ot_cost=None,
                sinkhorn_converged=None,
                mapped_radius_p95=None,
                mapped_radius_max=None,
                interpolation_residual=None,
            ),
        ),
    ]
    subregion_labels = np.array([0, 1], dtype=np.int32)
    transforms = [
        {"R": np.eye(2), "scale": 1.0, "t": np.zeros(2)},
        {"R": np.eye(2), "scale": 1.0, "t": np.zeros(2)},
    ]
    # Both cells are covered by the same two subregions, so the context alone is symmetric.
    subregion_cluster_costs = np.array(
        [
            [0.1, 1.0],
            [1.0, 0.1],
        ],
        dtype=np.float32,
    )

    labels, combined, feature_probs, context_probs = _project_cells_from_subregions(
        features=features,
        coords_um=coords,
        measures=measures,
        subregion_labels=subregion_labels,
        atom_coords=atom_coords,
        atom_features=atom_features,
        prototype_weights=prototype_weights,
        assigned_transforms=transforms,
        subregion_cluster_costs=subregion_cluster_costs,
        lambda_x=0.5,
        lambda_y=1.0,
        cost_scale_x=1.0,
        cost_scale_y=1.0,
        assignment_temperature=0.35,
        context_weight=0.5,
    )

    # Context is symmetric across the two cells because they share the same subregions.
    assert np.allclose(context_probs[0], context_probs[1])
    # The split must therefore come from latent geometry.
    assert labels.tolist() == [0, 1]
    assert feature_probs[0, 0] > feature_probs[0, 1]
    assert feature_probs[1, 1] > feature_probs[1, 0]
    assert np.allclose(combined.sum(axis=1), 1.0)


def test_initialize_cluster_atoms_respects_lambda_scaling() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ],
        dtype=np.float32,
    )
    features = np.array(
        [
            [1.0, 1.0],
            [1.2, 1.0],
            [50.0, 50.0],
            [49.8, 50.1],
        ],
        dtype=np.float32,
    )
    centers = np.array([[0.0, 0.0]], dtype=np.float32)
    regions = [
        RegionGeometry(
            region_id="r0", members=np.arange(coords.shape[0], dtype=np.int32)
        )
    ]
    q, w = make_reference_points_unit_disk(64)
    measures = _build_subregion_measures(
        features=features,
        coords_um=coords,
        centers_um=centers,
        region_geometries=regions,
        geometry_reference_points=q,
        geometry_reference_weights=w,
        geometry_eps=0.03,
        geometry_samples=64,
        compressed_support_size=4,
        lambda_x=0.25,
        lambda_y=4.0,
        seed=0,
        allow_convex_hull_fallback=True,
    )
    atom_coords, atom_features, betas = _initialize_cluster_atoms(
        measures=measures,
        labels=np.array([0], dtype=np.int32),
        n_clusters=1,
        atoms_per_cluster=2,
        lambda_x=0.25,
        lambda_y=4.0,
        random_state=0,
    )
    assert atom_coords.shape == (1, 2, 2)
    assert atom_features.shape == (1, 2, 2)
    assert np.max(np.abs(atom_coords[0])) < 5.0
    assert np.max(atom_features[0]) > 10.0
    assert np.allclose(betas.sum(axis=1), 1.0)


def test_sample_geometry_points_prefers_explicit_polygon_geometry() -> None:
    region = RegionGeometry(
        region_id="poly",
        members=np.arange(10, dtype=np.int32),
        polygon_vertices=np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 4.0],
                [0.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )
    observed = np.column_stack(
        [np.linspace(0.0, 0.5, 10), np.linspace(0.0, 0.5, 10)]
    ).astype(np.float32)
    pts, source, used_fallback = sample_geometry_points(
        region,
        observed_coords=observed,
        n_points=256,
        seed=11,
        allow_convex_hull_fallback=True,
    )
    assert source == "polygon"
    assert not used_fallback
    assert pts[:, 0].max() > 3.0
    assert pts[:, 1].max() > 3.0


def test_build_subregion_measures_handles_singleton_subregions() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ],
        dtype=np.float32,
    )
    features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    centers = coords.copy()
    regions = [
        RegionGeometry(region_id=f"r{i}", members=np.array([i], dtype=np.int32))
        for i in range(coords.shape[0])
    ]
    q, w = make_reference_points_unit_disk(64)

    measures = _build_subregion_measures(
        features=features,
        coords_um=coords,
        centers_um=centers,
        region_geometries=regions,
        geometry_reference_points=q,
        geometry_reference_weights=w,
        geometry_eps=0.03,
        geometry_samples=64,
        compressed_support_size=4,
        lambda_x=0.5,
        lambda_y=1.0,
        seed=0,
        allow_convex_hull_fallback=True,
    )

    assert len(measures) == 3
    for measure in measures:
        assert measure.geometry_point_count == 1
        assert measure.normalizer.interpolator is None
        assert measure.canonical_coords.shape == (1, 2)
        assert np.allclose(measure.canonical_coords, 0.0)


def test_returned_costs_match_returned_atoms() -> None:
    rng = np.random.default_rng(7)
    coords = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.6, size=(40, 2)),
            rng.normal(loc=[15.0, 0.0], scale=0.6, size=(40, 2)),
            rng.normal(loc=[0.0, 15.0], scale=0.6, size=(40, 2)),
        ]
    ).astype(np.float32)
    features = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0, 0.0], scale=0.15, size=(40, 3)),
            rng.normal(loc=[4.0, 4.0, 4.0], scale=0.15, size=(40, 3)),
            rng.normal(loc=[0.0, 4.0, 2.0], scale=0.15, size=(40, 3)),
        ]
    ).astype(np.float32)

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords,
        n_clusters=3,
        atoms_per_cluster=3,
        radius_um=10.0,
        stride_um=12.0,
        min_cells=15,
        max_subregions=20,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=64,
        compressed_support_size=16,
        align_iters=2,
        allow_reflection=False,
        allow_scale=False,
        allow_convex_hull_fallback=True,
        max_iter=3,
        tol=1e-4,
        basic_niche_size_um=None,
        subregion_clustering_method="ot_dictionary",
        seed=7,
        compute_device="cpu",
    )
    q, w = make_reference_points_unit_disk(64)
    rebuilt_centers, rebuilt_members, _, _, _ = (
        build_partition_subregions_from_grid_tiles(
            coords_um=coords,
            radius_um=10.0,
            stride_um=12.0,
            min_cells=15,
            max_subregions=20,
            partition_features=_standardize_features(features),
            seed=7,
        )
    )
    assert np.allclose(rebuilt_centers, result.subregion_centers_um)
    assert all(
        np.array_equal(a, b)
        for a, b in zip(rebuilt_members, result.subregion_members, strict=False)
    )
    regions = _region_geometries_from_observed_points(result.subregion_members)
    measures = _build_subregion_measures(
        features=_standardize_features(features),
        coords_um=coords,
        centers_um=result.subregion_centers_um,
        region_geometries=regions,
        geometry_reference_points=q,
        geometry_reference_weights=w,
        geometry_eps=0.03,
        geometry_samples=64,
        compressed_support_size=16,
        lambda_x=0.5,
        lambda_y=1.0,
        seed=7,
        allow_convex_hull_fallback=True,
    )
    recomputed, effective_eps_matrix, used_fallback_matrix = _compute_assignment_costs(
        measures=measures,
        atom_coords=result.cluster_atom_coords,
        atom_features=result.cluster_atom_features,
        betas=result.cluster_prototype_weights,
        lambda_x=0.5,
        lambda_y=1.0,
        eps=0.03,
        rho=0.5,
        align_iters=2,
        allow_reflection=False,
        allow_scale=False,
        cost_scale_x=result.cost_scale_x,
        cost_scale_y=result.cost_scale_y,
        min_scale=0.75,
        max_scale=1.33,
        scale_penalty=0.05,
        shift_penalty=0.05,
        compute_device=torch.device("cpu"),
        return_diagnostics=True,
    )
    recomputed, effective_eps_matrix, used_fallback_matrix = (
        _stabilize_mixed_candidate_assignment_costs(
            measures=measures,
            atom_coords=result.cluster_atom_coords,
            atom_features=result.cluster_atom_features,
            betas=result.cluster_prototype_weights,
            transport_costs=recomputed,
            candidate_effective_eps_matrix=effective_eps_matrix,
            candidate_used_fallback_matrix=used_fallback_matrix,
            lambda_x=0.5,
            lambda_y=1.0,
            ot_eps=0.03,
            rho=0.5,
            align_iters=2,
            allow_reflection=False,
            allow_scale=False,
            cost_scale_x=result.cost_scale_x,
            cost_scale_y=result.cost_scale_y,
            min_scale=0.75,
            max_scale=1.33,
            scale_penalty=0.05,
            shift_penalty=0.05,
            compute_device=torch.device("cpu"),
        )
    )
    assert recomputed.shape == result.subregion_cluster_costs.shape
    assert effective_eps_matrix.shape == recomputed.shape
    assert used_fallback_matrix.shape == recomputed.shape
    assert np.array_equal(
        result.subregion_cluster_labels, result.subregion_cluster_costs.argmin(axis=1)
    )
    assert np.allclose(recomputed, result.subregion_cluster_costs, atol=2e-3)
    assert result.subregion_assigned_used_ot_fallback.shape[0] == len(
        result.subregion_members
    )
    assert np.allclose(
        effective_eps_matrix, result.subregion_candidate_effective_eps_matrix
    )
    assert np.array_equal(
        used_fallback_matrix, result.subregion_candidate_used_ot_fallback_matrix
    )


def test_empty_mask_geometry_raises() -> None:
    region = RegionGeometry(
        region_id="empty",
        members=np.arange(5, dtype=np.int32),
        mask=np.zeros((8, 8), dtype=bool),
    )
    observed = np.random.default_rng(0).normal(size=(5, 2)).astype(np.float32)
    try:
        sample_geometry_points(
            region,
            observed_coords=observed,
            n_points=32,
            seed=0,
            allow_convex_hull_fallback=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected empty mask geometry to raise")


def test_fit_requires_explicit_fallback_flag_for_observed_hull_geometry() -> None:
    coords = np.array(
        [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [5.0, 5.0], [5.5, 5.0], [5.0, 5.5]],
        dtype=np.float32,
    )
    features = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], dtype=np.float32)
    subregion_members = [
        np.array([0, 1, 2], dtype=np.int32),
        np.array([3, 4, 5], dtype=np.int32),
    ]
    subregion_centers = np.vstack(
        [coords[members].mean(axis=0) for members in subregion_members]
    ).astype(np.float32)
    try:
        fit_multilevel_ot(
            features=features,
            coords_um=coords,
            subregion_members=subregion_members,
            subregion_centers_um=subregion_centers,
            n_clusters=2,
            atoms_per_cluster=1,
            radius_um=1.0,
            stride_um=5.0,
            min_cells=3,
            max_subregions=4,
            lambda_x=0.5,
            lambda_y=1.0,
            geometry_eps=0.03,
            ot_eps=0.03,
            rho=0.5,
            geometry_samples=32,
            compressed_support_size=3,
            align_iters=1,
            allow_reflection=False,
            allow_scale=False,
            max_iter=1,
            tol=1e-4,
            basic_niche_size_um=None,
            seed=0,
            compute_device="cpu",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected fit_multilevel_ot to require an explicit hull-fallback opt-in"
        )


def test_explicit_subregions_respect_min_cells_filter() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.0, 0.2],
            [3.0, 3.0],
            [3.2, 3.0],
            [3.0, 3.2],
            [6.0, 6.0],
            [6.1, 6.0],
        ],
        dtype=np.float32,
    )
    features = np.array(
        [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [2.0], [2.0]], dtype=np.float32
    )
    subregion_members = [
        np.array([0, 1, 2], dtype=np.int32),
        np.array([3, 4, 5], dtype=np.int32),
        np.array([6, 7], dtype=np.int32),
    ]
    result = fit_multilevel_ot(
        features=features,
        coords_um=coords,
        subregion_members=subregion_members,
        n_clusters=2,
        atoms_per_cluster=1,
        radius_um=1.0,
        stride_um=1.0,
        min_cells=3,
        max_subregions=0,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=32,
        compressed_support_size=2,
        align_iters=1,
        allow_convex_hull_fallback=True,
        max_iter=1,
        tol=1e-4,
        n_init=1,
        basic_niche_size_um=None,
        seed=12,
        compute_device="cpu",
    )
    assert len(result.subregion_members) == 2
    assert all(len(members) >= 3 for members in result.subregion_members)


def test_fit_rejects_mismatched_explicit_subregion_centers() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.0, 0.2],
            [3.0, 3.0],
            [3.2, 3.0],
            [3.0, 3.2],
        ],
        dtype=np.float32,
    )
    features = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], dtype=np.float32)
    subregion_members = [
        np.array([0, 1, 2], dtype=np.int32),
        np.array([3, 4, 5], dtype=np.int32),
    ]
    try:
        fit_multilevel_ot(
            features=features,
            coords_um=coords,
            subregion_members=subregion_members,
            subregion_centers_um=np.array([[0.0, 0.0]], dtype=np.float32),
            n_clusters=2,
            atoms_per_cluster=1,
            radius_um=1.0,
            stride_um=1.0,
            min_cells=3,
            max_subregions=0,
            lambda_x=0.5,
            lambda_y=1.0,
            geometry_eps=0.03,
            ot_eps=0.03,
            rho=0.5,
            geometry_samples=32,
            compressed_support_size=2,
            align_iters=1,
            allow_convex_hull_fallback=True,
            max_iter=1,
            tol=1e-4,
            n_init=1,
            basic_niche_size_um=None,
            seed=12,
            compute_device="cpu",
        )
    except ValueError as exc:
        assert "subregion_centers_um" in str(exc)
    else:
        raise AssertionError("Expected mismatched explicit centers to be rejected")


def test_shape_descriptor_frame_rejects_misaligned_geometries() -> None:
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [6.0, 5.0], [5.0, 6.0]],
        dtype=np.float32,
    )
    members = [np.array([0, 1, 2], dtype=np.int32), np.array([3, 4, 5], dtype=np.int32)]
    geometries = [
        RegionGeometry(
            region_id="r0",
            members=members[0],
            polygon_vertices=np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
            ),
        )
    ]
    try:
        _shape_descriptor_frame(members, coords, region_geometries=geometries)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected misaligned region geometries to be rejected")


def test_fit_rejects_degenerate_zero_geometry_and_feature_weights() -> None:
    coords = np.array(
        [[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]], dtype=np.float32
    )
    features = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
    try:
        fit_multilevel_ot(
            features=features,
            coords_um=coords,
            n_clusters=2,
            atoms_per_cluster=1,
            radius_um=1.0,
            stride_um=1.0,
            min_cells=2,
            max_subregions=0,
            lambda_x=0.0,
            lambda_y=0.0,
            geometry_eps=0.03,
            ot_eps=0.03,
            rho=0.5,
            geometry_samples=32,
            compressed_support_size=2,
            align_iters=1,
            allow_convex_hull_fallback=True,
            max_iter=1,
            tol=1e-4,
            n_init=1,
            basic_niche_size_um=None,
            seed=0,
            compute_device="cpu",
        )
    except ValueError as exc:
        assert "lambda_x or lambda_y" in str(exc)
    else:
        raise AssertionError("Expected zero lambda_x/lambda_y to be rejected")


def test_multilevel_ot_cuda_smoke_if_available() -> None:
    if not torch.cuda.is_available():
        return
    rng = np.random.default_rng(21)
    coords = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.3, size=(12, 2)),
            rng.normal(loc=[5.0, 5.0], scale=0.3, size=(12, 2)),
        ]
    ).astype(np.float32)
    features = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.1, size=(12, 2)),
            rng.normal(loc=[3.0, 3.0], scale=0.1, size=(12, 2)),
        ]
    ).astype(np.float32)
    subregion_members = [
        np.arange(0, 12, dtype=np.int32),
        np.arange(12, 24, dtype=np.int32),
    ]
    subregion_centers = np.vstack(
        [coords[m].mean(axis=0) for m in subregion_members]
    ).astype(np.float32)
    result = fit_multilevel_ot(
        features=features,
        coords_um=coords,
        subregion_members=subregion_members,
        subregion_centers_um=subregion_centers,
        n_clusters=2,
        atoms_per_cluster=2,
        radius_um=2.0,
        stride_um=4.0,
        min_cells=6,
        max_subregions=6,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=32,
        compressed_support_size=6,
        align_iters=1,
        allow_reflection=False,
        allow_scale=False,
        allow_convex_hull_fallback=True,
        max_iter=2,
        tol=1e-4,
        basic_niche_size_um=None,
        seed=21,
        compute_device="cuda",
    )
    assert result.subregion_cluster_costs.shape[1] == 2
    assert result.cell_cluster_probs.shape[1] == 2


def test_weighted_similarity_fit_no_reflection_keeps_positive_determinant() -> None:
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    y = np.array([[0.0, 0.0], [-1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    w = np.array([0.2, 0.4, 0.4], dtype=np.float64)
    transform = weighted_similarity_fit(
        x, y, w, allow_reflection=False, allow_scale=True
    )
    assert np.linalg.det(transform["R"]) > 0
    assert transform["scale"] > 0


def test_ensure_nonempty_clusters_returns_forced_mask() -> None:
    labels = np.array([0, 0, 0, 1], dtype=np.int32)
    costs = np.array(
        [
            [0.1, 1.0, 0.9],
            [0.2, 1.1, 0.8],
            [0.3, 0.9, 0.4],
            [1.0, 0.1, 0.6],
        ],
        dtype=np.float32,
    )
    repaired, forced = _ensure_nonempty_clusters(labels, costs, n_clusters=3)
    assert set(repaired.tolist()) == {0, 1, 2}
    assert forced.sum() == 1


def test_ensure_minimum_cluster_size_counts_subregions_not_cells_or_spots() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
    costs = np.array(
        [
            [0.0, 5.0, 5.0],
            [0.1, 5.0, 5.0],
            [0.2, 5.0, 0.3],
            [0.3, 5.0, 0.2],
            [5.0, 0.0, 5.0],
            [5.0, 0.1, 5.0],
        ],
        dtype=np.float32,
    )

    repaired, forced = _ensure_minimum_cluster_size(
        labels,
        costs,
        n_clusters=3,
        min_subregions_per_cluster=2,
    )

    counts = np.bincount(repaired, minlength=3)
    assert counts.tolist() == [2, 2, 2]
    assert forced.sum() == 2


def test_cluster_count_dict_keeps_missing_cluster_as_zero() -> None:
    observed = _cluster_count_dict(np.array([0, 0, 2], dtype=np.int32), n_clusters=4)

    assert observed == {"C0": 2, "C1": 0, "C2": 1, "C3": 0}


def test_cluster_count_dict_rejects_out_of_range_labels() -> None:
    with pytest.raises(ValueError, match=r"\[0, n_clusters\)"):
        _cluster_count_dict(np.array([0, 3], dtype=np.int32), n_clusters=3)
