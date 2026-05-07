from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
import pytest

from spatial_ot.pairwise_niche import (
    assign_high_contrast_colors,
    build_instance_neighbor_indices,
    build_local_measures,
    cluster_from_distance,
    compute_pairwise_ot_distance_matrix,
    estimate_pairwise_fgw_work,
    estimate_pairwise_ot_work,
    fit_expression_embedding,
    load_expression_embedding_state,
    run_pairwise_niche_on_h5ad,
    save_expression_embedding_state,
)
from spatial_ot.pairwise_niche.cluster import ot_knn_affinity
from spatial_ot.pairwise_niche.fgw import fused_gromov_wasserstein_block
from spatial_ot.pairwise_niche.local_measure import _cap_neighbors
from spatial_ot.cli import _parse_int_list


def _toy_cohort(n_per_sample: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    x_a = rng.normal(loc=0.0, scale=0.2, size=(n_per_sample, 5))
    x_b = rng.normal(loc=2.0, scale=0.2, size=(n_per_sample, 5))
    features = np.vstack([x_a, x_b]).astype(np.float32)
    coords_a = np.column_stack(
        [np.arange(n_per_sample, dtype=np.float32), np.zeros(n_per_sample, dtype=np.float32)]
    )
    coords_b = np.column_stack(
        [
            np.arange(n_per_sample, dtype=np.float32),
            np.full(n_per_sample, 100.0, dtype=np.float32),
        ]
    )
    coords = np.vstack([coords_a, coords_b]).astype(np.float32)
    samples = np.asarray(["A"] * n_per_sample + ["B"] * n_per_sample, dtype=object)
    return features, coords, samples


def test_expression_embedding_does_not_use_spatial_coordinates() -> None:
    features, _, _ = _toy_cohort()
    first = fit_expression_embedding(features, method="pca", embedding_dim=3, random_state=7)
    second = fit_expression_embedding(features, method="pca", embedding_dim=3, random_state=7)
    assert first.values.shape == (12, 3)
    assert not bool(first.metadata["uses_spatial_coordinates"])
    np.testing.assert_allclose(first.values, second.values)
    np.testing.assert_allclose(first.state.transform(features), first.values, atol=1e-5)


def test_expression_embedding_state_save_load_round_trip(tmp_path) -> None:
    features, _, _ = _toy_cohort()
    embedding = fit_expression_embedding(features, method="pca", embedding_dim=3, random_state=7)
    path = tmp_path / "expression_state.npz"
    save_expression_embedding_state(embedding.state, path)
    loaded = load_expression_embedding_state(path)
    np.testing.assert_allclose(loaded.transform(features), embedding.values, atol=1e-5)


def test_precomputed_embedding_standardization_can_be_disabled() -> None:
    features, _, _ = _toy_cohort()
    unchanged = fit_expression_embedding(
        features,
        method="precomputed",
        standardize_precomputed=False,
    )
    standardized = fit_expression_embedding(
        features,
        method="precomputed",
        standardize_precomputed=True,
    )
    np.testing.assert_allclose(unchanged.values, features)
    assert unchanged.metadata["standardization"] == "none"
    assert standardized.metadata["standardization"] == "mean_std"
    assert not np.allclose(standardized.values, features)


def test_local_measures_do_not_cross_samples_and_preserve_rare_state() -> None:
    features, coords, samples = _toy_cohort(n_per_sample=8)
    embedding = fit_expression_embedding(features, method="pca", embedding_dim=3)
    measures = build_local_measures(
        expression_embedding=embedding.values,
        coords_um=coords,
        sample_ids=samples,
        radius_um=10.0,
        max_neighbors=3,
        include_anchor=True,
        cap_mode="radial_shell_state",
        cap_state_clusters=3,
        seed=11,
    )
    assert measures.tokens.shape[0] == features.shape[0]
    assert np.all(measures.full_neighbor_counts >= measures.retained_neighbor_counts)
    assert measures.metadata["ground_cost_normalization"] == "sampled_median"
    scales = measures.metadata["ground_cost_component_scales"]
    assert all(float(value) > 0 for value in scales.values())
    for row, ids in enumerate(measures.neighbor_indices):
        valid = ids[ids >= 0]
        assert set(samples[valid]) == {samples[row]}


def test_radial_shell_state_cap_samples_high_mass_strata_when_overfull() -> None:
    local_indices = np.arange(12, dtype=np.int64)
    distances = np.linspace(0.1, 0.9, 12, dtype=np.float32)
    weights = np.ones(12, dtype=np.float32)
    weights[-1] = 1000.0
    state_labels = np.arange(12, dtype=np.int32)
    chosen = _cap_neighbors(
        local_indices=local_indices,
        distances=distances,
        weights=weights,
        state_labels=state_labels,
        radius_um=1.0,
        max_neighbors=3,
        radial_shells=1,
        cap_mode="radial_shell_state",
        rng=np.random.default_rng(1337),
    )
    assert 11 in local_indices[chosen]


def test_local_measure_fgw_structure_modes_are_recorded() -> None:
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    coords = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    samples = np.asarray(["A", "A", "A", "A"], dtype=object)
    complete = build_local_measures(
        expression_embedding=features,
        coords_um=coords,
        sample_ids=samples,
        radius_um=2.0,
        max_neighbors=3,
        include_anchor=True,
        fgw_structure_mode="complete_euclidean",
    )
    topology = build_local_measures(
        expression_embedding=features,
        coords_um=coords,
        sample_ids=samples,
        radius_um=2.0,
        max_neighbors=3,
        include_anchor=True,
        fgw_structure_mode="local_knn_shortest_path",
        fgw_structure_knn=1,
    )
    assert complete.metadata["uses_graph_topology_structure"] is False
    assert topology.metadata["uses_graph_topology_structure"] is True
    assert topology.metadata["fgw_structure_mode"] == "local_knn_shortest_path"
    assert not np.allclose(complete.structure_matrices, topology.structure_matrices)


def test_local_measure_rejects_invalid_radius_and_neighbor_cap() -> None:
    features, coords, samples = _toy_cohort(n_per_sample=2)
    with pytest.raises(ValueError, match="radius_um"):
        build_local_measures(
            expression_embedding=features,
            coords_um=coords,
            sample_ids=samples,
            radius_um=0.0,
        )
    with pytest.raises(ValueError, match="max_neighbors"):
        build_instance_neighbor_indices(
            coords_um=coords,
            sample_ids=samples,
            radius_um=1.0,
            max_neighbors=0,
        )


def test_isolated_no_anchor_uses_zero_dummy_not_anchor_expression() -> None:
    features = np.asarray([[10.0, 0.0], [0.0, 10.0]], dtype=np.float32)
    coords = np.asarray([[0.0, 0.0], [100.0, 0.0]], dtype=np.float32)
    samples = np.asarray(["A", "A"], dtype=object)
    measures = build_local_measures(
        expression_embedding=features,
        coords_um=coords,
        sample_ids=samples,
        radius_um=1.0,
        max_neighbors=1,
        include_anchor=False,
        isolated_policy="zero_dummy",
        seed=19,
    )
    assert np.all(measures.full_neighbor_counts == 0)
    assert np.all(measures.retained_neighbor_counts == 0)
    assert np.all(measures.neighbor_indices[:, 0] == -1)
    np.testing.assert_allclose(measures.tokens[:, 0, :], 0.0)
    np.testing.assert_allclose(measures.weights[:, 0], 1.0)

    fallback = build_local_measures(
        expression_embedding=features,
        coords_um=coords,
        sample_ids=samples,
        radius_um=1.0,
        max_neighbors=1,
        include_anchor=False,
        isolated_policy="anchor_fallback",
        seed=19,
    )
    assert np.all(fallback.neighbor_indices[:, 0] == np.arange(2))
    assert not np.allclose(fallback.tokens[:, 0, :], 0.0)


def test_pairwise_ot_matrix_is_symmetric_with_zero_diagonal() -> None:
    features, coords, samples = _toy_cohort(n_per_sample=4)
    embedding = fit_expression_embedding(features, method="pca", embedding_dim=3)
    measures = build_local_measures(
        expression_embedding=embedding.values,
        coords_um=coords,
        sample_ids=samples,
        radius_um=3.0,
        max_neighbors=2,
        include_anchor=True,
        seed=13,
    )
    distance, metadata = compute_pairwise_ot_distance_matrix(
        measures=measures,
        anchor_embedding=embedding.values,
        block_size=3,
        device="cpu",
        epsilon=0.05,
        n_iters=5,
        max_exact_cells=20,
    )
    assert metadata["distance_mode"] == "debiased_entropic_transport"
    assert metadata["returns_plan_transport_cost_only"]
    assert distance.shape == (8, 8)
    np.testing.assert_allclose(distance, distance.T, atol=1e-5)
    np.testing.assert_allclose(np.diag(distance), 0.0)
    assert np.isfinite(distance).all()


def test_local_measure_structure_matrix_and_fgw_distance() -> None:
    features, coords, samples = _toy_cohort(n_per_sample=4)
    embedding = fit_expression_embedding(features, method="pca", embedding_dim=3)
    measures = build_local_measures(
        expression_embedding=embedding.values,
        coords_um=coords,
        sample_ids=samples,
        radius_um=3.0,
        max_neighbors=2,
        include_anchor=True,
        seed=13,
    )
    assert measures.structure_matrices is not None
    assert measures.structure_matrices.shape == (8, 3, 3)
    np.testing.assert_allclose(
        measures.structure_matrices,
        np.swapaxes(measures.structure_matrices, 1, 2),
        atol=1e-6,
    )
    assert measures.metadata["max_radius_um"] == 3.0
    assert measures.metadata["max_neighbors_included"] == 2

    import torch

    a = torch.as_tensor(measures.tokens[:2], dtype=torch.float32)
    c = torch.as_tensor(measures.structure_matrices[:2], dtype=torch.float32)
    w = torch.as_tensor(measures.weights[:2], dtype=torch.float32)
    d = fused_gromov_wasserstein_block(
        a,
        c,
        w,
        a,
        c,
        w,
        alpha=0.5,
        epsilon=0.05,
        sinkhorn_iters=3,
        fgw_iters=2,
    )
    assert d.shape == (2, 2)
    assert torch.isfinite(d).all()


def test_pairwise_fgw_matrix_uses_graph_structure_and_scaled_features() -> None:
    features, coords, samples = _toy_cohort(n_per_sample=4)
    embedding = fit_expression_embedding(features, method="pca", embedding_dim=3)
    measures = build_local_measures(
        expression_embedding=embedding.values,
        coords_um=coords,
        sample_ids=samples,
        radius_um=3.0,
        max_neighbors=2,
        include_anchor=True,
        seed=13,
    )
    distance, metadata = compute_pairwise_ot_distance_matrix(
        measures=measures,
        anchor_embedding=embedding.values,
        block_size=3,
        device="cpu",
        epsilon=0.05,
        n_iters=3,
        distance_mode="fused_gromov_wasserstein",
        fgw_alpha=0.6,
        fgw_iters=2,
        max_exact_cells=20,
    )
    assert metadata["distance_mode"] == "fused_gromov_wasserstein"
    assert metadata["uses_graph_topology"] is True
    assert metadata["uses_complete_spatial_structure"] is False
    assert metadata["fgw_structure_mode"] == "local_knn_shortest_path"
    assert metadata["fgw_node_feature_mode"] == "expression_only"
    assert metadata["fgw_node_feature_dim"] == 3
    assert metadata["fgw_structure_normalization"] == "sampled_median"
    assert float(metadata["fgw_structure_cost_scale"]) > 0.0
    assert metadata["fgw_alpha"] == 0.6
    assert distance.shape == (8, 8)
    np.testing.assert_allclose(distance, distance.T, atol=1e-5)
    np.testing.assert_allclose(np.diag(distance), 0.0)
    assert np.isfinite(distance).all()


def test_pairwise_ot_rejects_unbounded_or_unknown_distance_modes() -> None:
    features, coords, samples = _toy_cohort(n_per_sample=3)
    embedding = fit_expression_embedding(features, method="pca", embedding_dim=2)
    measures = build_local_measures(
        expression_embedding=embedding.values,
        coords_um=coords,
        sample_ids=samples,
        radius_um=3.0,
        max_neighbors=2,
        include_anchor=True,
    )
    with pytest.raises(ValueError, match="distance_mode"):
        compute_pairwise_ot_distance_matrix(
            measures=measures,
            anchor_embedding=embedding.values,
            distance_mode="not_a_metric",
            max_exact_cells=20,
        )
    with pytest.raises(ValueError, match="Exact all-pairs OT requested"):
        compute_pairwise_ot_distance_matrix(
            measures=measures,
            anchor_embedding=embedding.values,
            max_exact_cells=2,
        )
    with pytest.raises(ValueError, match="work estimate"):
        compute_pairwise_ot_distance_matrix(
            measures=measures,
            anchor_embedding=embedding.values,
            max_exact_cells=20,
            max_ot_work_units=1.0,
        )
    with pytest.raises(ValueError, match="FGW work estimate"):
        compute_pairwise_ot_distance_matrix(
            measures=measures,
            anchor_embedding=embedding.values,
            distance_mode="fused_gromov_wasserstein",
            max_exact_cells=20,
            max_ot_work_units=1e12,
            max_fgw_work_units=1.0,
        )
    estimate = estimate_pairwise_ot_work(n_cells=6, support_size=3, sinkhorn_iters=5)
    assert estimate["n_pairs"] == 21.0
    assert estimate["work_units"] == 21.0 * 3.0 * 3.0 * 5.0
    fgw_estimate = estimate_pairwise_fgw_work(
        n_cells=6,
        support_size=3,
        sinkhorn_iters=5,
        fgw_iters=2,
    )
    assert fgw_estimate["fgw_structure_units"] == 21.0 * 2.0 * 27.0
    assert fgw_estimate["fgw_sinkhorn_units"] == 21.0 * 2.0 * 9.0 * 5.0


def test_clustering_uses_precomputed_distance_not_kmeans() -> None:
    distance = np.asarray(
        [
            [0.0, 0.1, 4.0, 4.2],
            [0.1, 0.0, 4.1, 4.0],
            [4.0, 4.1, 0.0, 0.2],
            [4.2, 4.0, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    result = cluster_from_distance(distance, method="agglomerative", n_clusters=2)
    assert set(result.labels[:2]) != set(result.labels[2:])
    assert result.metadata["assignment_score_type"] == "precomputed_distance_margin"
    affinity = ot_knn_affinity(distance, k=1, scaling="local")
    global_affinity = ot_knn_affinity(distance, k=1, scaling="global")
    np.testing.assert_allclose(affinity.toarray(), affinity.toarray().T)
    np.testing.assert_allclose(global_affinity.toarray(), global_affinity.toarray().T)


def test_cluster_model_selection_chooses_candidate_k() -> None:
    distance = np.asarray(
        [
            [0.0, 0.1, 0.2, 5.0, 5.1, 5.2],
            [0.1, 0.0, 0.2, 5.1, 5.0, 5.1],
            [0.2, 0.2, 0.0, 5.2, 5.1, 5.0],
            [5.0, 5.1, 5.2, 0.0, 0.1, 0.2],
            [5.1, 5.0, 5.1, 0.1, 0.0, 0.2],
            [5.2, 5.1, 5.0, 0.2, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    result = cluster_from_distance(
        distance,
        method="agglomerative",
        candidate_n_clusters=(2, 3),
    )
    assert result.metadata["model_selection"]["selected_n_clusters"] == 2
    assert result.metadata["model_selection"]["criterion"] == "rank_ensemble"
    assert result.metadata["model_selection"]["metrics"] == [
        "silhouette",
        "pseudo_calinski_harabasz",
        "medoid_davies_bouldin",
        "percentile_dunn",
    ]
    first_result = result.metadata["model_selection"]["results"][0]
    assert set(first_result["scores"]) == {
        "silhouette",
        "pseudo_calinski_harabasz",
        "medoid_davies_bouldin",
        "percentile_dunn",
    }
    assert set(first_result["ranks"]) == {
        "silhouette",
        "pseudo_calinski_harabasz",
        "medoid_davies_bouldin",
        "percentile_dunn",
    }
    assert first_result["cluster_size_summary"]["min_cluster_size"] >= 2
    assert first_result["within_between_distance_ratio"] is not None
    assert result.metadata["candidate_n_clusters"] == [2, 3]
    assert result.metadata["within_between_distance_ratio"] is not None
    assert int(np.unique(result.labels).size) == 2


def test_singleton_cluster_assignment_score_is_zero() -> None:
    distance = np.asarray(
        [
            [0.0, 0.1, 5.0],
            [0.1, 0.0, 5.1],
            [5.0, 5.1, 0.0],
        ],
        dtype=np.float32,
    )
    result = cluster_from_distance(distance, method="agglomerative", n_clusters=2)
    singleton_labels = [
        int(label)
        for label in np.unique(result.labels)
        if np.sum(result.labels == int(label)) == 1
    ]
    assert singleton_labels
    singleton_rows = result.labels == singleton_labels[0]
    assert np.all(result.assignment_score[singleton_rows] == 0.0)


def test_candidate_cluster_range_parser_is_inclusive() -> None:
    assert _parse_int_list("5:8") == (5, 6, 7, 8)
    assert _parse_int_list("8-5") == (8, 7, 6, 5)
    assert _parse_int_list("5:7,10") == (5, 6, 7, 10)


def test_high_contrast_colors_are_deterministic_and_keep_on12_orange() -> None:
    colors = assign_high_contrast_colors([f"ON{idx}" for idx in range(15)])
    assert colors["ON12"] == "#ff7f00"
    assert len(set(colors.values())) == 15
    assert colors == assign_high_contrast_colors(reversed([f"ON{idx}" for idx in range(15)]))


def test_pairwise_niche_h5ad_end_to_end(tmp_path) -> None:
    features, coords, samples = _toy_cohort(n_per_sample=5)
    adata = ad.AnnData(X=features)
    adata.obsm["X_expr"] = features
    adata.obs = pd.DataFrame(
        {
            "sample_id": samples,
            "x": coords[:, 0],
            "y": coords[:, 1],
        },
        index=[f"cell_{idx}" for idx in range(features.shape[0])],
    )
    input_path = tmp_path / "input.h5ad"
    output_dir = tmp_path / "pairwise"
    adata.write_h5ad(input_path)

    summary = run_pairwise_niche_on_h5ad(
        input_h5ad=input_path,
        output_dir=output_dir,
        feature_obsm_key="X_expr",
        spatial_x_key="x",
        spatial_y_key="y",
        sample_obs_key="sample_id",
        embedding_method="pca",
        embedding_dim=3,
        radius_um=3.0,
        max_neighbors=2,
        block_size=3,
        sinkhorn_iters=5,
        device="cpu",
        max_exact_cells=20,
        cluster_method="agglomerative",
        candidate_n_clusters=(2, 3),
        seed=17,
    )
    assert summary["active_path"] == "pairwise-niche"
    out = ad.read_h5ad(output_dir / "cells_pairwise_niche.h5ad")
    assert "X_gene_cohort" in out.obsm
    assert "cell_ot_dissimilarity" in out.obsp
    assert "cell_ot_affinity" in out.obsp
    assert "ot_niche" in out.obs
    assert "ot_niche_colors" in out.uns
    assert (output_dir / "ot_niche_colors.json").exists()
    assert "ot_niche_instance" in out.obs
    assert "n_neighbors_full_r3" in out.obs
    assert "neighbor_retention_fraction_r3" in out.obs
    assert (output_dir / "pairwise_niche_model" / "expression_embedding_state.npz").exists()
    assert summary["batch_embedding"]["batch_correction_applied_by_pairwise_niche"] is False
    assert summary["method_semantics"]["anchor_expression_enters_twice"] is False
    assert summary["clustering"]["model_selection"]["selected_n_clusters"] in {2, 3}
    np.testing.assert_allclose(out.obsp["cell_ot_dissimilarity"], out.obsp["cell_ot_dissimilarity"].T)


def test_pairwise_niche_umap_feature_requires_explicit_opt_in(tmp_path) -> None:
    features, coords, samples = _toy_cohort(n_per_sample=4)
    adata = ad.AnnData(X=features)
    adata.obsm["X_umap_marker_genes_3d"] = features[:, :3]
    adata.obs = pd.DataFrame(
        {
            "sample_id": samples,
            "x": coords[:, 0],
            "y": coords[:, 1],
        },
        index=[f"cell_{idx}" for idx in range(features.shape[0])],
    )
    input_path = tmp_path / "input_umap.h5ad"
    adata.write_h5ad(input_path)
    with pytest.raises(ValueError, match="UMAP"):
        run_pairwise_niche_on_h5ad(
            input_h5ad=input_path,
            output_dir=tmp_path / "pairwise_reject",
            feature_obsm_key="X_umap_marker_genes_3d",
            spatial_x_key="x",
            spatial_y_key="y",
            sample_obs_key="sample_id",
            embedding_method="precomputed",
            n_clusters=2,
            max_exact_cells=20,
            sinkhorn_iters=3,
            device="cpu",
        )
    summary = run_pairwise_niche_on_h5ad(
        input_h5ad=input_path,
        output_dir=tmp_path / "pairwise_accept",
        feature_obsm_key="X_umap_marker_genes_3d",
        spatial_x_key="x",
        spatial_y_key="y",
        sample_obs_key="sample_id",
        embedding_method="precomputed",
        allow_umap_as_feature=True,
        n_clusters=2,
        max_exact_cells=20,
        sinkhorn_iters=3,
        device="cpu",
    )
    assert summary["feature_source"]["feature_embedding_warning"] == "umap_exploratory"
    assert summary["config"]["allow_umap_as_feature"] is True


def test_pairwise_niche_memmap_distance_store(tmp_path) -> None:
    features, coords, samples = _toy_cohort(n_per_sample=4)
    adata = ad.AnnData(X=features)
    adata.obsm["X_expr"] = features
    adata.obs = pd.DataFrame(
        {
            "sample_id": samples,
            "x": coords[:, 0],
            "y": coords[:, 1],
        },
        index=[f"cell_{idx}" for idx in range(features.shape[0])],
    )
    input_path = tmp_path / "input_memmap.h5ad"
    output_dir = tmp_path / "pairwise_memmap"
    adata.write_h5ad(input_path)

    summary = run_pairwise_niche_on_h5ad(
        input_h5ad=input_path,
        output_dir=output_dir,
        feature_obsm_key="X_expr",
        spatial_x_key="x",
        spatial_y_key="y",
        sample_obs_key="sample_id",
        embedding_method="pca",
        embedding_dim=3,
        radius_um=3.0,
        max_neighbors=2,
        block_size=3,
        sinkhorn_iters=5,
        device="cpu",
        max_exact_cells=20,
        distance_store="npy_memmap",
        cluster_method="agglomerative",
        n_clusters=2,
    )
    distance_path = summary["outputs"]["distance_matrix"]
    assert distance_path is not None
    matrix = np.load(str(distance_path), mmap_mode="r")
    assert matrix.shape == (8, 8)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-5)
    out = ad.read_h5ad(output_dir / "cells_pairwise_niche.h5ad")
    assert "cell_ot_dissimilarity" not in out.obsp
    assert out.uns["cell_ot_dissimilarity_store"] == str(distance_path)
