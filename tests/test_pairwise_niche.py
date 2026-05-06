from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
import pytest

from spatial_ot.pairwise_niche import (
    build_local_measures,
    cluster_from_distance,
    compute_pairwise_ot_distance_matrix,
    fit_expression_embedding,
    run_pairwise_niche_on_h5ad,
)
from spatial_ot.pairwise_niche.cluster import ot_knn_affinity


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
    for row, ids in enumerate(measures.neighbor_indices):
        valid = ids[ids >= 0]
        assert set(samples[valid]) == {samples[row]}


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
        seed=19,
    )
    assert np.all(measures.full_neighbor_counts == 0)
    assert np.all(measures.retained_neighbor_counts == 0)
    assert np.all(measures.neighbor_indices[:, 0] == -1)
    np.testing.assert_allclose(measures.tokens[:, 0, :], 0.0)
    np.testing.assert_allclose(measures.weights[:, 0], 1.0)


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
    assert metadata["distance_mode"] == "sinkhorn_divergence"
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
    affinity = ot_knn_affinity(distance, k=1)
    np.testing.assert_allclose(affinity.toarray(), affinity.toarray().T)


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
        n_clusters=2,
        seed=17,
    )
    assert summary["active_path"] == "pairwise-niche"
    out = ad.read_h5ad(output_dir / "cells_pairwise_niche.h5ad")
    assert "X_gene_cohort" in out.obsm
    assert "cell_ot_dissimilarity" in out.obsp
    assert "cell_ot_affinity" in out.obsp
    assert "ot_niche" in out.obs
    assert "ot_niche_instance" in out.obs
    assert "n_neighbors_full_r3" in out.obs
    assert "neighbor_retention_fraction_r3" in out.obs
    np.testing.assert_allclose(out.obsp["cell_ot_dissimilarity"], out.obsp["cell_ot_dissimilarity"].T)


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
