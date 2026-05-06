from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from spatial_ot.cell_niche import (
    CellNicheDataset,
    OTDeepSHEModel,
    OTPrototypeHead,
    build_knn_graphs,
    build_radius_graphs,
    compute_cell_heterogeneity_descriptors,
    fit_state_codebook,
    run_cell_niche_on_h5ad,
    sinkhorn_balanced_distance,
)
from spatial_ot.cell_niche.losses import deepshe_loss


def test_radius_graphs_never_connect_across_samples() -> None:
    coords = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    samples = np.asarray(["a", "a", "b", "b"], dtype=object)

    graphs = build_radius_graphs(
        coords,
        samples,
        radii_um=(2.0,),
        max_neighbors=8,
        include_self=False,
    )

    graph = graphs["r2"].connectivities.tocsr()
    assert graph[0, 1] > 0
    assert graph[2, 3] > 0
    assert graph[0, 2] == 0
    assert graph[1, 3] == 0


def test_radius_graph_cap_preserves_outer_shell_neighbors() -> None:
    coords = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.2, 0.0],
            [1.4, 0.0],
            [4.8, 0.0],
            [5.1, 0.0],
            [9.0, 0.0],
        ],
        dtype=np.float32,
    )
    samples = np.asarray(["s"] * coords.shape[0], dtype=object)

    graphs = build_radius_graphs(
        coords,
        samples,
        radii_um=(10.0,),
        max_neighbors=3,
        include_self=False,
        kernel="uniform",
    )

    dist = graphs["r10"].distances.tocsr()
    row0_distances = dist.data[dist.indptr[0] : dist.indptr[1]]
    assert row0_distances.size == 3
    assert np.any(row0_distances > 8.0)


def test_knn_graphs_never_connect_across_samples() -> None:
    coords = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [100.0, 0.0],
            [101.0, 0.0],
        ],
        dtype=np.float32,
    )
    samples = np.asarray(["a", "a", "b", "b"], dtype=object)

    graphs = build_knn_graphs(
        coords,
        samples,
        k_values=(1,),
        max_neighbors=4,
        include_self=False,
    )

    graph = graphs["k1"].connectivities.tocsr()
    assert graph[0, 1] > 0
    assert graph[2, 3] > 0
    assert graph[0, 2] == 0
    assert graph[1, 3] == 0


def test_graph_kernel_aliases_are_canonicalized() -> None:
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    samples = np.asarray(["s", "s"], dtype=object)

    graphs = build_radius_graphs(
        coords,
        samples,
        radii_um=(2.0,),
        kernel="binary",
    )

    assert graphs["r2"].metadata["kernel"] == "uniform"


def test_soft_codebook_rows_sum_to_one() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(24, 5)).astype(np.float32)

    codebook = fit_state_codebook(
        features,
        n_codewords=4,
        sample_size=24,
        random_state=3,
    )

    assert codebook.posteriors.shape == (24, 4)
    np.testing.assert_allclose(codebook.posteriors.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(codebook.entropy >= 0.0)


def test_cell_niche_dataset_batch_shapes() -> None:
    coords = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    samples = np.asarray(["s"] * coords.shape[0], dtype=object)
    graphs = build_radius_graphs(coords, samples, radii_um=(2.1,), max_neighbors=4)
    features = np.arange(coords.shape[0] * 3, dtype=np.float32).reshape(coords.shape[0], 3)
    posteriors = np.eye(3, dtype=np.float32)[np.arange(coords.shape[0]) % 3]
    targets = np.ones((coords.shape[0], 7), dtype=np.float32)
    dataset = CellNicheDataset(
        features=features,
        posteriors=posteriors,
        coords_um=coords,
        graphs=graphs,
        descriptor_targets=targets,
        radial_shells=3,
        max_neighbors_per_graph=2,
    )

    batch = dataset.collate_fn([0, 1, 4])

    assert batch["tokens"].shape == (3, 1, 2, 12)
    assert batch["weights"].shape == (3, 1, 2)
    assert batch["mask"].dtype == torch.bool
    assert batch["descriptor_targets"].shape == (3, 7)
    assert batch["is_isolated"].shape == (3, 1)
    assert batch["n_neighbors"].shape == (3, 1)
    assert batch["local_density_per_um2"].shape == (3, 1)
    assert batch["local_density"].shape == (3, 1)
    torch.testing.assert_close(
        (batch["weights"] * batch["mask"]).sum(dim=2),
        torch.ones((3, 1), dtype=torch.float32),
    )


def test_cell_niche_dataset_isolated_cell_uses_zero_context_token() -> None:
    coords = np.asarray([[0.0, 0.0], [100.0, 0.0]], dtype=np.float32)
    samples = np.asarray(["s", "s"], dtype=object)
    graphs = build_radius_graphs(coords, samples, radii_um=(1.0,), max_neighbors=4)
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    posteriors = np.eye(2, dtype=np.float32)
    dataset = CellNicheDataset(
        features=features,
        posteriors=posteriors,
        coords_um=coords,
        graphs=graphs,
        radial_shells=2,
        max_neighbors_per_graph=3,
    )

    batch = dataset.collate_fn([0])

    assert bool(batch["is_isolated"][0, 0])
    assert float(batch["n_neighbors"][0, 0]) == 0.0
    assert float(batch["local_density_per_um2"][0, 0]) == 0.0
    assert float(batch["local_density"][0, 0]) == 0.0
    torch.testing.assert_close(batch["tokens"][0, 0, 0], torch.zeros(9))
    torch.testing.assert_close(batch["weights"][0, 0], torch.tensor([1.0, 0.0, 0.0]))


def test_covariance_descriptor_block_metadata_is_present() -> None:
    coords = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    samples = np.asarray(["s"] * coords.shape[0], dtype=object)
    graphs = build_radius_graphs(coords, samples, radii_um=(2.1,), max_neighbors=4)
    features = np.arange(coords.shape[0] * 4, dtype=np.float32).reshape(coords.shape[0], 4)
    posteriors = np.eye(3, dtype=np.float32)[np.arange(coords.shape[0]) % 3]

    descriptor = compute_cell_heterogeneity_descriptors(
        features=features,
        posteriors=posteriors,
        graphs=graphs,
        coords=coords,
        blocks=("covariance",),
        self_weight=0.0,
        covariance_dims=3,
    )

    assert descriptor.raw.shape == (coords.shape[0], 6)
    assert descriptor.metadata["family_counts"] == {"covariance": 1}
    assert descriptor.metadata["graph_diagnostics"]["r2p1"]["covariance_block"][
        "upper_triangular_dim"
    ] == 6


def test_ot_cost_identical_single_support_measures_is_zero() -> None:
    cost = torch.zeros((2, 1, 1), dtype=torch.float32)
    weights = torch.ones((2, 1), dtype=torch.float32)

    distances = sinkhorn_balanced_distance(
        cost,
        weights,
        weights,
        epsilon=0.05,
        n_iters=5,
    )

    torch.testing.assert_close(distances, torch.zeros(2), atol=1e-7, rtol=1e-7)


def test_sinkhorn_identical_measure_cost_is_finite_and_small() -> None:
    torch.manual_seed(17)
    points = torch.randn(1, 4, 3)
    cost = torch.cdist(points, points, p=2.0).pow(2)
    weights = torch.full((1, 4), 0.25)

    distance = sinkhorn_balanced_distance(
        cost,
        weights,
        weights,
        epsilon=0.01,
        n_iters=80,
    )

    assert torch.isfinite(distance).all()
    assert float(distance[0]) < 1e-3


def test_sinkhorn_gradients_are_finite() -> None:
    torch.manual_seed(19)
    left = torch.randn(1, 5, 3, requires_grad=True)
    right = torch.randn(1, 6, 3, requires_grad=True)
    cost = torch.cdist(left, right, p=2.0).pow(2)
    a = torch.full((1, 5), 0.2)
    b = torch.full((1, 6), 1.0 / 6.0)

    distance = sinkhorn_balanced_distance(cost, a, b, epsilon=0.05, n_iters=20)
    distance.sum().backward()

    assert left.grad is not None
    assert right.grad is not None
    assert torch.isfinite(left.grad).all()
    assert torch.isfinite(right.grad).all()


def test_ot_prototype_head_output_shape() -> None:
    torch.manual_seed(5)
    head = OTPrototypeHead(
        n_radii=2,
        token_dim=4,
        n_prototypes=3,
        support_size=2,
        epsilon=0.1,
        sinkhorn_iters=3,
    )
    token_embeddings = torch.randn(5, 2, 4, 4)
    weights = torch.full((5, 2, 4), 0.25)
    mask = torch.ones((5, 2, 4), dtype=torch.bool)

    distances, posterior = head(
        token_embeddings=token_embeddings,
        weights=weights,
        mask=mask,
    )

    assert distances.shape == (5, 3)
    assert posterior.shape == (5, 3)
    torch.testing.assert_close(posterior.sum(dim=1), torch.ones(5), atol=1e-5, rtol=1e-5)


def test_deepshe_forward_backward_runs() -> None:
    coords = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=np.float32,
    )
    samples = np.asarray(["s"] * coords.shape[0], dtype=object)
    graphs = build_radius_graphs(coords, samples, radii_um=(2.1,), max_neighbors=4)
    rng = np.random.default_rng(13)
    features = rng.normal(size=(coords.shape[0], 4)).astype(np.float32)
    posteriors = np.eye(3, dtype=np.float32)[np.arange(coords.shape[0]) % 3]
    targets = rng.normal(size=(coords.shape[0], 6)).astype(np.float32)
    dataset = CellNicheDataset(
        features=features,
        posteriors=posteriors,
        coords_um=coords,
        graphs=graphs,
        descriptor_targets=targets,
        radial_shells=2,
        max_neighbors_per_graph=3,
    )
    batch = dataset.collate_fn([0, 1, 2, 3])
    model = OTDeepSHEModel(
        z_dim=4,
        token_input_dim=dataset.token_input_dim,
        n_radii=1,
        descriptor_dim=6,
        token_dim=8,
        hidden_dim=16,
        embedding_dim=4,
        use_ot_prototypes=True,
        n_ot_prototypes=3,
        prototype_support_size=2,
        ot_epsilon=0.1,
    )

    outputs = model(batch)
    loss, metrics = deepshe_loss(outputs, batch)
    loss.backward()

    assert np.isfinite(metrics["loss"])
    assert outputs["embedding_raw"].shape == outputs["embedding"].shape
    torch.testing.assert_close(
        outputs["embedding"].norm(dim=1),
        torch.ones(outputs["embedding"].shape[0]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert model.encoder.fuse[-1].weight.grad is not None


def test_deepshe_keeps_isolated_context_tokens_zero() -> None:
    coords = np.asarray([[0.0, 0.0], [100.0, 0.0]], dtype=np.float32)
    samples = np.asarray(["s", "s"], dtype=object)
    graphs = build_radius_graphs(coords, samples, radii_um=(1.0,), max_neighbors=4)
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    posteriors = np.eye(2, dtype=np.float32)
    targets = np.zeros((2, 4), dtype=np.float32)
    dataset = CellNicheDataset(
        features=features,
        posteriors=posteriors,
        coords_um=coords,
        graphs=graphs,
        descriptor_targets=targets,
        radial_shells=2,
        max_neighbors_per_graph=3,
    )
    batch = dataset.collate_fn([0])
    model = OTDeepSHEModel(
        z_dim=2,
        token_input_dim=dataset.token_input_dim,
        n_radii=1,
        descriptor_dim=4,
        token_dim=8,
        hidden_dim=16,
        embedding_dim=4,
    )

    outputs = model(batch)

    assert bool(batch["is_isolated"][0, 0])
    torch.testing.assert_close(
        outputs["token_embeddings"][0, 0],
        torch.zeros_like(outputs["token_embeddings"][0, 0]),
    )


def test_cell_niche_descriptor_run_writes_cell_level_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    n_per_sample = 18
    sample = np.repeat(["s1", "s2"], n_per_sample)
    x = np.tile(np.arange(6, dtype=np.float32), 6)
    y = np.repeat(np.arange(6, dtype=np.float32), 6)
    coords = np.column_stack([x, y])[: 2 * n_per_sample].astype(np.float32)
    coords[n_per_sample:] += 20.0
    labels_hint = (coords[:, 0] > np.median(coords[:, 0])).astype(np.float32)
    features = np.column_stack(
        [
            labels_hint,
            1.0 - labels_hint,
            np.sin(coords[:, 0] / 3.0),
            np.cos(coords[:, 1] / 3.0),
            rng.normal(scale=0.05, size=coords.shape[0]),
        ]
    ).astype(np.float32)
    adata = ad.AnnData(
        X=np.ones((coords.shape[0], 3), dtype=np.float32),
        obs=pd.DataFrame(index=[f"cell_{idx}" for idx in range(coords.shape[0])]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    adata.obs["cell_x"] = coords[:, 0]
    adata.obs["cell_y"] = coords[:, 1]
    adata.obs["sample_id"] = sample
    adata.obsm["X_test_features"] = features
    input_h5ad = tmp_path / "input.h5ad"
    adata.write_h5ad(input_h5ad)

    summary = run_cell_niche_on_h5ad(
        input_h5ad=input_h5ad,
        output_dir=tmp_path / "out",
        feature_obsm_key="X_test_features",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        sample_obs_key="sample_id",
        radii_um=(2.5, 4.0),
        state_codebook_size=4,
        state_codebook_sample_size=32,
        feature_pca_dim=4,
        embedding_dim=4,
        cluster_method="kmeans",
        n_clusters=3,
        max_neighbors=8,
        pair_top_states=3,
        seed=7,
    )

    output_h5ad = Path(summary["outputs"]["h5ad"])
    assert output_h5ad.exists()
    fitted = ad.read_h5ad(output_h5ad)
    assert fitted.n_obs == adata.n_obs
    assert "X_spatial_heterogeneity_descriptor" in fitted.obsm
    assert "X_spatial_heterogeneity" in fitted.obsm
    assert fitted.obsm["X_spatial_heterogeneity"].shape[1] == 4
    assert "spatial_niche" in fitted.obs
    assert "spatial_niche_confidence" in fitted.obs
    assert "spatial_niche_assignment_score" in fitted.obs
    assert "spatial_niche_assignment_score_type" in fitted.obs
    assert "spatial_niche_instance" in fitted.obs
    assert "spatial_connectivities_r2p5" in fitted.obsp
    assert "spatial_distances_r4" in fitted.obsp
    assert "n_neighbors_r2p5" in fitted.obs
    assert "local_density_per_um2_r2p5" in fitted.obs
    np.testing.assert_allclose(
        fitted.obs["local_density_per_um2_r2p5"].to_numpy(dtype=np.float32),
        fitted.obs["n_neighbors_r2p5"].to_numpy(dtype=np.float32) / (np.pi * 2.5 * 2.5),
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        fitted.obs["local_density_r2p5"].to_numpy(dtype=np.float32),
        fitted.obs["local_density_per_um2_r2p5"].to_numpy(dtype=np.float32),
        rtol=1e-6,
        atol=1e-8,
    )
    assert "is_isolated_r2p5" in fitted.obs
    assert fitted.uns["spatial_niche_summary"]["primary_unit"] == "cell"
    assert summary["primary_unit"] == "cell"
    assert summary["n_niches"] == 3


def test_descriptor_mode_rejects_ot_prototype_flag(tmp_path: Path) -> None:
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    adata = ad.AnnData(
        X=np.ones((coords.shape[0], 2), dtype=np.float32),
        obs=pd.DataFrame(index=[f"cell_{idx}" for idx in range(coords.shape[0])]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    adata.obs["cell_x"] = coords[:, 0]
    adata.obs["cell_y"] = coords[:, 1]
    adata.obs["sample_id"] = "s"
    adata.obsm["X_test_features"] = np.column_stack(
        [coords[:, 0], coords[:, 1], np.ones(coords.shape[0])]
    ).astype(np.float32)
    input_h5ad = tmp_path / "input.h5ad"
    adata.write_h5ad(input_h5ad)

    try:
        run_cell_niche_on_h5ad(
            input_h5ad=input_h5ad,
            output_dir=tmp_path / "out",
            feature_obsm_key="X_test_features",
            spatial_x_key="cell_x",
            spatial_y_key="cell_y",
            sample_obs_key="sample_id",
            radii_um=(2.5,),
            state_codebook_size=2,
            state_codebook_sample_size=4,
            feature_pca_dim=3,
            embedding_dim=2,
            cluster_method="kmeans",
            n_clusters=2,
            use_ot_prototypes=True,
        )
    except ValueError as exc:
        assert "requires a DeepSHE encoder" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("descriptor mode should reject --use-ot-prototypes")


def test_cell_niche_deep_ot_run_writes_deepshe_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(321)
    n_cells = 24
    sample = np.repeat(["s1", "s2"], n_cells // 2)
    coords = np.column_stack(
        [
            np.tile(np.arange(6, dtype=np.float32), 4),
            np.repeat(np.arange(4, dtype=np.float32), 6),
        ]
    )
    coords[n_cells // 2 :] += 10.0
    features = np.column_stack(
        [
            np.sin(coords[:, 0]),
            np.cos(coords[:, 1]),
            (sample == "s1").astype(np.float32),
            rng.normal(scale=0.05, size=n_cells),
        ]
    ).astype(np.float32)
    adata = ad.AnnData(
        X=np.ones((n_cells, 3), dtype=np.float32),
        obs=pd.DataFrame(index=[f"cell_{idx}" for idx in range(n_cells)]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    adata.obs["cell_x"] = coords[:, 0]
    adata.obs["cell_y"] = coords[:, 1]
    adata.obs["sample_id"] = sample
    adata.obsm["X_test_features"] = features
    input_h5ad = tmp_path / "deep_input.h5ad"
    adata.write_h5ad(input_h5ad)

    summary = run_cell_niche_on_h5ad(
        input_h5ad=input_h5ad,
        output_dir=tmp_path / "deep_out",
        feature_obsm_key="X_test_features",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        sample_obs_key="sample_id",
        radii_um=(2.5,),
        state_codebook_size=4,
        state_codebook_sample_size=24,
        feature_pca_dim=4,
        embedding_dim=4,
        encoder="attention_deepsets",
        epochs=1,
        batch_size=12,
        token_dim=8,
        hidden_dim=16,
        use_ot_prototypes=True,
        n_ot_prototypes=3,
        prototype_support_size=2,
        ot_epsilon=0.1,
        cluster_method="kmeans",
        n_clusters=3,
        max_neighbors=6,
        max_neighbors_per_radius=3,
        pair_top_states=3,
        seed=9,
    )

    fitted = ad.read_h5ad(summary["outputs"]["h5ad"])
    assert "X_spatial_ot_deepshe" in fitted.obsm
    assert "X_spatial_ot_deepshe_raw" in fitted.obsm
    assert fitted.obsm["X_spatial_ot_deepshe_raw"].shape == fitted.obsm[
        "X_spatial_ot_deepshe"
    ].shape
    assert "X_spatial_ot_prototype_distances" in fitted.obsm
    assert fitted.obsm["X_spatial_ot_prototype_distances"].shape == (n_cells, 3)
    assert set(fitted.obs["spatial_niche_source"].astype(str)) == {"deep_plus_ot_kmeans"}
    assert summary["validation_status"]["learned_deepsets_encoder_implemented"] is True
    assert summary["validation_status"]["ot_prototype_diagnostics_implemented"] is True
