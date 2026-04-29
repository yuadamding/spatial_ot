from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from spatial_ot.cli import _resolve_deep_fit_config_from_args, build_parser
from spatial_ot.config import DeepFeatureConfig, load_multilevel_config
from spatial_ot.deep import graph as deep_graph
from spatial_ot.deep.graph import aggregate_neighbor_mean_torch, build_neighbor_graph
from spatial_ot.deep import (
    SpatialOTFeatureEncoder,
    fit_deep_features,
    fit_deep_features_on_h5ad,
    transform_h5ad_with_deep_model,
)
from spatial_ot.deep.features import _split_validation
from spatial_ot.deep.io import _extract_count_target as _extract_deep_count_target
from spatial_ot.multilevel.metadata import extract_count_target as _extract_multilevel_count_target
from spatial_ot.multilevel.io import _load_region_geometry_json
from spatial_ot.multilevel import run_multilevel_ot_on_h5ad


def test_deep_feature_encoder_fit_save_load(tmp_path) -> None:
    rng = np.random.default_rng(123)
    features = rng.normal(size=(48, 6)).astype(np.float32)
    coords = rng.normal(size=(48, 2)).astype(np.float32)
    config = DeepFeatureConfig(
        method="autoencoder",
        latent_dim=4,
        hidden_dim=24,
        layers=2,
        neighbor_k=4,
        epochs=3,
        batch_size=16,
        validation="none",
        output_embedding="intrinsic",
        save_model=True,
    )
    encoder = SpatialOTFeatureEncoder(config)
    encoder.fit(features=features, coords_um=coords, seed=7)
    embedding = encoder.transform(features=features, coords_um=coords)
    assert embedding.shape == (48, 4)
    assert encoder.uses_coordinate_input is False

    model_path = tmp_path / "deep_feature_model.pt"
    encoder.save(model_path)
    loaded = SpatialOTFeatureEncoder.load(model_path)
    loaded_embedding = loaded.transform(features=features, coords_um=coords)
    assert np.allclose(embedding, loaded_embedding, atol=1e-5)


def test_graph_autoencoder_fit_limit_checks_total_cells_before_training() -> None:
    rng = np.random.default_rng(321)
    features = rng.normal(size=(40, 4)).astype(np.float32)
    coords = rng.normal(size=(40, 2)).astype(np.float32)
    batch = np.repeat(["trainish", "heldoutish"], repeats=20)
    encoder = SpatialOTFeatureEncoder(
        DeepFeatureConfig(
            method="graph_autoencoder",
            latent_dim=3,
            hidden_dim=12,
            layers=1,
            graph_layers=1,
            validation="sample_holdout",
            full_batch_max_cells=30,
            output_embedding="context",
            epochs=1,
        )
    )
    with pytest.raises(ValueError, match="n_cells=40"):
        encoder.fit(features=features, coords_um=coords, batch=batch, seed=7)


def test_graph_deep_feature_encoder_fit_save_load(tmp_path) -> None:
    rng = np.random.default_rng(456)
    features = rng.normal(size=(40, 6)).astype(np.float32)
    coords = rng.normal(size=(40, 2)).astype(np.float32)
    config = DeepFeatureConfig(
        method="graph_autoencoder",
        latent_dim=4,
        hidden_dim=24,
        layers=2,
        neighbor_k=4,
        radius_um=0.8,
        short_radius_um=0.6,
        mid_radius_um=1.2,
        graph_layers=1,
        epochs=3,
        batch_size=16,
        validation="none",
        output_embedding="joint",
        save_model=True,
    )
    encoder = SpatialOTFeatureEncoder(config)
    encoder.fit(features=features, coords_um=coords, seed=5)
    embedding = encoder.transform(features=features, coords_um=coords)
    assert embedding.shape == (40, 4)

    model_path = tmp_path / "graph_deep_feature_model.pt"
    encoder.save(model_path)
    loaded = SpatialOTFeatureEncoder.load(model_path)
    loaded_embedding = loaded.transform(features=features, coords_um=coords)
    assert np.allclose(embedding, loaded_embedding, atol=1e-5)


def test_cuda_saved_encoder_can_load_on_cpu(tmp_path) -> None:
    rng = np.random.default_rng(2026)
    features = rng.normal(size=(24, 5)).astype(np.float32)
    coords = rng.normal(size=(24, 2)).astype(np.float32)
    config = DeepFeatureConfig(
        method="autoencoder",
        latent_dim=3,
        hidden_dim=16,
        layers=1,
        neighbor_k=3,
        epochs=2,
        batch_size=8,
        validation="none",
        device="cpu",
        output_embedding="intrinsic",
    )
    encoder = SpatialOTFeatureEncoder(config)
    encoder.fit(features=features, coords_um=coords, seed=17)
    model_path = tmp_path / "cpu_loadable_model.pt"
    encoder.save(model_path)

    metadata_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["config"]["device"] = "cuda"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    loaded = SpatialOTFeatureEncoder.load(model_path, map_location="cpu", device="cpu")
    assert loaded.device.type == "cpu"
    loaded_embedding = loaded.transform(features=features, coords_um=coords)
    assert loaded_embedding.shape == (24, 3)


def test_no_validation_restores_latest_epoch_not_epoch_one() -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(64, 8)).astype(np.float32)
    coords = rng.normal(size=(64, 2)).astype(np.float32)

    config_epoch1 = DeepFeatureConfig(
        method="autoencoder",
        latent_dim=4,
        hidden_dim=20,
        layers=2,
        neighbor_k=4,
        epochs=1,
        batch_size=16,
        validation="none",
        output_embedding="intrinsic",
        restore_best=True,
    )
    config_epoch5 = DeepFeatureConfig(
        method="autoencoder",
        latent_dim=4,
        hidden_dim=20,
        layers=2,
        neighbor_k=4,
        epochs=5,
        batch_size=16,
        validation="none",
        output_embedding="intrinsic",
        restore_best=True,
    )

    enc1 = SpatialOTFeatureEncoder(config_epoch1).fit(features=features, coords_um=coords, seed=11)
    enc5 = SpatialOTFeatureEncoder(config_epoch5).fit(features=features, coords_um=coords, seed=11)

    z1 = enc1.transform(features=features, coords_um=coords)
    z5 = enc5.transform(features=features, coords_um=coords)

    assert not np.allclose(z1, z5)


def test_seed_reproducibility_includes_weight_initialization() -> None:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(48, 6)).astype(np.float32)
    coords = rng.normal(size=(48, 2)).astype(np.float32)
    config = DeepFeatureConfig(
        method="graph_autoencoder",
        latent_dim=4,
        hidden_dim=24,
        layers=2,
        neighbor_k=4,
        radius_um=0.8,
        short_radius_um=0.6,
        mid_radius_um=1.2,
        graph_layers=1,
        epochs=3,
        batch_size=16,
        validation="none",
        output_embedding="joint",
        restore_best=True,
    )

    enc_a = SpatialOTFeatureEncoder(config).fit(features=features, coords_um=coords, seed=23)
    enc_b = SpatialOTFeatureEncoder(config).fit(features=features, coords_um=coords, seed=23)

    z_a = enc_a.transform(features=features, coords_um=coords)
    z_b = enc_b.transform(features=features, coords_um=coords)

    assert np.allclose(z_a, z_b, atol=1e-6)


def test_build_neighbor_graph_respects_max_neighbors() -> None:
    coords = np.stack(
        [
            np.linspace(0.0, 1.0, 12, dtype=np.float32),
            np.zeros(12, dtype=np.float32),
        ],
        axis=1,
    )
    edge_index = build_neighbor_graph(
        coords,
        neighbor_k=8,
        radius_um=10.0,
        max_neighbors=3,
    )
    if edge_index.shape[1] > 0:
        max_degree = np.bincount(edge_index[1], minlength=coords.shape[0]).max()
        assert int(max_degree) <= 3


def test_aggregate_neighbor_mean_torch_matches_full_gather_when_chunked(monkeypatch) -> None:
    features = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 1, 5],
            [1, 1, 2, 2, 3, 3, 4, 4, 7, 7],
        ],
        dtype=torch.long,
    )
    src, dst = edge_index
    expected = torch.zeros_like(features)
    deg = torch.zeros((features.shape[0], 1), dtype=features.dtype)
    expected.index_add_(0, dst, features[src])
    deg.index_add_(0, dst, torch.ones((src.numel(), 1), dtype=features.dtype))
    expected = expected / deg.clamp_min(1.0)
    isolated = deg.squeeze(1) == 0
    if torch.any(isolated):
        expected[isolated] = features[isolated]

    monkeypatch.setattr(deep_graph, "_NEIGHBOR_AGGREGATION_MAX_GATHER_BYTES", 8)
    actual = aggregate_neighbor_mean_torch(features, edge_index)
    assert torch.allclose(actual, expected)


def test_deep_transform_batched_equals_default(tmp_path) -> None:
    rng = np.random.default_rng(55)
    features = rng.normal(size=(64, 6)).astype(np.float32)
    coords = rng.normal(size=(64, 2)).astype(np.float32)
    config = DeepFeatureConfig(
        method="autoencoder",
        latent_dim=4,
        hidden_dim=20,
        layers=2,
        neighbor_k=4,
        epochs=2,
        batch_size=16,
        validation="none",
        output_embedding="intrinsic",
    )
    encoder = SpatialOTFeatureEncoder(config)
    encoder.fit(features=features, coords_um=coords, seed=9)
    emb_default = encoder.transform(features=features, coords_um=coords)
    emb_batched = encoder.transform(features=features, coords_um=coords, batch_size=7)
    assert np.allclose(emb_default, emb_batched, atol=1e-5)


def test_deep_scaler_fit_on_train_only() -> None:
    rng = np.random.default_rng(99)
    train = rng.normal(loc=0.0, scale=1.0, size=(20, 5)).astype(np.float32)
    val = rng.normal(loc=100.0, scale=1.0, size=(20, 5)).astype(np.float32)
    features = np.vstack([train, val]).astype(np.float32)
    coords = rng.normal(size=(40, 2)).astype(np.float32)
    batch = np.array(["train"] * 20 + ["holdout"] * 20)
    config = DeepFeatureConfig(
        method="autoencoder",
        latent_dim=3,
        hidden_dim=16,
        layers=1,
        neighbor_k=3,
        epochs=1,
        batch_size=8,
        validation="sample_holdout",
        batch_key="sample_id",
        output_embedding="intrinsic",
    )
    encoder = SpatialOTFeatureEncoder(config)
    encoder.fit(features=features, coords_um=coords, batch=batch, seed=3)
    val_mask = _split_validation(coords_um=coords, batch=batch, config=config, seed=3)
    train_mask = ~val_mask
    expected_mean = features[train_mask].mean(axis=0, keepdims=True)
    assert np.allclose(encoder.feature_mean, expected_mean.astype(np.float32), atol=1e-5)


def test_fit_deep_features_helper_returns_result(tmp_path) -> None:
    rng = np.random.default_rng(321)
    features = rng.normal(size=(36, 5)).astype(np.float32)
    coords = rng.normal(size=(36, 2)).astype(np.float32)
    batch = np.array(["a"] * 18 + ["b"] * 18)
    config = DeepFeatureConfig(
        method="autoencoder",
        latent_dim=3,
        hidden_dim=16,
        layers=1,
        neighbor_k=3,
        epochs=2,
        batch_size=12,
        validation="sample_holdout",
        batch_key="sample_id",
        output_embedding="intrinsic",
        save_model=True,
    )
    result = fit_deep_features(
        features=features,
        coords_um=coords,
        config=config,
        batch=batch,
        seed=11,
        save_path=tmp_path / "encoder.pt",
    )
    assert result.embedding.shape == (36, 3)
    assert result.model_path is not None
    assert len(result.history) == 2


def test_fit_deep_features_on_h5ad_outputs_artifacts(tmp_path) -> None:
    rng = np.random.default_rng(77)
    features = rng.normal(size=(32, 5)).astype(np.float32)
    coords = rng.normal(size=(32, 2)).astype(np.float32)
    adata = ad.AnnData(X=features.copy())
    adata.obsm["X_pca"] = features.copy()
    adata.obs["cell_x"] = coords[:, 0]
    adata.obs["cell_y"] = coords[:, 1]
    input_h5ad = tmp_path / "input_fit.h5ad"
    adata.write_h5ad(input_h5ad)

    summary = fit_deep_features_on_h5ad(
        input_h5ad=input_h5ad,
        output_dir=tmp_path / "deep_fit_out",
        feature_obsm_key="X_pca",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=2.0,
        config=DeepFeatureConfig(
            method="graph_autoencoder",
            latent_dim=3,
            hidden_dim=16,
            layers=1,
            neighbor_k=3,
            radius_um=1.0,
            short_radius_um=0.75,
            mid_radius_um=1.5,
            graph_layers=1,
            graph_max_neighbors=5,
            epochs=2,
            batch_size=16,
            validation="none",
            output_embedding="joint",
            save_model=True,
        ),
        seed=13,
    )

    assert summary["deep_features"]["method"] == "graph_autoencoder"
    assert summary["method_family"] == "deep_feature_adapter"
    assert summary["active_path"] == "deep-fit"
    assert summary["latent_source"] == "deep_joint"
    assert summary["communication_source"] == "none"
    assert summary["method_stack"]["deep_feature_adapter"] == "graph_autoencoder"
    assert summary["deep_features"]["graph_max_neighbors"] == 5
    assert summary["deep_features"]["full_batch_max_cells"] == 50000
    assert summary["spatial_scale"] == 2.0
    assert Path(summary["outputs"]["embedded_h5ad"]).exists()
    assert Path(summary["outputs"]["deep_feature_model"]).exists()
    saved = ad.read_h5ad(summary["outputs"]["embedded_h5ad"])
    assert "X_spatial_ot_deep" in saved.obsm
    assert saved.obsm["X_spatial_ot_deep"].shape == (32, 3)
    assert saved.uns["deep_features"]["summary_json"]
    saved_summary = json.loads(saved.uns["deep_features"]["summary_json"])
    assert saved_summary["method_family"] == "deep_feature_adapter"
    assert saved_summary["latent_source"] == "deep_joint"
    assert saved_summary["deep_features"]["feature_schema"]["spatial_scale"] == 2.0


def test_transform_h5ad_with_deep_model_writes_embedding(tmp_path) -> None:
    rng = np.random.default_rng(78)
    features_train = rng.normal(size=(24, 4)).astype(np.float32)
    coords_train = rng.normal(size=(24, 2)).astype(np.float32)
    train = ad.AnnData(X=features_train.copy())
    train.obsm["X_pca"] = features_train.copy()
    train.obs["cell_x"] = coords_train[:, 0]
    train.obs["cell_y"] = coords_train[:, 1]
    train_h5ad = tmp_path / "train.h5ad"
    train.write_h5ad(train_h5ad)

    fit_summary = fit_deep_features_on_h5ad(
        input_h5ad=train_h5ad,
        output_dir=tmp_path / "deep_model",
        feature_obsm_key="X_pca",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=1.0,
        config=DeepFeatureConfig(
            method="autoencoder",
            latent_dim=3,
            hidden_dim=12,
            layers=1,
            neighbor_k=3,
            epochs=2,
            batch_size=8,
            validation="none",
            output_embedding="intrinsic",
            save_model=True,
        ),
        seed=9,
    )

    features_new = rng.normal(size=(18, 4)).astype(np.float32)
    coords_new = rng.normal(size=(18, 2)).astype(np.float32)
    new = ad.AnnData(X=features_new.copy())
    new.obsm["X_pca"] = features_new.copy()
    new.obs["cell_x"] = coords_new[:, 0]
    new.obs["cell_y"] = coords_new[:, 1]
    new_h5ad = tmp_path / "new.h5ad"
    new.write_h5ad(new_h5ad)

    summary = transform_h5ad_with_deep_model(
        model_path=fit_summary["outputs"]["deep_feature_model"],
        input_h5ad=new_h5ad,
        output_h5ad=tmp_path / "new_embedded.h5ad",
        feature_obsm_key="X_pca",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=1.0,
        output_obsm_key="X_deep_test",
        batch_size=5,
    )

    assert summary["method_family"] == "deep_feature_adapter"
    assert summary["active_path"] == "deep-transform"
    assert summary["latent_source"] == "deep_intrinsic"
    assert summary["communication_source"] == "none"
    assert summary["method_stack"]["active_path"] == "deep-transform"
    assert summary["deep_features"]["pretrained_model_loaded"] is True
    assert summary["deep_features"]["graph_inference_mode"] == "batched"
    transformed = ad.read_h5ad(summary["outputs"]["embedded_h5ad"])
    assert "X_deep_test" in transformed.obsm
    assert transformed.obsm["X_deep_test"].shape == (18, 3)


def test_transform_h5ad_with_deep_model_rejects_graph_full_batch_oversize_input(tmp_path) -> None:
    rng = np.random.default_rng(7801)
    features_train = rng.normal(size=(6, 4)).astype(np.float32)
    coords_train = rng.normal(size=(6, 2)).astype(np.float32)
    train = ad.AnnData(X=features_train.copy())
    train.obsm["X_pca"] = features_train.copy()
    train.obs["cell_x"] = coords_train[:, 0]
    train.obs["cell_y"] = coords_train[:, 1]
    train_h5ad = tmp_path / "train_graph_guard.h5ad"
    train.write_h5ad(train_h5ad)

    fit_summary = fit_deep_features_on_h5ad(
        input_h5ad=train_h5ad,
        output_dir=tmp_path / "deep_graph_guard_model",
        feature_obsm_key="X_pca",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=1.0,
        config=DeepFeatureConfig(
            method="graph_autoencoder",
            latent_dim=3,
            hidden_dim=12,
            layers=1,
            neighbor_k=3,
            radius_um=1.0,
            short_radius_um=0.75,
            mid_radius_um=1.5,
            graph_layers=1,
            epochs=2,
            batch_size=8,
            validation="none",
            output_embedding="joint",
            allow_joint_ot_embedding=True,
            full_batch_max_cells=8,
            save_model=True,
        ),
        seed=9,
    )

    features_new = rng.normal(size=(10, 4)).astype(np.float32)
    coords_new = rng.normal(size=(10, 2)).astype(np.float32)
    new = ad.AnnData(X=features_new.copy())
    new.obsm["X_pca"] = features_new.copy()
    new.obs["cell_x"] = coords_new[:, 0]
    new.obs["cell_y"] = coords_new[:, 1]
    new_h5ad = tmp_path / "new_graph_guard.h5ad"
    new.write_h5ad(new_h5ad)

    with pytest.raises(ValueError, match="full-batch execution"):
        transform_h5ad_with_deep_model(
            model_path=fit_summary["outputs"]["deep_feature_model"],
            input_h5ad=new_h5ad,
            output_h5ad=tmp_path / "new_graph_guard_embedded.h5ad",
            feature_obsm_key="X_pca",
            spatial_x_key="cell_x",
            spatial_y_key="cell_y",
            spatial_scale=1.0,
        )


def test_transform_h5ad_with_deep_model_rejects_mismatched_input_obsm_key(tmp_path) -> None:
    rng = np.random.default_rng(79)
    features = rng.normal(size=(20, 4)).astype(np.float32)
    coords = rng.normal(size=(20, 2)).astype(np.float32)
    train = ad.AnnData(X=features.copy())
    train.obsm["X_pca"] = features.copy()
    train.obs["cell_x"] = coords[:, 0]
    train.obs["cell_y"] = coords[:, 1]
    train_h5ad = tmp_path / "train_schema.h5ad"
    train.write_h5ad(train_h5ad)

    fit_summary = fit_deep_features_on_h5ad(
        input_h5ad=train_h5ad,
        output_dir=tmp_path / "deep_model_schema",
        feature_obsm_key="X_pca",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=1.0,
        config=DeepFeatureConfig(
            method="autoencoder",
            latent_dim=3,
            hidden_dim=12,
            layers=1,
            neighbor_k=3,
            epochs=2,
            batch_size=8,
            validation="none",
            output_embedding="intrinsic",
            save_model=True,
        ),
        seed=9,
    )

    new = ad.AnnData(X=features.copy())
    new.obsm["X_other"] = features.copy()
    new.obs["cell_x"] = coords[:, 0]
    new.obs["cell_y"] = coords[:, 1]
    new_h5ad = tmp_path / "new_schema.h5ad"
    new.write_h5ad(new_h5ad)

    with pytest.raises(ValueError, match="Input obsm key mismatch"):
        transform_h5ad_with_deep_model(
            model_path=fit_summary["outputs"]["deep_feature_model"],
            input_h5ad=new_h5ad,
            output_h5ad=tmp_path / "new_schema_embedded.h5ad",
            feature_obsm_key="X_other",
            spatial_x_key="cell_x",
            spatial_y_key="cell_y",
            spatial_scale=1.0,
        )


def test_multilevel_pretrained_deep_model_rejects_mismatched_input_obsm_key(tmp_path) -> None:
    rng = np.random.default_rng(791)
    features = rng.normal(size=(20, 4)).astype(np.float32)
    coords = rng.normal(size=(20, 2)).astype(np.float32)
    train = ad.AnnData(X=features.copy())
    train.obsm["X_pca"] = features.copy()
    train.obs["cell_x"] = coords[:, 0]
    train.obs["cell_y"] = coords[:, 1]
    train_h5ad = tmp_path / "train_multilevel_pretrained_schema.h5ad"
    train.write_h5ad(train_h5ad)

    fit_summary = fit_deep_features_on_h5ad(
        input_h5ad=train_h5ad,
        output_dir=tmp_path / "deep_model_multilevel_schema",
        feature_obsm_key="X_pca",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=1.0,
        config=DeepFeatureConfig(
            method="autoencoder",
            latent_dim=3,
            hidden_dim=12,
            layers=1,
            neighbor_k=3,
            epochs=2,
            batch_size=8,
            validation="none",
            output_embedding="intrinsic",
            save_model=True,
        ),
        seed=9,
    )

    new = ad.AnnData(X=features.copy())
    new.obsm["X_other"] = features.copy()
    new.obs["cell_x"] = coords[:, 0]
    new.obs["cell_y"] = coords[:, 1]
    new_h5ad = tmp_path / "new_multilevel_pretrained_schema.h5ad"
    new.write_h5ad(new_h5ad)

    with pytest.raises(ValueError, match="Input obsm key mismatch"):
        run_multilevel_ot_on_h5ad(
            input_h5ad=new_h5ad,
            output_dir=tmp_path / "out_multilevel_pretrained_schema",
            feature_obsm_key="X_other",
            spatial_x_key="cell_x",
            spatial_y_key="cell_y",
            spatial_scale=1.0,
            n_clusters=2,
            atoms_per_cluster=2,
            radius_um=2.0,
            stride_um=4.0,
            min_cells=4,
            max_subregions=4,
            lambda_x=0.5,
            lambda_y=1.0,
            geometry_eps=0.03,
            ot_eps=0.03,
            rho=0.5,
            geometry_samples=32,
            compressed_support_size=8,
            align_iters=1,
            n_init=1,
            allow_convex_hull_fallback=True,
            max_iter=1,
            tol=1e-4,
            basic_niche_size_um=None,
            seed=5,
            compute_device="cpu",
            deep_config=DeepFeatureConfig(
                method="autoencoder",
                output_embedding="intrinsic",
                pretrained_model=str(fit_summary["outputs"]["deep_feature_model"]),
                device="cpu",
            ),
        )


def test_transform_h5ad_with_deep_model_rejects_spatial_scale_mismatch(tmp_path) -> None:
    rng = np.random.default_rng(80)
    features = rng.normal(size=(20, 4)).astype(np.float32)
    coords = rng.normal(size=(20, 2)).astype(np.float32)
    train = ad.AnnData(X=features.copy())
    train.obsm["X_pca"] = features.copy()
    train.obs["cell_x"] = coords[:, 0]
    train.obs["cell_y"] = coords[:, 1]
    train_h5ad = tmp_path / "train_scale.h5ad"
    train.write_h5ad(train_h5ad)

    fit_summary = fit_deep_features_on_h5ad(
        input_h5ad=train_h5ad,
        output_dir=tmp_path / "deep_model_scale",
        feature_obsm_key="X_pca",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=0.5,
        config=DeepFeatureConfig(
            method="graph_autoencoder",
            latent_dim=3,
            hidden_dim=12,
            layers=1,
            neighbor_k=3,
            radius_um=1.0,
            short_radius_um=0.75,
            mid_radius_um=1.5,
            graph_layers=1,
            epochs=2,
            batch_size=8,
            validation="none",
            output_embedding="joint",
            save_model=True,
        ),
        seed=9,
    )

    new = ad.AnnData(X=features.copy())
    new.obsm["X_pca"] = features.copy()
    new.obs["cell_x"] = coords[:, 0]
    new.obs["cell_y"] = coords[:, 1]
    new_h5ad = tmp_path / "new_scale.h5ad"
    new.write_h5ad(new_h5ad)

    with pytest.raises(ValueError, match="Spatial scale mismatch"):
        transform_h5ad_with_deep_model(
            model_path=fit_summary["outputs"]["deep_feature_model"],
            input_h5ad=new_h5ad,
            output_h5ad=tmp_path / "new_scale_embedded.h5ad",
            feature_obsm_key="X_pca",
            spatial_x_key="cell_x",
            spatial_y_key="cell_y",
            spatial_scale=1.0,
        )


def test_load_multilevel_config_active_path(tmp_path) -> None:
    config_path = tmp_path / "multilevel.toml"
    config_path.write_text(
        """
[paths]
input_h5ad = "input.h5ad"
output_dir = "outputs/run"
feature_obsm_key = "X_pca"

[ot]
n_clusters = 3
atoms_per_cluster = 4
compute_device = "cpu"

	[deep]
	method = "graph_autoencoder"
	latent_dim = 5
	epochs = 4
	output_embedding = "joint"
	validation_context_mode = "inductive"
	"""
    )
    config = load_multilevel_config(config_path)
    assert config.paths.feature_obsm_key == "X_pca"
    assert config.ot.n_clusters == 3
    assert config.deep.method == "graph_autoencoder"
    assert config.deep.latent_dim == 5
    assert config.deep.validation_context_mode == "inductive"


def test_active_deep_config_requires_explicit_output_embedding(tmp_path) -> None:
    config_path = tmp_path / "multilevel_missing_output.toml"
    config_path.write_text(
        """
[paths]
input_h5ad = "input.h5ad"
output_dir = "outputs/run"
feature_obsm_key = "X_pca"

[ot]
n_clusters = 3
atoms_per_cluster = 4

[deep]
method = "graph_autoencoder"
latent_dim = 5
"""
    )
    with pytest.raises(ValueError, match="output_embedding"):
        load_multilevel_config(config_path)


def test_count_layer_config_loads_now_that_count_reconstruction_is_supported(tmp_path) -> None:
    config_path = tmp_path / "multilevel_count.toml"
    config_path.write_text(
        """
[paths]
input_h5ad = "input.h5ad"
output_dir = "outputs/run"
feature_obsm_key = "X_pca"

[ot]
n_clusters = 2
atoms_per_cluster = 2

	[deep]
	method = "autoencoder"
	output_embedding = "intrinsic"
	count_layer = "counts"
	"""
    )
    config = load_multilevel_config(config_path)
    assert config.deep.count_layer == "counts"


def test_count_layer_counts_reads_named_layer_not_x() -> None:
    x = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    counts = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    adata = ad.AnnData(X=x.copy())
    adata.layers["counts"] = counts.copy()

    for extractor in (_extract_deep_count_target, _extract_multilevel_count_target):
        matrix, used = extractor(adata, count_layer="counts")
        assert used == "counts"
        assert np.array_equal(np.asarray(matrix), counts)


def test_fit_deep_features_supports_count_reconstruction_targets() -> None:
    rng = np.random.default_rng(2468)
    counts = rng.poisson(lam=3.0, size=(28, 9)).astype(np.float32)
    features = np.log1p(counts[:, :6]).astype(np.float32)
    coords = rng.normal(size=(28, 2)).astype(np.float32)
    result = fit_deep_features(
        features=features,
        coords_um=coords,
        config=DeepFeatureConfig(
            method="autoencoder",
            latent_dim=4,
            hidden_dim=20,
            layers=2,
            neighbor_k=4,
            epochs=2,
            batch_size=14,
            validation="none",
            output_embedding="intrinsic",
            count_layer="X",
            count_decoder_rank=5,
            count_chunk_size=4,
            count_loss_weight=0.25,
        ),
        count_matrix=counts,
        seed=23,
    )
    assert result.embedding.shape == (28, 4)
    assert result.feature_schema["count_dim"] == counts.shape[1]
    assert result.feature_schema["count_layer"] == "X"
    assert "count_reconstruction_nb_loss" in result.latent_diagnostics
    assert result.latent_diagnostics["count_target_dim"] == counts.shape[1]


def test_fit_deep_features_on_h5ad_reports_count_reconstruction(tmp_path) -> None:
    rng = np.random.default_rng(1357)
    counts = rng.poisson(lam=2.5, size=(24, 8)).astype(np.float32)
    features = np.log1p(counts[:, :5]).astype(np.float32)
    coords = rng.normal(size=(24, 2)).astype(np.float32)
    adata = ad.AnnData(X=counts.copy())
    adata.obsm["X_pca"] = features.copy()
    adata.obs["cell_x"] = coords[:, 0]
    adata.obs["cell_y"] = coords[:, 1]
    input_h5ad = tmp_path / "count_input.h5ad"
    adata.write_h5ad(input_h5ad)

    summary = fit_deep_features_on_h5ad(
        input_h5ad=input_h5ad,
        output_dir=tmp_path / "count_fit_out",
        feature_obsm_key="X_pca",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=1.0,
        config=DeepFeatureConfig(
            method="autoencoder",
            latent_dim=3,
            hidden_dim=16,
            layers=1,
            neighbor_k=3,
            epochs=2,
            batch_size=8,
            validation="none",
            output_embedding="intrinsic",
            count_layer="X",
            count_decoder_rank=4,
            count_chunk_size=3,
            count_loss_weight=0.4,
            save_model=True,
        ),
        seed=31,
    )

    count_summary = summary["deep_features"]["count_reconstruction"]
    assert isinstance(count_summary, dict)
    assert count_summary["enabled"] is True
    assert count_summary["target_layer"] == "X"
    assert count_summary["decoder_rank"] == 4
    assert count_summary["gene_chunk_size"] == 3
    assert summary["deep_features"]["latent_diagnostics"]["count_target_dim"] == counts.shape[1]


def test_deep_fit_cli_spatial_scale_overrides_config(tmp_path) -> None:
    config_path = tmp_path / "deep_fit_cli.toml"
    config_path.write_text(
        """
[paths]
input_h5ad = "input.h5ad"
output_dir = "outputs/run"
feature_obsm_key = "X_pca"
spatial_scale = 1.0
"""
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "deep-fit",
            "--config",
            str(config_path),
            "--input-h5ad",
            "cells.h5ad",
            "--output-dir",
            "out",
            "--feature-obsm-key",
            "X_pca",
            "--spatial-scale",
            "2.0",
        ]
    )
    config, _ = _resolve_deep_fit_config_from_args(args)
    assert config.paths.spatial_scale == 2.0


def test_deep_fit_cli_allows_explicit_joint_ot_embedding_opt_in() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "deep-fit",
            "--input-h5ad",
            "cells.h5ad",
            "--output-dir",
            "out",
            "--feature-obsm-key",
            "X_pca",
            "--deep-feature-method",
            "graph_autoencoder",
            "--deep-output-embedding",
            "joint",
            "--deep-allow-joint-ot-embedding",
        ]
    )
    config, _ = _resolve_deep_fit_config_from_args(args)
    assert config.deep.output_embedding == "joint"
    assert config.deep.allow_joint_ot_embedding is True


def test_basic_niche_zero_in_toml_disables_composition(tmp_path) -> None:
    config_path = tmp_path / "basic_niche_zero.toml"
    config_path.write_text(
        """
[paths]
input_h5ad = "input.h5ad"
output_dir = "outputs/run"
feature_obsm_key = "X_pca"

[ot]
n_clusters = 2
atoms_per_cluster = 2
basic_niche_size_um = 0
"""
    )
    config = load_multilevel_config(config_path)
    assert config.ot.basic_niche_size_um is None


def test_run_multilevel_ot_on_h5ad_with_deep_features(tmp_path) -> None:
    rng = np.random.default_rng(0)
    coords_a = rng.normal(loc=[0.0, 0.0], scale=0.4, size=(18, 2))
    coords_b = rng.normal(loc=[8.0, 8.0], scale=0.4, size=(18, 2))
    coords = np.vstack([coords_a, coords_b]).astype(np.float32)
    feat_a = rng.normal(loc=[0.0, 0.0, 0.0, 0.0], scale=0.15, size=(18, 4))
    feat_b = rng.normal(loc=[3.0, 3.0, 3.0, 3.0], scale=0.15, size=(18, 4))
    features = np.vstack([feat_a, feat_b]).astype(np.float32)

    adata = ad.AnnData(X=features.copy())
    adata.obsm["X_pca"] = features.copy()
    adata.obs["cell_x"] = coords[:, 0]
    adata.obs["cell_y"] = coords[:, 1]
    adata.obs["region_id"] = np.repeat(["r0", "r1"], repeats=18)
    input_h5ad = tmp_path / "tiny.h5ad"
    adata.write_h5ad(input_h5ad)

    output_dir = tmp_path / "out"
    summary = run_multilevel_ot_on_h5ad(
        input_h5ad=input_h5ad,
        output_dir=output_dir,
        feature_obsm_key="X_pca",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=1.0,
        region_obs_key="region_id",
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
        n_init=1,
        subregion_clustering_method="pooled_subregion_latent",
        allow_convex_hull_fallback=True,
        max_iter=2,
        tol=1e-4,
        basic_niche_size_um=None,
        seed=5,
        compute_device="cpu",
        deep_config=DeepFeatureConfig(
            method="graph_autoencoder",
            latent_dim=3,
            hidden_dim=16,
            layers=1,
            neighbor_k=3,
            radius_um=1.0,
            short_radius_um=0.75,
            mid_radius_um=1.5,
            graph_layers=1,
            epochs=2,
            batch_size=12,
            validation="none",
            output_embedding="joint",
            allow_joint_ot_embedding=True,
            save_model=True,
        ),
    )
    assert summary["deep_features"]["enabled"] is True
    assert summary["deep_features"]["method"] == "graph_autoencoder"
    assert summary["deep_features"]["allow_joint_ot_embedding"] is True
    assert summary["deep_features"]["ot_feature_view_warning"] == "joint_embedding_explicit_opt_in"
    assert summary["deep_features"]["batch_correction"] == "disabled"
    assert "runtime_memory" in summary["deep_features"]
    assert summary["deep_features"]["feature_schema"]["input_obsm_key"] == "X_pca"
    assert summary["deep_features"]["feature_schema"]["coordinate_keys"] == ["cell_x", "cell_y"]
    assert "deep_feature_model" in summary["outputs"]
    assert "deep_feature_history" in summary["outputs"]
    assert "deep_feature_model_meta" in summary["outputs"]
    assert "deep_feature_scaler" in summary["outputs"]
    assert "candidate_cost_diagnostics" in summary["outputs"]
    saved_summary = json.loads((output_dir / "summary.json").read_text())
    assert saved_summary["deep_features"]["enabled"] is True
    assert saved_summary["method_family"] == "multilevel_ot"
    assert saved_summary["latent_source"] == "deep_joint"
    assert saved_summary["communication_source"] == "none"
    assert saved_summary["method_stack"]["deep_feature_adapter"] == "graph_autoencoder"
    assert saved_summary["method_stack"]["cell_label_mode"] == "fitted_subregion_cluster_membership"
    assert saved_summary["method_stack"]["cell_projection_mode"] == "auxiliary_approximate_cell_scores"
    assigned_transport_cost_decomposition = saved_summary["assigned_transport_cost_decomposition"]
    assert assigned_transport_cost_decomposition["mean_transport_plus_transform_cost"] > 0.0
    assert assigned_transport_cost_decomposition["mean_transport_assignment_objective"] >= assigned_transport_cost_decomposition["mean_transport_plus_transform_cost"]
    assert np.isclose(
        assigned_transport_cost_decomposition["geometry_transport_fraction"]
        + assigned_transport_cost_decomposition["feature_transport_fraction"]
        + assigned_transport_cost_decomposition["transform_penalty_fraction"],
        1.0,
        atol=1e-5,
    )
    assert assigned_transport_cost_decomposition["mean_regularized_objective"] >= assigned_transport_cost_decomposition["mean_transport_plus_transform_cost"]
    assert saved_summary["cost_reliability"]["effective_eps_matrix_available"] is True
    assert "subregion_embedding_compactness" in saved_summary
    assert "boundary_separation" in saved_summary
    assert "transform_diagnostics" in saved_summary
    assert Path(saved_summary["outputs"]["candidate_cost_diagnostics"]).exists()
    assert any(item["code"] == "cell_projection_is_approximate_assigned_subregion" for item in saved_summary["qc_warnings"])
    assert any(item["code"] == "observed_hull_geometry_fallback_active" for item in saved_summary["qc_warnings"])
    required_summary_keys = {
        "summary_schema_version",
        "method_family",
        "latent_source",
        "communication_source",
        "deep_features",
        "geometry_source_counts",
        "convex_hull_fallback_fraction",
        "assigned_ot_fallback_fraction",
        "boundary_invariance_claim",
        "shape_leakage_diagnostics",
        "density_leakage_diagnostics",
        "subregion_construction",
        "realized_subregion_statistics",
        "normalizer_diagnostics",
    }
    assert required_summary_keys <= set(saved_summary)
    assert saved_summary["boundary_invariance_claim"] == "not_supported_observed_hull_fallback"
    assert saved_summary["subregion_construction"]["radius_used_for_membership"] is False
    assert saved_summary["radius_used_for_subregion_membership"] is False
    assert saved_summary["realized_subregion_statistics"]["n_cells"]["count"] == saved_summary["n_subregions"]
    assert saved_summary["compute_device_requested"] == "cpu"
    assert saved_summary["compute_device_used"] == "cpu"
    assert saved_summary["deep_features"]["validation_context_mode"] == "inductive"
    assert saved_summary["summary_schema_version"] == "1"
    assert saved_summary["deep_features"]["uses_absolute_coordinate_features"] is False
    assert saved_summary["deep_features"]["uses_spatial_graph"] is True
    feature_schema = saved_summary["deep_features"]["feature_schema"]
    assert feature_schema["uses_absolute_coordinate_features"] is False
    assert feature_schema["uses_spatial_graph"] is True
    assert feature_schema["graph_training_mode"] == "full_batch"
    assert feature_schema["short_graph"]["edges"] >= 0
    diag = np.load(saved_summary["outputs"]["candidate_cost_diagnostics"])
    assert "subregion_cluster_transport_costs" in diag.files
    assert "subregion_cluster_overlap_penalties" in diag.files
    assert "subregion_measure_summaries" in diag.files
    saved_cells = ad.read_h5ad(output_dir / "cells_multilevel_ot.h5ad")
    assert "mlot_projected_cluster_int" in saved_cells.obs
    assert "mlot_subregion_id" in saved_cells.obs
    assert "mlot_subregion_int" in saved_cells.obs
    assert "mlot_subregion_cluster_int" in saved_cells.obs
    assert saved_cells.obs["mlot_subregion_id"].to_numpy(dtype=np.int32).min() >= 0
    assert np.array_equal(
        saved_cells.obs["mlot_cluster_int"].to_numpy(dtype=np.int32),
        saved_cells.obs["mlot_subregion_cluster_int"].to_numpy(dtype=np.int32),
    )
    assert "mlot_projected_cluster_probs" in saved_cells.obsm
    assert "mlot_subregion_cluster_probs" in saved_cells.obsm


def test_run_multilevel_ot_on_h5ad_uses_region_geometry_json_without_hull_fallback(tmp_path) -> None:
    rng = np.random.default_rng(404)
    coords = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.2, size=(10, 2)),
            rng.normal(loc=[5.0, 5.0], scale=0.2, size=(10, 2)),
        ]
    ).astype(np.float32)
    features = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.1, size=(10, 2)),
            rng.normal(loc=[2.0, 2.0], scale=0.1, size=(10, 2)),
        ]
    ).astype(np.float32)
    adata = ad.AnnData(X=features.copy())
    adata.obsm["X_pca"] = features.copy()
    adata.obs["cell_x"] = coords[:, 0]
    adata.obs["cell_y"] = coords[:, 1]
    adata.obs["region_id"] = np.repeat(["r0", "r1"], repeats=10)
    input_h5ad = tmp_path / "regions.h5ad"
    adata.write_h5ad(input_h5ad)

    geometry_json = tmp_path / "region_geometry.json"
    geometry_json.write_text(
        json.dumps(
            {
                "coordinate_units": "um",
                "regions": [
                    {
                        "region_id": "r0",
                        "polygon_vertices": [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                    },
                    {
                        "region_id": "r1",
                        "polygon_vertices": [[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]],
                    },
                ],
            }
        )
    )

    summary = run_multilevel_ot_on_h5ad(
        input_h5ad=input_h5ad,
        output_dir=tmp_path / "region_out",
        feature_obsm_key="X_pca",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=1.0,
        region_obs_key="region_id",
        region_geometry_json=geometry_json,
        n_clusters=2,
        atoms_per_cluster=2,
        radius_um=2.0,
        stride_um=2.0,
        min_cells=8,
        max_subregions=4,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=32,
        compressed_support_size=4,
        align_iters=1,
        n_init=1,
        allow_convex_hull_fallback=False,
        max_iter=1,
        tol=1e-4,
        basic_niche_size_um=None,
        shape_diagnostics=False,
        seed=11,
        compute_device="cpu",
    )
    assert summary["geometry_fallback_fraction"] == 0.0
    assert summary["geometry_source_counts"] == {"polygon": 2}
    assert summary["boundary_invariance_claim"] == "supported_with_explicit_geometry"
    assert summary["shape_diagnostics_enabled"] is False
    assert summary["shape_leakage_balanced_accuracy"] is None
    assert "subregion" in summary["method_layers"]["layer_1_subregion_formation"]
    assert "pooled matrix" in summary["method_layers"]["layer_2_subregion_heterogeneity_clustering"]
    assert "downstream projections" in summary["method_layers"]["layer_3_projection_and_visualization"]
    assert summary["capabilities"]["spot_level_latent_charts_implemented"] is True
    assert summary["method_stack"]["spot_level_latent_projection"] == "balanced_ot_atom_barycentric_mds_over_cluster_atom_posteriors"
    assert "spot_level_latent" in summary["outputs"]
    spot_latent_path = Path(summary["outputs"]["spot_level_latent"])
    assert spot_latent_path.exists()
    spot_latent = np.load(spot_latent_path)
    assert spot_latent["latent_coords"].shape[1] == 2
    assert spot_latent["within_coords"].shape == spot_latent["latent_coords"].shape
    assert spot_latent["cluster_anchors"].shape[1] == 2
    assert spot_latent["atom_embedding"].shape[2] == 2
    assert spot_latent["aligned_coords"].shape[1] == 2
    assert np.allclose(spot_latent["atom_posteriors"].sum(axis=1), 1.0, atol=1e-5)
    assert spot_latent["posterior_entropy"].shape[0] == spot_latent["latent_coords"].shape[0]
    assert spot_latent["normalized_posterior_entropy"].shape[0] == spot_latent["latent_coords"].shape[0]
    assert spot_latent["atom_argmax"].shape[0] == spot_latent["latent_coords"].shape[0]
    assert spot_latent["temperature_used"].shape[0] == spot_latent["latent_coords"].shape[0]
    assert spot_latent["temperature_cost_gap"].shape[0] == spot_latent["latent_coords"].shape[0]
    assert spot_latent["temperature_fixed"].shape[0] == spot_latent["latent_coords"].shape[0]
    assert spot_latent["posterior_entropy_cost_gap"].shape[0] == spot_latent["latent_coords"].shape[0]
    assert spot_latent["posterior_entropy_fixed"].shape[0] == spot_latent["latent_coords"].shape[0]
    assert summary["spot_level_latent"]["coordinate_scope"] == "cluster_atom_measure_mds_anchors_plus_atom_posterior_barycentric_within_cluster_residual"
    assert summary["spot_level_latent"]["chart_learning_mode"] == "model_grounded_atom_distance_mds_without_fisher_labels"
    assert summary["spot_level_latent"]["validation_role"] == "diagnostic_visualization_not_independent_evidence"
    assert summary["spot_level_latent"]["unsupervised_baseline_required_for_validation"] is True
    assert summary["spot_level_latent"]["label_permutation_control_recommended"] is False
    assert summary["spot_level_latent"]["latent_refinement"] == "atom_posterior_barycenter_without_local_pca_radius_equalization"
    assert summary["spot_level_latent"]["includes_aligned_coordinates_in_chart_features"] is False
    assert summary["spot_level_latent"]["uses_forced_cluster_local_radius"] is False
    assert summary["spot_level_latent"]["temperature_mode"] == "auto_entropy"
    assert summary["spot_level_latent"]["cluster_anchor_distance_method"] == "balanced_ot"
    assert summary["spot_level_latent"]["cluster_anchor_distance_requested_method"] == "balanced_ot"
    assert summary["spot_level_latent"]["cluster_anchor_distance_effective_method"] == "balanced_ot"
    assert summary["spot_level_latent"]["cluster_anchor_ot_fallback_fraction"] == 0.0
    assert summary["spot_level_latent"]["cluster_anchor_mds_status"] in {
        "interpretable",
        "diagnostic_only",
        "not_geometrically_interpretable",
    }
    assert "cluster_anchor_mds_stress" in summary["spot_level_latent"]
    assert "temperature_cost_gap_summary" in summary["spot_level_latent"]
    assert "normalized_posterior_entropy_fixed_summary" in summary["spot_level_latent"]
    assert summary["spot_level_latent"]["posterior_entropy_summary"]["count"] == spot_latent["latent_coords"].shape[0]
    saved = ad.read_h5ad(summary["outputs"]["h5ad"])
    assert "mlot_spot_latent_coords" in saved.obsm
    assert "mlot_spot_latent_unweighted_coords" in saved.obsm
    assert "mlot_spot_latent_confidence_weighted_coords" in saved.obsm
    assert saved.obsm["mlot_spot_latent_coords"].shape == (adata.n_obs, 2)
    assert "mlot_spot_latent_cluster_int" in saved.obs
    assert "mlot_spot_latent_posterior_entropy" in saved.obs
    assert saved.uns["multilevel_ot"]["method_layers"] == summary["method_layers"]
    assert saved.uns["multilevel_ot"]["subregion_clustering_method"] == summary["subregion_clustering_method"]
    assert saved.uns["multilevel_ot"]["subregion_clustering_uses_spatial"] == summary["subregion_clustering_uses_spatial"]
    assert saved.uns["multilevel_ot"]["subregion_clustering_feature_space"] == summary["subregion_clustering_feature_space"]
    assert saved.uns["multilevel_ot"]["spot_level_latent_mode"] == "atom_barycentric_mds"
    assert (
        saved.uns["multilevel_ot"]["spot_level_latent_projection_mode"]
        == "balanced_ot_atom_barycentric_mds_over_cluster_atom_posteriors"
    )
    assert saved.uns["multilevel_ot"]["spot_level_latent_cluster_anchor_distance_method"] == "balanced_ot"
    assert (
        saved.uns["multilevel_ot"]["spot_level_latent_validation_role"]
        == "diagnostic_visualization_not_independent_evidence"
    )
    assert spot_latent["spot_latent_mode"].item() == "atom_barycentric_mds"
    assert spot_latent["latent_projection_mode"].item() == "balanced_ot_atom_barycentric_mds_over_cluster_atom_posteriors"
    assert spot_latent["chart_learning_mode"].item() == "model_grounded_atom_distance_mds_without_fisher_labels"
    assert spot_latent["validation_role"].item() == "diagnostic_visualization_not_independent_evidence"
    assert spot_latent["temperature_mode"].item() == "auto_entropy"
    assert spot_latent["cluster_anchor_distance_method"].item() == "balanced_ot"
    assert spot_latent["cluster_anchor_distance_requested_method"].item() == "balanced_ot"
    assert spot_latent["cluster_anchor_distance_effective_method"].item() == "balanced_ot"
    assert spot_latent["cluster_anchor_distance"].shape == (2, 2)
    assert spot_latent["cluster_anchor_ot_fallback_matrix"].shape == (2, 2)
    assert spot_latent["cluster_anchor_solver_status_matrix"].shape == (2, 2)
    assert float(spot_latent["cluster_anchor_ot_fallback_fraction"].item()) == 0.0
    assert spot_latent["atom_mds_stress"].shape == (2,)
    assert "cell_spot_latent_unweighted_coords" in spot_latent.files
    assert "cell_spot_latent_confidence_weighted_coords" in spot_latent.files
    assert bool(spot_latent["unsupervised_baseline_required_for_validation"].item()) is True
    assert bool(spot_latent["label_permutation_control_recommended"].item()) is False
    assert bool(spot_latent["includes_aligned_coordinates_in_chart_features"].item()) is False
    assert bool(spot_latent["uses_forced_cluster_local_radius"].item()) is False
    assert "latent_anchor_repulsion_min_distance" not in spot_latent.files


def test_region_geometry_json_rejects_unknown_units_and_bad_affine(tmp_path) -> None:
    bad_units = tmp_path / "bad_units.json"
    bad_units.write_text(
        json.dumps(
            {
                "coordinate_units": "pixels",
                "regions": [
                    {
                        "region_id": "r0",
                        "polygon_vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="coordinate_units"):
        _load_region_geometry_json(
            bad_units,
            region_ids=["r0"],
            subregion_members=[np.array([0, 1, 2], dtype=np.int32)],
            spatial_scale=2.0,
        )

    bad_affine = tmp_path / "bad_affine.json"
    bad_affine.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region_id": "r0",
                        "mask": [[1, 1], [1, 1]],
                        "affine": [[1.0, 0.0], [0.0, 1.0]],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="affine"):
        _load_region_geometry_json(
            bad_affine,
            region_ids=["r0"],
            subregion_members=[np.array([0, 1, 2], dtype=np.int32)],
            spatial_scale=2.0,
        )


def test_explicit_region_min_cell_filter_keeps_geometry_descriptors_aligned(tmp_path) -> None:
    rng = np.random.default_rng(405)
    coords = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.1, size=(4, 2)),
            rng.normal(loc=[100.0, 100.0], scale=0.1, size=(10, 2)),
            rng.normal(loc=[200.0, 200.0], scale=0.1, size=(10, 2)),
        ]
    ).astype(np.float32)
    features = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.1, size=(4, 2)),
            rng.normal(loc=[1.0, 1.0], scale=0.1, size=(10, 2)),
            rng.normal(loc=[2.0, 2.0], scale=0.1, size=(10, 2)),
        ]
    ).astype(np.float32)
    adata = ad.AnnData(X=features.copy())
    adata.obsm["X_pca"] = features.copy()
    adata.obs["cell_x"] = coords[:, 0]
    adata.obs["cell_y"] = coords[:, 1]
    adata.obs["region_id"] = ["drop"] * 4 + ["keep_a"] * 10 + ["keep_b"] * 10
    input_h5ad = tmp_path / "regions_with_small_first.h5ad"
    adata.write_h5ad(input_h5ad)

    geometry_json = tmp_path / "region_geometry.json"
    geometry_json.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region_id": "drop",
                        "polygon_vertices": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                    },
                    {
                        "region_id": "keep_a",
                        "polygon_vertices": [[95.0, 95.0], [105.0, 95.0], [105.0, 105.0], [95.0, 105.0]],
                    },
                    {
                        "region_id": "keep_b",
                        "polygon_vertices": [[195.0, 195.0], [205.0, 195.0], [205.0, 205.0], [195.0, 205.0]],
                    },
                ],
            }
        )
    )

    output_dir = tmp_path / "region_out"
    summary = run_multilevel_ot_on_h5ad(
        input_h5ad=input_h5ad,
        output_dir=output_dir,
        feature_obsm_key="X_pca",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=1.0,
        region_obs_key="region_id",
        region_geometry_json=geometry_json,
        n_clusters=2,
        atoms_per_cluster=2,
        radius_um=2.0,
        stride_um=2.0,
        min_cells=8,
        max_subregions=4,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=32,
        compressed_support_size=4,
        align_iters=1,
        n_init=1,
        allow_convex_hull_fallback=False,
        max_iter=1,
        tol=1e-4,
        basic_niche_size_um=None,
        shape_diagnostics=False,
        seed=12,
        compute_device="cpu",
    )

    subregions = pd.read_parquet(output_dir / "subregions_multilevel_ot.parquet")
    assert summary["n_subregions"] == 2
    assert summary["geometry_source_counts"] == {"polygon": 2}
    assert set(subregions["shape_descriptor_source"]) == {"explicit_polygon"}
    assert float(subregions["shape_area"].min()) > 50.0


def test_run_multilevel_ot_on_h5ad_reports_deep_count_reconstruction(tmp_path) -> None:
    rng = np.random.default_rng(2024)
    coords_a = rng.normal(loc=[0.0, 0.0], scale=0.5, size=(16, 2))
    coords_b = rng.normal(loc=[6.0, 6.0], scale=0.5, size=(16, 2))
    coords = np.vstack([coords_a, coords_b]).astype(np.float32)
    counts_a = rng.poisson(lam=2.0, size=(16, 7)).astype(np.float32)
    counts_b = rng.poisson(lam=5.0, size=(16, 7)).astype(np.float32)
    counts = np.vstack([counts_a, counts_b]).astype(np.float32)
    features = np.log1p(counts[:, :4]).astype(np.float32)

    adata = ad.AnnData(X=counts.copy())
    adata.obsm["X_pca"] = features.copy()
    adata.obs["cell_x"] = coords[:, 0]
    adata.obs["cell_y"] = coords[:, 1]
    input_h5ad = tmp_path / "count_multilevel.h5ad"
    adata.write_h5ad(input_h5ad)

    summary = run_multilevel_ot_on_h5ad(
        input_h5ad=input_h5ad,
        output_dir=tmp_path / "count_multilevel_out",
        feature_obsm_key="X_pca",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=1.0,
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
        n_init=1,
        allow_convex_hull_fallback=True,
        max_iter=2,
        tol=1e-4,
        basic_niche_size_um=None,
        seed=17,
        compute_device="cpu",
        deep_config=DeepFeatureConfig(
            method="autoencoder",
            latent_dim=3,
            hidden_dim=16,
            layers=1,
            neighbor_k=3,
            epochs=2,
            batch_size=8,
            validation="none",
            output_embedding="intrinsic",
            count_layer="X",
            count_decoder_rank=4,
            count_chunk_size=3,
            count_loss_weight=0.3,
        ),
    )

    count_summary = summary["deep_features"]["count_reconstruction"]
    assert isinstance(count_summary, dict)
    assert count_summary["enabled"] is True
    assert count_summary["target_layer"] == "X"
    assert summary["geometry_source_counts"] == {"observed_point_cloud": summary["n_subregions"]}
    assert summary["shape_descriptor_source_counts"] == {"observed_point_cloud": summary["n_subregions"]}
    assert summary["boundary_invariance_claim"] == "observed_geometry_normalized_not_full_shape_invariant"
    assert summary["deep_features"]["latent_diagnostics"]["count_target_dim"] == counts.shape[1]
    saved_summary = json.loads((tmp_path / "count_multilevel_out" / "summary.json").read_text())
    assert saved_summary["deep_features"]["count_reconstruction"]["target_layer"] == "X"
    assert "latent_diagnostics" in saved_summary["deep_features"]
    assert "runtime_memory" in saved_summary["deep_features"]


def test_multilevel_ot_marks_visualization_like_feature_space_in_summary(tmp_path) -> None:
    rng = np.random.default_rng(222)
    coords_a = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(16, 2))
    coords_b = rng.normal(loc=[5.0, 5.0], scale=0.3, size=(16, 2))
    coords = np.vstack([coords_a, coords_b]).astype(np.float32)
    features = np.vstack(
        [
            rng.normal(loc=[-1.0, -1.0], scale=0.15, size=(16, 2)),
            rng.normal(loc=[1.0, 1.0], scale=0.15, size=(16, 2)),
        ]
    ).astype(np.float32)

    adata = ad.AnnData(X=np.zeros((32, 2), dtype=np.float32))
    adata.obsm["X_tsne_demo"] = features.copy()
    adata.obs["cell_x"] = coords[:, 0]
    adata.obs["cell_y"] = coords[:, 1]
    input_h5ad = tmp_path / "tsne_like_input.h5ad"
    adata.write_h5ad(input_h5ad)

    summary = run_multilevel_ot_on_h5ad(
        input_h5ad=input_h5ad,
        output_dir=tmp_path / "out_tsne_like",
        feature_obsm_key="X_tsne_demo",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=1.0,
        n_clusters=2,
        atoms_per_cluster=2,
        radius_um=1.5,
        stride_um=3.0,
        min_cells=6,
        max_subregions=6,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=32,
        compressed_support_size=8,
        align_iters=1,
        n_init=1,
        allow_convex_hull_fallback=True,
        max_iter=2,
        tol=1e-4,
        basic_niche_size_um=None,
        seed=17,
        compute_device="cpu",
    )

    assert summary["feature_embedding_warning"] == "visualization_embedding_like"
    assert summary["method_stack"]["feature_space_kind"] == "visualization_like_embedding"
    assert any(item["code"] == "visualization_like_feature_space" for item in summary["qc_warnings"])


def test_run_multilevel_ot_on_h5ad_with_autoencoder_context_features(tmp_path) -> None:
    rng = np.random.default_rng(44)
    coords_a = rng.normal(loc=[0.0, 0.0], scale=0.35, size=(20, 2))
    coords_b = rng.normal(loc=[7.5, 7.5], scale=0.35, size=(20, 2))
    coords = np.vstack([coords_a, coords_b]).astype(np.float32)
    feat_a = rng.normal(loc=[0.0, 1.0, 0.0, 1.0, 0.5], scale=0.2, size=(20, 5))
    feat_b = rng.normal(loc=[3.0, 2.0, 3.0, 2.0, 2.5], scale=0.2, size=(20, 5))
    features = np.vstack([feat_a, feat_b]).astype(np.float32)

    adata = ad.AnnData(X=features.copy())
    adata.obsm["X_pca"] = features.copy()
    adata.obs["cell_x"] = coords[:, 0]
    adata.obs["cell_y"] = coords[:, 1]
    input_h5ad = tmp_path / "tiny_autoencoder_context.h5ad"
    adata.write_h5ad(input_h5ad)

    output_dir = tmp_path / "out_autoencoder_context"
    summary = run_multilevel_ot_on_h5ad(
        input_h5ad=input_h5ad,
        output_dir=output_dir,
        feature_obsm_key="X_pca",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=1.0,
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
        n_init=1,
        allow_convex_hull_fallback=True,
        max_iter=2,
        tol=1e-4,
        basic_niche_size_um=None,
        seed=11,
        compute_device="cpu",
        deep_config=DeepFeatureConfig(
            method="autoencoder",
            latent_dim=4,
            hidden_dim=24,
            layers=2,
            neighbor_k=4,
            epochs=3,
            batch_size=16,
            validation="none",
            output_embedding="context",
            save_model=False,
        ),
    )

    assert summary["deep_features"]["enabled"] is True
    assert summary["deep_features"]["method"] == "autoencoder"
    assert summary["deep_features"]["output_embedding"] == "context"
    assert summary["latent_source"] == "deep_context"
    assert summary["deep_features"]["uses_spatial_graph"] is False
    saved_summary = json.loads((output_dir / "summary.json").read_text())
    assert saved_summary["deep_features"]["method"] == "autoencoder"
    assert saved_summary["deep_features"]["output_embedding"] == "context"
    assert saved_summary["latent_source"] == "deep_context"
    assert saved_summary["deep_features"]["uses_spatial_graph"] is False
    assert "latent_diagnostics" in saved_summary["deep_features"]
    assert saved_summary["deep_features"]["ot_feature_view_warning"] is None


def test_run_multilevel_ot_on_h5ad_rejects_joint_ot_without_explicit_opt_in(tmp_path) -> None:
    rng = np.random.default_rng(101)
    coords = rng.normal(size=(24, 2)).astype(np.float32)
    features = rng.normal(size=(24, 4)).astype(np.float32)

    adata = ad.AnnData(X=features.copy())
    adata.obsm["X_pca"] = features.copy()
    adata.obs["cell_x"] = coords[:, 0]
    adata.obs["cell_y"] = coords[:, 1]
    input_h5ad = tmp_path / "joint_guard.h5ad"
    adata.write_h5ad(input_h5ad)

    with pytest.raises(ValueError, match="requires explicit opt-in"):
        run_multilevel_ot_on_h5ad(
            input_h5ad=input_h5ad,
            output_dir=tmp_path / "out",
            feature_obsm_key="X_pca",
            spatial_x_key="cell_x",
            spatial_y_key="cell_y",
            spatial_scale=1.0,
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
            compressed_support_size=8,
            align_iters=1,
            n_init=1,
            allow_convex_hull_fallback=True,
            max_iter=2,
            tol=1e-4,
            basic_niche_size_um=None,
            seed=5,
            compute_device="cpu",
            deep_config=DeepFeatureConfig(
                method="graph_autoencoder",
                latent_dim=3,
                hidden_dim=16,
                layers=1,
                neighbor_k=3,
                radius_um=1.0,
                short_radius_um=0.75,
                mid_radius_um=1.5,
                graph_layers=1,
                epochs=2,
                batch_size=12,
                validation="none",
                output_embedding="joint",
                save_model=False,
            ),
        )


def test_run_multilevel_ot_on_h5ad_accepts_full_gene_x(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(2026)
    coords_a = rng.normal(loc=[0.0, 0.0], scale=0.4, size=(16, 2))
    coords_b = rng.normal(loc=[7.0, 7.0], scale=0.4, size=(16, 2))
    coords = np.vstack([coords_a, coords_b]).astype(np.float32)
    counts_a = rng.poisson(lam=[8, 1, 0, 0, 2, 1], size=(16, 6)).astype(np.float32)
    counts_b = rng.poisson(lam=[0, 0, 7, 6, 1, 2], size=(16, 6)).astype(np.float32)
    counts = np.vstack([counts_a, counts_b]).astype(np.float32)

    adata = ad.AnnData(X=counts.copy())
    adata.obs["cell_x"] = coords[:, 0]
    adata.obs["cell_y"] = coords[:, 1]
    input_h5ad = tmp_path / "full_gene_input.h5ad"
    adata.write_h5ad(input_h5ad)

    monkeypatch.setenv("SPATIAL_OT_X_SVD_COMPONENTS", "3")
    output_dir = tmp_path / "full_gene_out"
    summary = run_multilevel_ot_on_h5ad(
        input_h5ad=input_h5ad,
        output_dir=output_dir,
        feature_obsm_key="X",
        spatial_x_key="cell_x",
        spatial_y_key="cell_y",
        spatial_scale=1.0,
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
        n_init=1,
        allow_convex_hull_fallback=True,
        max_iter=2,
        tol=1e-4,
        basic_niche_size_um=None,
        seed=5,
        compute_device="cpu",
        deep_config=DeepFeatureConfig(method="none"),
    )

    assert summary["feature_obsm_key_requested"] == "X"
    assert summary["feature_input_mode"] == "X"
    assert summary["feature_source"]["preprocessing"] == "library_size_normalize_log1p_truncated_svd"
    assert summary["feature_dim"] == 3
    saved_summary = json.loads((output_dir / "summary.json").read_text())
    assert saved_summary["feature_source"]["svd_components_used"] == 3


def test_autoencoder_latent_diagnostics_show_distinct_branches() -> None:
    rng = np.random.default_rng(1234)
    features = rng.normal(size=(48, 6)).astype(np.float32)
    coords = rng.normal(size=(48, 2)).astype(np.float32)
    encoder = SpatialOTFeatureEncoder(
        DeepFeatureConfig(
            method="autoencoder",
            latent_dim=4,
            hidden_dim=24,
            layers=2,
            neighbor_k=4,
            epochs=3,
            batch_size=16,
            validation="none",
            output_embedding="intrinsic",
        )
    )
    encoder.fit(features=features, coords_um=coords, seed=19)
    diagnostics = encoder.latent_diagnostics
    assert diagnostics["selected_embedding"] == "intrinsic"
    assert diagnostics["intrinsic_context_allclose"] is False
    assert diagnostics["intrinsic_context_mean_abs_correlation"] >= 0.0
    assert "intrinsic_context_top_canonical_correlation" in diagnostics
    assert "intrinsic_context_distance_correlation" in diagnostics
    assert "intrinsic_context_hsic_rbf" in diagnostics
    assert "intrinsic_input_r2" in diagnostics
    assert "context_context_target_r2" in diagnostics
    assert "intrinsic_coordinate_r2" in diagnostics
    assert "runtime_memory" in diagnostics


def test_graph_autoencoder_full_batch_guard_rejects_large_fit() -> None:
    rng = np.random.default_rng(4321)
    features = rng.normal(size=(10, 4)).astype(np.float32)
    coords = rng.normal(size=(10, 2)).astype(np.float32)
    encoder = SpatialOTFeatureEncoder(
        DeepFeatureConfig(
            method="graph_autoencoder",
            latent_dim=3,
            hidden_dim=12,
            layers=1,
            neighbor_k=3,
            radius_um=1.0,
            short_radius_um=0.75,
            mid_radius_um=1.5,
            graph_layers=1,
            epochs=1,
            batch_size=8,
            validation="none",
            output_embedding="joint",
            full_batch_max_cells=5,
        )
    )
    with pytest.raises(ValueError, match="full-batch execution"):
        encoder.fit(features=features, coords_um=coords, seed=7)


def test_multilevel_ot_requires_explicit_umap_opt_in(tmp_path) -> None:
    rng = np.random.default_rng(987)
    features = rng.normal(size=(12, 3)).astype(np.float32)
    coords = rng.normal(size=(12, 2)).astype(np.float32)
    adata = ad.AnnData(X=features.copy())
    adata.obsm["X_umap_demo"] = features.copy()
    adata.obs["cell_x"] = coords[:, 0]
    adata.obs["cell_y"] = coords[:, 1]
    input_h5ad = tmp_path / "umap_input.h5ad"
    adata.write_h5ad(input_h5ad)

    with pytest.raises(ValueError, match="requires explicit opt-in"):
        run_multilevel_ot_on_h5ad(
            input_h5ad=input_h5ad,
            output_dir=tmp_path / "out",
            feature_obsm_key="X_umap_demo",
            spatial_x_key="cell_x",
            spatial_y_key="cell_y",
            spatial_scale=1.0,
            n_clusters=2,
            atoms_per_cluster=2,
            radius_um=2.0,
            stride_um=4.0,
            min_cells=4,
            max_subregions=4,
            lambda_x=0.5,
            lambda_y=1.0,
            geometry_eps=0.03,
            ot_eps=0.03,
            rho=0.5,
            geometry_samples=32,
            compressed_support_size=8,
            align_iters=1,
            n_init=1,
            allow_convex_hull_fallback=True,
            max_iter=1,
            tol=1e-4,
            basic_niche_size_um=None,
            seed=5,
            compute_device="cpu",
        )
