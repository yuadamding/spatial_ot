from __future__ import annotations

import json

import anndata as ad
import numpy as np

from spatial_ot.config import DeepFeatureConfig, load_multilevel_config
from spatial_ot.deep_features import SpatialOTFeatureEncoder, fit_deep_features
from spatial_ot.multilevel_ot import run_multilevel_ot_on_h5ad


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

[deep]
method = "autoencoder"
latent_dim = 5
epochs = 4
"""
    )
    config = load_multilevel_config(config_path)
    assert config.paths.feature_obsm_key == "X_pca"
    assert config.ot.n_clusters == 3
    assert config.deep.method == "autoencoder"
    assert config.deep.latent_dim == 5


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
        seed=5,
        deep_config=DeepFeatureConfig(
            method="autoencoder",
            latent_dim=3,
            hidden_dim=16,
            layers=1,
            neighbor_k=3,
            epochs=2,
            batch_size=12,
            validation="none",
            save_model=True,
        ),
    )
    assert summary["deep_features"]["enabled"] is True
    assert summary["deep_features"]["method"] == "autoencoder"
    assert "deep_feature_model" in summary["outputs"]
    assert "deep_feature_history" in summary["outputs"]
    saved_summary = json.loads((output_dir / "summary.json").read_text())
    assert saved_summary["deep_features"]["enabled"] is True
