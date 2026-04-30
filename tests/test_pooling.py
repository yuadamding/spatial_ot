from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatial_ot.feature_source import (
    default_precomputed_x_feature_key,
    prepare_h5ad_feature_cache,
)
from spatial_ot.pooling import (
    distribute_pooled_feature_cache_to_inputs,
    pool_h5ads_in_directory,
)


def _write_demo_h5ad(path: Path, *, offset: float) -> None:
    x = np.asarray(
        [
            [1.0 + offset, 0.0, 3.0, 0.0],
            [0.0, 2.0 + offset, 0.0, 4.0],
            [5.0, 0.0, 6.0 + offset, 0.0],
        ],
        dtype=np.float32,
    )
    adata = ad.AnnData(
        X=x, var=pd.DataFrame(index=["gene_a", "gene_b", "gene_c", "gene_d"])
    )
    adata.obs["cell_id"] = [f"cell_{i}" for i in range(3)]
    adata.obs["cell_x"] = np.asarray(
        [offset + 0.0, offset + 1.0, offset + 2.0], dtype=np.float32
    )
    adata.obs["cell_y"] = np.asarray(
        [offset + 0.5, offset + 1.5, offset + 2.5], dtype=np.float32
    )
    adata.obs["nucleus_x"] = adata.obs["cell_x"].to_numpy(dtype=np.float32)
    adata.obs["nucleus_y"] = adata.obs["cell_y"].to_numpy(dtype=np.float32)
    adata.obs["library_size"] = np.asarray([10, 11, 12], dtype=np.int32)
    adata.obsm["X_umap_marker_genes_3d"] = np.asarray(
        [
            [offset + 0.1, offset + 0.2, offset + 0.3],
            [offset + 1.1, offset + 1.2, offset + 1.3],
            [offset + 2.1, offset + 2.2, offset + 2.3],
        ],
        dtype=np.float32,
    )
    adata.write_h5ad(path)


def test_pool_h5ads_in_directory_keeps_sample_labels_and_separates_coordinates(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_demo_h5ad(input_dir / "sample_a_cells_marker_genes_umap3d.h5ad", offset=0.0)
    _write_demo_h5ad(input_dir / "sample_b_cells_marker_genes_umap3d.h5ad", offset=0.0)

    output_h5ad = tmp_path / "pooled.h5ad"
    summary = pool_h5ads_in_directory(
        input_dir=input_dir,
        output_h5ad=output_h5ad,
        feature_obsm_keys=["X", "X_umap_marker_genes_3d"],
    )

    pooled = ad.read_h5ad(output_h5ad)
    assert pooled.n_obs == 6
    assert pooled.n_vars == 4
    assert summary["n_samples"] == 2
    assert set(pooled.obs["sample_id"].astype(str)) == {"sample_a", "sample_b"}
    assert "pooled_cell_x" in pooled.obs
    assert "pooled_cell_y" in pooled.obs
    assert "original_cell_x" in pooled.obs
    assert "original_cell_y" in pooled.obs
    assert pooled.X.shape == (6, 4)
    assert pooled.var_names.tolist() == ["gene_a", "gene_b", "gene_c", "gene_d"]
    assert "X_umap_marker_genes_3d" in pooled.obsm
    assert pooled.obsm["X_umap_marker_genes_3d"].shape == (6, 3)
    assert summary["feature_keys"] == ["X", "X_umap_marker_genes_3d"]
    assert pooled.uns["pooled_inputs"]["samples_are_physically_connected"] is False

    sample_a = pooled.obs["sample_id"].astype(str) == "sample_a"
    sample_b = pooled.obs["sample_id"].astype(str) == "sample_b"
    max_a_x = float(
        np.asarray(pooled.obs.loc[sample_a, "pooled_cell_x"], dtype=np.float32).max()
    )
    min_b_x = float(
        np.asarray(pooled.obs.loc[sample_b, "pooled_cell_x"], dtype=np.float32).min()
    )
    assert min_b_x > max_a_x

    assert all(
        str(index).startswith(("sample_a:", "sample_b:")) for index in pooled.obs_names
    )


def test_pool_h5ads_in_directory_normalizes_xenium_sample_ids(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "xenium"
    input_dir.mkdir()
    for sample, offset in [("P1_CRC", 0.0), ("P2_CRC", 10.0)]:
        path = input_dir / f"xenium_{sample}_processed.h5ad"
        adata = ad.AnnData(
            X=np.ones((2, 3), dtype=np.float32),
            var=pd.DataFrame(index=["g1", "g2", "g3"]),
        )
        adata.obs["x_centroid"] = np.asarray([offset, offset + 1.0], dtype=np.float32)
        adata.obs["y_centroid"] = np.asarray([offset + 2.0, offset + 3.0], dtype=np.float32)
        adata.write_h5ad(path)

    output_h5ad = tmp_path / "xenium_pooled.h5ad"
    summary = pool_h5ads_in_directory(
        input_dir=input_dir,
        output_h5ad=output_h5ad,
        feature_obsm_keys=["X"],
        sample_glob="xenium_*_processed.h5ad",
        spatial_x_key="x_centroid",
        spatial_y_key="y_centroid",
        sample_id_prefix="xenium_",
        sample_id_suffix="_processed",
        sample_id_case="lower",
    )

    pooled = ad.read_h5ad(output_h5ad)
    assert summary["sample_ids"] == ["p1_crc", "p2_crc"]
    assert set(pooled.obs["sample_id"].astype(str)) == {"p1_crc", "p2_crc"}
    assert "original_x_centroid" in pooled.obs
    assert "original_y_centroid" in pooled.obs
    assert all(
        str(index).startswith(("p1_crc:", "p2_crc:")) for index in pooled.obs_names
    )
    assert pooled.uns["pooled_inputs"]["sample_id_prefix"] == "xenium_"
    assert pooled.uns["pooled_inputs"]["sample_id_suffix"] == "_processed"
    assert pooled.uns["pooled_inputs"]["sample_id_case"] == "lower"


def test_prepare_h5ad_feature_cache_reuses_matching_precomputed_x_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_h5ad = tmp_path / "prepared_input.h5ad"
    _write_demo_h5ad(input_h5ad, offset=0.0)

    monkeypatch.setenv("SPATIAL_OT_X_SVD_COMPONENTS", "2")
    monkeypatch.setenv("SPATIAL_OT_X_TARGET_SUM", "1000")
    prepared_key = default_precomputed_x_feature_key(requested_components=2)

    first = prepare_h5ad_feature_cache(
        input_h5ad=input_h5ad,
        feature_obsm_key="X",
        output_obsm_key=prepared_key,
    )
    assert first["prepared_feature_obsm_key"] == prepared_key
    assert first["reused_existing"] is False

    prepared = ad.read_h5ad(input_h5ad)
    assert prepared_key in prepared.obsm
    assert prepared.obsm[prepared_key].shape == (3, 2)
    assert "spatial_ot_prepared_features" in prepared.uns

    second = prepare_h5ad_feature_cache(
        input_h5ad=input_h5ad,
        feature_obsm_key="X",
        output_obsm_key=prepared_key,
    )
    assert second["prepared_feature_obsm_key"] == prepared_key
    assert second["reused_existing"] is True


def test_distribute_pooled_feature_cache_to_inputs_writes_shared_cache_back_to_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    first = input_dir / "sample_a_cells_marker_genes_umap3d.h5ad"
    second = input_dir / "sample_b_cells_marker_genes_umap3d.h5ad"
    _write_demo_h5ad(first, offset=0.0)
    _write_demo_h5ad(second, offset=5.0)

    output_h5ad = tmp_path / "pooled.h5ad"
    pool_h5ads_in_directory(
        input_dir=input_dir,
        output_h5ad=output_h5ad,
        feature_obsm_keys=["X"],
    )

    monkeypatch.setenv("SPATIAL_OT_X_SVD_COMPONENTS", "2")
    monkeypatch.setenv("SPATIAL_OT_X_TARGET_SUM", "1000")
    prepared_key = default_precomputed_x_feature_key(requested_components=2)
    prepare_h5ad_feature_cache(
        input_h5ad=output_h5ad,
        feature_obsm_key="X",
        output_obsm_key=prepared_key,
    )

    summary = distribute_pooled_feature_cache_to_inputs(
        pooled_h5ad=output_h5ad,
        input_dir=input_dir,
        prepared_obsm_key=prepared_key,
    )

    assert summary["n_inputs"] == 2
    for sample_path in (first, second):
        sample = ad.read_h5ad(sample_path)
        assert prepared_key in sample.obsm
        assert sample.obsm[prepared_key].shape == (3, 2)
        assert (
            sample.uns["spatial_ot_prepared_features"][prepared_key][
                "distributed_from_pooled"
            ]
            is True
        )
