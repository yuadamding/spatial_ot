from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from spatial_ot.pooling import pool_h5ads_in_directory


def _write_demo_h5ad(path: Path, *, offset: float) -> None:
    x = np.asarray(
        [
            [1.0 + offset, 0.0, 3.0, 0.0],
            [0.0, 2.0 + offset, 0.0, 4.0],
            [5.0, 0.0, 6.0 + offset, 0.0],
        ],
        dtype=np.float32,
    )
    adata = ad.AnnData(X=x, var=pd.DataFrame(index=["gene_a", "gene_b", "gene_c", "gene_d"]))
    adata.obs["cell_id"] = [f"cell_{i}" for i in range(3)]
    adata.obs["cell_x"] = np.asarray([offset + 0.0, offset + 1.0, offset + 2.0], dtype=np.float32)
    adata.obs["cell_y"] = np.asarray([offset + 0.5, offset + 1.5, offset + 2.5], dtype=np.float32)
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


def test_pool_h5ads_in_directory_keeps_sample_labels_and_separates_coordinates(tmp_path: Path) -> None:
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
    max_a_x = float(np.asarray(pooled.obs.loc[sample_a, "pooled_cell_x"], dtype=np.float32).max())
    min_b_x = float(np.asarray(pooled.obs.loc[sample_b, "pooled_cell_x"], dtype=np.float32).min())
    assert min_b_x > max_a_x

    assert all(str(index).startswith(("sample_a:", "sample_b:")) for index in pooled.obs_names)
