from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from spatial_ot.multilevel import plot_sample_niche_maps_from_run_dir


def test_plot_sample_niche_maps_from_run_dir_writes_one_plot_per_sample(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    adata = ad.AnnData(X=np.zeros((6, 0), dtype=np.float32))
    adata.obs["sample_id"] = ["sample_a", "sample_a", "sample_a", "sample_b", "sample_b", "sample_b"]
    adata.obs["source_h5ad"] = [
        "sample_a_cells_marker_genes_umap3d.h5ad",
        "sample_a_cells_marker_genes_umap3d.h5ad",
        "sample_a_cells_marker_genes_umap3d.h5ad",
        "sample_b_cells_marker_genes_umap3d.h5ad",
        "sample_b_cells_marker_genes_umap3d.h5ad",
        "sample_b_cells_marker_genes_umap3d.h5ad",
    ]
    adata.obs["cell_x"] = np.asarray([0.0, 1.0, 2.0, 0.5, 1.5, 2.5], dtype=np.float32)
    adata.obs["cell_y"] = np.asarray([0.0, 0.5, 1.0, 2.0, 2.5, 3.0], dtype=np.float32)
    adata.obs["original_cell_x"] = adata.obs["cell_x"].to_numpy(dtype=np.float32)
    adata.obs["original_cell_y"] = adata.obs["cell_y"].to_numpy(dtype=np.float32)
    adata.obs["mlot_cluster_int"] = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int32)
    adata.obs["mlot_cluster_id"] = ["C0", "C1", "C0", "C1", "C1", "C0"]
    adata.obs["mlot_cluster_hex"] = ["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e", "#ff7f0e", "#1f77b4"]
    adata.uns["multilevel_ot"] = {
        "spatial_scale": 2.0,
        "spatial_x_key": "pooled_cell_x",
        "spatial_y_key": "pooled_cell_y",
    }
    adata.write_h5ad(run_dir / "cells_multilevel_ot.h5ad")

    manifest = plot_sample_niche_maps_from_run_dir(
        run_dir=run_dir,
        output_dir=run_dir / "sample_niche_plots",
        sample_obs_key="sample_id",
        plot_spatial_x_key="original_cell_x",
        plot_spatial_y_key="original_cell_y",
    )

    assert manifest["n_samples"] == 2
    assert manifest["plot_spatial_x_key"] == "original_cell_x"
    assert manifest["plot_spatial_y_key"] == "original_cell_y"
    assert Path(str(manifest["manifest_json"])).exists()

    plots = manifest["plots"]
    assert isinstance(plots, list)
    assert len(plots) == 2
    sample_ids = {str(item["sample_id"]) for item in plots}
    assert sample_ids == {"sample_a", "sample_b"}
    for item in plots:
        output_png = Path(str(item["output_png"]))
        assert output_png.exists()
        assert output_png.stat().st_size > 0
