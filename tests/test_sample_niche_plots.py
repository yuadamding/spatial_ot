from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from spatial_ot.multilevel import (
    plot_sample_niche_maps_from_run_dir,
    plot_sample_spot_latent_maps_from_run_dir,
    plot_sample_spatial_maps_from_run_dir,
)


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
    adata.obs["cell_y"] = np.asarray([0.0, 0.5, 0.1, 2.0, 2.5, 2.1], dtype=np.float32)
    adata.obs["original_cell_x"] = adata.obs["cell_x"].to_numpy(dtype=np.float32)
    adata.obs["original_cell_y"] = adata.obs["cell_y"].to_numpy(dtype=np.float32)
    adata.obs["mlot_cluster_int"] = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int32)
    adata.obs["mlot_cluster_id"] = ["C0", "C1", "C0", "C1", "C1", "C0"]
    adata.obs["mlot_cluster_hex"] = ["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e", "#ff7f0e", "#1f77b4"]
    adata.obs["mlot_subregion_id"] = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
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
    assert manifest["rendering"] == "subregion_polygons_and_cell_scatter"
    assert manifest["views"] == ["subregion_wise_filled_polygons", "cell_wise_inherited_label_scatter"]
    assert manifest["subregion_membership_source"] == "obs[mlot_subregion_id]"
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
        assert int(item["n_filled_subregions"]) == 1


def test_plot_sample_niche_maps_recovers_subregions_from_spot_latent_npz(tmp_path: Path) -> None:
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
    adata.obs["cell_y"] = np.asarray([0.0, 0.5, 0.1, 2.0, 2.5, 2.1], dtype=np.float32)
    adata.obs["original_cell_x"] = adata.obs["cell_x"].to_numpy(dtype=np.float32)
    adata.obs["original_cell_y"] = adata.obs["cell_y"].to_numpy(dtype=np.float32)
    adata.obs["mlot_cluster_int"] = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
    adata.obs["mlot_cluster_id"] = ["C0", "C0", "C0", "C1", "C1", "C1"]
    adata.obs["mlot_cluster_hex"] = ["#1f77b4", "#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e", "#ff7f0e"]
    adata.uns["multilevel_ot"] = {
        "spatial_scale": 1.0,
        "spatial_x_key": "cell_x",
        "spatial_y_key": "cell_y",
        "spot_level_latent_npz": str(run_dir / "spot_level_latent_multilevel_ot.npz"),
    }
    adata.write_h5ad(run_dir / "cells_multilevel_ot.h5ad")
    np.savez_compressed(
        run_dir / "spot_level_latent_multilevel_ot.npz",
        cell_indices=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32),
        subregion_ids=np.asarray([10, 10, 10, 11, 11, 11], dtype=np.int32),
        cluster_labels=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        latent_coords=np.zeros((6, 2), dtype=np.float32),
        weights=np.ones(6, dtype=np.float32),
    )

    manifest = plot_sample_niche_maps_from_run_dir(
        run_dir=run_dir,
        output_dir=run_dir / "sample_niche_plots",
        sample_obs_key="sample_id",
        plot_spatial_x_key="original_cell_x",
        plot_spatial_y_key="original_cell_y",
    )

    assert manifest["rendering"] == "subregion_polygons_and_cell_scatter"
    assert manifest["subregion_membership_source"] == "spot_level_latent_npz[cell_indices,subregion_ids]"
    assert str(manifest["subregion_membership_npz"]).endswith("spot_level_latent_multilevel_ot.npz")
    plots = manifest["plots"]
    assert isinstance(plots, list)
    assert len(plots) == 2
    for item in plots:
        assert int(item["n_subregions"]) == 1
        assert int(item["n_filled_subregions"]) == 1
        output_png = Path(str(item["output_png"]))
        assert output_png.exists()
        assert output_png.stat().st_size > 0


def test_plot_sample_spatial_maps_from_run_dir_writes_one_plot_per_sample(tmp_path: Path) -> None:
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

    manifest = plot_sample_spatial_maps_from_run_dir(
        run_dir=run_dir,
        output_dir=run_dir / "sample_spatial_maps",
        sample_obs_key="sample_id",
        plot_spatial_x_key="original_cell_x",
        plot_spatial_y_key="original_cell_y",
    )

    assert manifest["n_samples"] == 2
    assert manifest["output_filename_suffix"] == "_multilevel_ot_spatial_map.png"
    assert Path(str(manifest["manifest_json"])).exists()

    plots = manifest["plots"]
    assert isinstance(plots, list)
    assert len(plots) == 2
    for item in plots:
        output_png = Path(str(item["output_png"]))
        assert output_png.exists()
        assert output_png.name.endswith("_multilevel_ot_spatial_map.png")
        assert output_png.stat().st_size > 0


def test_plot_sample_spot_latent_maps_from_run_dir_writes_one_plot_per_sample(tmp_path: Path) -> None:
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
    adata.obs["mlot_spot_latent_cluster_int"] = np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int32)
    adata.obsm["mlot_spot_latent_coords"] = np.asarray(
        [
            [0.0, 0.0],
            [0.4, 0.1],
            [0.8, 0.2],
            [0.1, 0.7],
            [0.3, 0.9],
            [0.6, 1.1],
        ],
        dtype=np.float32,
    )
    adata.uns["multilevel_ot"] = {
        "spatial_scale": 2.0,
        "spatial_x_key": "pooled_cell_x",
        "spatial_y_key": "pooled_cell_y",
        "spot_level_latent_npz": str(run_dir / "spot_level_latent_multilevel_ot.npz"),
    }
    adata.write_h5ad(run_dir / "cells_multilevel_ot.h5ad")
    np.savez_compressed(
        run_dir / "spot_level_latent_multilevel_ot.npz",
        cell_indices=np.asarray([0, 1, 2, 3, 4, 5, 0, 4], dtype=np.int32),
        subregion_ids=np.asarray([0, 0, 0, 1, 1, 1, 2, 3], dtype=np.int32),
        cluster_labels=np.asarray([0, 0, 1, 1, 1, 0, 1, 0], dtype=np.int32),
        latent_coords=np.asarray(
            [
                [0.0, 0.0],
                [0.4, 0.1],
                [0.8, 0.2],
                [0.1, 0.7],
                [0.3, 0.9],
                [0.6, 1.1],
                [0.2, 0.4],
                [0.7, 0.8],
            ],
            dtype=np.float32,
        ),
        weights=np.ones(8, dtype=np.float32),
        spot_latent_mode=np.array("atom_barycentric_mds"),
        latent_projection_mode=np.array("ot_atom_barycentric_mds_over_cluster_atom_posteriors"),
        chart_learning_mode=np.array("model_grounded_atom_distance_mds_without_fisher_labels"),
        validation_role=np.array("diagnostic_visualization_not_independent_evidence"),
    )

    manifest = plot_sample_spot_latent_maps_from_run_dir(
        run_dir=run_dir,
        output_dir=run_dir / "sample_spot_latent_plots",
        sample_obs_key="sample_id",
        plot_spatial_x_key="original_cell_x",
        plot_spatial_y_key="original_cell_y",
    )

    assert manifest["n_samples"] == 2
    assert manifest["n_plots"] == 2
    assert manifest["rendering"] == "whole_sample_within_niche_latent_rgb"
    assert manifest["max_occurrences_per_sample"] == 0
    assert (
        manifest["coordinate_scope"]
        == "cluster_atom_measure_mds_anchors_plus_atom_posterior_barycentric_within_cluster_residual"
    )
    assert manifest["chart_learning_mode"] == "model_grounded_atom_distance_mds_without_fisher_labels"
    assert manifest["validation_role"] == "diagnostic_visualization_not_independent_evidence"
    assert manifest["spot_latent_mode"] == "atom_barycentric_mds"
    assert manifest["includes_aligned_coordinates_in_chart_features"] is False
    assert manifest["uses_forced_cluster_local_radius"] is False
    assert "diagnostic visualization" in manifest["color_encoding"]
    assert manifest["latent_source"] == "occurrence_npz"
    assert manifest["subregion_id_source"] == "occurrence_npz[subregion_ids]"
    assert manifest["subregion_boundary_overlay"] == "concave_hull_of_sample_occurrence_subregion_members"
    assert manifest["latent_obsm_key"] == "mlot_spot_latent_coords"
    assert set(manifest["within_niche_latent_color_limits"]) == {"0", "1"}
    assert "display_latent_limits" in manifest
    assert set(manifest["cluster_display_anchors"]) == {"0", "1"}
    assert Path(str(manifest["manifest_json"])).exists()

    plots = manifest["plots"]
    assert isinstance(plots, list)
    assert len(plots) == 2
    for item in plots:
        output_png = Path(str(item["output_png"]))
        assert output_png.exists()
        assert output_png.name.endswith("_spot_latent_field.png")
        assert "_C" not in output_png.name
        assert int(item["n_latent_occurrences"]) == 4
        assert int(item["n_plotted_occurrences"]) == 4
        assert "n_subregion_boundary_outlines" in item
        assert output_png.stat().st_size > 0
