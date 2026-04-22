from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd


def cluster_palette(n_clusters: int) -> np.ndarray:
    cmap_name = "tab20" if n_clusters <= 20 else "gist_ncar"
    cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(n_clusters)
    rgba = np.asarray([cmap(i) for i in range(n_clusters)], dtype=np.float32)
    return np.clip(np.rint(rgba[:, :3] * 255.0), 0, 255).astype(np.uint8)


def _marker_size(n_points: int, *, low: float = 0.5, high: float = 8.0) -> float:
    if n_points <= 1000:
        return high
    if n_points >= 250000:
        return low
    scale = (np.log10(n_points) - np.log10(1000)) / (np.log10(250000) - np.log10(1000))
    scale = float(np.clip(scale, 0.0, 1.0))
    return high - (high - low) * scale


def _safe_filename_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "sample"


def _resolve_sample_plot_coordinate_keys(
    obs: pd.DataFrame,
    *,
    requested_x_key: str | None,
    requested_y_key: str | None,
    metadata_x_key: str | None,
    metadata_y_key: str | None,
) -> tuple[str, str]:
    candidate_pairs: list[tuple[str | None, str | None]] = [
        (requested_x_key, requested_y_key),
        ("original_cell_x", "original_cell_y"),
        ("cell_x", "cell_y"),
        (metadata_x_key, metadata_y_key),
    ]
    for x_key, y_key in candidate_pairs:
        if x_key is None or y_key is None:
            continue
        if x_key in obs.columns and y_key in obs.columns:
            return str(x_key), str(y_key)
    raise KeyError(
        "Could not find a usable spatial coordinate pair for sample niche plots. "
        "Pass --plot-spatial-x-key and --plot-spatial-y-key explicitly."
    )


def _plot_sample_cluster_maps(
    cells_h5ad: str | Path,
    output_dir: str | Path | None = None,
    *,
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    cluster_obs_key: str = "mlot_cluster_int",
    cluster_label_obs_key: str = "mlot_cluster_id",
    cluster_hex_obs_key: str = "mlot_cluster_hex",
    plot_spatial_x_key: str | None = None,
    plot_spatial_y_key: str | None = None,
    default_sample_id: str = "all_cells",
    spatial_scale: float | None = None,
    title_prefix: str,
    output_filename_suffix: str,
    manifest_filename: str,
) -> dict[str, object]:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    cells_h5ad = Path(cells_h5ad)
    output_dir = Path(output_dir) if output_dir is not None else cells_h5ad.parent / "sample_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(cells_h5ad, backed="r")
    try:
        obs = adata.obs.copy()
        metadata = dict(adata.uns["multilevel_ot"]) if "multilevel_ot" in adata.uns else {}
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()

    if cluster_obs_key not in obs.columns and cluster_label_obs_key not in obs.columns:
        raise KeyError(
            f"Expected either cluster obs key '{cluster_obs_key}' or '{cluster_label_obs_key}' in {cells_h5ad}."
        )

    metadata_x_key = metadata.get("spatial_x_key")
    metadata_y_key = metadata.get("spatial_y_key")
    resolved_x_key, resolved_y_key = _resolve_sample_plot_coordinate_keys(
        obs,
        requested_x_key=plot_spatial_x_key,
        requested_y_key=plot_spatial_y_key,
        metadata_x_key=str(metadata_x_key) if metadata_x_key is not None else None,
        metadata_y_key=str(metadata_y_key) if metadata_y_key is not None else None,
    )
    resolved_scale = float(spatial_scale) if spatial_scale is not None else float(metadata.get("spatial_scale", 1.0))

    if sample_obs_key in obs.columns:
        sample_ids = [str(value) for value in pd.unique(obs[sample_obs_key].astype(str))]
        sample_values = obs[sample_obs_key].astype(str).to_numpy()
    else:
        sample_ids = [str(default_sample_id)]
        sample_values = np.full(obs.shape[0], str(default_sample_id), dtype=object)

    if cluster_label_obs_key in obs.columns:
        cluster_names = obs[cluster_label_obs_key].astype(str).to_numpy()
    elif cluster_obs_key in obs.columns:
        cluster_names = np.asarray([f"C{int(value)}" for value in np.asarray(obs[cluster_obs_key], dtype=np.int32)], dtype=object)
    else:
        raise KeyError("No cluster label information was available for sample plotting.")

    if cluster_obs_key in obs.columns:
        cluster_ids = np.asarray(obs[cluster_obs_key], dtype=np.int32)
    else:
        category = pd.Categorical(cluster_names)
        cluster_ids = category.codes.astype(np.int32)
        cluster_names = np.asarray([str(category.categories[idx]) for idx in cluster_ids], dtype=object)

    if cluster_hex_obs_key in obs.columns:
        cluster_hex = obs[cluster_hex_obs_key].astype(str).to_numpy()
    else:
        palette = cluster_palette(int(cluster_ids.max()) + 1)
        cluster_hex = np.asarray(
            [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette[cluster_ids].tolist()],
            dtype=object,
        )

    coords_um = np.stack(
        [
            np.asarray(obs[resolved_x_key], dtype=np.float32) * resolved_scale,
            np.asarray(obs[resolved_y_key], dtype=np.float32) * resolved_scale,
        ],
        axis=1,
    )

    cluster_display: dict[int, tuple[str, str]] = {}
    for idx, cluster_id in enumerate(cluster_ids.tolist()):
        cluster_display.setdefault(int(cluster_id), (str(cluster_names[idx]), str(cluster_hex[idx])))

    plots: list[dict[str, object]] = []
    for sample_id in sample_ids:
        sample_mask = sample_values == sample_id
        sample_count = int(np.sum(sample_mask))
        if sample_count == 0:
            continue

        fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
        point_size = _marker_size(sample_count)
        sample_cluster_ids = cluster_ids[sample_mask]
        sample_coords = coords_um[sample_mask]
        for cluster_id in np.unique(sample_cluster_ids):
            cluster_mask = sample_cluster_ids == cluster_id
            label_name, color_hex = cluster_display[int(cluster_id)]
            ax.scatter(
                sample_coords[cluster_mask, 0],
                sample_coords[cluster_mask, 1],
                s=point_size,
                color=color_hex,
                linewidths=0,
                alpha=0.85,
                rasterized=sample_count > 20000,
                label=f"{label_name} ({int(np.sum(cluster_mask))})",
            )

        source_name: str | list[str] | None = None
        if source_file_obs_key in obs.columns:
            sources = [str(value) for value in pd.unique(obs.loc[sample_mask, source_file_obs_key].astype(str))]
            if len(sources) == 1:
                source_name = sources[0]
            elif sources:
                source_name = sources

        title = f"{title_prefix}: {sample_id}"
        if isinstance(source_name, str):
            title = f"{title}\n{source_name}"
        ax.set_title(title)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)

        output_png = output_dir / f"{_safe_filename_component(sample_id)}{output_filename_suffix}"
        fig.savefig(output_png, dpi=250, bbox_inches="tight")
        plt.close(fig)

        plots.append(
            {
                "sample_id": str(sample_id),
                "source_h5ad": source_name,
                "n_cells": sample_count,
                "output_png": str(output_png),
            }
        )

    manifest: dict[str, object] = {
        "cells_h5ad": str(cells_h5ad),
        "output_dir": str(output_dir),
        "n_samples": int(len(plots)),
        "sample_obs_key": str(sample_obs_key),
        "source_file_obs_key": str(source_file_obs_key),
        "cluster_obs_key": str(cluster_obs_key),
        "cluster_label_obs_key": str(cluster_label_obs_key),
        "cluster_hex_obs_key": str(cluster_hex_obs_key),
        "plot_spatial_x_key": str(resolved_x_key),
        "plot_spatial_y_key": str(resolved_y_key),
        "spatial_scale": float(resolved_scale),
        "title_prefix": str(title_prefix),
        "output_filename_suffix": str(output_filename_suffix),
        "plots": plots,
    }
    manifest_path = output_dir / manifest_filename
    manifest["manifest_json"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def plot_sample_niche_maps(
    cells_h5ad: str | Path,
    output_dir: str | Path | None = None,
    *,
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    cluster_obs_key: str = "mlot_cluster_int",
    cluster_label_obs_key: str = "mlot_cluster_id",
    cluster_hex_obs_key: str = "mlot_cluster_hex",
    plot_spatial_x_key: str | None = None,
    plot_spatial_y_key: str | None = None,
    default_sample_id: str = "all_cells",
    spatial_scale: float | None = None,
) -> dict[str, object]:
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(cells_h5ad).parent / "sample_niche_plots"
    return _plot_sample_cluster_maps(
        cells_h5ad=cells_h5ad,
        output_dir=resolved_output_dir,
        sample_obs_key=sample_obs_key,
        source_file_obs_key=source_file_obs_key,
        cluster_obs_key=cluster_obs_key,
        cluster_label_obs_key=cluster_label_obs_key,
        cluster_hex_obs_key=cluster_hex_obs_key,
        plot_spatial_x_key=plot_spatial_x_key,
        plot_spatial_y_key=plot_spatial_y_key,
        default_sample_id=default_sample_id,
        spatial_scale=spatial_scale,
        title_prefix="Spatial niche map",
        output_filename_suffix="_spatial_niche_map.png",
        manifest_filename="sample_niche_plots_manifest.json",
    )


def plot_sample_spatial_maps(
    cells_h5ad: str | Path,
    output_dir: str | Path | None = None,
    *,
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    cluster_obs_key: str = "mlot_cluster_int",
    cluster_label_obs_key: str = "mlot_cluster_id",
    cluster_hex_obs_key: str = "mlot_cluster_hex",
    plot_spatial_x_key: str | None = None,
    plot_spatial_y_key: str | None = None,
    default_sample_id: str = "all_cells",
    spatial_scale: float | None = None,
) -> dict[str, object]:
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(cells_h5ad).parent / "sample_spatial_maps"
    return _plot_sample_cluster_maps(
        cells_h5ad=cells_h5ad,
        output_dir=resolved_output_dir,
        sample_obs_key=sample_obs_key,
        source_file_obs_key=source_file_obs_key,
        cluster_obs_key=cluster_obs_key,
        cluster_label_obs_key=cluster_label_obs_key,
        cluster_hex_obs_key=cluster_hex_obs_key,
        plot_spatial_x_key=plot_spatial_x_key,
        plot_spatial_y_key=plot_spatial_y_key,
        default_sample_id=default_sample_id,
        spatial_scale=spatial_scale,
        title_prefix="Shape-normalized multilevel OT cell labels",
        output_filename_suffix="_multilevel_ot_spatial_map.png",
        manifest_filename="sample_spatial_maps_manifest.json",
    )


def plot_sample_niche_maps_from_run_dir(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    cluster_obs_key: str = "mlot_cluster_int",
    cluster_label_obs_key: str = "mlot_cluster_id",
    cluster_hex_obs_key: str = "mlot_cluster_hex",
    plot_spatial_x_key: str | None = None,
    plot_spatial_y_key: str | None = None,
    default_sample_id: str = "all_cells",
    spatial_scale: float | None = None,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    cells_h5ad = run_dir / "cells_multilevel_ot.h5ad"
    if not cells_h5ad.exists():
        raise FileNotFoundError(f"Expected multilevel OT cell output under {cells_h5ad}.")
    resolved_output_dir = Path(output_dir) if output_dir is not None else run_dir / "sample_niche_plots"
    return plot_sample_niche_maps(
        cells_h5ad=cells_h5ad,
        output_dir=resolved_output_dir,
        sample_obs_key=sample_obs_key,
        source_file_obs_key=source_file_obs_key,
        cluster_obs_key=cluster_obs_key,
        cluster_label_obs_key=cluster_label_obs_key,
        cluster_hex_obs_key=cluster_hex_obs_key,
        plot_spatial_x_key=plot_spatial_x_key,
        plot_spatial_y_key=plot_spatial_y_key,
        default_sample_id=default_sample_id,
        spatial_scale=spatial_scale,
    )


def plot_sample_spatial_maps_from_run_dir(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    cluster_obs_key: str = "mlot_cluster_int",
    cluster_label_obs_key: str = "mlot_cluster_id",
    cluster_hex_obs_key: str = "mlot_cluster_hex",
    plot_spatial_x_key: str | None = None,
    plot_spatial_y_key: str | None = None,
    default_sample_id: str = "all_cells",
    spatial_scale: float | None = None,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    cells_h5ad = run_dir / "cells_multilevel_ot.h5ad"
    if not cells_h5ad.exists():
        raise FileNotFoundError(f"Expected multilevel OT cell output under {cells_h5ad}.")
    resolved_output_dir = Path(output_dir) if output_dir is not None else run_dir / "sample_spatial_maps"
    return plot_sample_spatial_maps(
        cells_h5ad=cells_h5ad,
        output_dir=resolved_output_dir,
        sample_obs_key=sample_obs_key,
        source_file_obs_key=source_file_obs_key,
        cluster_obs_key=cluster_obs_key,
        cluster_label_obs_key=cluster_label_obs_key,
        cluster_hex_obs_key=cluster_hex_obs_key,
        plot_spatial_x_key=plot_spatial_x_key,
        plot_spatial_y_key=plot_spatial_y_key,
        default_sample_id=default_sample_id,
        spatial_scale=spatial_scale,
    )
