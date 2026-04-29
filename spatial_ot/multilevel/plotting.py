from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError


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


def _mode_int(values: np.ndarray) -> int:
    values_arr = np.asarray(values, dtype=np.int64)
    valid = values_arr[values_arr >= 0]
    if valid.size == 0:
        return -1
    counts = np.bincount(valid)
    return int(np.argmax(counts))


def _resolve_spot_latent_npz(
    *,
    cells_h5ad: Path,
    metadata: dict,
    spot_latent_npz: str | Path | None,
) -> Path | None:
    candidates: list[Path] = []
    if spot_latent_npz is not None:
        candidates.append(Path(spot_latent_npz))
    metadata_npz = metadata.get("spot_level_latent_npz")
    if metadata_npz:
        candidates.append(Path(str(metadata_npz)))
    candidates.append(cells_h5ad.parent / "spot_level_latent_multilevel_ot.npz")
    for candidate in candidates:
        if candidate.exists():
            return candidate
        resolved = candidate if candidate.is_absolute() else cells_h5ad.parent / candidate
        if resolved.exists():
            return resolved
    return None


def _subregion_memberships_from_obs(obs: pd.DataFrame) -> tuple[dict[int, np.ndarray], str | None]:
    for key in ("mlot_subregion_id", "mlot_subregion_int"):
        if key not in obs.columns:
            continue
        subregion_ids = np.asarray(obs[key], dtype=np.int32)
        valid_ids = np.unique(subregion_ids[subregion_ids >= 0])
        if valid_ids.size == 0:
            continue
        groups = {
            int(subregion_id): np.flatnonzero(subregion_ids == int(subregion_id)).astype(np.int64)
            for subregion_id in valid_ids.tolist()
        }
        return groups, f"obs[{key}]"
    return {}, None


def _subregion_memberships_from_npz(npz_path: Path, *, n_obs: int) -> dict[int, np.ndarray]:
    with np.load(npz_path) as payload:
        if "cell_indices" not in payload.files or "subregion_ids" not in payload.files:
            return {}
        cell_indices = np.asarray(payload["cell_indices"], dtype=np.int64)
        subregion_ids = np.asarray(payload["subregion_ids"], dtype=np.int32)
    if cell_indices.ndim != 1 or subregion_ids.ndim != 1 or cell_indices.shape[0] != subregion_ids.shape[0]:
        raise ValueError(f"Subregion membership arrays in '{npz_path}' are inconsistent.")
    valid = (
        (cell_indices >= 0)
        & (cell_indices < int(n_obs))
        & (subregion_ids >= 0)
    )
    if not np.any(valid):
        return {}
    valid_cells = cell_indices[valid]
    valid_subregions = subregion_ids[valid]
    groups: dict[int, np.ndarray] = {}
    for subregion_id in np.unique(valid_subregions).tolist():
        members = np.unique(valid_cells[valid_subregions == int(subregion_id)]).astype(np.int64)
        if members.size:
            groups[int(subregion_id)] = members
    return groups


def _load_subregion_memberships(
    *,
    obs: pd.DataFrame,
    cells_h5ad: Path,
    metadata: dict,
    spot_latent_npz: str | Path | None,
) -> tuple[dict[int, np.ndarray], str, Path | None]:
    obs_groups, obs_source = _subregion_memberships_from_obs(obs)
    if obs_groups:
        return obs_groups, str(obs_source), None
    resolved_npz = _resolve_spot_latent_npz(
        cells_h5ad=cells_h5ad,
        metadata=metadata,
        spot_latent_npz=spot_latent_npz,
    )
    if resolved_npz is None:
        return {}, "unavailable", None
    npz_groups = _subregion_memberships_from_npz(resolved_npz, n_obs=obs.shape[0])
    return npz_groups, "spot_level_latent_npz[cell_indices,subregion_ids]", resolved_npz


def _polygon_exteriors_from_geometry(geometry) -> list[np.ndarray]:
    if geometry is None or getattr(geometry, "is_empty", False):
        return []
    if geometry.geom_type == "Polygon":
        exterior = np.asarray(geometry.exterior.coords, dtype=np.float32)
        return [exterior] if exterior.shape[0] >= 4 else []
    if geometry.geom_type == "MultiPolygon":
        polygons: list[np.ndarray] = []
        for part in geometry.geoms:
            exterior = np.asarray(part.exterior.coords, dtype=np.float32)
            if exterior.shape[0] >= 4:
                polygons.append(exterior)
        return polygons
    return []


def _subregion_boundary_polygons(coords: np.ndarray, *, concave_ratio: float = 0.35) -> list[np.ndarray]:
    coords_arr = np.asarray(coords, dtype=np.float32)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
        return []
    finite = np.all(np.isfinite(coords_arr), axis=1)
    unique = np.unique(coords_arr[finite], axis=0)
    if unique.shape[0] < 3:
        return []

    try:
        from shapely import concave_hull
        from shapely.geometry import MultiPoint

        points = MultiPoint(unique)
        hull = concave_hull(points, ratio=float(np.clip(concave_ratio, 0.0, 1.0)), allow_holes=False)
        polygons = _polygon_exteriors_from_geometry(hull)
        if polygons:
            return polygons
        polygons = _polygon_exteriors_from_geometry(points.convex_hull)
        if polygons:
            return polygons
    except Exception:
        pass

    try:
        hull = ConvexHull(unique)
    except QhullError:
        return []
    polygon = unique[hull.vertices]
    return [np.vstack([polygon, polygon[0]]).astype(np.float32)]


def _latent_color_limits(latent: np.ndarray) -> dict[str, list[float]]:
    latent_arr = np.asarray(latent, dtype=np.float32)
    if latent_arr.ndim != 2 or latent_arr.shape[1] != 2:
        return {"lower": [0.0, 0.0], "upper": [1.0, 1.0]}
    finite = np.all(np.isfinite(latent_arr), axis=1)
    if not np.any(finite):
        return {"lower": [0.0, 0.0], "upper": [1.0, 1.0]}
    reference = latent_arr[finite]
    lower = np.nanpercentile(reference, 1.0, axis=0).astype(np.float32)
    upper = np.nanpercentile(reference, 99.0, axis=0).astype(np.float32)
    span = upper - lower
    for dim in range(2):
        if not np.isfinite(span[dim]) or float(span[dim]) <= 1e-6:
            center = float(np.nanmean(reference[:, dim]))
            lower[dim] = center - 0.5
            upper[dim] = center + 0.5
    return {
        "lower": [float(lower[0]), float(lower[1])],
        "upper": [float(upper[0]), float(upper[1])],
    }


def _latent_to_rgb(latent: np.ndarray, *, limits: dict[str, list[float]]) -> np.ndarray:
    from matplotlib.colors import hsv_to_rgb

    latent_arr = np.asarray(latent, dtype=np.float32)
    lower = np.asarray(limits["lower"], dtype=np.float32)
    upper = np.asarray(limits["upper"], dtype=np.float32)
    span = np.maximum(upper - lower, 1e-6)
    scaled = np.clip((latent_arr - lower[None, :]) / span[None, :], 0.0, 1.0)
    centered = scaled - 0.5
    hue = (np.arctan2(centered[:, 1], centered[:, 0]) / (2.0 * np.pi) + 1.0) % 1.0
    radius = np.sqrt(np.sum(centered * centered, axis=1)) / np.sqrt(0.5)
    saturation = np.clip(0.2 + 0.8 * radius, 0.2, 1.0)
    value = np.full(latent_arr.shape[0], 0.95, dtype=np.float32)
    rgb = hsv_to_rgb(np.stack([hue, saturation, value], axis=1)).astype(np.float32)
    finite = np.all(np.isfinite(latent_arr), axis=1)
    rgb[~finite] = np.asarray([0.82, 0.82, 0.82], dtype=np.float32)
    return rgb


def _latent_color_limits_by_cluster(latent: np.ndarray, cluster_ids: np.ndarray) -> dict[str, dict[str, list[float]]]:
    latent_arr = np.asarray(latent, dtype=np.float32)
    cluster_arr = np.asarray(cluster_ids, dtype=np.int32)
    if latent_arr.ndim != 2 or latent_arr.shape[1] != 2 or cluster_arr.shape[0] != latent_arr.shape[0]:
        return {}
    finite = np.all(np.isfinite(latent_arr), axis=1)
    limits_by_cluster: dict[str, dict[str, list[float]]] = {}
    for cluster_id in np.unique(cluster_arr[(cluster_arr >= 0) & finite]).tolist():
        mask = (cluster_arr == int(cluster_id)) & finite
        if np.any(mask):
            limits_by_cluster[str(int(cluster_id))] = _latent_color_limits(latent_arr[mask])
    return limits_by_cluster


def _latent_to_within_cluster_rgb(
    latent: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    limits_by_cluster: dict[str, dict[str, list[float]]],
) -> np.ndarray:
    latent_arr = np.asarray(latent, dtype=np.float32)
    cluster_arr = np.asarray(cluster_ids, dtype=np.int32)
    rgb = np.full((latent_arr.shape[0], 3), 0.82, dtype=np.float32)
    if latent_arr.ndim != 2 or latent_arr.shape[1] != 2 or cluster_arr.shape[0] != latent_arr.shape[0]:
        return rgb
    finite = np.all(np.isfinite(latent_arr), axis=1)
    for cluster_key, limits in limits_by_cluster.items():
        cluster_id = int(cluster_key)
        mask = (cluster_arr == cluster_id) & finite
        if np.any(mask):
            rgb[mask] = _latent_to_rgb(latent_arr[mask], limits=limits)
    return rgb


def _cluster_mean_latent_anchors(latent: np.ndarray, cluster_ids: np.ndarray) -> dict[int, list[float]]:
    latent_arr = np.asarray(latent, dtype=np.float32)
    cluster_arr = np.asarray(cluster_ids, dtype=np.int32)
    anchors: dict[int, list[float]] = {}
    if latent_arr.ndim != 2 or latent_arr.shape[1] != 2 or cluster_arr.shape[0] != latent_arr.shape[0]:
        return anchors
    finite = np.all(np.isfinite(latent_arr), axis=1)
    for cluster_id in np.unique(cluster_arr[(cluster_arr >= 0) & finite]).tolist():
        idx = np.flatnonzero((cluster_arr == int(cluster_id)) & finite)
        if idx.size:
            center = np.nanmean(latent_arr[idx], axis=0)
            anchors[int(cluster_id)] = [float(center[0]), float(center[1])]
    return anchors


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


def _plot_sample_subregion_cluster_maps(
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
    spot_latent_npz: str | Path | None = None,
) -> dict[str, object]:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    cells_h5ad = Path(cells_h5ad)
    output_dir = Path(output_dir) if output_dir is not None else cells_h5ad.parent / "sample_niche_plots"
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
        n_palette = int(np.max(cluster_ids[cluster_ids >= 0])) + 1 if np.any(cluster_ids >= 0) else 1
        palette = cluster_palette(n_palette)
        cluster_hex = np.asarray(
            [
                f"#{r:02x}{g:02x}{b:02x}" if int(cluster_id) >= 0 else "#d0d0d0"
                for cluster_id, (r, g, b) in zip(
                    cluster_ids.tolist(),
                    palette[np.clip(cluster_ids, 0, palette.shape[0] - 1)].tolist(),
                    strict=False,
                )
            ],
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

    subregion_groups, membership_source, resolved_npz = _load_subregion_memberships(
        obs=obs,
        cells_h5ad=cells_h5ad,
        metadata=metadata,
        spot_latent_npz=spot_latent_npz,
    )
    if not subregion_groups:
        manifest = _plot_sample_cluster_maps(
            cells_h5ad=cells_h5ad,
            output_dir=output_dir,
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
        manifest["rendering"] = "cell_scatter_fallback_no_subregion_membership"
        manifest["subregion_membership_source"] = membership_source
        Path(str(manifest["manifest_json"])).write_text(json.dumps(manifest, indent=2))
        return manifest

    plots: list[dict[str, object]] = []
    for sample_id in sample_ids:
        sample_mask = sample_values == sample_id
        sample_indices = np.flatnonzero(sample_mask)
        sample_count = int(sample_indices.size)
        if sample_count == 0:
            continue

        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(15.5, 7.4),
            constrained_layout=True,
        )
        subregion_ax, cell_ax = axes
        sample_coords = coords_um[sample_indices]
        subregion_ax.scatter(
            sample_coords[:, 0],
            sample_coords[:, 1],
            s=max(_marker_size(sample_count) * 0.12, 0.05),
            color="#d9d9d9",
            linewidths=0,
            alpha=0.25,
            rasterized=sample_count > 20000,
        )

        polygons_by_cluster: dict[int, list[np.ndarray]] = {}
        cells_by_cluster: dict[int, int] = {}
        subregions_by_cluster: dict[int, int] = {}
        degenerate_centers_by_cluster: dict[int, list[np.ndarray]] = {}
        n_sample_subregions = 0
        n_filled_subregions = 0
        n_degenerate_subregions = 0
        for members in subregion_groups.values():
            member_arr = np.asarray(members, dtype=np.int64)
            member_arr = member_arr[(member_arr >= 0) & (member_arr < obs.shape[0])]
            if member_arr.size == 0:
                continue
            member_arr = member_arr[sample_values[member_arr] == sample_id]
            if member_arr.size == 0:
                continue
            n_sample_subregions += 1
            cid = _mode_int(cluster_ids[member_arr])
            subregions_by_cluster[cid] = subregions_by_cluster.get(cid, 0) + 1
            cells_by_cluster[cid] = cells_by_cluster.get(cid, 0) + int(member_arr.size)
            polygons = _subregion_boundary_polygons(coords_um[member_arr])
            if polygons:
                polygons_by_cluster.setdefault(cid, []).extend(polygons)
                n_filled_subregions += 1
            else:
                n_degenerate_subregions += 1
                degenerate_centers_by_cluster.setdefault(cid, []).append(coords_um[member_arr].mean(axis=0))

        for cid in sorted(polygons_by_cluster):
            _, color_hex = cluster_display.get(int(cid), (f"C{int(cid)}", "#4d4d4d"))
            collection = PolyCollection(
                polygons_by_cluster[cid],
                facecolors=color_hex,
                edgecolors="#ffffff",
                linewidths=0.12,
                alpha=0.78,
                antialiaseds=False,
                rasterized=len(polygons_by_cluster[cid]) > 2000,
            )
            subregion_ax.add_collection(collection)

        for cid, centers in degenerate_centers_by_cluster.items():
            _, color_hex = cluster_display.get(int(cid), (f"C{int(cid)}", "#4d4d4d"))
            center_arr = np.asarray(centers, dtype=np.float32)
            subregion_ax.scatter(
                center_arr[:, 0],
                center_arr[:, 1],
                s=4.0,
                marker="s",
                color=color_hex,
                linewidths=0,
                alpha=0.9,
                rasterized=True,
            )

        source_name: str | list[str] | None = None
        if source_file_obs_key in obs.columns:
            sources = [str(value) for value in pd.unique(obs.loc[sample_mask, source_file_obs_key].astype(str))]
            if len(sources) == 1:
                source_name = sources[0]
            elif sources:
                source_name = sources

        title = f"Spatial niche map: {sample_id}"
        if isinstance(source_name, str):
            title = f"{title}\n{source_name}"
        fig.suptitle(title)
        subregion_ax.set_title(f"Subregion-wise\n{n_filled_subregions}/{n_sample_subregions} filled subregions")
        subregion_ax.set_xlabel("x (µm)")
        subregion_ax.set_ylabel("y (µm)")
        subregion_ax.set_aspect("equal")
        subregion_ax.invert_yaxis()
        subregion_ax.update_datalim(sample_coords)
        subregion_ax.autoscale_view()

        cell_point_size = _marker_size(sample_count)
        sample_cluster_ids = cluster_ids[sample_indices]
        for cid in np.unique(sample_cluster_ids):
            cell_cluster_mask = sample_cluster_ids == int(cid)
            label_name, color_hex = cluster_display.get(int(cid), (f"C{int(cid)}", "#4d4d4d"))
            cell_ax.scatter(
                sample_coords[cell_cluster_mask, 0],
                sample_coords[cell_cluster_mask, 1],
                s=cell_point_size,
                color=color_hex,
                linewidths=0,
                alpha=0.85,
                rasterized=sample_count > 20000,
                label=f"{label_name} ({int(np.sum(cell_cluster_mask))} cells)",
            )
        cell_ax.set_title(f"Cell-wise inherited labels\n{sample_count} cells")
        cell_ax.set_xlabel("x (µm)")
        cell_ax.set_ylabel("y (µm)")
        cell_ax.set_aspect("equal")
        cell_ax.invert_yaxis()
        cell_ax.update_datalim(sample_coords)
        cell_ax.autoscale_view()

        legend_handles = []
        for cid in sorted(cid for cid in subregions_by_cluster if cid >= 0):
            label_name, color_hex = cluster_display.get(int(cid), (f"C{int(cid)}", "#4d4d4d"))
            legend_handles.append(
                Patch(
                    facecolor=color_hex,
                    edgecolor="#ffffff",
                    label=f"{label_name} ({subregions_by_cluster[cid]} regions, {cells_by_cluster[cid]} cells)",
                    alpha=0.78,
                )
            )
        if legend_handles:
            cell_ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True, title="Clusters")

        output_png = output_dir / f"{_safe_filename_component(sample_id)}_spatial_niche_map.png"
        fig.savefig(output_png, dpi=250, bbox_inches="tight")
        plt.close(fig)

        plots.append(
            {
                "sample_id": str(sample_id),
                "source_h5ad": source_name,
                "n_cells": sample_count,
                "n_subregions": int(n_sample_subregions),
                "n_filled_subregions": int(n_filled_subregions),
                "n_degenerate_subregions": int(n_degenerate_subregions),
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
        "title_prefix": "Spatial niche map",
        "output_filename_suffix": "_spatial_niche_map.png",
        "rendering": "subregion_polygons_and_cell_scatter",
        "views": ["subregion_wise_filled_polygons", "cell_wise_inherited_label_scatter"],
        "subregion_membership_source": membership_source,
        "subregion_membership_npz": str(resolved_npz) if resolved_npz is not None else None,
        "polygon_boundary": "concave_hull_of_subregion_member_cells_with_convex_hull_fallback",
        "plots": plots,
    }
    manifest_path = output_dir / "sample_niche_plots_manifest.json"
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
    spot_latent_npz: str | Path | None = None,
) -> dict[str, object]:
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(cells_h5ad).parent / "sample_niche_plots"
    return _plot_sample_subregion_cluster_maps(
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
        spot_latent_npz=spot_latent_npz,
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


def plot_sample_spot_latent_maps(
    cells_h5ad: str | Path,
    output_dir: str | Path | None = None,
    *,
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    spot_latent_npz: str | Path | None = None,
    latent_obsm_key: str = "mlot_spot_latent_coords",
    latent_cluster_obs_key: str = "mlot_spot_latent_cluster_int",
    plot_spatial_x_key: str | None = None,
    plot_spatial_y_key: str | None = None,
    default_sample_id: str = "all_cells",
    spatial_scale: float | None = None,
    max_occurrences_per_cluster: int = 0,
    random_state: int = 0,
) -> dict[str, object]:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    cells_h5ad = Path(cells_h5ad)
    output_dir = Path(output_dir) if output_dir is not None else cells_h5ad.parent / "sample_spot_latent_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(cells_h5ad, backed="r")
    try:
        obs = adata.obs.copy()
        metadata = dict(adata.uns["multilevel_ot"]) if "multilevel_ot" in adata.uns else {}
        fallback_latent = None
        if latent_obsm_key in adata.obsm:
            fallback_latent = np.asarray(adata.obsm[latent_obsm_key], dtype=np.float32)
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()

    resolved_spot_latent_npz: Path | None = Path(spot_latent_npz) if spot_latent_npz is not None else None
    if resolved_spot_latent_npz is None and metadata.get("spot_level_latent_npz"):
        candidate = Path(str(metadata["spot_level_latent_npz"]))
        resolved_spot_latent_npz = candidate if candidate.is_absolute() else cells_h5ad.parent / candidate

    latent_source = "occurrence_npz"
    occurrence_weights: np.ndarray
    occurrence_subregion_ids: np.ndarray
    subregion_id_source = "unavailable"
    if resolved_spot_latent_npz is not None and resolved_spot_latent_npz.exists():
        with np.load(resolved_spot_latent_npz) as payload:
            occurrence_cell_indices = np.asarray(payload["cell_indices"], dtype=np.int64)
            latent = np.asarray(payload["latent_coords"], dtype=np.float32)
            cluster_ids = np.asarray(payload["cluster_labels"], dtype=np.int32)
            occurrence_weights = np.asarray(payload["weights"], dtype=np.float32) if "weights" in payload.files else np.ones(latent.shape[0], dtype=np.float32)
            if "subregion_ids" in payload.files:
                occurrence_subregion_ids = np.asarray(payload["subregion_ids"], dtype=np.int32)
                subregion_id_source = "occurrence_npz[subregion_ids]"
            else:
                occurrence_subregion_ids = np.full(latent.shape[0], -1, dtype=np.int32)
        if occurrence_cell_indices.ndim != 1 or occurrence_cell_indices.shape[0] != latent.shape[0]:
            raise ValueError(f"Spot latent NPZ '{resolved_spot_latent_npz}' has inconsistent cell_indices and latent_coords.")
    else:
        latent_source = "cell_preview_obsm"
        if fallback_latent is None:
            raise KeyError(
                f"Expected either spot latent NPZ '{resolved_spot_latent_npz}' or obsm key '{latent_obsm_key}' in {cells_h5ad}."
            )
        latent = fallback_latent
        if latent_cluster_obs_key in obs.columns:
            cluster_ids = np.asarray(obs[latent_cluster_obs_key], dtype=np.int32)
        elif "mlot_cluster_int" in obs.columns:
            cluster_ids = np.asarray(obs["mlot_cluster_int"], dtype=np.int32)
        else:
            cluster_ids = np.zeros(obs.shape[0], dtype=np.int32)
        occurrence_cell_indices = np.arange(obs.shape[0], dtype=np.int64)
        occurrence_weights = np.ones(latent.shape[0], dtype=np.float32)
        if "mlot_subregion_id" in obs.columns:
            occurrence_subregion_ids = np.asarray(obs["mlot_subregion_id"], dtype=np.int32)
            subregion_id_source = "obs[mlot_subregion_id]"
        elif "mlot_subregion_int" in obs.columns:
            occurrence_subregion_ids = np.asarray(obs["mlot_subregion_int"], dtype=np.int32)
            subregion_id_source = "obs[mlot_subregion_int]"
        else:
            occurrence_subregion_ids = np.full(latent.shape[0], -1, dtype=np.int32)

    if latent.ndim != 2 or latent.shape[1] != 2:
        raise ValueError("Spot latent coordinates must have shape (n_observations, 2).")
    if cluster_ids.shape[0] != latent.shape[0]:
        raise ValueError("Spot latent cluster labels must have one entry per latent coordinate.")
    if occurrence_subregion_ids.shape[0] != latent.shape[0]:
        raise ValueError("Spot latent subregion labels must have one entry per latent coordinate.")
    if occurrence_cell_indices.size and (
        int(occurrence_cell_indices.min()) < 0 or int(occurrence_cell_indices.max()) >= obs.shape[0]
    ):
        raise ValueError("Spot latent cell_indices contain rows outside the H5AD obs table.")

    if sample_obs_key in obs.columns:
        sample_ids = [str(value) for value in pd.unique(obs[sample_obs_key].astype(str))]
        sample_values = obs[sample_obs_key].astype(str).to_numpy()
    else:
        sample_ids = [str(default_sample_id)]
        sample_values = np.full(obs.shape[0], str(default_sample_id), dtype=object)

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

    coords_um = np.stack(
        [
            np.asarray(obs[resolved_x_key], dtype=np.float32) * resolved_scale,
            np.asarray(obs[resolved_y_key], dtype=np.float32) * resolved_scale,
        ],
        axis=1,
    )
    occurrence_sample_values = sample_values[occurrence_cell_indices]
    occurrence_coords = coords_um[occurrence_cell_indices]
    finite_latent = np.all(np.isfinite(latent), axis=1)
    valid_cluster = cluster_ids >= 0
    if occurrence_weights.shape[0] != latent.shape[0]:
        raise ValueError("Spot latent weights must have one entry per latent coordinate.")
    rng = np.random.default_rng(int(random_state))
    display_latent = np.asarray(latent, dtype=np.float32)
    cluster_display_anchors = _cluster_mean_latent_anchors(display_latent, cluster_ids)
    display_latent_limits = _latent_color_limits(display_latent[finite_latent & valid_cluster])
    within_niche_color_limits = _latent_color_limits_by_cluster(latent, cluster_ids)

    plots: list[dict[str, object]] = []
    for sample_id in sample_ids:
        sample_cell_mask = sample_values == sample_id
        sample_occurrence_mask = (occurrence_sample_values == sample_id) & finite_latent & valid_cluster
        sample_count = int(np.sum(sample_cell_mask))
        if sample_count == 0 or not np.any(sample_occurrence_mask):
            continue
        background_coords = coords_um[sample_cell_mask]
        point_size = _marker_size(sample_count)
        sample_occurrence_idx = np.flatnonzero(sample_occurrence_mask)
        occurrence_idx = sample_occurrence_idx
        total_occurrences = int(sample_occurrence_idx.size)
        if max_occurrences_per_cluster > 0 and total_occurrences > int(max_occurrences_per_cluster):
            occurrence_idx = np.sort(
                rng.choice(occurrence_idx, size=int(max_occurrences_per_cluster), replace=False)
            )
        display_latent_values = display_latent[occurrence_idx]
        local_latent_values = latent[occurrence_idx]
        local_cluster_ids = cluster_ids[occurrence_idx]
        latent_coords_um = occurrence_coords[occurrence_idx]
        colors = _latent_to_within_cluster_rgb(
            local_latent_values,
            local_cluster_ids,
            limits_by_cluster=within_niche_color_limits,
        )
        boundary_polygons: list[np.ndarray] = []
        boundary_subregion_ids = occurrence_subregion_ids[sample_occurrence_idx]
        valid_boundary_ids = np.unique(boundary_subregion_ids[boundary_subregion_ids >= 0])
        if valid_boundary_ids.size:
            boundary_coords_um = occurrence_coords[sample_occurrence_idx]
            for subregion_id in valid_boundary_ids.tolist():
                subregion_mask = boundary_subregion_ids == int(subregion_id)
                boundary_polygons.extend(_subregion_boundary_polygons(boundary_coords_um[subregion_mask]))

        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(12.5, 7.2),
            gridspec_kw={"width_ratios": [4.2, 1.25]},
            constrained_layout=True,
        )
        ax = axes[0]
        ax.scatter(
            background_coords[:, 0],
            background_coords[:, 1],
            s=max(point_size * 0.25, 0.08),
            color="#d0d0d0",
            linewidths=0,
            alpha=0.25,
            rasterized=sample_count > 20000,
        )
        ax.scatter(
            latent_coords_um[:, 0],
            latent_coords_um[:, 1],
            s=point_size,
            color=colors,
            linewidths=0,
            alpha=0.9,
            rasterized=display_latent_values.shape[0] > 20000,
        )
        if boundary_polygons:
            ax.add_collection(
                PolyCollection(
                    boundary_polygons,
                    facecolors="none",
                    edgecolors="#222222",
                    linewidths=0.08,
                    alpha=0.35,
                    antialiaseds=False,
                    rasterized=len(boundary_polygons) > 2000,
                )
            )

        source_name: str | list[str] | None = None
        if source_file_obs_key in obs.columns:
            sources = [str(value) for value in pd.unique(obs.loc[sample_cell_mask, source_file_obs_key].astype(str))]
            if len(sources) == 1:
                source_name = sources[0]
            elif sources:
                source_name = sources

        title = f"Within-niche spot latent heterogeneity: {sample_id}\n{int(occurrence_idx.size)}/{total_occurrences} latent occurrences"
        if isinstance(source_name, str):
            title = f"{title}\n{source_name}"
        ax.set_title(title)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_aspect("equal")
        ax.invert_yaxis()

        key_ax = axes[1]
        key_ax.scatter(
            display_latent_values[:, 0],
            display_latent_values[:, 1],
            s=max(point_size * 0.8, 0.4),
            color=colors,
            linewidths=0,
            alpha=0.85,
            rasterized=display_latent_values.shape[0] > 20000,
        )
        key_ax.set_title("global latent key")
        key_ax.set_xlabel("latent 1")
        key_ax.set_ylabel("latent 2")
        key_ax.set_aspect("equal", adjustable="box")
        key_ax.set_xlim(float(display_latent_limits["lower"][0]), float(display_latent_limits["upper"][0]))
        key_ax.set_ylim(float(display_latent_limits["lower"][1]), float(display_latent_limits["upper"][1]))

        present_clusters = np.unique(cluster_ids[occurrence_idx][cluster_ids[occurrence_idx] >= 0])
        for cluster_id in present_clusters.tolist():
            anchor = cluster_display_anchors.get(int(cluster_id))
            if anchor is None:
                continue
            key_ax.text(
                float(anchor[0]),
                float(anchor[1]),
                f"C{int(cluster_id)}",
                fontsize=8,
                ha="center",
                va="center",
                color="black",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
            )
        output_png = output_dir / f"{_safe_filename_component(sample_id)}_spot_latent_field.png"
        fig.savefig(output_png, dpi=250, bbox_inches="tight")
        plt.close(fig)
        plots.append(
            {
                "sample_id": str(sample_id),
                "source_h5ad": source_name,
                "n_cells": sample_count,
                "n_clusters_present": int(present_clusters.size),
                "n_latent_occurrences": total_occurrences,
                "n_plotted_occurrences": int(occurrence_idx.size),
                "n_subregion_boundary_outlines": int(len(boundary_polygons)),
                "local_latent1_min": float(np.nanmin(local_latent_values[:, 0])),
                "local_latent1_max": float(np.nanmax(local_latent_values[:, 0])),
                "local_latent2_min": float(np.nanmin(local_latent_values[:, 1])),
                "local_latent2_max": float(np.nanmax(local_latent_values[:, 1])),
                "output_png": str(output_png),
            }
        )

    manifest: dict[str, object] = {
        "cells_h5ad": str(cells_h5ad),
        "output_dir": str(output_dir),
        "n_samples": int(len({str(item["sample_id"]) for item in plots})),
        "n_plots": int(len(plots)),
        "sample_obs_key": str(sample_obs_key),
        "source_file_obs_key": str(source_file_obs_key),
        "latent_source": latent_source,
        "spot_latent_npz": str(resolved_spot_latent_npz) if resolved_spot_latent_npz is not None else None,
        "subregion_id_source": subregion_id_source,
        "subregion_boundary_overlay": "concave_hull_of_sample_occurrence_subregion_members",
        "latent_obsm_key": str(latent_obsm_key),
        "latent_cluster_obs_key": str(latent_cluster_obs_key),
        "plot_spatial_x_key": str(resolved_x_key),
        "plot_spatial_y_key": str(resolved_y_key),
        "spatial_scale": float(resolved_scale),
        "coordinate_scope": "global_fisher_latent_with_per_niche_color_scaling",
        "rendering": "whole_sample_within_niche_latent_rgb",
        "color_encoding": "The side key uses the learned global Fisher/discriminative 2D latent coordinates to show OT-supported between-cluster separation without forcing a minimum or target cluster distance. Slide colors are robustly rescaled within each niche/cluster before RGB conversion so continuous within-niche heterogeneity uses the full color range.",
        "latent_color_limits": within_niche_color_limits,
        "within_niche_latent_color_limits": within_niche_color_limits,
        "display_latent_limits": display_latent_limits,
        "cluster_display_anchors": {str(key): value for key, value in cluster_display_anchors.items()},
        "max_occurrences_per_sample": int(max_occurrences_per_cluster),
        "max_occurrences_per_cluster": int(max_occurrences_per_cluster),
        "plots": plots,
    }
    manifest_path = output_dir / "sample_spot_latent_plots_manifest.json"
    manifest["manifest_json"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


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
    spot_latent_npz: str | Path | None = None,
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
        spot_latent_npz=spot_latent_npz if spot_latent_npz is not None else run_dir / "spot_level_latent_multilevel_ot.npz",
    )


def plot_sample_spot_latent_maps_from_run_dir(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    spot_latent_npz: str | Path | None = None,
    latent_obsm_key: str = "mlot_spot_latent_coords",
    latent_cluster_obs_key: str = "mlot_spot_latent_cluster_int",
    plot_spatial_x_key: str | None = None,
    plot_spatial_y_key: str | None = None,
    default_sample_id: str = "all_cells",
    spatial_scale: float | None = None,
    max_occurrences_per_cluster: int = 0,
    random_state: int = 0,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    cells_h5ad = run_dir / "cells_multilevel_ot.h5ad"
    if not cells_h5ad.exists():
        raise FileNotFoundError(f"Expected multilevel OT cell output under {cells_h5ad}.")
    resolved_output_dir = Path(output_dir) if output_dir is not None else run_dir / "sample_spot_latent_plots"
    resolved_spot_latent_npz = Path(spot_latent_npz) if spot_latent_npz is not None else run_dir / "spot_level_latent_multilevel_ot.npz"
    return plot_sample_spot_latent_maps(
        cells_h5ad=cells_h5ad,
        output_dir=resolved_output_dir,
        sample_obs_key=sample_obs_key,
        source_file_obs_key=source_file_obs_key,
        spot_latent_npz=resolved_spot_latent_npz,
        latent_obsm_key=latent_obsm_key,
        latent_cluster_obs_key=latent_cluster_obs_key,
        plot_spatial_x_key=plot_spatial_x_key,
        plot_spatial_y_key=plot_spatial_y_key,
        default_sample_id=default_sample_id,
        spatial_scale=spatial_scale,
        max_occurrences_per_cluster=max_occurrences_per_cluster,
        random_state=random_state,
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
