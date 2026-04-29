from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import warnings

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import torch

from ..config import DeepFeatureConfig, MultilevelExperimentConfig
from ..deep.features import SpatialOTFeatureEncoder, fit_deep_features, save_deep_feature_history
from ..feature_source import resolve_h5ad_features
from .._runtime import runtime_memory_snapshot as _runtime_memory_snapshot
from .core import _resolve_compute_device, fit_multilevel_ot
from .diagnostics import (
    LEAKAGE_BALANCED_ACCURACY_WARNING,
    LEAKAGE_PERMUTATION_MEAN_EXCESS_WARNING,
    LEAKAGE_PERMUTATION_P95_MARGIN_WARNING,
    assigned_transport_cost_decomposition as _assigned_transport_cost_decomposition,
    build_qc_warnings as _build_qc_warnings,
    cell_subregion_coverage as _cell_subregion_coverage,
    cost_reliability_metrics as _cost_reliability_metrics,
    probability_diagnostics as _probability_diagnostics,
    transform_diagnostics as _transform_diagnostics,
)
from .embedding import (
    compute_subregion_embedding as _compute_subregion_embedding,
    subregion_embedding_compactness as _subregion_embedding_compactness,
    subregion_graph_metrics as _subregion_graph_metrics,
)
from .geometry import (
    _region_geometries_from_observed_points,
    _shape_descriptor_frame,
    _shape_leakage_balanced_accuracy,
    _shape_leakage_permutation_baseline,
    _shape_leakage_spatial_block_accuracy,
    _validate_mutually_exclusive_memberships,
)
from .metadata import (
    extract_count_target as _extract_count_target,
    git_sha as _git_sha,
    latent_source_label as _latent_source_label,
    package_version as _package_version,
)
from .plotting import (
    cluster_palette as _cluster_palette,
    plot_sample_niche_maps as plot_sample_niche_maps,
    plot_sample_niche_maps_from_run_dir as plot_sample_niche_maps_from_run_dir,
    plot_sample_spot_latent_maps as plot_sample_spot_latent_maps,
    plot_sample_spot_latent_maps_from_run_dir as plot_sample_spot_latent_maps_from_run_dir,
    plot_sample_spatial_maps,
    plot_sample_spatial_maps_from_run_dir as plot_sample_spatial_maps_from_run_dir,
)
from .spot_latent import (
    spot_latent_mode_metadata,
    spot_latent_separation_diagnostics as _spot_latent_separation_diagnostics,
)
from .types import MultilevelOTResult, RegionGeometry


_IO_PROGRESS_START = time.perf_counter()


def _io_progress(message: str) -> None:
    raw = os.environ.get("SPATIAL_OT_PROGRESS", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        elapsed = time.perf_counter() - _IO_PROGRESS_START
        print(f"[spatial_ot io {elapsed:8.1f}s] {message}", file=sys.stderr, flush=True)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _h5ad_compression_from_env() -> str | None:
    raw = os.environ.get("SPATIAL_OT_H5AD_COMPRESSION", "gzip").strip().lower()
    if raw in {"", "none", "0", "false", "no", "off"}:
        return None
    return raw


def _method_stack_summary(
    *,
    feature_source: dict,
    deep_summary: dict,
    feature_obsm_key: str,
    subregion_clustering_method: str,
    subregion_clustering_uses_spatial: bool,
) -> dict[str, object]:
    return {
        "method_family": "multilevel_ot",
        "active_path": "multilevel-ot",
        "core_model": (
            "pooled_raw_member_feature_distribution_subregion_latent_clustering"
            if not bool(subregion_clustering_uses_spatial)
            else "shape_normalized_multilevel_semi_relaxed_ot"
        ),
        "deep_feature_adapter": (
            str(deep_summary.get("method", "none"))
            if bool(deep_summary.get("enabled"))
            else "none"
        ),
        "latent_used_for_ot": _latent_source_label(feature_source, deep_summary),
        "ot_feature_obsm_key": str(feature_obsm_key),
        "feature_input_mode": str(feature_source.get("input_mode", "obsm")),
        "feature_space_kind": str(feature_source.get("feature_space_kind", "unknown")),
        "communication_source": "none",
        "cell_label_mode": "fitted_subregion_cluster_membership",
        "subregion_clustering_method": str(subregion_clustering_method),
        "subregion_clustering_uses_spatial": bool(subregion_clustering_uses_spatial),
        "cell_projection_mode": "auxiliary_approximate_cell_scores",
    }


def _cluster_count_dict(labels: np.ndarray, n_clusters: int) -> dict[str, int]:
    labels_arr = np.asarray(labels, dtype=np.int32)
    if int(n_clusters) < 1:
        raise ValueError("n_clusters must be at least 1.")
    if labels_arr.size and (int(labels_arr.min()) < 0 or int(labels_arr.max()) >= int(n_clusters)):
        raise ValueError("cluster labels must be in [0, n_clusters).")
    counts = np.bincount(labels_arr, minlength=int(n_clusters))
    return {f"C{idx}": int(counts[idx]) for idx in range(int(n_clusters))}


def _numeric_summary(values: object) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q10": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _mds_status(
    stress: float | None,
    *,
    strong_threshold: float = 0.10,
    diagnostic_threshold: float = 0.20,
) -> str:
    try:
        value = float(stress)
    except (TypeError, ValueError):
        return "missing"
    if not np.isfinite(value):
        return "missing"
    if value < float(strong_threshold):
        return "interpretable"
    if value <= float(diagnostic_threshold):
        return "diagnostic_only"
    return "not_geometrically_interpretable"


def _fraction_above(values: object, threshold: float) -> float | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr > float(threshold)))


def _temperature_spread_ratio(values: object) -> float | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return None
    q05, q95 = np.quantile(arr, [0.05, 0.95])
    if q05 <= 0.0:
        return None
    return float(q95 / q05)


def _subregion_center_nn_distances(centers_um: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers_um, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[0] < 2:
        return np.asarray([], dtype=np.float32)
    distances, _ = NearestNeighbors(n_neighbors=2).fit(centers).kneighbors(centers)
    return np.asarray(distances[:, 1], dtype=np.float32)


def _density_descriptor_frame(shape_df: pd.DataFrame, result: MultilevelOTResult) -> pd.DataFrame:
    n_subregions = len(result.subregion_members)
    n_cells = np.asarray([len(members) for members in result.subregion_members], dtype=np.float64)
    geometry_counts = np.asarray(result.subregion_geometry_point_counts, dtype=np.float64)
    base = pd.DataFrame(
        {
            "subregion_id": np.arange(n_subregions, dtype=np.int32),
            "n_cells": n_cells,
            "log1p_n_cells": np.log1p(n_cells),
            "geometry_point_count": geometry_counts,
            "log1p_geometry_point_count": np.log1p(np.maximum(geometry_counts, 0.0)),
        }
    )
    if "shape_area" in shape_df.columns:
        area_df = shape_df[["subregion_id", "shape_area"]].copy()
        base = base.merge(area_df, on="subregion_id", how="left")
    else:
        base["shape_area"] = np.nan
    area = base["shape_area"].to_numpy(dtype=np.float64)
    valid_area = area[np.isfinite(area) & (area > 0.0)]
    fallback_area = float(np.median(valid_area)) if valid_area.size else 1.0
    safe_area = np.where(np.isfinite(area) & (area > 0.0), area, fallback_area)
    density = n_cells / np.maximum(safe_area, 1e-8)
    base["shape_area"] = safe_area
    base["log1p_shape_area"] = np.log1p(safe_area)
    base["cell_density_per_um2"] = density
    base["log1p_cell_density_per_um2"] = np.log1p(density)
    return base


def _realized_subregion_statistics(
    *,
    result: MultilevelOTResult,
    shape_df: pd.DataFrame,
    min_cells: int,
    max_subregion_area_um2: float | None = None,
) -> dict[str, object]:
    n_cells = np.asarray([len(members) for members in result.subregion_members], dtype=np.float64)
    basic_counts = np.asarray([len(ids) for ids in result.subregion_basic_niche_ids], dtype=np.float64)
    geometry_counts = np.asarray(result.subregion_geometry_point_counts, dtype=np.float64)
    density_df = _density_descriptor_frame(shape_df, result)
    stats: dict[str, object] = {
        "subregion_count": int(len(result.subregion_members)),
        "minimum_cell_constraint": int(min_cells),
        "minimum_cell_constraint_satisfied": bool(n_cells.size == 0 or np.min(n_cells) >= int(min_cells)),
        "maximum_area_constraint_um2": float(max_subregion_area_um2)
        if max_subregion_area_um2 is not None
        else None,
        "n_cells": _numeric_summary(n_cells),
        "basic_niche_count": _numeric_summary(basic_counts),
        "geometry_point_count": _numeric_summary(geometry_counts),
        "center_nearest_neighbor_distance_um": _numeric_summary(
            _subregion_center_nn_distances(result.subregion_centers_um)
        ),
        "cell_density_per_um2": _numeric_summary(density_df["cell_density_per_um2"].to_numpy(dtype=np.float64)),
    }
    shape_columns = {
        "shape_area": "shape_area_um2",
        "shape_perimeter": "shape_perimeter_um",
        "shape_compactness": "shape_compactness",
        "shape_aspect_ratio": "shape_aspect_ratio",
        "shape_eccentricity": "shape_eccentricity",
        "shape_radius_mean": "shape_radius_mean_um",
        "shape_radius_std": "shape_radius_std_um",
    }
    for column, key in shape_columns.items():
        if column in shape_df.columns:
            stats[key] = _numeric_summary(shape_df[column].to_numpy(dtype=np.float64))
    if max_subregion_area_um2 is not None and "shape_area" in shape_df.columns:
        area = shape_df["shape_area"].to_numpy(dtype=np.float64)
        finite_area = area[np.isfinite(area)]
        stats["maximum_area_constraint_satisfied"] = bool(
            finite_area.size == 0 or np.max(finite_area) <= float(max_subregion_area_um2) * (1.0 + 1e-6)
        )
        stats["maximum_area_constraint_violation_count"] = int(
            np.sum(finite_area > float(max_subregion_area_um2) * (1.0 + 1e-6))
        )
    return stats


def _subregion_construction_summary(
    *,
    build_generated_subregions: bool,
    region_obs_key: str | None,
    region_geometry_json: str | Path | None,
    result: MultilevelOTResult,
    radius_um: float,
    stride_um: float,
    min_cells: int,
    max_subregions: int,
    max_subregion_area_um2: float | None,
    subregion_construction_method: str,
    subregion_feature_weight: float,
    subregion_feature_dims: int,
    deep_segmentation_knn: int,
    deep_segmentation_feature_dims: int,
    deep_segmentation_feature_weight: float,
    deep_segmentation_spatial_weight: float,
    deep_summary: dict,
) -> dict[str, object]:
    generated = bool(build_generated_subregions)
    target_scale = float(result.basic_niche_size_um) if result.basic_niche_size_um is not None else float(stride_um)
    construction_method = str(subregion_construction_method).strip().lower()
    if generated:
        if construction_method == "deep_segmentation":
            mode = "generated_deep_graph_segmentation"
            membership_source = (
                "observed_coordinates_plus_deep_embedding_affinity"
                if bool(deep_summary.get("enabled"))
                else "observed_coordinates_plus_ot_feature_affinity_deep_segmentation_mode"
            )
        else:
            if float(subregion_feature_weight) > 0.0 and int(subregion_feature_dims) > 0:
                mode = "generated_data_driven_spatial_feature_partition"
                membership_source = "observed_coordinates_plus_ot_feature_view"
            else:
                mode = "generated_data_driven_coordinate_partition"
                membership_source = "observed_coordinates_only"
        boundary_shape_source = "observed_member_point_cloud"
        radius_semantics = (
            "radius_um is retained for backward compatibility and graph diagnostics; it is not used as a fixed "
            "ball/window membership radius in generated data-driven subregion construction."
        )
    else:
        mode = "explicit_region_obs_membership"
        membership_source = f"obs[{region_obs_key}]" if region_obs_key is not None else "provided_subregion_members"
        boundary_shape_source = "explicit_mask_or_polygon" if region_geometry_json is not None else "observed_member_point_cloud"
        radius_semantics = (
            "radius_um is retained for compatibility and graph diagnostics; explicit memberships come from the "
            "provided region labels, not a radius query."
        )
    return {
        "mode": mode,
        "membership_source": membership_source,
        "mutually_exclusive": True,
        "requires_full_cell_coverage_for_generated_partitions": bool(generated),
        "radius_used_for_membership": False,
        "radius_um": float(radius_um),
        "radius_um_semantics": radius_semantics,
        "stride_um": float(stride_um),
        "target_subregion_scale_um": target_scale,
        "basic_niche_size_um": float(result.basic_niche_size_um) if result.basic_niche_size_um is not None else None,
        "boundary_shape_source": boundary_shape_source,
        "min_cells": int(min_cells),
        "max_subregions": int(max_subregions),
        "max_subregion_area_um2": float(max_subregion_area_um2)
        if max_subregion_area_um2 is not None
        else None,
        "construction_method": construction_method,
        "partition_feature_weight": float(subregion_feature_weight),
        "partition_feature_dims": int(subregion_feature_dims),
        "coordinate_only_baseline": bool(
            construction_method == "data_driven"
            and float(subregion_feature_weight) == 0.0
        ),
        "feature_boundary_circularity_risk": bool(
            generated
            and (
                (construction_method == "data_driven" and float(subregion_feature_weight) > 0.0 and int(subregion_feature_dims) > 0)
                or construction_method == "deep_segmentation"
            )
        ),
        "recommended_primary_claim_mode": "coordinate_only_data_driven",
        "boundary_refinement_iters": _env_int("SPATIAL_OT_SUBREGION_BOUNDARY_REFINEMENT_ITERS", 2),
        "boundary_refinement_knn": _env_int("SPATIAL_OT_SUBREGION_BOUNDARY_KNN", 12),
        "seed_partition_kmeans_max_iter": _env_int("SPATIAL_OT_SUBREGION_KMEANS_MAX_ITER", 25),
        "seed_partition_kmeans_batch_multiplier": _env_int("SPATIAL_OT_SUBREGION_KMEANS_BATCH_MULTIPLIER", 2),
        "deep_segmentation": {
            "enabled": bool(generated and construction_method == "deep_segmentation"),
            "knn": int(deep_segmentation_knn),
            "feature_dims": int(deep_segmentation_feature_dims),
            "feature_weight": float(deep_segmentation_feature_weight),
            "spatial_weight": float(deep_segmentation_spatial_weight),
            "refinement_iters": _env_int("SPATIAL_OT_DEEP_SEGMENTATION_REFINEMENT_ITERS", 6),
            "embedding_source": (
                f"deep_{deep_summary.get('output_embedding')}"
                if bool(deep_summary.get("enabled"))
                else "ot_feature_view"
            ),
            "algorithm": "coordinate_seeded_spatial_knn_boundary_refinement_by_deep_affinity_then_connected_min_size_merge",
        },
    }


def _cell_subregion_cluster_projection(
    *,
    n_cells: int,
    result: MultilevelOTResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        _validate_mutually_exclusive_memberships(int(n_cells), result.subregion_members)
    except RuntimeError as exc:
        raise ValueError("Cannot write primary cell niche labels because subregion memberships overlap or are invalid.") from exc
    n_clusters = int(result.cluster_supports.shape[0])
    labels = np.full(int(n_cells), -1, dtype=np.int32)
    probs = np.zeros((int(n_cells), n_clusters), dtype=np.float32)
    membership_counts = np.zeros(int(n_cells), dtype=np.int32)
    for rid, members in enumerate(result.subregion_members):
        member_arr = np.asarray(members, dtype=np.int64)
        if member_arr.size == 0:
            continue
        labels[member_arr] = int(result.subregion_cluster_labels[rid])
        probs[member_arr] = np.asarray(result.subregion_cluster_probs[rid], dtype=np.float32)
        np.add.at(membership_counts, member_arr, 1)
    if int(membership_counts.max(initial=0)) > 1:
        raise ValueError("Cannot write primary cell niche labels because subregion memberships overlap.")
    return labels, probs, membership_counts


def _nonnegative_cluster_count_dict(labels: np.ndarray, n_clusters: int) -> dict[str, int]:
    labels_arr = np.asarray(labels, dtype=np.int32)
    return _cluster_count_dict(labels_arr[labels_arr >= 0], n_clusters)


def _deep_feature_schema_extra(
    *,
    feature_obsm_key: str,
    feature_source: dict,
    spatial_x_key: str,
    spatial_y_key: str,
    spatial_scale: float,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "input_mode": str(feature_source.get("input_mode", "obsm")),
        "input_obsm_key": str(feature_obsm_key),
        "input_feature_key": str(feature_obsm_key),
        "coordinate_keys": [str(spatial_x_key), str(spatial_y_key)],
        "preprocessing": str(feature_source.get("preprocessing", "train_only_standardization")),
        "spatial_scale": float(spatial_scale),
        "spatial_units_after_scaling": "um",
        "source_feature_dim": int(feature_source.get("source_feature_dim", feature_source.get("feature_dim", 0))),
        "feature_dim": int(feature_source.get("feature_dim", 0)),
    }
    for key in [
        "target_sum",
        "svd_components_requested",
        "svd_components_used",
        "svd_random_state",
        "svd_n_iter",
        "svd_explained_variance_ratio_sum",
    ]:
        payload[key] = feature_source.get(key)
    return payload


def _geometry_array(value: object, *, scale: float) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        raise ValueError("Region geometry polygons must be arrays with shape (n_vertices, 2) and at least 3 vertices.")
    return (arr * float(scale)).astype(np.float32)


def _coordinate_scale_for_region_geometry_units(coordinate_units: object, *, spatial_scale: float) -> float:
    units = str(coordinate_units).strip().lower()
    if units in {"um", "micron", "microns"}:
        return 1.0
    if units in {"obs", "raw", "input"}:
        return float(spatial_scale)
    raise ValueError(
        "Region geometry JSON coordinate_units must be one of 'um', 'micron', 'microns', 'obs', 'raw', or 'input'."
    )


def _load_region_geometry_json(
    path: str | Path,
    *,
    region_ids: list[str],
    subregion_members: list[np.ndarray],
    spatial_scale: float,
) -> list[RegionGeometry]:
    source = Path(path)
    payload = json.loads(source.read_text())
    coordinate_units = "um"
    if isinstance(payload, dict):
        coordinate_units = str(payload.get("coordinate_units", payload.get("units", "um"))).lower()
    coordinate_scale = _coordinate_scale_for_region_geometry_units(coordinate_units, spatial_scale=spatial_scale)

    if isinstance(payload, dict) and "regions" in payload:
        raw_regions = payload["regions"]
    elif isinstance(payload, dict):
        raw_regions = [
            {"region_id": str(region_id), **(spec if isinstance(spec, dict) else {"polygon_vertices": spec})}
            for region_id, spec in payload.items()
            if region_id not in {"coordinate_units", "units"}
        ]
    else:
        raw_regions = payload
    if not isinstance(raw_regions, list):
        raise ValueError("Region geometry JSON must contain a list under 'regions' or a mapping keyed by region id.")

    by_id: dict[str, dict[str, object]] = {}
    for item in raw_regions:
        if not isinstance(item, dict):
            raise ValueError("Each region geometry entry must be an object.")
        region_id = item.get("region_id", item.get("id"))
        if region_id is None:
            raise ValueError("Each region geometry entry must include 'region_id' or 'id'.")
        by_id[str(region_id)] = item

    geometries: list[RegionGeometry] = []
    missing = [region_id for region_id in region_ids if region_id not in by_id]
    if missing:
        raise KeyError(f"Region geometry JSON is missing geometry for region ids: {', '.join(missing)}")

    for region_id, members in zip(region_ids, subregion_members, strict=False):
        item = by_id[str(region_id)]
        polygon_vertices = None
        polygon_components = None
        mask = None
        affine = None
        vertices = item.get("polygon_vertices", item.get("polygon", item.get("vertices")))
        components = item.get("polygon_components", item.get("components"))
        if vertices is not None:
            polygon_vertices = _geometry_array(vertices, scale=coordinate_scale)
        if components is not None:
            if not isinstance(components, list):
                raise ValueError(f"Region '{region_id}' polygon_components must be a list of polygons.")
            polygon_components = [_geometry_array(component, scale=coordinate_scale) for component in components]
        if item.get("mask") is not None:
            mask = np.asarray(item["mask"], dtype=bool)
            if mask.ndim != 2:
                raise ValueError(f"Region '{region_id}' mask geometry must be a 2D boolean array.")
            affine = np.asarray(item["affine"], dtype=np.float32) if item.get("affine") is not None else None
            if affine is not None and affine.shape != (3, 3):
                raise ValueError(f"Region '{region_id}' mask affine must be a 3x3 matrix.")
            if coordinate_scale != 1.0:
                if affine is None:
                    affine = np.diag([coordinate_scale, coordinate_scale, 1.0]).astype(np.float32)
                else:
                    affine = affine.copy()
                    affine[:2, :] *= float(coordinate_scale)
        if polygon_vertices is None and polygon_components is None and mask is None:
            raise ValueError(f"Region '{region_id}' has no polygon_vertices, polygon_components, or mask geometry.")
        geometries.append(
            RegionGeometry(
                region_id=str(region_id),
                members=np.asarray(members, dtype=np.int32),
                polygon_vertices=polygon_vertices,
                polygon_components=polygon_components,
                mask=mask,
                affine=affine,
            )
        )
    return geometries


def _filter_explicit_regions_by_min_cells(
    *,
    subregion_members: list[np.ndarray],
    subregion_centers_um: np.ndarray,
    region_geometries: list[RegionGeometry],
    min_cells: int,
) -> tuple[list[np.ndarray], np.ndarray, list[RegionGeometry]]:
    keep_idx = [idx for idx, members in enumerate(subregion_members) if np.asarray(members).size >= int(min_cells)]
    if len(keep_idx) == len(subregion_members):
        return subregion_members, subregion_centers_um, region_geometries
    if not keep_idx:
        raise RuntimeError("No valid subregions remain after applying min_cells.")
    keep_arr = np.asarray(keep_idx, dtype=np.int32)
    return (
        [subregion_members[int(idx)] for idx in keep_idx],
        np.asarray(subregion_centers_um, dtype=np.float32)[keep_arr],
        [region_geometries[int(idx)] for idx in keep_idx],
    )


def _save_multilevel_outputs(
    adata: ad.AnnData,
    result: MultilevelOTResult,
    output_dir: Path,
    feature_obsm_key: str,
    spatial_x_key: str,
    spatial_y_key: str,
    spatial_scale: float,
    radius_um: float,
    stride_um: float,
    embedding_2d: np.ndarray,
    embedding_name: str,
    shape_df: pd.DataFrame,
    summary: dict,
    deep_embedding: np.ndarray | None = None,
    deep_obsm_key: str | None = None,
    extra_outputs: dict[str, str] | None = None,
) -> dict[str, str]:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = output_dir / "cells_multilevel_ot.h5ad"
    subregions_path = output_dir / "subregions_multilevel_ot.parquet"
    supports_path = output_dir / "cluster_supports_multilevel_ot.npz"
    spot_latent_path = output_dir / "spot_level_latent_multilevel_ot.npz"
    candidate_diag_path = output_dir / "multilevel_ot_candidate_cost_diagnostics.npz"
    map_path = output_dir / "multilevel_ot_spatial_map.png"
    emb_path = output_dir / "multilevel_ot_subregion_embedding.png"
    atom_path = output_dir / "multilevel_ot_atom_layouts.png"
    summary_path = output_dir / "summary.json"
    auto_k_path = output_dir / "auto_k_selection.json"
    auto_k_scores_path = output_dir / "auto_k_scores.csv"
    outputs = {
        "h5ad": str(h5ad_path),
        "subregions": str(subregions_path),
        "supports": str(supports_path),
        "spot_level_latent": str(spot_latent_path),
        "candidate_cost_diagnostics": str(candidate_diag_path),
        "spatial_map": str(map_path),
        "subregion_embedding": str(emb_path),
        "atom_layouts": str(atom_path),
        "summary": str(summary_path),
    }
    if extra_outputs:
        outputs.update(extra_outputs)
    if result.auto_k_selection is not None:
        outputs["auto_k_selection"] = str(auto_k_path)
        outputs["auto_k_scores"] = str(auto_k_scores_path)
    write_sample_spatial_maps = _env_bool("SPATIAL_OT_WRITE_SAMPLE_SPATIAL_MAPS", True)
    if write_sample_spatial_maps:
        outputs["sample_spatial_maps_dir"] = str(output_dir / "sample_spatial_maps")
        outputs["sample_spatial_maps_manifest"] = str(
            output_dir / "sample_spatial_maps" / "sample_spatial_maps_manifest.json"
        )
    summary["outputs"] = outputs
    spot_latent_computed = bool(summary.get("compute_spot_latent", True)) and result.spot_latent_coords.shape[0] > 0
    spot_latent_metadata = spot_latent_mode_metadata(result.spot_latent_mode)
    summary.setdefault("capabilities", {})["spot_level_latent_charts_implemented"] = bool(spot_latent_computed)
    summary.setdefault("method_stack", {})["spot_level_latent_projection"] = (
        str(spot_latent_metadata["latent_projection_mode"])
        if spot_latent_computed
        else "disabled"
    )
    summary.setdefault("method_notes", {})["spot_level_latent"] = (
        "Every row/spot occurrence inside every fitted regional niche is projected into a shared 2D diagnostic chart. "
        "The default chart is OT-grounded atom-barycentric MDS: cluster anchors are learned from balanced OT distances between "
        "the fitted cluster atom measures, and each occurrence is placed inside its assigned cluster by barycentering "
        "the cluster's atom embedding with the occurrence's entropy-calibrated OT atom posterior. Raw aligned x/y coordinates are not "
        "concatenated into the default chart features, and cluster-local variance is not equalized to a fixed radius. "
        "The Fisher/local-PCA chart remains available only as an explicit diagnostic mode. Primary OT subregion "
        "labels remain authoritative; the latent projection changes the chart geometry, not the fitted subregion labels. "
        "The H5AD stores a collapsed per-row preview; the NPZ stores the full occurrence-level latent field, posterior "
        "entropy, atom argmax, effective temperature, atom embeddings, and cluster anchors."
    )
    summary.setdefault("method_notes", {})["cell_cluster_labels"] = (
        "Primary H5AD labels mlot_cluster_int and mlot_subregion_cluster_int inherit each row's fitted mutually exclusive "
        "subregion cluster. The mlot_projected_* fields are auxiliary per-row scores and may vary inside a fitted "
        "subregion. Soft cluster probabilities are unconstrained cost-based scores; hard labels remain authoritative "
        "when the minimum-subregion-per-cluster constraint forces a label."
    )
    covered_spot_rows = int(np.sum(result.cell_spot_latent_cluster_labels >= 0))
    spot_latent_diagnostics = _spot_latent_separation_diagnostics(
        result.spot_latent_coords,
        result.spot_latent_cluster_labels,
        result.spot_latent_weights,
        result.spot_latent_subregion_ids,
    )
    cluster_anchor_stress = (
        float(result.spot_latent_cluster_mds_stress)
        if np.isfinite(result.spot_latent_cluster_mds_stress)
        else None
    )
    cluster_anchor_positive_mass = (
        float(result.spot_latent_cluster_mds_positive_eigenvalue_mass_2d)
        if np.isfinite(result.spot_latent_cluster_mds_positive_eigenvalue_mass_2d)
        else None
    )
    cluster_anchor_negative_mass = (
        float(result.spot_latent_cluster_mds_negative_eigenvalue_mass_fraction)
        if np.isfinite(result.spot_latent_cluster_mds_negative_eigenvalue_mass_fraction)
        else None
    )
    atom_high_stress_fraction = _fraction_above(result.spot_latent_atom_mds_stress, 0.25)
    summary["spot_level_latent"] = {
        "implemented": bool(spot_latent_computed),
        "projection": summary["method_stack"]["spot_level_latent_projection"],
        "occurrence_count": int(result.spot_latent_coords.shape[0]),
        "covered_row_count": covered_spot_rows,
        "covered_row_fraction": float(covered_spot_rows / max(int(adata.n_obs), 1)),
        "atoms_per_cluster": int(result.spot_latent_atom_posteriors.shape[1])
        if result.spot_latent_atom_posteriors.ndim == 2
        else 0,
        "occurrence_arrays": [
            "cell_indices",
            "subregion_ids",
            "cluster_labels",
            "latent_coords",
            "within_coords",
            "cluster_anchors",
            "atom_embedding",
            "aligned_coords",
            "cluster_probs",
            "atom_confidence",
            "posterior_entropy",
            "normalized_posterior_entropy",
            "atom_argmax",
            "temperature_used",
            "temperature_cost_gap",
            "temperature_fixed",
            "weights",
            "atom_posteriors",
            "posterior_entropy_cost_gap",
            "normalized_posterior_entropy_cost_gap",
            "posterior_entropy_fixed",
            "normalized_posterior_entropy_fixed",
            "cluster_anchor_distance",
            "cluster_anchor_ot_fallback_matrix",
            "cluster_anchor_solver_status_matrix",
            "atom_mds_stress",
            "atom_mds_positive_eigenvalue_mass_2d",
            "atom_mds_negative_eigenvalue_mass_fraction",
        ],
        "cell_preview_obsm_key": "mlot_spot_latent_coords",
        "coordinate_scope": str(spot_latent_metadata["coordinate_scope"]),
        "chart_learning_mode": str(spot_latent_metadata["chart_learning_mode"]),
        "validation_role": str(spot_latent_metadata["validation_role"]),
        "unsupervised_baseline_required_for_validation": bool(
            spot_latent_metadata["unsupervised_baseline_required_for_validation"]
        ),
        "label_permutation_control_recommended": bool(
            spot_latent_metadata["label_permutation_control_recommended"]
        ),
        "latent_refinement": str(spot_latent_metadata["latent_refinement"]),
        "includes_aligned_coordinates_in_chart_features": bool(
            spot_latent_metadata["includes_aligned_coordinates_in_chart_features"]
        ),
        "uses_forced_cluster_local_radius": bool(spot_latent_metadata["uses_forced_cluster_local_radius"]),
        "posterior_entropy_obs_key": "mlot_spot_latent_posterior_entropy",
        "atom_argmax_occurrence_array": "atom_argmax",
        "assignment_temperature": float(result.spot_latent_assignment_temperature)
        if np.isfinite(result.spot_latent_assignment_temperature)
        else None,
        "temperature_mode": str(result.spot_latent_temperature_mode),
        "temperature_calibration": "default auto_entropy mode solves for a target median normalized atom-posterior entropy; fixed/manual and auto_cost_gap modes remain available for diagnostics",
        "temperature_used_summary": _numeric_summary(result.spot_latent_temperature_used),
        "temperature_cost_gap_summary": _numeric_summary(result.spot_latent_temperature_cost_gap),
        "temperature_fixed_summary": _numeric_summary(result.spot_latent_temperature_fixed),
        "temperature_used_q95_q05_ratio": _temperature_spread_ratio(result.spot_latent_temperature_used),
        "posterior_entropy_summary": _numeric_summary(result.spot_latent_posterior_entropy),
        "normalized_posterior_entropy_summary": _numeric_summary(
            result.spot_latent_normalized_posterior_entropy
        ),
        "posterior_entropy_cost_gap_summary": _numeric_summary(result.spot_latent_posterior_entropy_cost_gap),
        "normalized_posterior_entropy_cost_gap_summary": _numeric_summary(
            result.spot_latent_normalized_posterior_entropy_cost_gap
        ),
        "posterior_entropy_fixed_summary": _numeric_summary(result.spot_latent_posterior_entropy_fixed),
        "normalized_posterior_entropy_fixed_summary": _numeric_summary(
            result.spot_latent_normalized_posterior_entropy_fixed
        ),
        "atom_confidence_summary": _numeric_summary(result.spot_latent_atom_confidence),
        "cluster_anchor_distance_method": str(result.spot_latent_cluster_anchor_distance_method),
        "cluster_anchor_distance_requested_method": str(
            result.spot_latent_cluster_anchor_distance_requested_method
        ),
        "cluster_anchor_distance_effective_method": str(
            result.spot_latent_cluster_anchor_distance_effective_method
        ),
        "cluster_anchor_ot_fallback_fraction": float(result.spot_latent_cluster_anchor_ot_fallback_fraction),
        "cluster_anchor_ot_fallback_count": int(
            np.sum(np.triu(result.spot_latent_cluster_anchor_ot_fallback_matrix, k=1))
        ),
        "cluster_anchor_solver_status_codebook": {
            "diagonal": 0,
            "expected_cross_cost": 1,
            "ot_success": 2,
            "expected_cross_cost_fallback": 3,
        },
        "cluster_anchor_mds_stress": cluster_anchor_stress,
        "cluster_anchor_mds_status": _mds_status(cluster_anchor_stress),
        "cluster_anchor_mds_positive_eigenvalue_mass_2d": cluster_anchor_positive_mass,
        "cluster_anchor_mds_negative_eigenvalue_mass_fraction": cluster_anchor_negative_mass,
        "mds_qc_thresholds": {
            "cluster_anchor_stress_interpretable": 0.10,
            "cluster_anchor_stress_diagnostic_upper": 0.20,
            "atom_stress_warning": 0.25,
            "positive_eigenvalue_mass_2d_warning": 0.70,
            "negative_eigenvalue_mass_fraction_warning": 0.10,
        },
        "atom_mds_stress_summary": _numeric_summary(result.spot_latent_atom_mds_stress),
        "atom_mds_high_stress_fraction": atom_high_stress_fraction,
        "atom_mds_high_stress_cluster_count": (
            int(
                round(
                    float(atom_high_stress_fraction)
                    * result.spot_latent_atom_mds_stress[np.isfinite(result.spot_latent_atom_mds_stress)].size
                )
            )
            if atom_high_stress_fraction is not None
            else None
        ),
        "atom_mds_positive_eigenvalue_mass_2d_summary": _numeric_summary(
            result.spot_latent_atom_mds_positive_eigenvalue_mass_2d
        ),
        "atom_mds_negative_eigenvalue_mass_fraction_summary": _numeric_summary(
            result.spot_latent_atom_mds_negative_eigenvalue_mass_fraction
        ),
        "cell_preview_aggregation": "unweighted_occurrence_average_by_default; confidence-weighted preview coordinates are saved separately and weights should be used for opacity/QC rather than suppressing transitional cells",
        "global_within_scale": float(result.spot_latent_global_within_scale)
        if np.isfinite(result.spot_latent_global_within_scale)
        else None,
        "local_context_radii_canonical": [0.25, 0.5, 1.0]
        if bool(spot_latent_metadata["includes_aligned_coordinates_in_chart_features"])
        else [],
        "honest_separation_diagnostics": spot_latent_diagnostics,
    }

    primary_cell_labels, primary_cell_probs, _ = _cell_subregion_cluster_projection(
        n_cells=int(adata.n_obs),
        result=result,
    )
    primary_cell_subregion_ids = np.full(int(adata.n_obs), -1, dtype=np.int32)
    for rid, members in enumerate(result.subregion_members):
        member_arr = np.asarray(members, dtype=np.int64)
        if member_arr.size:
            primary_cell_subregion_ids[member_arr] = int(rid)
    palette = _cluster_palette(result.cluster_supports.shape[0])
    label_names = [f"C{int(x)}" if int(x) >= 0 else "uncovered" for x in primary_cell_labels]
    label_hex = [
        f"#{r:02x}{g:02x}{b:02x}" if int(label) >= 0 else "#d0d0d0"
        for label, (r, g, b) in zip(
            primary_cell_labels.tolist(),
            palette[np.clip(primary_cell_labels, 0, palette.shape[0] - 1)].tolist(),
            strict=False,
        )
    ]
    projected_label_names = [f"C{int(x)}" for x in result.cell_cluster_labels]

    light_cell_h5ad = _env_bool("SPATIAL_OT_LIGHT_CELL_H5AD", False)
    h5ad_compression = _h5ad_compression_from_env()
    summary.setdefault("output_options", {})["light_cell_h5ad"] = bool(light_cell_h5ad)
    summary.setdefault("output_options", {})["h5ad_compression"] = h5ad_compression or "none"
    _io_progress("writing cell-level output h5ad")
    if light_cell_h5ad:
        cells_out = ad.AnnData(
            X=np.zeros((int(adata.n_obs), 0), dtype=np.float32),
            obs=adata.obs.copy(),
        )
    else:
        cells_out = adata.copy()
    cells_out.obs["mlot_cluster_id"] = pd.Categorical(label_names)
    cells_out.obs["mlot_cluster_int"] = primary_cell_labels.astype(np.int32)
    cells_out.obs["mlot_cluster_hex"] = label_hex
    cells_out.obs["mlot_subregion_id"] = primary_cell_subregion_ids.astype(np.int32)
    cells_out.obs["mlot_subregion_int"] = primary_cell_subregion_ids.astype(np.int32)
    cells_out.obs["mlot_subregion_cluster_id"] = pd.Categorical(label_names)
    cells_out.obs["mlot_subregion_cluster_int"] = primary_cell_labels.astype(np.int32)
    cells_out.obs["mlot_projected_cluster_id"] = pd.Categorical(projected_label_names)
    cells_out.obs["mlot_projected_cluster_int"] = result.cell_cluster_labels.astype(np.int32)
    cells_out.obsm["mlot_cluster_probs"] = primary_cell_probs.astype(np.float32)
    cells_out.obsm["mlot_subregion_cluster_probs"] = primary_cell_probs.astype(np.float32)
    cells_out.obsm["mlot_projected_cluster_probs"] = result.cell_cluster_probs.astype(np.float32)
    cells_out.obsm["mlot_cell_cluster_scores"] = result.cell_cluster_probs.astype(np.float32)
    cells_out.obsm["mlot_feature_cluster_probs"] = result.cell_feature_cluster_probs.astype(np.float32)
    cells_out.obsm["mlot_context_cluster_probs"] = result.cell_context_cluster_probs.astype(np.float32)
    spot_label_names = [
        f"C{int(label)}" if int(label) >= 0 else "uncovered"
        for label in result.cell_spot_latent_cluster_labels.tolist()
    ]
    cells_out.obs["mlot_spot_latent_cluster_id"] = pd.Categorical(spot_label_names)
    cells_out.obs["mlot_spot_latent_cluster_int"] = result.cell_spot_latent_cluster_labels.astype(np.int32)
    cells_out.obs["mlot_spot_latent_weight"] = result.cell_spot_latent_weights.astype(np.float32)
    cells_out.obs["mlot_spot_latent_posterior_entropy"] = (
        result.cell_spot_latent_posterior_entropy.astype(np.float32)
    )
    cells_out.obsm["mlot_spot_latent_coords"] = result.cell_spot_latent_coords.astype(np.float32)
    cells_out.obsm["mlot_spot_latent_unweighted_coords"] = result.cell_spot_latent_unweighted_coords.astype(np.float32)
    cells_out.obsm["mlot_spot_latent_confidence_weighted_coords"] = (
        result.cell_spot_latent_confidence_weighted_coords.astype(np.float32)
    )
    if deep_embedding is not None and deep_obsm_key:
        cells_out.obsm[deep_obsm_key] = np.asarray(deep_embedding, dtype=np.float32)
    cells_out.uns["multilevel_ot"] = {
        "feature_obsm_key": feature_obsm_key,
        "feature_input_mode": summary.get("feature_input_mode"),
        "feature_source": summary.get("feature_source"),
        "spatial_x_key": spatial_x_key,
        "spatial_y_key": spatial_y_key,
        "spatial_scale": float(spatial_scale),
        "radius_um": float(radius_um),
        "stride_um": float(stride_um),
        "basic_niche_size_um": float(result.basic_niche_size_um) if result.basic_niche_size_um is not None else None,
        "subregion_clustering_method": str(result.subregion_clustering_method),
        "subregion_clustering_uses_spatial": bool(result.subregion_clustering_uses_spatial),
        "subregion_clustering_feature_space": summary.get("subregion_clustering_feature_space"),
        "cell_label_mode": "fitted_subregion_cluster_membership",
        "cell_projection_mode": "auxiliary_approximate_cell_scores",
        "primary_cluster_obs_key": "mlot_cluster_int",
        "subregion_cluster_obs_key": "mlot_subregion_cluster_int",
        "projected_cluster_obs_key": "mlot_projected_cluster_int",
        "spot_level_latent_mode": result.spot_latent_mode,
        "spot_level_latent_projection_mode": result.spot_latent_projection_mode,
        "spot_level_latent_chart_learning_mode": result.spot_latent_chart_learning_mode,
        "spot_level_latent_validation_role": result.spot_latent_validation_role,
        "spot_level_latent_cluster_anchor_distance_method": result.spot_latent_cluster_anchor_distance_method,
        "spot_level_latent_cluster_anchor_distance_requested_method": (
            result.spot_latent_cluster_anchor_distance_requested_method
        ),
        "spot_level_latent_cluster_anchor_distance_effective_method": (
            result.spot_latent_cluster_anchor_distance_effective_method
        ),
        "spot_level_latent_cluster_anchor_ot_fallback_fraction": (
            float(result.spot_latent_cluster_anchor_ot_fallback_fraction)
        ),
        "spot_level_latent_cluster_anchor_mds_stress": (
            float(result.spot_latent_cluster_mds_stress)
            if np.isfinite(result.spot_latent_cluster_mds_stress)
            else None
        ),
        "spot_level_latent_npz": str(spot_latent_path),
        "deep_obsm_key": deep_obsm_key,
        "method_layers": summary.get("method_layers"),
        "summary_json": json.dumps(summary),
    }
    cells_out.write_h5ad(h5ad_path, compression=h5ad_compression)
    _io_progress("cell-level output h5ad written")

    subregion_rows = []
    sorted_costs = np.sort(result.subregion_cluster_costs, axis=1)
    subregion_margin = (
        sorted_costs[:, 1] - sorted_costs[:, 0]
        if sorted_costs.shape[1] >= 2
        else np.full(sorted_costs.shape[0], np.nan, dtype=np.float32)
    )
    for idx, members in enumerate(result.subregion_members):
        row = {
            "subregion_id": int(idx),
            "center_x_um": float(result.subregion_centers_um[idx, 0]),
            "center_y_um": float(result.subregion_centers_um[idx, 1]),
            "n_cells": int(len(members)),
            "basic_niche_count": int(len(result.subregion_basic_niche_ids[idx])),
            "geometry_point_count": int(result.subregion_geometry_point_counts[idx]),
            "geometry_source": result.subregion_geometry_sources[idx],
            "geometry_used_fallback": bool(result.subregion_geometry_used_fallback[idx]),
            "forced_label": bool(result.subregion_forced_label_mask[idx]),
            "argmin_cluster_int": int(result.subregion_argmin_labels[idx]),
            "assigned_effective_eps": float(result.subregion_assigned_effective_eps[idx]),
            "assigned_ot_used_fallback": bool(result.subregion_assigned_used_ot_fallback[idx]),
            "candidate_effective_eps_min": float(np.min(result.subregion_candidate_effective_eps_matrix[idx])),
            "candidate_effective_eps_max": float(np.max(result.subregion_candidate_effective_eps_matrix[idx])),
            "candidate_ot_used_fallback_any": bool(np.any(result.subregion_candidate_used_ot_fallback_matrix[idx])),
            "normalizer_radius_p95": float(result.subregion_normalizer_radius_p95[idx]) if np.isfinite(result.subregion_normalizer_radius_p95[idx]) else np.nan,
            "normalizer_radius_max": float(result.subregion_normalizer_radius_max[idx]) if np.isfinite(result.subregion_normalizer_radius_max[idx]) else np.nan,
            "normalizer_interpolation_residual": float(result.subregion_normalizer_interpolation_residual[idx]) if np.isfinite(result.subregion_normalizer_interpolation_residual[idx]) else np.nan,
            "cluster_id": f"C{int(result.subregion_cluster_labels[idx])}",
            "cluster_int": int(result.subregion_cluster_labels[idx]),
            "cluster_assignment_source": str(result.subregion_clustering_method),
            "cluster_assignment_uses_spatial": bool(result.subregion_clustering_uses_spatial),
            "objective": float(result.subregion_cluster_costs[idx, result.subregion_cluster_labels[idx]]),
            "transport_objective": float(result.subregion_cluster_transport_costs[idx, result.subregion_cluster_labels[idx]]),
            "overlap_consistency_penalty": float(
                result.subregion_cluster_overlap_penalties[idx, result.subregion_cluster_labels[idx]]
            ),
            "assigned_geometry_transport_cost": float(result.subregion_assigned_geometry_transport_costs[idx]),
            "assigned_feature_transport_cost": float(result.subregion_assigned_feature_transport_costs[idx]),
            "assigned_transform_penalty": float(result.subregion_assigned_transform_penalties[idx]),
            "assigned_overlap_consistency_penalty": float(result.subregion_assigned_overlap_consistency_penalties[idx]),
            "assigned_transform_rotation_deg": float(result.subregion_assigned_transform_rotation_deg[idx]),
            "assigned_transform_reflection": bool(result.subregion_assigned_transform_reflection[idx]),
            "assigned_transform_scale": float(result.subregion_assigned_transform_scale[idx]),
            "assigned_transform_translation_norm": float(result.subregion_assigned_transform_translation_norm[idx]),
            "assigned_reconstructed_transport_cost": float(
                result.subregion_assigned_geometry_transport_costs[idx]
                + result.subregion_assigned_feature_transport_costs[idx]
                + result.subregion_assigned_transform_penalties[idx]
            ),
            "assignment_margin": float(subregion_margin[idx]) if np.isfinite(subregion_margin[idx]) else np.nan,
        }
        for j, prob in enumerate(result.subregion_cluster_probs[idx]):
            row[f"cluster_prob_{j:02d}"] = float(prob)
        for j, weight in enumerate(result.subregion_atom_weights[idx]):
            row[f"atom_weight_{j:02d}"] = float(weight)
        for j, value in enumerate(result.subregion_latent_embeddings[idx, : min(result.subregion_latent_embeddings.shape[1], 8)]):
            row[f"subregion_latent_{j + 1:02d}"] = float(value)
        row["embed1"] = float(embedding_2d[idx, 0])
        row["embed2"] = float(embedding_2d[idx, 1])
        subregion_rows.append(row)
    subregions_df = pd.DataFrame(subregion_rows)
    if not shape_df.empty:
        subregions_df = subregions_df.merge(shape_df, on="subregion_id", how="left")
    _io_progress("writing subregion table")
    subregions_df.to_parquet(subregions_path, index=False)
    _io_progress("subregion table written")

    _io_progress("writing support and diagnostic arrays")
    np.savez_compressed(
        supports_path,
        cluster_supports=result.cluster_supports.astype(np.float32),
        cluster_atom_coords=result.cluster_atom_coords.astype(np.float32),
        cluster_atom_features=result.cluster_atom_features.astype(np.float32),
        cluster_prototype_weights=result.cluster_prototype_weights.astype(np.float32),
        subregion_atom_weights=result.subregion_atom_weights.astype(np.float32),
    )
    np.savez_compressed(
        spot_latent_path,
        cell_indices=result.spot_latent_cell_indices.astype(np.int32),
        subregion_ids=result.spot_latent_subregion_ids.astype(np.int32),
        cluster_labels=result.spot_latent_cluster_labels.astype(np.int32),
        latent_coords=result.spot_latent_coords.astype(np.float32),
        within_coords=result.spot_latent_within_coords.astype(np.float32),
        cluster_anchors=result.spot_latent_cluster_anchors.astype(np.float32),
        atom_embedding=result.spot_latent_atom_embedding.astype(np.float32),
        aligned_coords=result.spot_latent_aligned_coords.astype(np.float32),
        cluster_probs=result.spot_latent_cluster_probs.astype(np.float32),
        atom_confidence=result.spot_latent_atom_confidence.astype(np.float32),
        posterior_entropy=result.spot_latent_posterior_entropy.astype(np.float32),
        normalized_posterior_entropy=result.spot_latent_normalized_posterior_entropy.astype(np.float32),
        atom_argmax=result.spot_latent_atom_argmax.astype(np.int32),
        temperature_used=result.spot_latent_temperature_used.astype(np.float32),
        weights=result.spot_latent_weights.astype(np.float32),
        atom_posteriors=result.spot_latent_atom_posteriors.astype(np.float32),
        cluster_anchor_distance=result.spot_latent_cluster_anchor_distance.astype(np.float32),
        cluster_anchor_ot_fallback_matrix=result.spot_latent_cluster_anchor_ot_fallback_matrix.astype(bool),
        cluster_anchor_solver_status_matrix=result.spot_latent_cluster_anchor_solver_status_matrix.astype(np.int8),
        cluster_anchor_ot_fallback_fraction=np.array(
            float(result.spot_latent_cluster_anchor_ot_fallback_fraction),
            dtype=np.float32,
        ),
        atom_mds_stress=result.spot_latent_atom_mds_stress.astype(np.float32),
        atom_mds_positive_eigenvalue_mass_2d=result.spot_latent_atom_mds_positive_eigenvalue_mass_2d.astype(np.float32),
        atom_mds_negative_eigenvalue_mass_fraction=result.spot_latent_atom_mds_negative_eigenvalue_mass_fraction.astype(np.float32),
        posterior_entropy_cost_gap=result.spot_latent_posterior_entropy_cost_gap.astype(np.float32),
        normalized_posterior_entropy_cost_gap=result.spot_latent_normalized_posterior_entropy_cost_gap.astype(np.float32),
        posterior_entropy_fixed=result.spot_latent_posterior_entropy_fixed.astype(np.float32),
        normalized_posterior_entropy_fixed=result.spot_latent_normalized_posterior_entropy_fixed.astype(np.float32),
        temperature_cost_gap=result.spot_latent_temperature_cost_gap.astype(np.float32),
        temperature_fixed=result.spot_latent_temperature_fixed.astype(np.float32),
        cell_spot_latent_unweighted_coords=result.cell_spot_latent_unweighted_coords.astype(np.float32),
        cell_spot_latent_confidence_weighted_coords=result.cell_spot_latent_confidence_weighted_coords.astype(np.float32),
        cell_spot_latent_coords=result.cell_spot_latent_coords.astype(np.float32),
        cell_spot_latent_cluster_labels=result.cell_spot_latent_cluster_labels.astype(np.int32),
        cell_spot_latent_weights=result.cell_spot_latent_weights.astype(np.float32),
        cell_spot_latent_posterior_entropy=result.cell_spot_latent_posterior_entropy.astype(np.float32),
        spot_latent_mode=np.array(result.spot_latent_mode),
        latent_projection_mode=np.array(result.spot_latent_projection_mode),
        chart_learning_mode=np.array(result.spot_latent_chart_learning_mode),
        validation_role=np.array(result.spot_latent_validation_role),
        unsupervised_baseline_required_for_validation=np.array(
            bool(spot_latent_metadata["unsupervised_baseline_required_for_validation"])
        ),
        label_permutation_control_recommended=np.array(
            bool(spot_latent_metadata["label_permutation_control_recommended"])
        ),
        latent_refinement=np.array(str(spot_latent_metadata["latent_refinement"])),
        includes_aligned_coordinates_in_chart_features=np.array(
            bool(spot_latent_metadata["includes_aligned_coordinates_in_chart_features"])
        ),
        uses_forced_cluster_local_radius=np.array(bool(spot_latent_metadata["uses_forced_cluster_local_radius"])),
        global_within_scale=np.array(float(result.spot_latent_global_within_scale), dtype=np.float32),
        assignment_temperature=np.array(float(result.spot_latent_assignment_temperature), dtype=np.float32),
        temperature_mode=np.array(result.spot_latent_temperature_mode),
        cluster_anchor_distance_method=np.array(result.spot_latent_cluster_anchor_distance_method),
        cluster_anchor_distance_requested_method=np.array(
            result.spot_latent_cluster_anchor_distance_requested_method
        ),
        cluster_anchor_distance_effective_method=np.array(
            result.spot_latent_cluster_anchor_distance_effective_method
        ),
        cluster_anchor_mds_stress=np.array(float(result.spot_latent_cluster_mds_stress), dtype=np.float32),
        cluster_anchor_mds_positive_eigenvalue_mass_2d=np.array(
            float(result.spot_latent_cluster_mds_positive_eigenvalue_mass_2d),
            dtype=np.float32,
        ),
        cluster_anchor_mds_negative_eigenvalue_mass_fraction=np.array(
            float(result.spot_latent_cluster_mds_negative_eigenvalue_mass_fraction),
            dtype=np.float32,
        ),
    )
    np.savez_compressed(
        candidate_diag_path,
        subregion_cluster_costs=result.subregion_cluster_costs.astype(np.float32),
        subregion_cluster_transport_costs=result.subregion_cluster_transport_costs.astype(np.float32),
        subregion_cluster_overlap_penalties=result.subregion_cluster_overlap_penalties.astype(np.float32),
        subregion_measure_summaries=result.subregion_measure_summaries.astype(np.float32),
        subregion_latent_embeddings=result.subregion_latent_embeddings.astype(np.float32),
        subregion_clustering_method=np.array(result.subregion_clustering_method),
        subregion_clustering_uses_spatial=np.array(bool(result.subregion_clustering_uses_spatial)),
        candidate_effective_eps_matrix=result.subregion_candidate_effective_eps_matrix.astype(np.float32),
        candidate_used_ot_fallback_matrix=result.subregion_candidate_used_ot_fallback_matrix.astype(bool),
    )
    if result.auto_k_selection is not None:
        auto_k_path.write_text(json.dumps(result.auto_k_selection, indent=2))
        scores = result.auto_k_selection.get("scores")
        if isinstance(scores, list) and scores:
            pd.DataFrame(scores).to_csv(auto_k_scores_path, index=False)
    _io_progress("support and diagnostic arrays written")

    coords = np.stack(
        [
            np.asarray(adata.obs[spatial_x_key], dtype=np.float32) * spatial_scale,
            np.asarray(adata.obs[spatial_y_key], dtype=np.float32) * spatial_scale,
        ],
        axis=1,
    )
    point_size = 4.0 if coords.shape[0] > 100000 else 8.0
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    for cid in range(result.cluster_supports.shape[0]):
        mask = primary_cell_labels == cid
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            color=palette[cid] / 255.0,
            linewidths=0,
            alpha=0.85,
            rasterized=coords.shape[0] > 20000,
            label=f"C{cid} ({int(mask.sum())})",
        )
    ax.set_title("Shape-normalized multilevel OT cell labels")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.savefig(map_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    for cid in range(result.cluster_supports.shape[0]):
        mask = result.subregion_cluster_labels == cid
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            s=16,
            color=palette[cid] / 255.0,
            linewidths=0,
            alpha=0.9,
            label=f"C{cid} ({int(mask.sum())})",
        )
    ax.set_title(f"Subregion {embedding_name} from learned mixture weights")
    ax.set_xlabel(f"{embedding_name} 1")
    ax.set_ylabel(f"{embedding_name} 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.savefig(emb_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        nrows=result.cluster_supports.shape[0],
        ncols=1,
        figsize=(6.5, max(3.0, 2.4 * result.cluster_supports.shape[0])),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    feat_norm = np.linalg.norm(result.cluster_atom_features, axis=2)
    for cid, ax in enumerate(axes):
        sizes = 200.0 * np.clip(result.cluster_prototype_weights[cid], 0.05, None)
        sc = ax.scatter(
            result.cluster_atom_coords[cid, :, 0],
            result.cluster_atom_coords[cid, :, 1],
            s=sizes,
            c=feat_norm[cid],
            cmap="viridis",
            linewidths=0.5,
            edgecolors="black",
        )
        for atom_idx in range(result.cluster_atom_coords.shape[1]):
            ax.text(
                result.cluster_atom_coords[cid, atom_idx, 0],
                result.cluster_atom_coords[cid, atom_idx, 1],
                str(atom_idx),
                fontsize=7,
                ha="center",
                va="center",
                color="white",
            )
        ax.set_title(f"Cluster C{cid} canonical atom layout")
        ax.set_xlabel("canonical x")
        ax.set_ylabel("canonical y")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="feature norm")
    fig.savefig(atom_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    if write_sample_spatial_maps:
        sample_spatial_manifest = plot_sample_spatial_maps(
            cells_h5ad=h5ad_path,
            output_dir=output_dir / "sample_spatial_maps",
        )
        outputs["sample_spatial_maps_manifest"] = str(sample_spatial_manifest["manifest_json"])

    summary_path.write_text(json.dumps(summary, indent=2))
    return outputs


def run_multilevel_ot_on_h5ad(
    input_h5ad: str | Path,
    output_dir: str | Path,
    feature_obsm_key: str,
    spatial_x_key: str,
    spatial_y_key: str,
    spatial_scale: float,
    *,
    region_obs_key: str | None = None,
    region_geometry_json: str | Path | None = None,
    allow_umap_as_feature: bool = False,
    n_clusters: int,
    atoms_per_cluster: int,
    radius_um: float,
    stride_um: float,
    min_cells: int,
    max_subregions: int,
    lambda_x: float,
    lambda_y: float,
    geometry_eps: float,
    ot_eps: float,
    rho: float,
    geometry_samples: int,
    compressed_support_size: int,
    align_iters: int,
    allow_reflection: bool = False,
    allow_scale: bool = False,
    min_scale: float = 0.75,
    max_scale: float = 1.33,
    scale_penalty: float = 0.05,
    shift_penalty: float = 0.05,
    n_init: int = 5,
    allow_convex_hull_fallback: bool = False,
    max_iter: int = 10,
    tol: float = 1e-4,
    overlap_consistency_weight: float = 0.0,
    overlap_jaccard_min: float = 0.15,
    overlap_contrast_scale: float = 1.0,
    basic_niche_size_um: float | None = 200.0,
    max_subregion_area_um2: float | None = None,
    subregion_construction_method: str = "data_driven",
    subregion_feature_weight: float = 0.0,
    subregion_feature_dims: int = 16,
    deep_segmentation_knn: int = 12,
    deep_segmentation_feature_dims: int = 32,
    deep_segmentation_feature_weight: float = 1.0,
    deep_segmentation_spatial_weight: float = 0.05,
    subregion_clustering_method: str = "pooled_subregion_latent",
    shape_diagnostics: bool = True,
    shape_leakage_permutations: int = 64,
    compute_spot_latent: bool = True,
    auto_n_clusters: bool = False,
    candidate_n_clusters: tuple[int, ...] | list[int] | str | None = None,
    auto_k_max_score_subregions: int = 2500,
    auto_k_gap_references: int = 8,
    auto_k_mds_components: int = 8,
    auto_k_pilot_n_init: int = 1,
    auto_k_pilot_max_iter: int = 3,
    min_subregions_per_cluster: int = 50,
    seed: int = 1337,
    compute_device: str = "auto",
    deep_config: DeepFeatureConfig | None = None,
) -> dict:
    input_h5ad = Path(input_h5ad)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(input_h5ad)
    if spatial_x_key not in adata.obs or spatial_y_key not in adata.obs:
        raise KeyError(f"Spatial keys '{spatial_x_key}' and/or '{spatial_y_key}' not found in obs.")
    deep_config = deep_config or DeepFeatureConfig()
    features, feature_source = resolve_h5ad_features(
        adata,
        feature_obsm_key=feature_obsm_key,
        allow_umap_as_feature=allow_umap_as_feature,
    )
    feature_embedding_warning = feature_source.get("feature_embedding_warning")
    coords_um = np.stack(
        [
            np.asarray(adata.obs[spatial_x_key], dtype=np.float32) * spatial_scale,
            np.asarray(adata.obs[spatial_y_key], dtype=np.float32) * spatial_scale,
        ],
        axis=1,
    )
    feature_obsm_key_used = str(feature_source.get("feature_key", feature_obsm_key))
    resolved_compute_device = _resolve_compute_device(compute_device)
    if resolved_compute_device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(resolved_compute_device)
        except Exception:
            pass
    deep_embedding: np.ndarray | None = None
    deep_outputs: dict[str, str] = {}
    deep_summary = {
        "enabled": False,
        "method": "none",
    }
    if deep_config.method != "none":
        active_deep_config = deep_config
        batch = None
        count_layer_used = None
        if deep_config.batch_key is not None:
            if deep_config.batch_key not in adata.obs:
                raise KeyError(f"Deep-feature batch key '{deep_config.batch_key}' not found in obs.")
            batch = np.asarray(adata.obs[deep_config.batch_key].astype(str))
        if deep_config.pretrained_model is not None:
            encoder = SpatialOTFeatureEncoder.load(deep_config.pretrained_model, device=deep_config.device)
            active_deep_config = encoder.config
            allow_joint_ot_embedding = bool(active_deep_config.allow_joint_ot_embedding or deep_config.allow_joint_ot_embedding)
            if active_deep_config.output_embedding == "joint" and not allow_joint_ot_embedding:
                raise ValueError(
                    "Using deep.output_embedding='joint' as the OT feature view requires explicit opt-in. "
                    "Set deep.allow_joint_ot_embedding=true or pass --deep-allow-joint-ot-embedding."
                )
            encoder._validate_transform_schema(
                input_obsm_key=feature_obsm_key,
                coordinate_keys=(spatial_x_key, spatial_y_key),
                spatial_scale=spatial_scale,
            )
            deep_embedding = encoder.transform(features=features, coords_um=coords_um)
            history = list(encoder.history)
            model_path = str(Path(deep_config.pretrained_model))
            validation_report = dict(getattr(encoder, "validation_report", {}))
            feature_schema = dict(getattr(encoder, "feature_schema", {}))
            latent_diagnostics = dict(getattr(encoder, "latent_diagnostics", {}))
            if active_deep_config.count_layer is not None:
                count_layer_used = str(active_deep_config.count_layer)
        else:
            allow_joint_ot_embedding = bool(deep_config.allow_joint_ot_embedding)
            if deep_config.output_embedding == "joint" and not allow_joint_ot_embedding:
                raise ValueError(
                    "Using deep.output_embedding='joint' as the OT feature view requires explicit opt-in. "
                    "Set deep.allow_joint_ot_embedding=true or pass --deep-allow-joint-ot-embedding."
                )
            model_path = str(output_dir / "deep_feature_model.pt") if deep_config.save_model else None
            count_matrix, count_layer_used = _extract_count_target(adata, count_layer=deep_config.count_layer)
            deep_result = fit_deep_features(
                features=features,
                coords_um=coords_um,
                config=deep_config,
                batch=batch,
                count_matrix=count_matrix,
                seed=seed,
                save_path=model_path,
                feature_schema_extra=_deep_feature_schema_extra(
                    feature_obsm_key=feature_obsm_key,
                    feature_source=feature_source,
                    spatial_x_key=spatial_x_key,
                    spatial_y_key=spatial_y_key,
                    spatial_scale=spatial_scale,
                ),
            )
            deep_embedding = deep_result.embedding.astype(np.float32)
            history = list(deep_result.history)
            validation_report = dict(deep_result.validation_report)
            feature_schema = dict(deep_result.feature_schema)
            latent_diagnostics = dict(deep_result.latent_diagnostics)
        features = np.asarray(deep_embedding, dtype=np.float32)
        feature_obsm_key_used = active_deep_config.output_obsm_key
        adata.obsm[feature_obsm_key_used] = features.astype(np.float32)
        history_path = output_dir / "deep_feature_history.csv"
        save_deep_feature_history(history, history_path)
        config_path = output_dir / "deep_feature_config.json"
        config_path.write_text(json.dumps(asdict(active_deep_config), indent=2))
        deep_outputs["deep_feature_history"] = str(history_path)
        deep_outputs["deep_feature_config"] = str(config_path)
        if model_path is not None:
            deep_outputs["deep_feature_model"] = str(model_path)
            meta_path = Path(model_path).with_suffix(Path(model_path).suffix + ".meta.json")
            scaler_path = Path(model_path).with_suffix(Path(model_path).suffix + ".scaler.npz")
            if meta_path.exists():
                deep_outputs["deep_feature_model_meta"] = str(meta_path)
            if scaler_path.exists():
                deep_outputs["deep_feature_scaler"] = str(scaler_path)
        final_train_loss = history[-1].get("train_loss") if history else None
        final_val_loss = history[-1].get("val_loss") if history and "val_loss" in history[-1] else None
        count_reconstruction_summary: str | dict[str, object]
        if active_deep_config.count_layer is None:
            count_reconstruction_summary = "disabled"
        else:
            count_reconstruction_summary = {
                "enabled": True,
                "target_layer": str(count_layer_used or active_deep_config.count_layer),
                "decoder_rank": int(active_deep_config.count_decoder_rank),
                "gene_chunk_size": int(active_deep_config.count_chunk_size),
                "loss_weight": float(active_deep_config.count_loss_weight),
            }
        deep_summary = {
            "enabled": True,
            "method": active_deep_config.method,
            "input_feature_obsm_key": feature_obsm_key,
            "output_feature_obsm_key": feature_obsm_key_used,
            "latent_dim": int(features.shape[1]),
            "epochs": int(active_deep_config.epochs),
            "batch_key": active_deep_config.batch_key,
            "neighbor_k": int(active_deep_config.neighbor_k),
            "radius_um": float(active_deep_config.radius_um) if active_deep_config.radius_um is not None else None,
            "short_radius_um": float(active_deep_config.short_radius_um) if active_deep_config.short_radius_um is not None else None,
            "mid_radius_um": float(active_deep_config.mid_radius_um) if active_deep_config.mid_radius_um is not None else None,
            "graph_layers": int(active_deep_config.graph_layers),
            "graph_aggr": active_deep_config.graph_aggr,
            "graph_max_neighbors": int(active_deep_config.graph_max_neighbors),
            "full_batch_max_cells": int(active_deep_config.full_batch_max_cells),
            "validation": active_deep_config.validation,
            "validation_context_mode": active_deep_config.validation_context_mode,
            "allow_joint_ot_embedding": bool(allow_joint_ot_embedding),
            "uses_absolute_coordinate_features": False,
            "uses_spatial_graph": bool(active_deep_config.method == "graph_autoencoder"),
            "output_embedding": active_deep_config.output_embedding,
            "ot_feature_view_warning": (
                "joint_embedding_explicit_opt_in"
                if active_deep_config.output_embedding == "joint"
                else None
            ),
            "final_train_loss": float(final_train_loss) if final_train_loss is not None else None,
            "final_val_loss": float(final_val_loss) if final_val_loss is not None else None,
            "model_path": model_path,
            "batch_correction": "disabled",
            "count_reconstruction": count_reconstruction_summary,
            "pretrained_model_loaded": bool(deep_config.pretrained_model is not None),
            "validation_used_for_early_stopping": bool(
                deep_config.pretrained_model is None and active_deep_config.validation != "none"
            ),
            "runtime_memory": latent_diagnostics.get("runtime_memory"),
            "feature_schema": feature_schema,
            "validation_report": validation_report,
            "latent_diagnostics": latent_diagnostics,
        }
    region_geometries = None
    subregion_members = None
    subregion_centers_um = None
    build_generated_subregions = True
    if region_obs_key is not None:
        if region_obs_key not in adata.obs:
            raise KeyError(f"Region obs key '{region_obs_key}' not found in obs.")
        grouped = pd.Series(np.arange(adata.n_obs), index=adata.obs[region_obs_key].astype(str))
        grouped_items = list(grouped.groupby(level=0))
        region_ids = [str(region_id) for region_id, _ in grouped_items]
        subregion_members = [group.to_numpy(dtype=np.int32) for _, group in grouped_items]
        subregion_centers_um = np.vstack([coords_um[members].mean(axis=0) for members in subregion_members]).astype(np.float32)
        if region_geometry_json is not None:
            region_geometries = _load_region_geometry_json(
                region_geometry_json,
                region_ids=region_ids,
                subregion_members=subregion_members,
                spatial_scale=spatial_scale,
            )
        else:
            region_geometries = [
                RegionGeometry(region_id=str(region_id), members=np.asarray(members, dtype=np.int32))
                for region_id, members in zip(region_ids, subregion_members, strict=False)
            ]
        subregion_members, subregion_centers_um, region_geometries = _filter_explicit_regions_by_min_cells(
            subregion_members=subregion_members,
            subregion_centers_um=subregion_centers_um,
            region_geometries=region_geometries,
            min_cells=min_cells,
        )
        build_generated_subregions = False
    elif region_geometry_json is not None:
        raise ValueError("--region-geometry-json requires --region-obs-key so geometries can be matched to cells.")

    _io_progress("calling multilevel fit")
    result = fit_multilevel_ot(
        features=features,
        coords_um=coords_um,
        subregion_members=subregion_members,
        subregion_centers_um=subregion_centers_um,
        n_clusters=n_clusters,
        atoms_per_cluster=atoms_per_cluster,
        radius_um=radius_um,
        stride_um=stride_um,
        basic_niche_size_um=basic_niche_size_um,
        min_cells=min_cells,
        max_subregions=max_subregions,
        max_subregion_area_um2=max_subregion_area_um2,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        geometry_eps=geometry_eps,
        ot_eps=ot_eps,
        rho=rho,
        geometry_samples=geometry_samples,
        compressed_support_size=compressed_support_size,
        align_iters=align_iters,
        allow_reflection=allow_reflection,
        allow_scale=allow_scale,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_penalty=scale_penalty,
        shift_penalty=shift_penalty,
        n_init=n_init,
        region_geometries=region_geometries,
        build_generated_subregions=build_generated_subregions,
        allow_convex_hull_fallback=allow_convex_hull_fallback,
        max_iter=max_iter,
        tol=tol,
        overlap_consistency_weight=overlap_consistency_weight,
        overlap_jaccard_min=overlap_jaccard_min,
        overlap_contrast_scale=overlap_contrast_scale,
        compute_spot_latent=compute_spot_latent,
        subregion_construction_method=subregion_construction_method,
        subregion_feature_weight=subregion_feature_weight,
        subregion_feature_dims=subregion_feature_dims,
        deep_segmentation_knn=deep_segmentation_knn,
        deep_segmentation_feature_dims=deep_segmentation_feature_dims,
        deep_segmentation_feature_weight=deep_segmentation_feature_weight,
        deep_segmentation_spatial_weight=deep_segmentation_spatial_weight,
        subregion_clustering_method=subregion_clustering_method,
        auto_n_clusters=auto_n_clusters,
        candidate_n_clusters=candidate_n_clusters,
        auto_k_max_score_subregions=auto_k_max_score_subregions,
        auto_k_gap_references=auto_k_gap_references,
        auto_k_mds_components=auto_k_mds_components,
        auto_k_pilot_n_init=auto_k_pilot_n_init,
        auto_k_pilot_max_iter=auto_k_pilot_max_iter,
        min_subregions_per_cluster=min_subregions_per_cluster,
        seed=seed,
        compute_device=str(resolved_compute_device),
    )
    _io_progress("multilevel fit returned; computing fallback and shape summaries")
    fallback_fraction = float(np.mean(result.subregion_geometry_used_fallback.astype(np.float32)))
    if fallback_fraction > 0:
        warnings.warn(
            f"{int(result.subregion_geometry_used_fallback.sum())}/{len(result.subregion_members)} subregions used observed-coordinate convex-hull geometry fallback. Treat this run as exploratory rather than boundary-shape-invariant.",
            RuntimeWarning,
            stacklevel=2,
        )
    shape_region_geometries = region_geometries
    if shape_region_geometries is None and set(result.subregion_geometry_sources) == {"observed_point_cloud"}:
        shape_region_geometries = _region_geometries_from_observed_points(result.subregion_members)
    shape_df = _shape_descriptor_frame(result.subregion_members, coords_um, region_geometries=shape_region_geometries)
    _io_progress("shape descriptor frame complete")
    density_df = _density_descriptor_frame(shape_df, result)
    if shape_diagnostics:
        shape_leakage = _shape_leakage_balanced_accuracy(shape_df, result.subregion_cluster_labels, seed=seed)
        shape_leakage_block = _shape_leakage_spatial_block_accuracy(
            shape_df=shape_df,
            labels=result.subregion_cluster_labels,
            centers_um=result.subregion_centers_um,
            seed=seed,
        )
        shape_leakage_perm = _shape_leakage_permutation_baseline(
            shape_df,
            result.subregion_cluster_labels,
            seed=seed,
            n_perm=shape_leakage_permutations,
        )
        density_leakage = _shape_leakage_balanced_accuracy(density_df, result.subregion_cluster_labels, seed=seed)
        density_leakage_block = _shape_leakage_spatial_block_accuracy(
            shape_df=density_df,
            labels=result.subregion_cluster_labels,
            centers_um=result.subregion_centers_um,
            seed=seed,
        )
        density_leakage_perm = _shape_leakage_permutation_baseline(
            density_df,
            result.subregion_cluster_labels,
            seed=seed,
            n_perm=shape_leakage_permutations,
        )
    else:
        shape_leakage = None
        shape_leakage_block = None
        shape_leakage_perm = None
        density_leakage = None
        density_leakage_block = None
        density_leakage_perm = None
    shape_leakage_diagnostics = {
        "balanced_accuracy": shape_leakage,
        "spatial_block_accuracy": shape_leakage_block,
        "permutation": shape_leakage_perm,
    }
    density_leakage_diagnostics = {
        "balanced_accuracy": density_leakage,
        "spatial_block_accuracy": density_leakage_block,
        "permutation": density_leakage_perm,
    }
    _io_progress("computing subregion diagnostic embedding")
    embedding_2d, embedding_name = _compute_subregion_embedding(result.subregion_atom_weights, seed=seed)
    _io_progress("subregion diagnostic embedding complete")
    silhouette = None
    n_unique_labels = np.unique(result.subregion_cluster_labels).size
    if 1 < n_unique_labels < result.subregion_atom_weights.shape[0]:
        score_weights = np.asarray(result.subregion_atom_weights, dtype=np.float32)
        score_labels = np.asarray(result.subregion_cluster_labels, dtype=np.int32)
        if score_weights.shape[0] > 5000:
            keep = np.linspace(0, score_weights.shape[0] - 1, num=5000, dtype=np.int64)
            score_weights = score_weights[keep]
            score_labels = score_labels[keep]
        if 1 < np.unique(score_labels).size < score_weights.shape[0]:
            silhouette = float(
                silhouette_score(
                    score_weights,
                    score_labels,
                    metric="euclidean",
                )
            )
    _io_progress("summary silhouette diagnostic complete")
    sorted_costs = np.sort(result.subregion_cluster_costs, axis=1)
    margin = None
    if sorted_costs.shape[1] >= 2:
        margin = float(np.mean(sorted_costs[:, 1] - sorted_costs[:, 0]))
    coverage_summary = _cell_subregion_coverage(int(adata.n_obs), result.subregion_members)
    _io_progress("cell coverage summary complete")
    membership_validation_summary = {
        "mutually_exclusive": bool(
            int(coverage_summary["cell_subregion_duplicate_count"]) == 0
            and int(coverage_summary["cell_subregion_max_memberships"]) <= 1
        ),
        "full_partition": bool(coverage_summary["cell_subregion_partition_complete"]),
        "covered_cell_count": int(coverage_summary["covered_cell_count"]),
        "uncovered_cell_count": int(coverage_summary["uncovered_cell_count"]),
        "overlapping_cell_count": int(coverage_summary["cell_subregion_duplicate_count"]),
        "max_memberships_per_cell": int(coverage_summary["cell_subregion_max_memberships"]),
        "membership_mode": str(coverage_summary["subregion_membership_mode"]),
        "enforcement": "hard_fail_on_empty_duplicate_out_of_range_or_cross_subregion_overlap",
    }
    primary_cell_labels, primary_cell_probs, primary_cell_membership_counts = _cell_subregion_cluster_projection(
        n_cells=int(adata.n_obs),
        result=result,
    )
    cell_prob_summary = _probability_diagnostics(primary_cell_probs, prefix="cell")
    cell_projection_prob_summary = _probability_diagnostics(result.cell_cluster_probs, prefix="cell_projection")
    subregion_prob_summary = _probability_diagnostics(result.subregion_cluster_probs, prefix="subregion")
    compactness_summary = _subregion_embedding_compactness(result)
    _io_progress("subregion compactness diagnostics complete")
    boundary_summary = _subregion_graph_metrics(
        n_cells=int(adata.n_obs),
        result=result,
        radius_um=radius_um,
        stride_um=stride_um,
        coords_um=coords_um,
    )
    _io_progress("subregion boundary diagnostics complete")
    cost_reliability = _cost_reliability_metrics(result)
    transform_summary = _transform_diagnostics(result)
    cost_scale_summary = {
        "coordinate_scale": float(result.cost_scale_x),
        "feature_scale": float(result.cost_scale_y),
        "feature_to_coordinate_scale_ratio": float(result.cost_scale_y / max(result.cost_scale_x, 1e-8)),
        "effective_feature_to_geometry_weight_ratio": (
            float((lambda_y / max(result.cost_scale_y, 1e-8)) / (lambda_x / max(result.cost_scale_x, 1e-8)))
            if float(lambda_x) > 0
            else None
        ),
    }
    assigned_transport_cost_summary = _assigned_transport_cost_decomposition(result)
    runtime_memory = _runtime_memory_snapshot(resolved_compute_device)
    assigned_effective_eps_values = [float(x) for x in np.unique(np.round(result.subregion_assigned_effective_eps.astype(np.float64), 8))]
    method_stack = _method_stack_summary(
        feature_source=feature_source,
        deep_summary=deep_summary,
        feature_obsm_key=feature_obsm_key_used,
        subregion_clustering_method=result.subregion_clustering_method,
        subregion_clustering_uses_spatial=result.subregion_clustering_uses_spatial,
    )
    subregion_construction = _subregion_construction_summary(
        build_generated_subregions=build_generated_subregions,
        region_obs_key=region_obs_key,
        region_geometry_json=region_geometry_json,
        result=result,
        radius_um=radius_um,
        stride_um=stride_um,
        min_cells=min_cells,
        max_subregions=max_subregions,
        max_subregion_area_um2=max_subregion_area_um2,
        subregion_construction_method=subregion_construction_method,
        subregion_feature_weight=subregion_feature_weight,
        subregion_feature_dims=subregion_feature_dims,
        deep_segmentation_knn=deep_segmentation_knn,
        deep_segmentation_feature_dims=deep_segmentation_feature_dims,
        deep_segmentation_feature_weight=deep_segmentation_feature_weight,
        deep_segmentation_spatial_weight=deep_segmentation_spatial_weight,
        deep_summary=deep_summary,
    )
    subregion_construction["membership_validation"] = membership_validation_summary
    realized_subregion_statistics = _realized_subregion_statistics(
        result=result,
        shape_df=shape_df,
        min_cells=min_cells,
        max_subregion_area_um2=max_subregion_area_um2,
    )
    qc_warnings = _build_qc_warnings(
        feature_embedding_warning=feature_embedding_warning,
        fallback_fraction=float(fallback_fraction),
        assigned_ot_fallback_fraction=float(np.mean(result.subregion_assigned_used_ot_fallback.astype(np.float32))),
        assigned_effective_eps_values=assigned_effective_eps_values,
        requested_ot_eps=float(ot_eps),
        coverage_fraction=float(coverage_summary["cell_subregion_coverage_fraction"]),
        mean_assignment_margin=margin,
        assigned_transport_cost_decomposition=assigned_transport_cost_summary,
        cost_reliability=cost_reliability,
        transform_diagnostics=transform_summary,
        forced_label_fraction=float(result.subregion_forced_label_mask.sum() / max(len(result.subregion_members), 1)),
        deep_summary=deep_summary,
        shape_leakage_diagnostics=shape_leakage_diagnostics,
        density_leakage_diagnostics=density_leakage_diagnostics,
        subregion_construction=subregion_construction,
        realized_subregion_statistics=realized_subregion_statistics,
        auto_k_enabled=bool(auto_n_clusters),
    )
    result_n_clusters = int(result.cluster_supports.shape[0])
    subregion_cluster_counts = _cluster_count_dict(result.subregion_cluster_labels, result_n_clusters)
    subregion_cluster_count_min = min(subregion_cluster_counts.values()) if subregion_cluster_counts else 0
    geometry_source_counts = {
        key: int(value)
        for key, value in pd.Series(result.subregion_geometry_sources).value_counts().sort_index().items()
    }
    shape_descriptor_source_counts = (
        {
            key: int(value)
            for key, value in shape_df["shape_descriptor_source"].value_counts().sort_index().items()
        }
        if "shape_descriptor_source" in shape_df.columns
        else {}
    )
    geometry_sources = set(geometry_source_counts)
    no_geometry_fallback = float(np.mean(result.subregion_geometry_used_fallback.astype(np.float32))) == 0.0
    if no_geometry_fallback and geometry_sources == {"observed_point_cloud"}:
        boundary_invariance_claim = "observed_geometry_normalized_not_full_shape_invariant"
    elif no_geometry_fallback:
        boundary_invariance_claim = "supported_with_explicit_geometry"
    else:
        boundary_invariance_claim = "not_supported_observed_hull_fallback"
    covered_primary_cell_mask = primary_cell_labels >= 0
    projected_disagreement_count = int(
        np.sum(primary_cell_labels[covered_primary_cell_mask] != result.cell_cluster_labels[covered_primary_cell_mask])
    )
    projected_disagreement_fraction = float(
        projected_disagreement_count / max(int(np.sum(covered_primary_cell_mask)), 1)
    )
    primary_prob_argmax = (
        np.argmax(primary_cell_probs[covered_primary_cell_mask], axis=1)
        if np.any(covered_primary_cell_mask)
        else np.empty(0, dtype=np.int32)
    )
    primary_prob_argmax_disagreement_count = int(
        np.sum(primary_prob_argmax != primary_cell_labels[covered_primary_cell_mask])
    )
    primary_prob_argmax_disagreement_fraction = float(
        primary_prob_argmax_disagreement_count / max(int(np.sum(covered_primary_cell_mask)), 1)
    )
    summary = {
        "summary_schema_version": "1",
        "spatial_ot_version": _package_version(),
        "git_sha": _git_sha(),
        "method_family": "multilevel_ot",
        "active_path": "multilevel-ot",
        "capabilities": {
            "count_layer_implemented": True,
            "count_aware_denoising_implemented": True,
            "graph_autoencoder_implemented": True,
            "graph_autoencoder_mini_batch_implemented": False,
            "batch_adversarial_correction_implemented": False,
            "ot_aware_finetuning_implemented": False,
            "multilevel_prediction_bundle_implemented": False,
            "observed_hull_geometry_default_off": True,
            "spot_level_latent_charts_implemented": bool(compute_spot_latent),
            "auto_k_selection_implemented": True,
            "mutually_exclusive_subregions_implemented": True,
            "deep_graph_subregion_segmentation_implemented": True,
        },
        "latent_source": _latent_source_label(feature_source, deep_summary),
        "communication_source": "none",
        "input_h5ad": str(input_h5ad),
        "output_dir": str(output_dir),
        "feature_obsm_key": feature_obsm_key_used,
        "feature_obsm_key_requested": feature_obsm_key,
        "feature_input_mode": str(feature_source.get("input_mode", "obsm")),
        "feature_source": dict(feature_source),
        "feature_embedding_warning": feature_embedding_warning,
        "allow_umap_as_feature": bool(allow_umap_as_feature),
        "spatial_x_key": spatial_x_key,
        "spatial_y_key": spatial_y_key,
        "spatial_scale": float(spatial_scale),
        "region_obs_key": region_obs_key,
        "region_geometry_json": str(region_geometry_json) if region_geometry_json is not None else None,
        "n_cells": int(adata.n_obs),
        "feature_dim": int(features.shape[1]),
        "n_subregions": int(len(result.subregion_members)),
        "n_clusters": result_n_clusters,
        "requested_n_clusters": int(n_clusters),
        "auto_n_clusters": bool(auto_n_clusters),
        "auto_k_selection": result.auto_k_selection,
        "auto_k_selection_role": "exploratory_shortlist_final_refit_requires_stability_confirmation",
        "atoms_per_cluster": int(atoms_per_cluster),
        "subregion_construction": subregion_construction,
        "subregion_construction_mode": str(subregion_construction["mode"]),
        "subregion_construction_method": str(subregion_construction_method),
        "deep_segmentation_knn": int(deep_segmentation_knn),
        "subregion_feature_weight": float(subregion_feature_weight),
        "subregion_feature_dims": int(subregion_feature_dims),
        "deep_segmentation_feature_dims": int(deep_segmentation_feature_dims),
        "deep_segmentation_feature_weight": float(deep_segmentation_feature_weight),
        "deep_segmentation_spatial_weight": float(deep_segmentation_spatial_weight),
        "subregion_clustering_method": str(result.subregion_clustering_method),
        "subregion_clustering_uses_spatial": bool(result.subregion_clustering_uses_spatial),
        "subregion_clustering_feature_space": (
            "pooled raw-member feature-distribution subregion latent embeddings"
            if not result.subregion_clustering_uses_spatial
            else "shape-normalized OT dictionary candidate costs"
        ),
        "radius_used_for_subregion_membership": bool(subregion_construction["radius_used_for_membership"]),
        "radius_um_semantics": str(subregion_construction["radius_um_semantics"]),
        "radius_um": float(radius_um),
        "stride_um": float(stride_um),
        "basic_niche_size_um": float(result.basic_niche_size_um) if result.basic_niche_size_um is not None else None,
        "basic_niche_radius_um": (
            0.5 * float(result.basic_niche_size_um) * float(np.sqrt(2.0))
        ) if result.basic_niche_size_um is not None else None,
        "n_basic_niches": int(result.basic_niche_centers_um.shape[0]),
        "mean_basic_niches_per_subregion": (
            float(np.mean([len(niche_ids) for niche_ids in result.subregion_basic_niche_ids]))
            if result.subregion_basic_niche_ids
            else 0.0
        ),
        "min_cells": int(min_cells),
        "max_subregions": int(max_subregions),
        "max_subregion_area_um2": float(max_subregion_area_um2)
        if max_subregion_area_um2 is not None
        else None,
        "lambda_x": float(lambda_x),
        "lambda_y": float(lambda_y),
        "geometry_eps": float(geometry_eps),
        "ot_eps": float(ot_eps),
        "rho": float(rho),
        "geometry_samples": int(geometry_samples),
        "compressed_support_size": int(compressed_support_size),
        "align_iters": int(align_iters),
        "allow_reflection": bool(allow_reflection),
        "allow_scale": bool(allow_scale),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "scale_penalty": float(scale_penalty),
        "shift_penalty": float(shift_penalty),
        "n_init": int(n_init),
        "allow_convex_hull_fallback": bool(allow_convex_hull_fallback),
        "shape_diagnostics_enabled": bool(shape_diagnostics),
        "shape_leakage_permutations": int(shape_leakage_permutations),
        "compute_spot_latent": bool(compute_spot_latent),
        "auto_k_max_score_subregions": int(auto_k_max_score_subregions),
        "auto_k_gap_references": int(auto_k_gap_references),
        "auto_k_mds_components": int(auto_k_mds_components),
        "auto_k_pilot_n_init": int(auto_k_pilot_n_init),
        "auto_k_pilot_max_iter": int(auto_k_pilot_max_iter),
        "min_subregions_per_cluster": int(min_subregions_per_cluster),
        "effective_min_subregions_per_cluster": int(result.effective_min_subregions_per_cluster),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "overlap_consistency_weight": float(overlap_consistency_weight),
        "overlap_jaccard_min": float(overlap_jaccard_min),
        "overlap_contrast_scale": float(overlap_contrast_scale),
        "seed": int(seed),
        "compute_device_requested": str(compute_device),
        "compute_device_used": str(resolved_compute_device),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_list_env": os.environ.get("SPATIAL_OT_CUDA_DEVICE_LIST"),
        "parallel_restarts_env": os.environ.get("SPATIAL_OT_PARALLEL_RESTARTS"),
        "cuda_target_vram_gb_env": os.environ.get("SPATIAL_OT_CUDA_TARGET_VRAM_GB"),
        "torch_num_threads_env": os.environ.get("SPATIAL_OT_TORCH_NUM_THREADS"),
        "torch_num_interop_threads_env": os.environ.get("SPATIAL_OT_TORCH_NUM_INTEROP_THREADS"),
        "cost_scale_x": float(result.cost_scale_x),
        "cost_scale_y": float(result.cost_scale_y),
        "cost_scale_diagnostics": cost_scale_summary,
        "method_stack": method_stack,
        "requested_ot_eps": float(ot_eps),
        "assigned_ot_fallback_fraction": float(np.mean(result.subregion_assigned_used_ot_fallback.astype(np.float32))),
        "assigned_effective_eps_values": assigned_effective_eps_values,
        "assigned_transport_cost_decomposition": assigned_transport_cost_summary,
        "subregion_embedding_compactness": compactness_summary,
        "boundary_separation": boundary_summary,
        "cost_reliability": cost_reliability,
        "transform_diagnostics": transform_summary,
        "realized_subregion_statistics": realized_subregion_statistics,
        "subregion_membership_validation": membership_validation_summary,
        "selected_restart": int(result.selected_restart),
        "restart_summaries": result.restart_summaries,
        "subregion_cluster_counts": subregion_cluster_counts,
        "subregion_cluster_count_min": subregion_cluster_count_min,
        "subregion_cluster_size_constraint_satisfied": bool(
            subregion_cluster_count_min >= int(result.effective_min_subregions_per_cluster)
        ),
        "cell_cluster_assignment_source": "fitted_subregion_cluster_membership",
        "cell_cluster_counts": _nonnegative_cluster_count_dict(primary_cell_labels, result_n_clusters),
        "cell_projected_cluster_counts": _cluster_count_dict(result.cell_cluster_labels, result_n_clusters),
        "cell_projected_cluster_disagreement_count": projected_disagreement_count,
        "cell_projected_cluster_disagreement_fraction": projected_disagreement_fraction,
        "cell_cluster_probability_argmax_disagreement_count": primary_prob_argmax_disagreement_count,
        "cell_cluster_probability_argmax_disagreement_fraction": primary_prob_argmax_disagreement_fraction,
        "cell_subregion_assignment_uncovered_count": int(np.sum(primary_cell_membership_counts == 0)),
        "objective_history": result.objective_history,
        "subregion_embedding_method": embedding_name,
        "subregion_weight_silhouette": silhouette,
        "mean_assignment_margin": margin,
        "shape_leakage_balanced_accuracy": shape_leakage,
        "shape_leakage_spatial_block_accuracy": shape_leakage_block,
        "shape_leakage_permutation": shape_leakage_perm,
        "shape_leakage_diagnostics": shape_leakage_diagnostics,
        "density_leakage_balanced_accuracy": density_leakage,
        "density_leakage_spatial_block_accuracy": density_leakage_block,
        "density_leakage_permutation": density_leakage_perm,
        "density_leakage_diagnostics": density_leakage_diagnostics,
        "leakage_qc_thresholds": {
            "balanced_accuracy_warning": LEAKAGE_BALANCED_ACCURACY_WARNING,
            "permutation_p95_margin_warning": LEAKAGE_PERMUTATION_P95_MARGIN_WARNING,
            "permutation_mean_excess_warning": LEAKAGE_PERMUTATION_MEAN_EXCESS_WARNING,
        },
        **coverage_summary,
        **cell_prob_summary,
        **cell_projection_prob_summary,
        **subregion_prob_summary,
        "geometry_point_count_range": [
            int(result.subregion_geometry_point_counts.min()),
            int(result.subregion_geometry_point_counts.max()),
        ],
        "geometry_fallback_fraction": fallback_fraction,
        "convex_hull_fallback_fraction": fallback_fraction,
        "degenerate_geometry_subregion_count": int(np.sum(result.subregion_geometry_point_counts < 3)),
        "degenerate_geometry_subregion_fraction": float(np.mean((result.subregion_geometry_point_counts < 3).astype(np.float32))),
        "geometry_source_counts": geometry_source_counts,
        "shape_descriptor_source_counts": shape_descriptor_source_counts,
        "forced_label_count": int(result.subregion_forced_label_mask.sum()),
        "forced_label_fraction": float(result.subregion_forced_label_mask.sum() / max(len(result.subregion_members), 1)),
        "normalizer_radius_p95_mean": float(np.nanmean(result.subregion_normalizer_radius_p95)),
        "normalizer_radius_max_mean": float(np.nanmean(result.subregion_normalizer_radius_max)),
        "normalizer_interpolation_residual_mean": float(np.nanmean(result.subregion_normalizer_interpolation_residual)),
        "normalizer_diagnostics": {
            "radius_p95_mean": float(np.nanmean(result.subregion_normalizer_radius_p95)),
            "radius_max_mean": float(np.nanmean(result.subregion_normalizer_radius_max)),
            "interpolation_residual_mean": float(np.nanmean(result.subregion_normalizer_interpolation_residual)),
        },
        "boundary_invariance_claim": boundary_invariance_claim,
        "qc_warnings": qc_warnings,
        "qc_warning_count": int(len(qc_warnings)),
        "qc_has_warnings": bool(any(item.get("severity") == "warning" for item in qc_warnings)),
        "runtime_memory": runtime_memory,
        "method_layers": {
            "layer_1_subregion_formation": (
                "The clustered biological unit is a fitted mutually exclusive subregion containing many cells. "
                "Generated boundaries are data-driven spatial components, optionally feature-aware or deep-graph refined; "
                "explicit-region runs may supply external region geometry."
            ),
            "layer_2_subregion_heterogeneity_clustering": (
                "Each subregion is converted into a raw member-cell feature-distribution latent embedding summarizing "
                "its internal cell-state distribution. Niche clusters are learned by pooling all subregion latent "
                "embeddings across the cohort and clustering that pooled matrix; this label-assignment step does not "
                "use spatial coordinates, subregion centers, overlap edges, compressed OT supports, or OT candidate costs."
            ),
            "layer_3_projection_and_visualization": (
                "Cell labels, spot-level latent fields, and sample maps are downstream projections of fitted subregion "
                "clusters for interpretation and QC; they do not redefine the primary subregion labels."
            ),
        },
        "method_notes": {
            "core": "pooled raw-member feature-distribution subregion latent clustering with fixed-label OT atom diagnostics",
            "geometry_normalization": "data-driven geometry samples from each subregion are OT-mapped into a shared unit-disk reference domain before clustering; generated subregions use their observed cell point cloud, explicit regions use supplied mask/polygon geometry, and degenerate 1-2 point subregions fall back to centered-and-scaled local coordinates without OT interpolation",
            "geometry_proxy": "generated subregion memberships are learned from observed coordinates plus the OT feature view by data-driven spatial partitioning and graph-aware minimum-size merging; generated subregion boundary and shape are taken from observed member coordinates rather than a hand-coded template geometry",
            "deep_segmentation": "when subregion_construction_method='deep_segmentation', generated memberships start from many coordinate seeds for full tissue coverage, then a spatial kNN boundary-refinement pass moves boundaries using learned/deep feature affinity before connected small pieces are merged to satisfy min_cells and max_subregions",
            "subregion_membership_radius": "radius_um is not a generated-subregion membership radius; generated memberships are a full-coverage mutually exclusive spatial/feature partition controlled by target scale, min_cells, max_subregions, and graph-aware merging",
            "basic_niches": "when basic_niche_size_um is set, it is only a target scale hint for data-driven atomic membership seeds; fitted subregions are mutually exclusive spatial-graph connected pieces with sparse pieces merged to satisfy min_cells, while geometry remains data-driven from observed cells",
            "subregion_latent_clustering": "primary niche labels are assigned by KMeans/model selection on pooled subregion latent embeddings built from feature-distribution moments only; spatial coordinates are not used for this clustering step",
            "local_measure": "compressed empirical measures over canonical coordinates and standardized cell-level features retained for fixed-label atom dictionaries, projection, and QC diagnostics",
            "local_matching": "semi-relaxed unbalanced Sinkhorn with fixed source marginal and relaxed target marginal is used after labels are fixed to fit diagnostic atoms and projections",
            "overlap_consistency": "the overlap-penalty outputs are retained for backward-compatible diagnostics, but fitted subregion memberships are required to be mutually exclusive, so generated runs should report zero true-overlap edges",
            "residual_alignment": "weighted similarity transform is optimized during subregion-to-cluster matching",
            "support_sharing": "subregions assigned to the same cluster reuse the same shared atom dictionary but keep subregion-specific mixture weights",
            "subregion_cluster_size": "cluster-size constraints are applied to the number of fitted subregions assigned to each subregion cluster, not to projected cell or spot labels",
            "cell_boundary_projection": "cell-level scores are an approximate projection from canonical-coordinate plus feature fit to assigned cluster atoms, modulated by fitted-subregion cluster evidence; they are not an exact posterior under the OT model",
            "spot_level_latent": "spot-level latent charts are learned after the regional OT fit. The default chart is OT atom-barycentric MDS: fitted cluster atom measures define global anchors, and each occurrence is placed by barycentering its assigned cluster's atom embedding with a cost-gap-calibrated OT atom posterior. Raw aligned coordinates are not concatenated into the default chart features, cluster-local variance is not forced to a fixed radius, and posterior entropy/atom-argmax/effective-temperature diagnostics are saved. Treat this as diagnostic visualization, not independent validation",
            "auto_k_selection": "when enabled under pooled_subregion_latent clustering, K is selected directly from pooled raw-member feature-distribution subregion latent embeddings using Silhouette, CH, DB, and Gap. The historical OT-landmark selector remains only for ot_dictionary mode. Treat auto-K as exploratory until full fixed-K stability checks are run around the selected K",
        },
        "deep_features": deep_summary,
    }
    _save_multilevel_outputs(
        adata=adata,
        result=result,
        output_dir=output_dir,
        feature_obsm_key=feature_obsm_key_used,
        spatial_x_key=spatial_x_key,
        spatial_y_key=spatial_y_key,
        spatial_scale=spatial_scale,
        radius_um=radius_um,
        stride_um=stride_um,
        embedding_2d=embedding_2d,
        embedding_name=embedding_name,
        shape_df=shape_df,
        summary=summary,
        deep_embedding=deep_embedding,
        deep_obsm_key=feature_obsm_key_used if deep_embedding is not None else None,
        extra_outputs=deep_outputs,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_multilevel_ot_with_config(config: MultilevelExperimentConfig) -> dict:
    return run_multilevel_ot_on_h5ad(
        input_h5ad=config.paths.input_h5ad,
        output_dir=config.paths.output_dir,
        feature_obsm_key=config.paths.feature_obsm_key,
        spatial_x_key=config.paths.spatial_x_key,
        spatial_y_key=config.paths.spatial_y_key,
        spatial_scale=config.paths.spatial_scale,
        region_obs_key=config.paths.region_obs_key,
        region_geometry_json=config.paths.region_geometry_json,
        allow_umap_as_feature=config.paths.allow_umap_as_feature,
        n_clusters=config.ot.n_clusters,
        atoms_per_cluster=config.ot.atoms_per_cluster,
        radius_um=config.ot.radius_um,
        stride_um=config.ot.stride_um,
        basic_niche_size_um=config.ot.basic_niche_size_um,
        min_cells=config.ot.min_cells,
        max_subregions=config.ot.max_subregions,
        max_subregion_area_um2=config.ot.max_subregion_area_um2,
        lambda_x=config.ot.lambda_x,
        lambda_y=config.ot.lambda_y,
        geometry_eps=config.ot.geometry_eps,
        ot_eps=config.ot.ot_eps,
        rho=config.ot.rho,
        geometry_samples=config.ot.geometry_samples,
        compressed_support_size=config.ot.compressed_support_size,
        align_iters=config.ot.align_iters,
        allow_reflection=config.ot.allow_reflection,
        allow_scale=config.ot.allow_scale,
        min_scale=config.ot.min_scale,
        max_scale=config.ot.max_scale,
        scale_penalty=config.ot.scale_penalty,
        shift_penalty=config.ot.shift_penalty,
        n_init=config.ot.n_init,
        allow_convex_hull_fallback=config.ot.allow_convex_hull_fallback,
        max_iter=config.ot.max_iter,
        tol=config.ot.tol,
        overlap_consistency_weight=config.ot.overlap_consistency_weight,
        overlap_jaccard_min=config.ot.overlap_jaccard_min,
        overlap_contrast_scale=config.ot.overlap_contrast_scale,
        subregion_construction_method=config.ot.subregion_construction_method,
        subregion_feature_weight=config.ot.subregion_feature_weight,
        subregion_feature_dims=config.ot.subregion_feature_dims,
        deep_segmentation_knn=config.ot.deep_segmentation_knn,
        deep_segmentation_feature_dims=config.ot.deep_segmentation_feature_dims,
        deep_segmentation_feature_weight=config.ot.deep_segmentation_feature_weight,
        deep_segmentation_spatial_weight=config.ot.deep_segmentation_spatial_weight,
        subregion_clustering_method=config.ot.subregion_clustering_method,
        shape_diagnostics=config.ot.shape_diagnostics,
        shape_leakage_permutations=config.ot.shape_leakage_permutations,
        compute_spot_latent=config.ot.compute_spot_latent,
        auto_n_clusters=config.ot.auto_n_clusters,
        candidate_n_clusters=config.ot.candidate_n_clusters,
        auto_k_max_score_subregions=config.ot.auto_k_max_score_subregions,
        auto_k_gap_references=config.ot.auto_k_gap_references,
        auto_k_mds_components=config.ot.auto_k_mds_components,
        auto_k_pilot_n_init=config.ot.auto_k_pilot_n_init,
        auto_k_pilot_max_iter=config.ot.auto_k_pilot_max_iter,
        min_subregions_per_cluster=config.ot.min_subregions_per_cluster,
        seed=config.ot.seed,
        compute_device=config.ot.compute_device,
        deep_config=config.deep,
    )
