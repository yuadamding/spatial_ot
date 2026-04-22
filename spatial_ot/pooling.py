from __future__ import annotations

import json
import os
from math import ceil, sqrt
from pathlib import Path
import tempfile

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .feature_source import PREPARED_FEATURES_UNS_KEY


def _sample_id_from_path(path: Path, suffix: str) -> str:
    stem = path.stem
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _pooled_offsets(
    spans_x: list[float],
    spans_y: list[float],
    *,
    n_items: int,
    layout_columns: int | None,
    layout_gap: float | None,
) -> tuple[list[tuple[float, float]], dict[str, float | int]]:
    if n_items <= 0:
        return [], {"layout_columns": 0, "layout_rows": 0, "layout_gap": 0.0}

    max_span_x = max(max(spans_x), 1.0)
    max_span_y = max(max(spans_y), 1.0)
    cols = int(layout_columns) if layout_columns is not None and int(layout_columns) > 0 else int(ceil(sqrt(n_items)))
    rows = int(ceil(n_items / max(cols, 1)))
    gap = float(layout_gap) if layout_gap is not None else max(10000.0, 10.0 * max(max_span_x, max_span_y))
    tile_w = max_span_x + gap
    tile_h = max_span_y + gap

    offsets: list[tuple[float, float]] = []
    for idx in range(n_items):
        row = idx // cols
        col = idx % cols
        offsets.append((float(col) * tile_w, float(row) * tile_h))
    return offsets, {
        "layout_columns": int(cols),
        "layout_rows": int(rows),
        "layout_gap": float(gap),
        "tile_width": float(tile_w),
        "tile_height": float(tile_h),
    }


def _write_h5ad_atomically(adata: ad.AnnData, output_path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{output_path.name}.",
        suffix=".tmp.h5ad",
        dir=str(output_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        adata.write_h5ad(tmp_path, compression="gzip")
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def pool_h5ad_files(
    input_paths: list[str | Path],
    output_h5ad: str | Path,
    *,
    feature_obsm_keys: list[str],
    spatial_x_key: str = "cell_x",
    spatial_y_key: str = "cell_y",
    pooled_spatial_x_key: str = "pooled_cell_x",
    pooled_spatial_y_key: str = "pooled_cell_y",
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    sample_id_suffix: str = "_cells_marker_genes_umap3d",
    layout_columns: int | None = None,
    layout_gap: float | None = None,
) -> dict:
    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("At least one input H5AD path is required.")
    feature_keys = [str(key) for key in feature_obsm_keys]
    if not feature_keys:
        raise ValueError("At least one feature key must be provided for pooling.")
    preserve_x = "X" in feature_keys
    obsm_feature_keys = [key for key in feature_keys if key != "X"]

    output_path = Path(output_h5ad)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_frames: list[pd.DataFrame] = []
    sample_obsm: list[dict[str, np.ndarray]] = []
    sample_x: list[sparse.csr_matrix] = []
    sample_ids: list[str] = []
    spans_x: list[float] = []
    spans_y: list[float] = []
    summaries: dict[str, dict[str, object]] = {}
    reference_var: pd.DataFrame | None = None

    for input_path in paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input H5AD not found: {input_path}")
        sample_id = _sample_id_from_path(input_path, sample_id_suffix)
        adata = ad.read_h5ad(input_path)
        if spatial_x_key not in adata.obs or spatial_y_key not in adata.obs:
            raise KeyError(
                f"Spatial keys '{spatial_x_key}' and/or '{spatial_y_key}' not found in obs for {input_path.name}."
            )
        for key in obsm_feature_keys:
            if key not in adata.obsm:
                raise KeyError(f"Feature obsm key '{key}' not found in {input_path.name}.")
        if preserve_x:
            if adata.X is None or int(adata.n_vars) <= 0:
                raise ValueError(f"Feature key 'X' was requested, but {input_path.name} does not contain a usable gene matrix.")
            if reference_var is None:
                reference_var = adata.var.copy()
            elif not reference_var.index.equals(adata.var_names):
                raise ValueError(
                    f"Gene features do not align across inputs. Expected var_names matching the first sample, but {input_path.name} differs."
                )

        obs = adata.obs.copy()
        obs[sample_obs_key] = str(sample_id)
        obs[source_file_obs_key] = str(input_path.name)
        obs[f"original_{spatial_x_key}"] = np.asarray(obs[spatial_x_key], dtype=np.float32)
        obs[f"original_{spatial_y_key}"] = np.asarray(obs[spatial_y_key], dtype=np.float32)
        base_ids = obs["cell_id"].astype(str).to_numpy() if "cell_id" in obs else obs.index.astype(str).to_numpy()
        obs.index = pd.Index([f"{sample_id}:{cell_id}" for cell_id in base_ids], dtype="object")

        x = np.asarray(obs[spatial_x_key], dtype=np.float32)
        y = np.asarray(obs[spatial_y_key], dtype=np.float32)
        spans_x.append(float(x.max() - x.min()) if x.size else 0.0)
        spans_y.append(float(y.max() - y.min()) if y.size else 0.0)

        obsm_payload = {
            key: np.asarray(adata.obsm[key], dtype=np.float32)
            for key in obsm_feature_keys
        }
        if preserve_x:
            x_matrix = adata.X
            if sparse.issparse(x_matrix):
                sample_x.append(x_matrix.tocsr().astype(np.float32))
            else:
                sample_x.append(sparse.csr_matrix(np.asarray(x_matrix, dtype=np.float32)))
        sample_frames.append(obs)
        sample_obsm.append(obsm_payload)
        sample_ids.append(sample_id)
        summaries[str(sample_id)] = {
            "sample_id": str(sample_id),
            "source_h5ad": str(input_path.name),
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "spatial_x_min": float(x.min()) if x.size else None,
            "spatial_x_max": float(x.max()) if x.size else None,
            "spatial_y_min": float(y.min()) if y.size else None,
            "spatial_y_max": float(y.max()) if y.size else None,
        }

    offsets, layout_meta = _pooled_offsets(
        spans_x,
        spans_y,
        n_items=len(sample_frames),
        layout_columns=layout_columns,
        layout_gap=layout_gap,
    )

    pooled_obs_parts: list[pd.DataFrame] = []
    pooled_obsm_parts: dict[str, list[np.ndarray]] = {key: [] for key in obsm_feature_keys}
    for idx, (obs, obsm_payload) in enumerate(zip(sample_frames, sample_obsm, strict=False)):
        x = np.asarray(obs[spatial_x_key], dtype=np.float32)
        y = np.asarray(obs[spatial_y_key], dtype=np.float32)
        x_shift, y_shift = offsets[idx]
        pooled_obs = obs.copy()
        pooled_obs[pooled_spatial_x_key] = x + float(x_shift) - (float(x.min()) if x.size else 0.0)
        pooled_obs[pooled_spatial_y_key] = y + float(y_shift) - (float(y.min()) if y.size else 0.0)
        pooled_obs["sample_layout_col"] = int(idx % max(int(layout_meta["layout_columns"]), 1))
        pooled_obs["sample_layout_row"] = int(idx // max(int(layout_meta["layout_columns"]), 1))
        pooled_obs["sample_x_shift"] = float(x_shift) - (float(x.min()) if x.size else 0.0)
        pooled_obs["sample_y_shift"] = float(y_shift) - (float(y.min()) if y.size else 0.0)
        pooled_obs_parts.append(pooled_obs)
        for key, value in obsm_payload.items():
            pooled_obsm_parts[key].append(value)

    pooled_obs = pd.concat(pooled_obs_parts, axis=0)
    pooled_x = sparse.vstack(sample_x, format="csr").astype(np.float32) if preserve_x else sparse.csr_matrix((pooled_obs.shape[0], 0), dtype=np.float32)
    if reference_var is not None:
        pooled = ad.AnnData(X=pooled_x, obs=pooled_obs, var=reference_var.copy())
    else:
        pooled = ad.AnnData(X=pooled_x, obs=pooled_obs)
    for key, chunks in pooled_obsm_parts.items():
        pooled.obsm[key] = np.vstack(chunks).astype(np.float32)
    pooled.uns["pooled_inputs"] = {
        "n_samples": int(len(sample_ids)),
        "n_cells": int(pooled.n_obs),
        "n_genes": int(pooled.n_vars),
        "input_files": [str(path.name) for path in paths],
        "sample_ids": [str(sample_id) for sample_id in sample_ids],
        "feature_keys": [str(key) for key in feature_keys],
        "feature_obsm_keys": [str(key) for key in feature_obsm_keys],
        "spatial_keys": {
            "original_x": str(spatial_x_key),
            "original_y": str(spatial_y_key),
            "pooled_x": str(pooled_spatial_x_key),
            "pooled_y": str(pooled_spatial_y_key),
            "sample_obs_key": str(sample_obs_key),
            "source_file_obs_key": str(source_file_obs_key),
        },
        "layout": layout_meta,
        "samples_are_physically_connected": False,
        "pooled_coordinate_note": (
            "Samples share a pooled latent/OT input table, but pooled coordinates are translated onto separate tiles "
            "so graph neighborhoods and OT subregions do not cross between physical specimens."
        ),
        "sample_summaries": summaries,
    }
    pooled.write_h5ad(output_path, compression="gzip")

    return {
        "output_h5ad": str(output_path),
        "n_samples": int(len(sample_ids)),
        "n_cells": int(pooled.n_obs),
        "n_genes": int(pooled.n_vars),
        "sample_ids": [str(sample_id) for sample_id in sample_ids],
        "feature_keys": [str(key) for key in feature_keys],
        "feature_obsm_keys": [str(key) for key in feature_obsm_keys],
        "spatial_x_key": str(pooled_spatial_x_key),
        "spatial_y_key": str(pooled_spatial_y_key),
        "sample_obs_key": str(sample_obs_key),
        "source_file_obs_key": str(source_file_obs_key),
        "layout": layout_meta,
        "pooled_inputs_json": str(output_path.with_suffix(output_path.suffix + ".summary.json")),
    }


def pool_h5ads_in_directory(
    input_dir: str | Path,
    output_h5ad: str | Path,
    *,
    feature_obsm_keys: list[str],
    sample_glob: str = "*_cells_marker_genes_umap3d.h5ad",
    spatial_x_key: str = "cell_x",
    spatial_y_key: str = "cell_y",
    pooled_spatial_x_key: str = "pooled_cell_x",
    pooled_spatial_y_key: str = "pooled_cell_y",
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    sample_id_suffix: str = "_cells_marker_genes_umap3d",
    layout_columns: int | None = None,
    layout_gap: float | None = None,
) -> dict:
    input_root = Path(input_dir)
    paths = sorted(input_root.glob(sample_glob))
    if not paths:
        raise FileNotFoundError(f"No input H5AD files matched '{sample_glob}' under {input_root}.")
    summary = pool_h5ad_files(
        paths,
        output_h5ad=output_h5ad,
        feature_obsm_keys=feature_obsm_keys,
        spatial_x_key=spatial_x_key,
        spatial_y_key=spatial_y_key,
        pooled_spatial_x_key=pooled_spatial_x_key,
        pooled_spatial_y_key=pooled_spatial_y_key,
        sample_obs_key=sample_obs_key,
        source_file_obs_key=source_file_obs_key,
        sample_id_suffix=sample_id_suffix,
        layout_columns=layout_columns,
        layout_gap=layout_gap,
    )
    summary_path = Path(output_h5ad).with_suffix(Path(output_h5ad).suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def distribute_pooled_feature_cache_to_inputs(
    pooled_h5ad: str | Path,
    input_dir: str | Path,
    *,
    prepared_obsm_key: str,
    sample_glob: str = "*_cells_marker_genes_umap3d.h5ad",
    sample_obs_key: str = "sample_id",
    source_file_obs_key: str = "source_h5ad",
    overwrite: bool = False,
) -> dict:
    pooled_path = Path(pooled_h5ad)
    input_root = Path(input_dir)
    paths = sorted(input_root.glob(sample_glob))
    if not pooled_path.exists():
        raise FileNotFoundError(f"Pooled H5AD not found: {pooled_path}")
    if not paths:
        raise FileNotFoundError(f"No input H5AD files matched '{sample_glob}' under {input_root}.")

    pooled = ad.read_h5ad(pooled_path, backed="r")
    try:
        pooled_obs = pooled.obs.copy()
        pooled_uns = dict(pooled.uns.get(PREPARED_FEATURES_UNS_KEY, {}))
    finally:
        pooled.file.close()

    if source_file_obs_key not in pooled_obs.columns:
        raise KeyError(f"Pooled H5AD is missing source-file obs key '{source_file_obs_key}'.")
    if prepared_obsm_key not in pooled_uns:
        raise KeyError(
            f"Pooled H5AD does not contain prepared feature metadata for '{prepared_obsm_key}'. "
            "Run prepare-inputs on the pooled H5AD first."
        )

    prepared_metadata = dict(pooled_uns[prepared_obsm_key])
    source_files = pooled_obs[source_file_obs_key].astype(str).to_numpy()
    pooled_cell_ids = (
        pooled_obs["cell_id"].astype(str).to_numpy()
        if "cell_id" in pooled_obs.columns
        else np.asarray([str(index).split(":", 1)[-1] for index in pooled_obs.index], dtype=object)
    )
    row_indices_by_source = {
        str(source_name): np.flatnonzero(source_files == str(source_name))
        for source_name in pd.unique(source_files)
    }

    distributed: list[dict[str, object]] = []
    with h5py.File(pooled_path, "r") as handle:
        if "obsm" not in handle or prepared_obsm_key not in handle["obsm"]:
            raise KeyError(f"Pooled H5AD is missing obsm['{prepared_obsm_key}'].")
        feature_dataset = handle["obsm"][prepared_obsm_key]
        feature_dim = int(feature_dataset.shape[1])
        for input_path in paths:
            row_indices = np.asarray(row_indices_by_source.get(str(input_path.name), np.asarray([], dtype=np.int64)), dtype=np.int64)
            if row_indices.size == 0:
                raise KeyError(
                    f"No pooled rows were tagged with source file '{input_path.name}'. "
                    f"Check '{source_file_obs_key}' in {pooled_path.name}."
                )

            sample_adata = ad.read_h5ad(input_path)
            expected_metadata = dict(sample_adata.uns.get(PREPARED_FEATURES_UNS_KEY, {})).get(prepared_obsm_key, {})
            reusable = (
                not bool(overwrite)
                and prepared_obsm_key in sample_adata.obsm
                and bool(expected_metadata)
                and bool(expected_metadata.get("distributed_from_pooled", False))
                and str(expected_metadata.get("pooled_source_h5ad", "")) == str(pooled_path.name)
                and int(expected_metadata.get("svd_components_requested", -1)) == int(prepared_metadata.get("svd_components_requested", -1))
                and float(expected_metadata.get("target_sum", float("nan"))) == float(prepared_metadata.get("target_sum", float("nan")))
                and np.asarray(sample_adata.obsm[prepared_obsm_key]).shape == (int(sample_adata.n_obs), feature_dim)
            )
            if reusable:
                distributed.append(
                    {
                        "input_h5ad": str(input_path),
                        "sample_id": str(sample_adata.obs[sample_obs_key].iloc[0]) if sample_obs_key in sample_adata.obs and sample_adata.n_obs > 0 else input_path.stem,
                        "n_cells": int(sample_adata.n_obs),
                        "feature_dim": int(feature_dim),
                        "reused_existing": True,
                    }
                )
                continue

            sample_cell_ids = (
                sample_adata.obs["cell_id"].astype(str).to_numpy()
                if "cell_id" in sample_adata.obs.columns
                else sample_adata.obs.index.astype(str).to_numpy()
            )
            pooled_source_cell_ids = pooled_cell_ids[row_indices]
            if pooled_source_cell_ids.shape[0] != sample_cell_ids.shape[0]:
                raise ValueError(
                    f"Pooled/source cell count mismatch for {input_path.name}: "
                    f"{pooled_source_cell_ids.shape[0]} pooled rows vs {sample_cell_ids.shape[0]} sample rows."
                )
            if np.array_equal(sample_cell_ids, pooled_source_cell_ids):
                ordered_rows = row_indices
            else:
                lookup = pd.Index(pooled_source_cell_ids)
                mapped = lookup.get_indexer(sample_cell_ids)
                if np.any(mapped < 0):
                    raise ValueError(f"Could not align pooled prepared features back to {input_path.name} by cell_id/obs_name.")
                ordered_rows = row_indices[mapped]

            order = np.argsort(ordered_rows)
            sorted_rows = ordered_rows[order]
            sorted_features = np.asarray(feature_dataset[sorted_rows, :], dtype=np.float32)
            features = np.empty_like(sorted_features)
            features[order] = sorted_features
            sample_adata.obsm[prepared_obsm_key] = features
            sample_prepared_uns = dict(sample_adata.uns.get(PREPARED_FEATURES_UNS_KEY, {}))
            sample_prepared_uns[prepared_obsm_key] = {
                **prepared_metadata,
                "output_feature_key": str(prepared_obsm_key),
                "distributed_from_pooled": True,
                "pooled_source_h5ad": str(pooled_path.name),
                "pooled_source_feature_key": str(prepared_obsm_key),
            }
            sample_adata.uns[PREPARED_FEATURES_UNS_KEY] = sample_prepared_uns
            _write_h5ad_atomically(sample_adata, input_path)
            distributed.append(
                {
                    "input_h5ad": str(input_path),
                    "sample_id": str(sample_adata.obs[sample_obs_key].iloc[0]) if sample_obs_key in sample_adata.obs and sample_adata.n_obs > 0 else input_path.stem,
                    "n_cells": int(sample_adata.n_obs),
                    "feature_dim": int(feature_dim),
                    "reused_existing": False,
                }
            )

    return {
        "pooled_h5ad": str(pooled_path),
        "prepared_feature_obsm_key": str(prepared_obsm_key),
        "n_inputs": int(len(distributed)),
        "distributed_inputs": distributed,
    }


__all__ = [
    "pool_h5ad_files",
    "pool_h5ads_in_directory",
    "distribute_pooled_feature_cache_to_inputs",
]
