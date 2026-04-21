from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path
import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


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
    if not feature_obsm_keys:
        raise ValueError("At least one feature obsm key must be provided for pooling.")

    output_path = Path(output_h5ad)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_frames: list[pd.DataFrame] = []
    sample_obsm: list[dict[str, np.ndarray]] = []
    sample_ids: list[str] = []
    spans_x: list[float] = []
    spans_y: list[float] = []
    summaries: dict[str, dict[str, object]] = {}

    for input_path in paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input H5AD not found: {input_path}")
        sample_id = _sample_id_from_path(input_path, sample_id_suffix)
        adata = ad.read_h5ad(input_path)
        if spatial_x_key not in adata.obs or spatial_y_key not in adata.obs:
            raise KeyError(
                f"Spatial keys '{spatial_x_key}' and/or '{spatial_y_key}' not found in obs for {input_path.name}."
            )
        for key in feature_obsm_keys:
            if key not in adata.obsm:
                raise KeyError(f"Feature obsm key '{key}' not found in {input_path.name}.")

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
            for key in feature_obsm_keys
        }
        sample_frames.append(obs)
        sample_obsm.append(obsm_payload)
        sample_ids.append(sample_id)
        summaries[str(sample_id)] = {
            "sample_id": str(sample_id),
            "source_h5ad": str(input_path.name),
            "n_cells": int(adata.n_obs),
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
    pooled_obsm_parts: dict[str, list[np.ndarray]] = {key: [] for key in feature_obsm_keys}
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
    pooled = ad.AnnData(
        X=sparse.csr_matrix((pooled_obs.shape[0], 0), dtype=np.float32),
        obs=pooled_obs,
    )
    for key, chunks in pooled_obsm_parts.items():
        pooled.obsm[key] = np.vstack(chunks).astype(np.float32)
    pooled.uns["pooled_inputs"] = {
        "n_samples": int(len(sample_ids)),
        "n_cells": int(pooled.n_obs),
        "input_files": [str(path.name) for path in paths],
        "sample_ids": [str(sample_id) for sample_id in sample_ids],
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
        "sample_ids": [str(sample_id) for sample_id in sample_ids],
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
