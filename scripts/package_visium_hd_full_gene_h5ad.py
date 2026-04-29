from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


DEFAULT_SAMPLES: tuple[tuple[str, str, str], ...] = (
    (
        "p1_crc",
        "../work/visium_hd_p1_crc/exports/p1_crc_cells_marker_genes_umap3d.h5ad",
        "../work/visium_hd_p1_crc/runs/visium_hd_h1_yd7cdzk_a1_sr410_retry1/outs/segmented_outputs/spatial/scalefactors_json.json",
    ),
    (
        "p2_crc",
        "../work/visium_hd_p2_crc/exports/p2_crc_cells_marker_genes_umap3d.h5ad",
        "../work/visium_hd_p2_crc/runs/visium_hd_h1_vm2jxxk_a1_sr410_current/outs/segmented_outputs/spatial/scalefactors_json.json",
    ),
    (
        "p3_nat",
        "../work/visium_hd_p3_nat/exports/p3_nat_cells_marker_genes_umap3d.h5ad",
        "../work/visium_hd_p3_nat/runs/visium_hd_h1_vm2jxxk_d1_sr410_current/outs/segmented_outputs/spatial/scalefactors_json.json",
    ),
    (
        "p5_crc",
        "../work/visium_hd_p5_crc/exports/p5_crc_cells_marker_genes_umap3d.h5ad",
        "../work/visium_hd_p5_crc/runs/visium_hd_h1_b77777h_d1_sr410_current/outs/segmented_outputs/spatial/scalefactors_json.json",
    ),
    (
        "p5_nat",
        "../work/visium_hd_p5_nat/exports/p5_nat_cells_marker_genes_umap3d.h5ad",
        "../work/visium_hd_p5_nat/runs/visium_hd_h1_b77777h_a1_sr410_current/outs/segmented_outputs/spatial/scalefactors_json.json",
    ),
)


def _resolve_repo_relative(path: str | Path, *, base_dir: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base_dir / p).resolve()


def _relative_to_base(path: Path, *, base_dir: Path) -> str:
    return os.path.relpath(path, start=base_dir)


def _sample_label_from_id(sample_id: str) -> tuple[str, str]:
    parts = sample_id.split("_", 1)
    patient = parts[0] if parts else sample_id
    condition = parts[1] if len(parts) > 1 else ""
    return patient, condition


def _read_microns_per_pixel(path: Path, *, fallback: float) -> float:
    if not path.exists():
        return float(fallback)
    with path.open("r", encoding="utf-8") as handle:
        scalefactors = json.load(handle)
    scale = scalefactors.get("microns_per_pixel")
    if scale is None:
        return float(fallback)
    return float(scale)


def _prepare_sample(
    sample_id: str,
    path: Path,
    *,
    base_dir: Path,
    scalefactors_json: Path,
    spatial_scale_um_per_unit: float,
) -> ad.AnnData:
    sample = ad.read_h5ad(path)
    if sample.n_obs == 0 or sample.n_vars == 0:
        raise ValueError(f"Input sample is empty: {path}")
    missing = [key for key in ("cell_x", "cell_y") if key not in sample.obs.columns]
    if missing:
        raise KeyError(f"{path} is missing required coordinate columns: {missing}")

    patient_id, tissue_condition = _sample_label_from_id(sample_id)
    original_obs_names = sample.obs_names.astype(str).to_numpy()
    sample.obs_names = pd.Index([f"{sample_id}:{name}" for name in original_obs_names], name="cell_barcode")

    sample.obs["sample_id"] = str(sample_id)
    sample.obs["patient_id"] = str(patient_id)
    sample.obs["tissue_condition"] = str(tissue_condition)
    sample.obs["source_h5ad"] = path.name
    sample.obs["source_path"] = _relative_to_base(path, base_dir=base_dir)
    sample.obs["source_scalefactors_json"] = _relative_to_base(scalefactors_json, base_dir=base_dir)
    sample.obs["original_obs_id"] = original_obs_names
    if "cell_id" not in sample.obs.columns:
        sample.obs["cell_id"] = original_obs_names

    scale_um_per_unit = _read_microns_per_pixel(scalefactors_json, fallback=spatial_scale_um_per_unit)
    x = np.asarray(sample.obs["cell_x"], dtype=np.float32)
    y = np.asarray(sample.obs["cell_y"], dtype=np.float32)
    sample.obs["x"] = x
    sample.obs["y"] = y
    sample.obs["spatial_scale_um_per_unit"] = float(scale_um_per_unit)
    sample.obs["x_um"] = x * float(scale_um_per_unit)
    sample.obs["y_um"] = y * float(scale_um_per_unit)
    for key in list(sample.obsm.keys()):
        del sample.obsm[key]
    sample.obsm["spatial"] = np.column_stack([x, y]).astype(np.float32)
    sample.obsm["spatial_um"] = np.column_stack([sample.obs["x_um"], sample.obs["y_um"]]).astype(np.float32)

    if sparse.issparse(sample.X):
        sample.X = sample.X.tocsr()
    return sample


def package_visium_hd_full_gene_h5ad(
    *,
    output_h5ad: Path,
    manifest_json: Path | None,
    base_dir: Path,
    spatial_scale_um_per_unit: float,
    compression: str | None,
) -> dict[str, object]:
    inputs = [
        (
            sample_id,
            _resolve_repo_relative(path, base_dir=base_dir),
            _resolve_repo_relative(scalefactors_json, base_dir=base_dir),
        )
        for sample_id, path, scalefactors_json in DEFAULT_SAMPLES
    ]
    missing = [str(path) for _, path, _ in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Visium HD input H5AD(s): " + ", ".join(missing))

    samples = [
        _prepare_sample(
            sample_id,
            path,
            base_dir=base_dir,
            scalefactors_json=scalefactors_json,
            spatial_scale_um_per_unit=spatial_scale_um_per_unit,
        )
        for sample_id, path, scalefactors_json in inputs
    ]

    first_var_names = samples[0].var_names.astype(str).to_numpy()
    for sample_id, sample in zip([item[0] for item in inputs], samples, strict=False):
        if sample.n_vars != first_var_names.size or not np.array_equal(sample.var_names.astype(str).to_numpy(), first_var_names):
            raise ValueError(f"Gene axis for sample {sample_id} does not match the first sample.")

    cohort = ad.concat(
        samples,
        axis=0,
        join="inner",
        merge="same",
        uns_merge=None,
        label=None,
        index_unique=None,
    )
    cohort.var = samples[0].var.copy()
    cohort.var_names_make_unique()
    if sparse.issparse(cohort.X):
        cohort.X = cohort.X.tocsr()

    cohort.uns["visium_hd_full_gene_package"] = {
        "description": "Cell-level full-gene Visium HD cohort package.",
        "matrix": "X stores full-gene cell-by-gene expression counts from exported Visium HD cell H5AD files.",
        "n_samples": int(len(inputs)),
        "sample_ids": [sample_id for sample_id, _, _ in inputs],
        "source_h5ad": [_relative_to_base(path, base_dir=base_dir) for _, path, _ in inputs],
        "source_scalefactors_json": [
            _relative_to_base(scalefactors_json, base_dir=base_dir) for _, _, scalefactors_json in inputs
        ],
        "coordinate_columns": {
            "raw": ["x", "y"],
            "microns": ["x_um", "y_um"],
        },
        "obsm": {
            "spatial": "raw cell_x/cell_y coordinates from the source files",
            "spatial_um": "cell_x/cell_y multiplied by the sample-specific microns_per_pixel value",
        },
        "fallback_spatial_scale_um_per_unit": float(spatial_scale_um_per_unit),
        "spatial_scale_um_per_unit_by_sample": {
            sample_id: float(sample.obs["spatial_scale_um_per_unit"].iloc[0])
            for sample_id, sample in zip([item[0] for item in inputs], samples, strict=False)
        },
    }

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    cohort.write_h5ad(output_h5ad, compression=compression)

    summary = {
        "output_h5ad": _relative_to_base(output_h5ad, base_dir=base_dir),
        "n_obs": int(cohort.n_obs),
        "n_vars": int(cohort.n_vars),
        "samples": [
            {
                "sample_id": sample_id,
                "source_h5ad": _relative_to_base(path, base_dir=base_dir),
                "source_scalefactors_json": _relative_to_base(scalefactors_json, base_dir=base_dir),
                "n_cells": int(sample.n_obs),
                "spatial_scale_um_per_unit": float(sample.obs["spatial_scale_um_per_unit"].iloc[0]),
            }
            for (sample_id, path, scalefactors_json), sample in zip(inputs, samples, strict=False)
        ],
        "obs_required_columns": ["sample_id", "x", "y", "x_um", "y_um", "spatial_scale_um_per_unit"],
        "obsm_keys": ["spatial", "spatial_um"],
        "x_dtype": str(cohort.X.dtype),
        "x_sparse": bool(sparse.issparse(cohort.X)),
        "compression": compression,
        "fallback_spatial_scale_um_per_unit": float(spatial_scale_um_per_unit),
    }
    if manifest_json is None:
        manifest_json = output_h5ad.with_suffix(".manifest.json")
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Package full-gene cell-level Visium HD cohort expression into one H5AD.")
    parser.add_argument(
        "--output-h5ad",
        default="../outputs/visium_hd/visium_hd_cohort_cell_full_gene_expression.h5ad",
        help="Output cohort H5AD path.",
    )
    parser.add_argument(
        "--manifest-json",
        default=None,
        help="Optional manifest JSON path. Defaults to output path with .manifest.json suffix.",
    )
    parser.add_argument(
        "--spatial-scale-um-per-unit",
        type=float,
        default=0.2737012522439323,
        help="Fallback scale factor for coordinate conversion if a sample scalefactors JSON is unavailable.",
    )
    parser.add_argument(
        "--compression",
        default="lzf",
        choices=["lzf", "gzip", "none"],
        help="H5AD compression. Use 'none' to disable.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    compression = None if args.compression == "none" else str(args.compression)
    summary = package_visium_hd_full_gene_h5ad(
        output_h5ad=_resolve_repo_relative(args.output_h5ad, base_dir=base_dir),
        manifest_json=_resolve_repo_relative(args.manifest_json, base_dir=base_dir) if args.manifest_json else None,
        base_dir=base_dir,
        spatial_scale_um_per_unit=float(args.spatial_scale_um_per_unit),
        compression=compression,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
