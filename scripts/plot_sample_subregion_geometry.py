from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from spatial_ot.multilevel.plotting import cluster_palette


def _safe_filename_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "sample"


def _resolve_coordinate_keys(obs, requested_x: str | None, requested_y: str | None) -> tuple[str, str]:
    candidates = [
        (requested_x, requested_y),
        ("original_cell_x", "original_cell_y"),
        ("cell_x", "cell_y"),
        ("pooled_cell_x", "pooled_cell_y"),
    ]
    for x_key, y_key in candidates:
        if x_key is not None and y_key is not None and x_key in obs.columns and y_key in obs.columns:
            return str(x_key), str(y_key)
    raise KeyError("No usable coordinate pair found; pass --x-key and --y-key.")


def _convex_hull_polygon(coords: np.ndarray) -> np.ndarray | None:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[0] < 3:
        return None
    unique = np.unique(coords, axis=0)
    if unique.shape[0] < 3:
        return None
    try:
        hull = ConvexHull(unique)
    except QhullError:
        return None
    polygon = unique[hull.vertices]
    return np.vstack([polygon, polygon[0]]).astype(np.float32)


def plot_sample_subregion_geometry(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    sample_key: str = "sample_id",
    source_key: str = "source_h5ad",
    x_key: str | None = None,
    y_key: str | None = None,
    spatial_scale: float | None = None,
    max_background_points: int = 80000,
) -> dict[str, object]:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "sample_subregion_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)

    cells_h5ad = run_dir / "cells_multilevel_ot.h5ad"
    spot_latent_npz = run_dir / "spot_level_latent_multilevel_ot.npz"
    if not cells_h5ad.exists():
        raise FileNotFoundError(f"Missing cells H5AD: {cells_h5ad}")
    if not spot_latent_npz.exists():
        raise FileNotFoundError(f"Missing spot-level latent NPZ with subregion IDs: {spot_latent_npz}")

    adata = ad.read_h5ad(cells_h5ad, backed="r")
    try:
        obs = adata.obs.copy()
        metadata = dict(adata.uns["multilevel_ot"]) if "multilevel_ot" in adata.uns else {}
    finally:
        adata.file.close()

    resolved_x, resolved_y = _resolve_coordinate_keys(obs, x_key, y_key)
    resolved_scale = float(spatial_scale) if spatial_scale is not None else float(metadata.get("spatial_scale", 1.0))
    coords = np.column_stack(
        [
            np.asarray(obs[resolved_x], dtype=np.float32) * resolved_scale,
            np.asarray(obs[resolved_y], dtype=np.float32) * resolved_scale,
        ]
    ).astype(np.float32)

    if sample_key in obs.columns:
        sample_values = obs[sample_key].astype(str).to_numpy()
        sample_ids = [str(value) for value in dict.fromkeys(sample_values.tolist())]
    else:
        sample_values = np.full(obs.shape[0], "all_cells", dtype=object)
        sample_ids = ["all_cells"]

    source_values = obs[source_key].astype(str).to_numpy() if source_key in obs.columns else np.full(obs.shape[0], "", dtype=object)
    cluster_ids = np.asarray(obs["mlot_cluster_int"], dtype=np.int32) if "mlot_cluster_int" in obs.columns else np.zeros(obs.shape[0], dtype=np.int32)
    if "mlot_cluster_hex" in obs.columns:
        cluster_hex_values = obs["mlot_cluster_hex"].astype(str).to_numpy()
        cluster_colors: dict[int, str] = {}
        for cid, color in zip(cluster_ids.tolist(), cluster_hex_values.tolist(), strict=False):
            cluster_colors.setdefault(int(cid), str(color))
    else:
        palette = cluster_palette(int(cluster_ids.max(initial=0)) + 1)
        cluster_colors = {idx: f"#{r:02x}{g:02x}{b:02x}" for idx, (r, g, b) in enumerate(palette.tolist())}

    payload = np.load(spot_latent_npz)
    cell_indices = np.asarray(payload["cell_indices"], dtype=np.int64)
    occurrence_subregion_ids = np.asarray(payload["subregion_ids"], dtype=np.int32)
    subregion_by_cell = np.full(obs.shape[0], -1, dtype=np.int32)
    subregion_by_cell[cell_indices] = occurrence_subregion_ids

    n_samples = len(sample_ids)
    ncols = min(2, max(1, n_samples))
    nrows = int(np.ceil(n_samples / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.0 * ncols, 7.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    rng = np.random.default_rng(0)
    sample_summaries: list[dict[str, object]] = []
    used_cluster_ids: set[int] = set()
    for sample_idx, sample_id in enumerate(sample_ids):
        ax = axes[sample_idx // ncols][sample_idx % ncols]
        sample_mask = sample_values == sample_id
        sample_cell_indices = np.flatnonzero(sample_mask)
        covered_indices = sample_cell_indices[subregion_by_cell[sample_cell_indices] >= 0]
        if covered_indices.size == 0:
            ax.set_axis_off()
            continue

        background_idx = covered_indices
        if max_background_points > 0 and background_idx.size > max_background_points:
            background_idx = rng.choice(background_idx, size=int(max_background_points), replace=False)
        ax.scatter(
            coords[background_idx, 0],
            coords[background_idx, 1],
            s=0.08,
            color="#d9d9d9",
            linewidths=0,
            alpha=0.45,
            rasterized=True,
        )

        order = np.argsort(subregion_by_cell[covered_indices], kind="stable")
        sorted_indices = covered_indices[order]
        sorted_subregions = subregion_by_cell[sorted_indices]
        starts = np.r_[0, np.flatnonzero(sorted_subregions[1:] != sorted_subregions[:-1]) + 1]
        stops = np.r_[starts[1:], sorted_indices.size]

        segments_by_cluster: dict[int, list[np.ndarray]] = {}
        singleton_x: list[float] = []
        singleton_y: list[float] = []
        subregion_sizes: list[int] = []
        for start, stop in zip(starts.tolist(), stops.tolist(), strict=False):
            group_idx = sorted_indices[start:stop]
            subregion_sizes.append(int(group_idx.size))
            cid_values = cluster_ids[group_idx]
            valid = cid_values[cid_values >= 0]
            if valid.size:
                cid = int(np.bincount(valid).argmax())
            else:
                cid = -1
            used_cluster_ids.add(cid)
            polygon = _convex_hull_polygon(coords[group_idx])
            if polygon is None:
                center = coords[group_idx].mean(axis=0)
                singleton_x.append(float(center[0]))
                singleton_y.append(float(center[1]))
                continue
            segments_by_cluster.setdefault(cid, []).append(polygon)

        for cid, segments in segments_by_cluster.items():
            color = cluster_colors.get(int(cid), "#4d4d4d")
            lc = LineCollection(
                segments,
                colors=[color],
                linewidths=0.35,
                alpha=0.9,
                rasterized=True,
            )
            ax.add_collection(lc)
        if singleton_x:
            ax.scatter(singleton_x, singleton_y, s=2.0, color="#4d4d4d", linewidths=0, alpha=0.8, rasterized=True)

        sources = [str(value) for value in dict.fromkeys(source_values[sample_mask].tolist()) if str(value)]
        title = f"{sample_id}: {len(starts)} subregions, {covered_indices.size} cells"
        if len(sources) == 1:
            title = f"{title}\n{sources[0]}"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.autoscale()

        sample_summaries.append(
            {
                "sample_id": str(sample_id),
                "source_h5ad": sources[0] if len(sources) == 1 else sources,
                "n_cells": int(covered_indices.size),
                "n_subregions": int(len(starts)),
                "subregion_size_min": int(np.min(subregion_sizes)),
                "subregion_size_median": float(np.median(subregion_sizes)),
                "subregion_size_max": int(np.max(subregion_sizes)),
            }
        )

    for empty_idx in range(n_samples, nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].set_axis_off()

    legend_handles = [
        Line2D([0], [0], color=cluster_colors.get(cid, "#4d4d4d"), lw=2, label=f"C{cid}")
        for cid in sorted(cid for cid in used_cluster_ids if cid >= 0)
    ]
    if legend_handles:
        fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="lower center", ncol=min(len(legend_handles), 15), frameon=False)
    fig.suptitle("Fitted Subregion Geometry by Sample", fontsize=16)

    output_png = output_dir / "sample_subregion_geometry_overview.png"
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "run_dir": str(run_dir),
        "cells_h5ad": str(cells_h5ad),
        "spot_latent_npz": str(spot_latent_npz),
        "output_png": str(output_png),
        "sample_key": str(sample_key),
        "source_key": str(source_key),
        "plot_spatial_x_key": str(resolved_x),
        "plot_spatial_y_key": str(resolved_y),
        "spatial_scale": float(resolved_scale),
        "geometry": "convex_hull_of_fitted_subregion_member_cells_recovered_from_spot_level_latent_npz",
        "n_samples": int(len(sample_summaries)),
        "samples": sample_summaries,
    }
    manifest_path = output_dir / "sample_subregion_geometry_manifest.json"
    manifest["manifest_json"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot fitted sample-level subregion geometry from a multilevel OT run.")
    parser.add_argument("--run-dir", default="../outputs/spatial_ot/cohort_multilevel_ot_deep_segmentation_20260428_161254")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-key", default="sample_id")
    parser.add_argument("--source-key", default="source_h5ad")
    parser.add_argument("--x-key", default=None)
    parser.add_argument("--y-key", default=None)
    parser.add_argument("--spatial-scale", type=float, default=None)
    parser.add_argument("--max-background-points", type=int, default=80000)
    args = parser.parse_args()

    manifest = plot_sample_subregion_geometry(
        Path(args.run_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        sample_key=args.sample_key,
        source_key=args.source_key,
        x_key=args.x_key,
        y_key=args.y_key,
        spatial_scale=args.spatial_scale,
        max_background_points=args.max_background_points,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
