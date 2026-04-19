from __future__ import annotations

from dataclasses import asdict
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import warnings

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch

from ..config import DeepFeatureConfig, MultilevelExperimentConfig
from ..deep.features import SpatialOTFeatureEncoder, fit_deep_features, save_deep_feature_history
from .core import _resolve_compute_device, fit_multilevel_ot
from .geometry import (
    _shape_descriptor_frame,
    _shape_leakage_balanced_accuracy,
    _shape_leakage_permutation_baseline,
    _shape_leakage_spatial_block_accuracy,
)
from .types import MultilevelOTResult, RegionGeometry


def _compute_subregion_embedding(weights: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    try:
        import umap.umap_ as umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(30, max(3, weights.shape[0] - 1)),
            min_dist=0.2,
            metric="euclidean",
            random_state=seed,
            transform_seed=seed,
        )
        return reducer.fit_transform(weights).astype(np.float32), "UMAP"
    except Exception:
        pca = PCA(n_components=2, random_state=seed)
        return pca.fit_transform(weights).astype(np.float32), "PCA"


def _package_version() -> str:
    try:
        return version("spatial-ot")
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if pyproject.exists():
            import tomllib

            payload = tomllib.loads(pyproject.read_text())
            return str(payload.get("project", {}).get("version", "unknown"))
        return "unknown"


def _git_sha() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return completed.stdout.strip() or None
    return None


def _cluster_palette(n_clusters: int) -> np.ndarray:
    cmap_name = "tab20" if n_clusters <= 20 else "gist_ncar"
    cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(n_clusters)
    rgba = np.asarray([cmap(i) for i in range(n_clusters)], dtype=np.float32)
    return np.clip(np.rint(rgba[:, :3] * 255.0), 0, 255).astype(np.uint8)


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
    map_path = output_dir / "multilevel_ot_spatial_map.png"
    emb_path = output_dir / "multilevel_ot_subregion_embedding.png"
    atom_path = output_dir / "multilevel_ot_atom_layouts.png"
    summary_path = output_dir / "summary.json"
    outputs = {
        "h5ad": str(h5ad_path),
        "subregions": str(subregions_path),
        "supports": str(supports_path),
        "spatial_map": str(map_path),
        "subregion_embedding": str(emb_path),
        "atom_layouts": str(atom_path),
        "summary": str(summary_path),
    }
    if extra_outputs:
        outputs.update(extra_outputs)
    summary["outputs"] = outputs

    palette = _cluster_palette(result.cluster_supports.shape[0])
    label_names = [f"C{int(x)}" for x in result.cell_cluster_labels]
    label_hex = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette[result.cell_cluster_labels].tolist()]

    cells_out = adata.copy()
    cells_out.obs["mlot_cluster_id"] = pd.Categorical(label_names)
    cells_out.obs["mlot_cluster_int"] = result.cell_cluster_labels.astype(np.int32)
    cells_out.obs["mlot_cluster_hex"] = label_hex
    cells_out.obsm["mlot_cluster_probs"] = result.cell_cluster_probs.astype(np.float32)
    cells_out.obsm["mlot_cell_cluster_scores"] = result.cell_cluster_probs.astype(np.float32)
    cells_out.obsm["mlot_feature_cluster_probs"] = result.cell_feature_cluster_probs.astype(np.float32)
    cells_out.obsm["mlot_context_cluster_probs"] = result.cell_context_cluster_probs.astype(np.float32)
    if deep_embedding is not None and deep_obsm_key:
        cells_out.obsm[deep_obsm_key] = np.asarray(deep_embedding, dtype=np.float32)
    cells_out.uns["multilevel_ot"] = {
        "feature_obsm_key": feature_obsm_key,
        "spatial_x_key": spatial_x_key,
        "spatial_y_key": spatial_y_key,
        "spatial_scale": float(spatial_scale),
        "radius_um": float(radius_um),
        "stride_um": float(stride_um),
        "cell_projection_mode": "approximate_assigned_subregion",
        "deep_obsm_key": deep_obsm_key,
        "summary_json": json.dumps(summary),
    }
    cells_out.write_h5ad(h5ad_path, compression="gzip")

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
            "geometry_point_count": int(result.subregion_geometry_point_counts[idx]),
            "geometry_source": result.subregion_geometry_sources[idx],
            "geometry_used_fallback": bool(result.subregion_geometry_used_fallback[idx]),
            "forced_label": bool(result.subregion_forced_label_mask[idx]),
            "argmin_cluster_int": int(result.subregion_argmin_labels[idx]),
            "assigned_effective_eps": float(result.subregion_assigned_effective_eps[idx]),
            "assigned_ot_used_fallback": bool(result.subregion_assigned_used_ot_fallback[idx]),
            "normalizer_radius_p95": float(result.subregion_normalizer_radius_p95[idx]) if np.isfinite(result.subregion_normalizer_radius_p95[idx]) else np.nan,
            "normalizer_radius_max": float(result.subregion_normalizer_radius_max[idx]) if np.isfinite(result.subregion_normalizer_radius_max[idx]) else np.nan,
            "normalizer_interpolation_residual": float(result.subregion_normalizer_interpolation_residual[idx]) if np.isfinite(result.subregion_normalizer_interpolation_residual[idx]) else np.nan,
            "cluster_id": f"C{int(result.subregion_cluster_labels[idx])}",
            "cluster_int": int(result.subregion_cluster_labels[idx]),
            "objective": float(result.subregion_cluster_costs[idx, result.subregion_cluster_labels[idx]]),
            "assignment_margin": float(subregion_margin[idx]) if np.isfinite(subregion_margin[idx]) else np.nan,
        }
        for j, prob in enumerate(result.subregion_cluster_probs[idx]):
            row[f"cluster_prob_{j:02d}"] = float(prob)
        for j, weight in enumerate(result.subregion_atom_weights[idx]):
            row[f"atom_weight_{j:02d}"] = float(weight)
        row["embed1"] = float(embedding_2d[idx, 0])
        row["embed2"] = float(embedding_2d[idx, 1])
        subregion_rows.append(row)
    subregions_df = pd.DataFrame(subregion_rows)
    if not shape_df.empty:
        subregions_df = subregions_df.merge(shape_df, on="subregion_id", how="left")
    subregions_df.to_parquet(subregions_path, index=False)

    np.savez_compressed(
        supports_path,
        cluster_supports=result.cluster_supports.astype(np.float32),
        cluster_atom_coords=result.cluster_atom_coords.astype(np.float32),
        cluster_atom_features=result.cluster_atom_features.astype(np.float32),
        cluster_prototype_weights=result.cluster_prototype_weights.astype(np.float32),
        subregion_atom_weights=result.subregion_atom_weights.astype(np.float32),
    )

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
        mask = result.cell_cluster_labels == cid
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
    seed: int = 1337,
    compute_device: str = "auto",
    deep_config: DeepFeatureConfig | None = None,
) -> dict:
    input_h5ad = Path(input_h5ad)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(input_h5ad)
    if feature_obsm_key not in adata.obsm:
        raise KeyError(f"Feature key '{feature_obsm_key}' not found in obsm.")
    if spatial_x_key not in adata.obs or spatial_y_key not in adata.obs:
        raise KeyError(f"Spatial keys '{spatial_x_key}' and/or '{spatial_y_key}' not found in obs.")
    feature_embedding_warning = None
    if "umap" in feature_obsm_key.lower():
        warnings.warn(
            "Using UMAP coordinates as the OT feature space. UMAP is not generally metric-preserving; prefer PCA or standardized markers for validated runs.",
            RuntimeWarning,
            stacklevel=2,
        )
        feature_embedding_warning = "umap_exploratory"

    deep_config = deep_config or DeepFeatureConfig()
    features = np.asarray(adata.obsm[feature_obsm_key], dtype=np.float32)
    coords_um = np.stack(
        [
            np.asarray(adata.obs[spatial_x_key], dtype=np.float32) * spatial_scale,
            np.asarray(adata.obs[spatial_y_key], dtype=np.float32) * spatial_scale,
        ],
        axis=1,
    )
    feature_obsm_key_used = feature_obsm_key
    resolved_compute_device = _resolve_compute_device(compute_device)
    deep_embedding: np.ndarray | None = None
    deep_outputs: dict[str, str] = {}
    deep_summary = {
        "enabled": False,
        "method": "none",
    }
    if deep_config.method != "none":
        active_deep_config = deep_config
        batch = None
        if deep_config.batch_key is not None:
            if deep_config.batch_key not in adata.obs:
                raise KeyError(f"Deep-feature batch key '{deep_config.batch_key}' not found in obs.")
            batch = np.asarray(adata.obs[deep_config.batch_key].astype(str))
        if deep_config.pretrained_model is not None:
            encoder = SpatialOTFeatureEncoder.load(deep_config.pretrained_model)
            active_deep_config = encoder.config
            deep_embedding = encoder.transform(features=features, coords_um=coords_um)
            history = list(encoder.history)
            model_path = str(Path(deep_config.pretrained_model))
            validation_report = dict(getattr(encoder, "validation_report", {}))
            feature_schema = dict(getattr(encoder, "feature_schema", {}))
        else:
            model_path = str(output_dir / "deep_feature_model.pt") if deep_config.save_model else None
            deep_result = fit_deep_features(
                features=features,
                coords_um=coords_um,
                config=deep_config,
                batch=batch,
                seed=seed,
                save_path=model_path,
            )
            deep_embedding = deep_result.embedding.astype(np.float32)
            history = list(deep_result.history)
            validation_report = dict(deep_result.validation_report)
            feature_schema = dict(deep_result.feature_schema)
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
            "validation": active_deep_config.validation,
            "validation_context_mode": active_deep_config.validation_context_mode,
            "uses_absolute_coordinate_features": False,
            "uses_spatial_graph": bool(active_deep_config.method == "graph_autoencoder"),
            "output_embedding": active_deep_config.output_embedding,
            "final_train_loss": float(final_train_loss) if final_train_loss is not None else None,
            "final_val_loss": float(final_val_loss) if final_val_loss is not None else None,
            "model_path": model_path,
            "batch_correction": "not_implemented",
            "count_reconstruction": "not_implemented",
            "pretrained_model_loaded": bool(deep_config.pretrained_model is not None),
            "validation_used_for_early_stopping": bool(deep_config.validation != "none"),
            "feature_schema": feature_schema,
            "validation_report": validation_report,
        }
    region_geometries = None
    subregion_members = None
    subregion_centers_um = None
    build_grid_subregions = True
    if region_obs_key is not None:
        if region_obs_key not in adata.obs:
            raise KeyError(f"Region obs key '{region_obs_key}' not found in obs.")
        grouped = pd.Series(np.arange(adata.n_obs), index=adata.obs[region_obs_key].astype(str))
        subregion_members = [group.to_numpy(dtype=np.int32) for _, group in grouped.groupby(level=0)]
        subregion_centers_um = np.vstack([coords_um[members].mean(axis=0) for members in subregion_members]).astype(np.float32)
        region_geometries = [
            RegionGeometry(region_id=str(region_id), members=np.asarray(members, dtype=np.int32))
            for region_id, members in grouped.groupby(level=0)
        ]
        build_grid_subregions = False

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords_um,
        subregion_members=subregion_members,
        subregion_centers_um=subregion_centers_um,
        n_clusters=n_clusters,
        atoms_per_cluster=atoms_per_cluster,
        radius_um=radius_um,
        stride_um=stride_um,
        min_cells=min_cells,
        max_subregions=max_subregions,
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
        build_grid_subregions=build_grid_subregions,
        allow_convex_hull_fallback=allow_convex_hull_fallback,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        compute_device=str(resolved_compute_device),
    )
    fallback_fraction = float(np.mean(result.subregion_geometry_used_fallback.astype(np.float32)))
    if fallback_fraction > 0:
        warnings.warn(
            f"{int(result.subregion_geometry_used_fallback.sum())}/{len(result.subregion_members)} subregions used observed-coordinate convex-hull geometry fallback. Treat this run as exploratory rather than boundary-shape-invariant.",
            RuntimeWarning,
            stacklevel=2,
        )
    shape_df = _shape_descriptor_frame(result.subregion_members, coords_um, region_geometries=region_geometries)
    shape_leakage = _shape_leakage_balanced_accuracy(shape_df, result.subregion_cluster_labels, seed=seed)
    shape_leakage_block = _shape_leakage_spatial_block_accuracy(
        shape_df=shape_df,
        labels=result.subregion_cluster_labels,
        centers_um=result.subregion_centers_um,
        seed=seed,
    )
    shape_leakage_perm = _shape_leakage_permutation_baseline(shape_df, result.subregion_cluster_labels, seed=seed)
    embedding_2d, embedding_name = _compute_subregion_embedding(result.subregion_atom_weights, seed=seed)
    silhouette = None
    n_unique_labels = np.unique(result.subregion_cluster_labels).size
    if 1 < n_unique_labels < result.subregion_atom_weights.shape[0]:
        silhouette = float(
            silhouette_score(
                result.subregion_atom_weights,
                result.subregion_cluster_labels,
                metric="euclidean",
            )
        )
    sorted_costs = np.sort(result.subregion_cluster_costs, axis=1)
    margin = None
    if sorted_costs.shape[1] >= 2:
        margin = float(np.mean(sorted_costs[:, 1] - sorted_costs[:, 0]))
    summary = {
        "summary_schema_version": "1",
        "spatial_ot_version": _package_version(),
        "git_sha": _git_sha(),
        "input_h5ad": str(input_h5ad),
        "output_dir": str(output_dir),
        "feature_obsm_key": feature_obsm_key_used,
        "feature_obsm_key_requested": feature_obsm_key,
        "feature_embedding_warning": feature_embedding_warning,
        "spatial_x_key": spatial_x_key,
        "spatial_y_key": spatial_y_key,
        "spatial_scale": float(spatial_scale),
        "region_obs_key": region_obs_key,
        "n_cells": int(adata.n_obs),
        "feature_dim": int(features.shape[1]),
        "n_subregions": int(len(result.subregion_members)),
        "n_clusters": int(n_clusters),
        "atoms_per_cluster": int(atoms_per_cluster),
        "radius_um": float(radius_um),
        "stride_um": float(stride_um),
        "min_cells": int(min_cells),
        "max_subregions": int(max_subregions),
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
        "max_iter": int(max_iter),
        "tol": float(tol),
        "seed": int(seed),
        "compute_device_requested": str(compute_device),
        "compute_device_used": str(resolved_compute_device),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cost_scale_x": float(result.cost_scale_x),
        "cost_scale_y": float(result.cost_scale_y),
        "requested_ot_eps": float(ot_eps),
        "assigned_ot_fallback_fraction": float(np.mean(result.subregion_assigned_used_ot_fallback.astype(np.float32))),
        "assigned_effective_eps_values": [float(x) for x in np.unique(np.round(result.subregion_assigned_effective_eps.astype(np.float64), 8))],
        "selected_restart": int(result.selected_restart),
        "restart_summaries": result.restart_summaries,
        "subregion_cluster_counts": {f"C{int(k)}": int(v) for k, v in pd.Series(result.subregion_cluster_labels).value_counts().sort_index().items()},
        "cell_cluster_counts": {f"C{int(k)}": int(v) for k, v in pd.Series(result.cell_cluster_labels).value_counts().sort_index().items()},
        "objective_history": result.objective_history,
        "subregion_embedding_method": embedding_name,
        "subregion_weight_silhouette": silhouette,
        "mean_assignment_margin": margin,
        "shape_leakage_balanced_accuracy": shape_leakage,
        "shape_leakage_spatial_block_accuracy": shape_leakage_block,
        "shape_leakage_permutation": shape_leakage_perm,
        "shape_leakage_diagnostics": {
            "balanced_accuracy": shape_leakage,
            "spatial_block_accuracy": shape_leakage_block,
            "permutation": shape_leakage_perm,
        },
        "geometry_point_count_range": [
            int(result.subregion_geometry_point_counts.min()),
            int(result.subregion_geometry_point_counts.max()),
        ],
        "geometry_fallback_fraction": fallback_fraction,
        "convex_hull_fallback_fraction": fallback_fraction,
        "geometry_source_counts": {
            key: int(value)
            for key, value in pd.Series(result.subregion_geometry_sources).value_counts().sort_index().items()
        },
        "shape_descriptor_source_counts": {
            key: int(value)
            for key, value in shape_df["shape_descriptor_source"].value_counts().sort_index().items()
        }
        if "shape_descriptor_source" in shape_df.columns
        else {},
        "forced_label_count": int(result.subregion_forced_label_mask.sum()),
        "normalizer_radius_p95_mean": float(np.nanmean(result.subregion_normalizer_radius_p95)),
        "normalizer_radius_max_mean": float(np.nanmean(result.subregion_normalizer_radius_max)),
        "normalizer_interpolation_residual_mean": float(np.nanmean(result.subregion_normalizer_interpolation_residual)),
        "normalizer_diagnostics": {
            "radius_p95_mean": float(np.nanmean(result.subregion_normalizer_radius_p95)),
            "radius_max_mean": float(np.nanmean(result.subregion_normalizer_radius_max)),
            "interpolation_residual_mean": float(np.nanmean(result.subregion_normalizer_interpolation_residual)),
        },
        "boundary_invariance_claim": (
            "supported_with_explicit_geometry"
            if float(np.mean(result.subregion_geometry_used_fallback.astype(np.float32))) == 0.0
            else "not_supported_observed_hull_fallback"
        ),
        "method_notes": {
            "core": "shape-normalized cluster-specific semi-relaxed Wasserstein dictionary clustering",
            "geometry_normalization": "uniform geometry samples from each subregion are OT-mapped into a shared unit-disk reference domain before clustering",
            "geometry_proxy": "when explicit masks are unavailable and convex hull fallback is allowed, geometry samples are drawn from the convex hull of local cell coordinates",
            "local_measure": "compressed empirical measures over canonical coordinates and standardized cell-level features",
            "local_matching": "semi-relaxed unbalanced Sinkhorn with fixed source marginal and relaxed target marginal",
            "residual_alignment": "weighted similarity transform is optimized during subregion-to-cluster matching",
            "support_sharing": "subregions assigned to the same cluster reuse the same shared atom dictionary but keep subregion-specific mixture weights",
            "cell_boundary_projection": "cell-level scores are an approximate projection from canonical-coordinate plus feature fit to assigned cluster atoms, modulated by overlapping-subregion cluster evidence; they are not an exact posterior under the OT model",
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
        n_clusters=config.ot.n_clusters,
        atoms_per_cluster=config.ot.atoms_per_cluster,
        radius_um=config.ot.radius_um,
        stride_um=config.ot.stride_um,
        min_cells=config.ot.min_cells,
        max_subregions=config.ot.max_subregions,
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
        seed=config.ot.seed,
        compute_device=config.ot.compute_device,
        deep_config=config.deep,
    )
