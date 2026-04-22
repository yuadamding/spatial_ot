from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import warnings

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from ..config import DeepFeatureConfig, MultilevelExperimentConfig
from ..deep.features import SpatialOTFeatureEncoder, fit_deep_features, save_deep_feature_history
from ..feature_source import resolve_h5ad_features
from .._runtime import runtime_memory_snapshot as _runtime_memory_snapshot
from .core import _resolve_compute_device, fit_multilevel_ot
from .diagnostics import (
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
    _shape_descriptor_frame,
    _shape_leakage_balanced_accuracy,
    _shape_leakage_permutation_baseline,
    _shape_leakage_spatial_block_accuracy,
)
from .metadata import (
    extract_count_target as _extract_count_target,
    git_sha as _git_sha,
    latent_source_label as _latent_source_label,
    package_version as _package_version,
)
from .plotting import (
    cluster_palette as _cluster_palette,
    plot_sample_niche_maps,
    plot_sample_niche_maps_from_run_dir,
    plot_sample_spatial_maps,
    plot_sample_spatial_maps_from_run_dir,
)
from .types import MultilevelOTResult, RegionGeometry


def _method_stack_summary(
    *,
    feature_source: dict,
    deep_summary: dict,
    feature_obsm_key: str,
) -> dict[str, object]:
    return {
        "method_family": "multilevel_ot",
        "active_path": "multilevel-ot",
        "core_model": "shape_normalized_multilevel_semi_relaxed_ot",
        "deep_feature_adapter": (
            str(deep_summary.get("method", "none"))
            if bool(deep_summary.get("enabled"))
            else "none"
        ),
        "latent_used_for_ot": _latent_source_label(feature_source, deep_summary),
        "ot_feature_obsm_key": str(feature_obsm_key),
        "feature_input_mode": str(feature_source.get("input_mode", "obsm")),
        "feature_space_kind": str(feature_source.get("feature_space_kind", "unknown")),
        "legacy_teacher_student_used": False,
        "communication_source": "none",
        "cell_projection_mode": "approximate_assigned_subregion",
    }


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
    candidate_diag_path = output_dir / "multilevel_ot_candidate_cost_diagnostics.npz"
    map_path = output_dir / "multilevel_ot_spatial_map.png"
    emb_path = output_dir / "multilevel_ot_subregion_embedding.png"
    atom_path = output_dir / "multilevel_ot_atom_layouts.png"
    summary_path = output_dir / "summary.json"
    outputs = {
        "h5ad": str(h5ad_path),
        "subregions": str(subregions_path),
        "supports": str(supports_path),
        "candidate_cost_diagnostics": str(candidate_diag_path),
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
        "feature_input_mode": summary.get("feature_input_mode"),
        "feature_source": summary.get("feature_source"),
        "spatial_x_key": spatial_x_key,
        "spatial_y_key": spatial_y_key,
        "spatial_scale": float(spatial_scale),
        "radius_um": float(radius_um),
        "stride_um": float(stride_um),
        "basic_niche_size_um": float(result.basic_niche_size_um) if result.basic_niche_size_um is not None else None,
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
    np.savez_compressed(
        candidate_diag_path,
        subregion_cluster_costs=result.subregion_cluster_costs.astype(np.float32),
        subregion_cluster_transport_costs=result.subregion_cluster_transport_costs.astype(np.float32),
        subregion_cluster_overlap_penalties=result.subregion_cluster_overlap_penalties.astype(np.float32),
        subregion_measure_summaries=result.subregion_measure_summaries.astype(np.float32),
        candidate_effective_eps_matrix=result.subregion_candidate_effective_eps_matrix.astype(np.float32),
        candidate_used_ot_fallback_matrix=result.subregion_candidate_used_ot_fallback_matrix.astype(bool),
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

    sample_spatial_manifest = plot_sample_spatial_maps(
        cells_h5ad=h5ad_path,
        output_dir=output_dir / "sample_spatial_maps",
    )
    outputs["sample_spatial_maps_dir"] = str(output_dir / "sample_spatial_maps")
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
            encoder = SpatialOTFeatureEncoder.load(deep_config.pretrained_model)
            active_deep_config = encoder.config
            allow_joint_ot_embedding = bool(active_deep_config.allow_joint_ot_embedding or deep_config.allow_joint_ot_embedding)
            if active_deep_config.output_embedding == "joint" and not allow_joint_ot_embedding:
                raise ValueError(
                    "Using deep.output_embedding='joint' as the OT feature view requires explicit opt-in. "
                    "Set deep.allow_joint_ot_embedding=true or pass --deep-allow-joint-ot-embedding."
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
            "validation_used_for_early_stopping": bool(deep_config.validation != "none"),
            "runtime_memory": latent_diagnostics.get("runtime_memory"),
            "feature_schema": feature_schema,
            "validation_report": validation_report,
            "latent_diagnostics": latent_diagnostics,
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
        basic_niche_size_um=basic_niche_size_um,
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
        overlap_consistency_weight=overlap_consistency_weight,
        overlap_jaccard_min=overlap_jaccard_min,
        overlap_contrast_scale=overlap_contrast_scale,
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
    coverage_summary = _cell_subregion_coverage(int(adata.n_obs), result.subregion_members)
    cell_prob_summary = _probability_diagnostics(result.cell_cluster_probs, prefix="cell")
    subregion_prob_summary = _probability_diagnostics(result.subregion_cluster_probs, prefix="subregion")
    compactness_summary = _subregion_embedding_compactness(result)
    boundary_summary = _subregion_graph_metrics(
        n_cells=int(adata.n_obs),
        result=result,
        radius_um=radius_um,
        stride_um=stride_um,
        coords_um=coords_um,
    )
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
        "n_cells": int(adata.n_obs),
        "feature_dim": int(features.shape[1]),
        "n_subregions": int(len(result.subregion_members)),
        "n_clusters": int(n_clusters),
        "atoms_per_cluster": int(atoms_per_cluster),
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
        **coverage_summary,
        **cell_prob_summary,
        **subregion_prob_summary,
        "geometry_point_count_range": [
            int(result.subregion_geometry_point_counts.min()),
            int(result.subregion_geometry_point_counts.max()),
        ],
        "geometry_fallback_fraction": fallback_fraction,
        "convex_hull_fallback_fraction": fallback_fraction,
        "degenerate_geometry_subregion_count": int(np.sum(result.subregion_geometry_point_counts < 3)),
        "degenerate_geometry_subregion_fraction": float(np.mean((result.subregion_geometry_point_counts < 3).astype(np.float32))),
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
        "forced_label_fraction": float(result.subregion_forced_label_mask.sum() / max(len(result.subregion_members), 1)),
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
        "qc_warnings": qc_warnings,
        "qc_warning_count": int(len(qc_warnings)),
        "qc_has_warnings": bool(any(item.get("severity") == "warning" for item in qc_warnings)),
        "runtime_memory": runtime_memory,
        "method_notes": {
            "core": "shape-normalized cluster-specific semi-relaxed Wasserstein dictionary clustering",
            "geometry_normalization": "uniform geometry samples from each subregion are OT-mapped into a shared unit-disk reference domain before clustering; degenerate 1-2 point subregions fall back to centered-and-scaled local coordinates without OT interpolation",
            "geometry_proxy": "when explicit masks are unavailable and convex hull fallback is allowed, geometry samples are drawn from the convex hull of local cell coordinates",
            "basic_niches": "when basic_niche_size_um is set, grid-built subregions are unions of fixed-size basic niches rather than direct raw-cell radius windows",
            "local_measure": "compressed empirical measures over canonical coordinates and standardized cell-level features",
            "local_matching": "semi-relaxed unbalanced Sinkhorn with fixed source marginal and relaxed target marginal",
            "overlap_consistency": "overlapping subregions can be regularized toward compatible cluster assignments using Jaccard overlap gated by subregion-summary contrast",
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
        allow_umap_as_feature=config.paths.allow_umap_as_feature,
        n_clusters=config.ot.n_clusters,
        atoms_per_cluster=config.ot.atoms_per_cluster,
        radius_um=config.ot.radius_um,
        stride_um=config.ot.stride_um,
        basic_niche_size_um=config.ot.basic_niche_size_um,
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
        overlap_consistency_weight=config.ot.overlap_consistency_weight,
        overlap_jaccard_min=config.ot.overlap_jaccard_min,
        overlap_contrast_scale=config.ot.overlap_contrast_scale,
        seed=config.ot.seed,
        compute_device=config.ot.compute_device,
        deep_config=config.deep,
    )
