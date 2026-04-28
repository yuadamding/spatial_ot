from __future__ import annotations

from dataclasses import asdict
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess

import anndata as ad
import numpy as np

from ..config import DeepFeatureConfig
from ..feature_source import resolve_h5ad_features
from .features import SpatialOTFeatureEncoder, fit_deep_features, save_deep_feature_history

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _package_version() -> str:
    try:
        return version("spatial-ot")
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if pyproject.exists():
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


def _extract_features_and_coords(
    adata: ad.AnnData,
    *,
    feature_obsm_key: str,
    spatial_x_key: str,
    spatial_y_key: str,
    spatial_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if spatial_x_key not in adata.obs or spatial_y_key not in adata.obs:
        raise KeyError(f"Spatial keys '{spatial_x_key}' and '{spatial_y_key}' must both be present in obs.")
    features, feature_source = resolve_h5ad_features(
        adata,
        feature_obsm_key=feature_obsm_key,
        allow_umap_as_feature=True,
    )
    coords_um = np.stack(
        [
            np.asarray(adata.obs[spatial_x_key], dtype=np.float32),
            np.asarray(adata.obs[spatial_y_key], dtype=np.float32),
        ],
        axis=1,
    ) * float(spatial_scale)
    return features, coords_um, feature_source


def _feature_schema_extra(
    *,
    feature_obsm_key: str,
    feature_source: dict,
    spatial_x_key: str,
    spatial_y_key: str,
    spatial_scale: float,
) -> dict:
    payload = {
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


def _extract_count_target(adata: ad.AnnData, *, count_layer: str | None):
    if count_layer is None:
        return None, None
    layer_key = str(count_layer)
    if layer_key == "X":
        if adata.X is None:
            raise ValueError("deep.count_layer requested the primary count matrix, but adata.X is missing.")
        return adata.X, "X"
    if layer_key not in adata.layers:
        raise KeyError(f"deep.count_layer '{layer_key}' was not found in adata.layers.")
    return adata.layers[layer_key], layer_key


def _deep_summary(
    *,
    config: DeepFeatureConfig,
    feature_obsm_key: str,
    output_obsm_key: str,
    embedding_dim: int,
    history: list[dict[str, float]],
    feature_schema: dict,
    validation_report: dict,
    latent_diagnostics: dict,
    model_path: str | None,
    pretrained_model_loaded: bool,
    extra: dict | None = None,
) -> dict:
    final_train_loss = history[-1].get("train_loss") if history else None
    final_val_loss = history[-1].get("val_loss") if history and "val_loss" in history[-1] else None
    count_layer_used = None if extra is None else extra.get("count_layer_used")
    payload = {
        "enabled": True,
        "method": config.method,
        "input_feature_obsm_key": feature_obsm_key,
        "output_feature_obsm_key": output_obsm_key,
        "latent_dim": int(embedding_dim),
        "full_batch_max_cells": int(config.full_batch_max_cells),
        "epochs": int(config.epochs),
        "batch_key": config.batch_key,
        "neighbor_k": int(config.neighbor_k),
        "radius_um": float(config.radius_um) if config.radius_um is not None else None,
        "short_radius_um": float(config.short_radius_um) if config.short_radius_um is not None else None,
        "mid_radius_um": float(config.mid_radius_um) if config.mid_radius_um is not None else None,
        "graph_layers": int(config.graph_layers),
        "graph_aggr": config.graph_aggr,
        "graph_max_neighbors": int(config.graph_max_neighbors),
        "validation": config.validation,
        "validation_context_mode": config.validation_context_mode,
        "allow_joint_ot_embedding": bool(config.allow_joint_ot_embedding),
        "uses_absolute_coordinate_features": False,
        "uses_spatial_graph": bool(config.method == "graph_autoencoder"),
        "output_embedding": config.output_embedding,
        "ot_feature_view_warning": (
            "joint_embedding_explicit_opt_in"
            if config.output_embedding == "joint" and bool(config.allow_joint_ot_embedding)
            else None
        ),
        "final_train_loss": float(final_train_loss) if final_train_loss is not None else None,
        "final_val_loss": float(final_val_loss) if final_val_loss is not None else None,
        "model_path": model_path,
        "batch_correction": "disabled",
        "count_reconstruction": (
            {
                "enabled": True,
                "target_layer": str(count_layer_used or config.count_layer),
                "decoder_rank": int(config.count_decoder_rank),
                "gene_chunk_size": int(config.count_chunk_size),
                "loss_weight": float(config.count_loss_weight),
            }
            if config.count_layer is not None
            else "disabled"
        ),
        "pretrained_model_loaded": bool(pretrained_model_loaded),
        "validation_used_for_early_stopping": bool(config.validation != "none"),
        "runtime_memory": latent_diagnostics.get("runtime_memory"),
        "feature_schema": feature_schema,
        "validation_report": validation_report,
        "latent_diagnostics": latent_diagnostics,
    }
    if extra:
        payload.update(extra)
    return payload


def _deep_method_stack(
    *,
    active_path: str,
    config: DeepFeatureConfig,
    feature_source: dict,
    output_obsm_key: str,
) -> dict[str, object]:
    return {
        "method_family": "deep_feature_adapter",
        "active_path": str(active_path),
        "deep_feature_adapter": str(config.method),
        "latent_output": (
            f"deep_{config.output_embedding}"
            if config.output_embedding is not None
            else "deep_unspecified"
        ),
        "input_feature_key": str(feature_source.get("requested_feature_key", feature_source.get("feature_key", ""))),
        "input_feature_mode": str(feature_source.get("input_mode", "obsm")),
        "feature_space_kind": str(feature_source.get("feature_space_kind", "unknown")),
        "output_obsm_key": str(output_obsm_key),
        "legacy_teacher_student_used": False,
        "communication_source": "none",
    }


def _deep_qc_warnings(*, feature_embedding_warning: str | None, config: DeepFeatureConfig) -> list[dict[str, object]]:
    warnings_out: list[dict[str, object]] = []
    if feature_embedding_warning == "umap_exploratory":
        warnings_out.append(
            {
                "code": "umap_feature_space_exploratory",
                "severity": "warning",
                "message": "The feature space came from UMAP coordinates, which are exploratory rather than metric-preserving.",
            }
        )
    elif feature_embedding_warning == "visualization_embedding_like":
        warnings_out.append(
            {
                "code": "visualization_like_feature_space",
                "severity": "warning",
                "message": "The feature space came from a visualization-like embedding and should be treated as exploratory.",
            }
        )
    if config.method == "graph_autoencoder":
        warnings_out.append(
            {
                "code": "graph_autoencoder_full_batch_only",
                "severity": "info",
                "message": "graph_autoencoder remains a full-batch encoder guarded by deep.full_batch_max_cells.",
            }
        )
    return warnings_out


def fit_deep_features_on_h5ad(
    *,
    input_h5ad: str | Path,
    output_dir: str | Path,
    feature_obsm_key: str,
    spatial_x_key: str,
    spatial_y_key: str,
    spatial_scale: float,
    config: DeepFeatureConfig,
    seed: int = 1337,
) -> dict:
    input_h5ad = Path(input_h5ad)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.pretrained_model is not None:
        raise ValueError("deep-fit trains a new encoder bundle. Use deep-transform to apply a pretrained model.")

    adata = ad.read_h5ad(input_h5ad)
    features, coords_um, feature_source = _extract_features_and_coords(
        adata,
        feature_obsm_key=feature_obsm_key,
        spatial_x_key=spatial_x_key,
        spatial_y_key=spatial_y_key,
        spatial_scale=spatial_scale,
    )
    batch = None
    if config.batch_key is not None:
        if config.batch_key not in adata.obs:
            raise KeyError(f"Deep-feature batch key '{config.batch_key}' not found in obs.")
        batch = np.asarray(adata.obs[config.batch_key].astype(str))
    count_matrix, count_layer_used = _extract_count_target(adata, count_layer=config.count_layer)

    model_path = output_dir / "deep_feature_model.pt" if config.save_model else None
    result = fit_deep_features(
        features=features,
        coords_um=coords_um,
        config=config,
        batch=batch,
        count_matrix=count_matrix,
        seed=seed,
        save_path=model_path,
        feature_schema_extra=_feature_schema_extra(
            feature_obsm_key=feature_obsm_key,
            feature_source=feature_source,
            spatial_x_key=spatial_x_key,
            spatial_y_key=spatial_y_key,
            spatial_scale=spatial_scale,
        ),
    )
    output_obsm_key = config.output_obsm_key
    adata.obsm[output_obsm_key] = np.asarray(result.embedding, dtype=np.float32)

    embedded_h5ad_path = output_dir / "cells_deep_features.h5ad"
    history_path = output_dir / "deep_feature_history.csv"
    config_path = output_dir / "deep_feature_config.json"
    summary_path = output_dir / "summary.json"

    save_deep_feature_history(result.history, history_path)
    config_path.write_text(json.dumps(asdict(config), indent=2))

    outputs = {
        "embedded_h5ad": str(embedded_h5ad_path),
        "deep_feature_history": str(history_path),
        "deep_feature_config": str(config_path),
        "summary": str(summary_path),
    }
    if model_path is not None:
        outputs["deep_feature_model"] = str(model_path)
        meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
        scaler_path = model_path.with_suffix(model_path.suffix + ".scaler.npz")
        if meta_path.exists():
            outputs["deep_feature_model_meta"] = str(meta_path)
        if scaler_path.exists():
            outputs["deep_feature_scaler"] = str(scaler_path)

    deep_summary = _deep_summary(
        config=config,
        feature_obsm_key=feature_obsm_key,
        output_obsm_key=output_obsm_key,
        embedding_dim=int(result.embedding.shape[1]),
        history=result.history,
        feature_schema=result.feature_schema,
        validation_report=result.validation_report,
        latent_diagnostics=result.latent_diagnostics,
        model_path=str(model_path) if model_path is not None else None,
        pretrained_model_loaded=False,
        extra={"count_layer_used": count_layer_used},
    )
    summary = {
        "summary_schema_version": "1",
        "spatial_ot_version": _package_version(),
        "git_sha": _git_sha(),
        "method_family": "deep_feature_adapter",
        "active_path": "deep-fit",
        "latent_source": f"deep_{config.output_embedding}" if config.output_embedding is not None else "deep_unspecified",
        "communication_source": "none",
        "input_h5ad": str(input_h5ad),
        "output_dir": str(output_dir),
        "feature_obsm_key": feature_obsm_key,
        "feature_input_mode": str(feature_source.get("input_mode", "obsm")),
        "feature_source": dict(feature_source),
        "feature_embedding_warning": feature_source.get("feature_embedding_warning"),
        "output_obsm_key": output_obsm_key,
        "spatial_x_key": spatial_x_key,
        "spatial_y_key": spatial_y_key,
        "spatial_scale": float(spatial_scale),
        "n_cells": int(adata.n_obs),
        "feature_dim": int(features.shape[1]),
        "method_stack": _deep_method_stack(
            active_path="deep-fit",
            config=config,
            feature_source=feature_source,
            output_obsm_key=output_obsm_key,
        ),
        "qc_warnings": _deep_qc_warnings(
            feature_embedding_warning=feature_source.get("feature_embedding_warning"),
            config=config,
        ),
        "deep_features": deep_summary,
        "outputs": outputs,
    }

    adata.uns["deep_features"] = {
        "feature_obsm_key": feature_obsm_key,
        "output_obsm_key": output_obsm_key,
        "summary_json": json.dumps(summary),
    }
    adata.write_h5ad(embedded_h5ad_path, compression="gzip")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def transform_h5ad_with_deep_model(
    *,
    model_path: str | Path,
    input_h5ad: str | Path,
    output_h5ad: str | Path,
    feature_obsm_key: str,
    spatial_x_key: str,
    spatial_y_key: str,
    spatial_scale: float,
    output_obsm_key: str | None = None,
    batch_size: int | None = None,
) -> dict:
    model_path = Path(model_path)
    input_h5ad = Path(input_h5ad)
    output_h5ad = Path(output_h5ad)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)

    encoder = SpatialOTFeatureEncoder.load(model_path)
    adata = ad.read_h5ad(input_h5ad)
    features, coords_um, feature_source = _extract_features_and_coords(
        adata,
        feature_obsm_key=feature_obsm_key,
        spatial_x_key=spatial_x_key,
        spatial_y_key=spatial_y_key,
        spatial_scale=spatial_scale,
    )
    encoder._validate_transform_schema(
        input_obsm_key=feature_obsm_key,
        coordinate_keys=(spatial_x_key, spatial_y_key),
        spatial_scale=spatial_scale,
    )
    resolved_output_obsm_key = output_obsm_key or encoder.config.output_obsm_key
    embedding = encoder.transform(features=features, coords_um=coords_um, batch_size=batch_size)
    adata.obsm[resolved_output_obsm_key] = np.asarray(embedding, dtype=np.float32)

    summary_path = output_h5ad.with_suffix(output_h5ad.suffix + ".summary.json")
    deep_summary = _deep_summary(
        config=encoder.config,
        feature_obsm_key=feature_obsm_key,
        output_obsm_key=resolved_output_obsm_key,
        embedding_dim=int(embedding.shape[1]),
        history=encoder.history,
        feature_schema=encoder.feature_schema,
        validation_report=encoder.validation_report,
        latent_diagnostics=encoder.latent_diagnostics,
        model_path=str(model_path),
        pretrained_model_loaded=True,
        extra={
            "transform_batch_size": int(batch_size) if batch_size is not None else int(encoder.config.batch_size),
            "graph_inference_mode": "full_batch" if encoder.config.method == "graph_autoencoder" else "batched",
        },
    )
    summary = {
        "summary_schema_version": "1",
        "spatial_ot_version": _package_version(),
        "git_sha": _git_sha(),
        "method_family": "deep_feature_adapter",
        "active_path": "deep-transform",
        "latent_source": (
            f"deep_{encoder.config.output_embedding}"
            if encoder.config.output_embedding is not None
            else "deep_unspecified"
        ),
        "communication_source": "none",
        "input_h5ad": str(input_h5ad),
        "output_h5ad": str(output_h5ad),
        "model_path": str(model_path),
        "feature_obsm_key": feature_obsm_key,
        "feature_input_mode": str(feature_source.get("input_mode", "obsm")),
        "feature_source": dict(feature_source),
        "feature_embedding_warning": feature_source.get("feature_embedding_warning"),
        "output_obsm_key": resolved_output_obsm_key,
        "spatial_x_key": spatial_x_key,
        "spatial_y_key": spatial_y_key,
        "spatial_scale": float(spatial_scale),
        "n_cells": int(adata.n_obs),
        "feature_dim": int(features.shape[1]),
        "method_stack": _deep_method_stack(
            active_path="deep-transform",
            config=encoder.config,
            feature_source=feature_source,
            output_obsm_key=resolved_output_obsm_key,
        ),
        "qc_warnings": _deep_qc_warnings(
            feature_embedding_warning=feature_source.get("feature_embedding_warning"),
            config=encoder.config,
        ),
        "deep_features": deep_summary,
        "outputs": {
            "embedded_h5ad": str(output_h5ad),
            "summary": str(summary_path),
        },
    }

    adata.uns["deep_features_transform"] = {
        "feature_obsm_key": feature_obsm_key,
        "output_obsm_key": resolved_output_obsm_key,
        "summary_json": json.dumps(summary),
    }
    adata.write_h5ad(output_h5ad, compression="gzip")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


__all__ = [
    "fit_deep_features_on_h5ad",
    "transform_h5ad_with_deep_model",
]
