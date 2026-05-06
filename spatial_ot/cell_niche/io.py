from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

from ..feature_source import resolve_h5ad_features
from .cluster import cluster_embeddings, connected_components_by_label
from .codebook import fit_state_codebook, prepare_feature_space
from .config import CellNicheConfig
from .descriptors import (
    DEFAULT_BLOCK_WEIGHTS,
    EmbeddingResult,
    compute_cell_heterogeneity_descriptors,
    reduce_descriptor_embedding,
)
from .graph import NeighborhoodGraph, build_knn_graphs, build_radius_graphs
from .train import fit_deepshe_embedding


def _parse_float_tuple(value: str | tuple[float, ...] | list[float] | None) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(float(part.strip()) for part in value.split(",") if part.strip())
    return tuple(float(item) for item in value)


def _parse_int_tuple(value: str | tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    return tuple(int(item) for item in value)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sanitize_for_anndata(value):
    if value is None:
        return "none"
    if isinstance(value, dict):
        return {str(key): _sanitize_for_anndata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, dict) for item in value):
            return {
                str(idx): _sanitize_for_anndata(item)
                for idx, item in enumerate(value)
            }
        return [_sanitize_for_anndata(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return _sanitize_for_anndata(value.tolist())
    return value


def _cluster_counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels, dtype=np.int32), return_counts=True)
    return {f"N{int(label)}": int(count) for label, count in zip(values, counts, strict=False)}


def _choose_instance_graph(graphs: dict[str, NeighborhoodGraph]) -> NeighborhoodGraph:
    radius_graphs = [graph for graph in graphs.values() if graph.mode == "radius"]
    if radius_graphs:
        return radius_graphs[len(radius_graphs) // 2]
    return next(iter(graphs.values()))


def _write_spatial_plot(
    *,
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    xy = np.asarray(coords, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    scatter = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=y,
        s=4,
        cmap="tab20",
        linewidths=0,
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(scatter, ax=ax, label="spatial niche")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_embedding_plot(
    *,
    embedding: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    h = np.asarray(embedding, dtype=np.float32)
    if h.shape[1] < 2:
        coords = np.column_stack([h[:, 0], np.zeros(h.shape[0], dtype=np.float32)])
    else:
        coords = h[:, :2]
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=np.asarray(labels, dtype=np.int32),
        s=4,
        cmap="tab20",
        linewidths=0,
        alpha=0.85,
    )
    ax.set_title("Cell-Centered Spatial Heterogeneity Embedding")
    ax.set_xlabel("SHE 1")
    ax.set_ylabel("SHE 2")
    fig.colorbar(scatter, ax=ax, label="spatial niche")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _store_graphs(adata: ad.AnnData, graphs: dict[str, NeighborhoodGraph]) -> dict[str, str]:
    keys: dict[str, str] = {}
    for key, graph in graphs.items():
        conn_key = f"spatial_connectivities_{key}"
        dist_key = f"spatial_distances_{key}"
        adata.obsp[conn_key] = graph.connectivities
        adata.obsp[dist_key] = graph.distances
        keys[key] = conn_key
    return keys


def _block_weights_from_args(
    *,
    self_weight: float,
    composition_weight: float,
    diversity_weight: float,
    moments_weight: float,
    radial_weight: float,
    pair_weight: float,
    covariance_weight: float,
    gradient_weight: float,
) -> dict[str, float]:
    weights = dict(DEFAULT_BLOCK_WEIGHTS)
    weights.update(
        {
            "self": float(self_weight),
            "composition": float(composition_weight),
            "diversity": float(diversity_weight),
            "moments": float(moments_weight),
            "radial": float(radial_weight),
            "pair": float(pair_weight),
            "covariance": float(covariance_weight),
            "gradient": float(gradient_weight),
        }
    )
    return weights


def _standardize_for_feature_concat(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    mean = np.mean(x, axis=0, dtype=np.float64)
    scale = np.std(x, axis=0, dtype=np.float64)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return ((x.astype(np.float64, copy=False) - mean[None, :]) / scale[None, :]).astype(
        np.float32
    )


def run_cell_niche_on_h5ad(
    *,
    input_h5ad: str | Path,
    output_dir: str | Path,
    feature_obsm_key: str,
    spatial_x_key: str,
    spatial_y_key: str,
    sample_obs_key: str | None = "sample_id",
    spatial_scale: float = 1.0,
    radii_um: str | tuple[float, ...] | list[float] = (20.0, 50.0, 100.0),
    knn_values: str | tuple[int, ...] | list[int] | None = None,
    max_neighbors: int = 256,
    graph_kernel: str = "gaussian",
    density_correction: float = 0.5,
    state_codebook_size: int = 64,
    state_codebook_sample_size: int = 50000,
    feature_pca_dim: int = 128,
    descriptor_blocks: str | tuple[str, ...] | list[str] | None = None,
    radial_shells: int = 3,
    pair_mode: str = "anchor_neighbor",
    pair_top_states: int = 16,
    covariance_dims: int = 8,
    embedding_method: str = "descriptor_pca",
    embedding_dim: int = 64,
    encoder: str = "descriptor",
    max_neighbors_per_radius: int | None = None,
    token_dim: int = 128,
    hidden_dim: int = 256,
    epochs: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    use_ot_prototypes: bool = False,
    n_ot_prototypes: int = 20,
    prototype_support_size: int = 32,
    ot_epsilon: float = 0.05,
    ot_temperature: float = 0.25,
    ot_distance_feature_weight: float = 1.0,
    context_reconstruction_weight: float = 1.0,
    ot_prototype_weight: float = 0.5,
    prototype_balance_weight: float = 0.05,
    variance_weight: float = 0.02,
    decorrelation_weight: float = 0.005,
    cluster_method: str = "kmeans",
    n_clusters: int | None = None,
    resolution: float = 1.0,
    candidate_resolutions: str | tuple[float, ...] | list[float] | None = None,
    embedding_neighbors: int = 15,
    self_weight: float = 0.25,
    composition_weight: float = 0.25,
    diversity_weight: float = 0.25,
    moments_weight: float = 0.15,
    radial_weight: float = 0.15,
    pair_weight: float = 0.15,
    covariance_weight: float = 0.15,
    gradient_weight: float = 0.10,
    allow_umap_as_feature: bool = False,
    run_null_checks: bool = False,
    run_ablation_report: bool = False,
    seed: int = 1337,
) -> dict[str, object]:
    """Run descriptor-first cell-centered spatial heterogeneity niche discovery."""

    input_path = Path(input_h5ad)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(input_path)
    if spatial_x_key not in adata.obs or spatial_y_key not in adata.obs:
        raise KeyError(
            f"Spatial keys '{spatial_x_key}' and/or '{spatial_y_key}' were not found in obs."
        )

    features, feature_source = resolve_h5ad_features(
        adata,
        feature_obsm_key=feature_obsm_key,
        allow_umap_as_feature=bool(allow_umap_as_feature),
    )
    coords_um = np.column_stack(
        [
            np.asarray(adata.obs[spatial_x_key], dtype=np.float32) * float(spatial_scale),
            np.asarray(adata.obs[spatial_y_key], dtype=np.float32) * float(spatial_scale),
        ]
    ).astype(np.float32)
    sample_ids = (
        adata.obs[str(sample_obs_key)].astype(str).to_numpy()
        if sample_obs_key and str(sample_obs_key) in adata.obs
        else np.full(int(adata.n_obs), "sample_0", dtype=object)
    )

    feature_space = prepare_feature_space(
        features,
        n_components=int(feature_pca_dim),
        random_state=int(seed),
    )
    z = feature_space.values
    codebook = fit_state_codebook(
        z,
        n_codewords=int(state_codebook_size),
        sample_size=int(state_codebook_sample_size),
        random_state=int(seed),
    )

    radius_values = _parse_float_tuple(radii_um)
    k_values = _parse_int_tuple(knn_values)
    graphs: dict[str, NeighborhoodGraph] = {}
    if radius_values:
        graphs.update(
            build_radius_graphs(
                coords_um,
                sample_ids,
                radius_values,
                max_neighbors=int(max_neighbors),
                kernel=str(graph_kernel),
                density_correction=float(density_correction),
                include_self=False,
            )
        )
    if k_values:
        graphs.update(
            build_knn_graphs(
                coords_um,
                sample_ids,
                k_values,
                max_neighbors=int(max_neighbors),
                kernel=str(graph_kernel),
                density_correction=float(density_correction),
                include_self=False,
            )
        )
    if not graphs:
        raise ValueError("At least one radius or kNN graph must be requested.")

    block_weights = _block_weights_from_args(
        self_weight=self_weight,
        composition_weight=composition_weight,
        diversity_weight=diversity_weight,
        moments_weight=moments_weight,
        radial_weight=radial_weight,
        pair_weight=pair_weight,
        covariance_weight=covariance_weight,
        gradient_weight=gradient_weight,
    )
    descriptor = compute_cell_heterogeneity_descriptors(
        features=z,
        posteriors=codebook.posteriors,
        graphs=graphs,
        coords=coords_um,
        blocks=descriptor_blocks,
        block_weights=block_weights,
        self_weight=float(self_weight),
        radial_shells=int(radial_shells),
        pair_mode=pair_mode,
        pair_top_states=int(pair_top_states),
        covariance_dims=int(covariance_dims),
    )
    requested_encoder = str(encoder or "descriptor").strip().lower()
    if requested_encoder not in {"descriptor", "deepsets", "attention_deepsets", "ot_deepshe"}:
        raise ValueError("encoder must be descriptor, deepsets, attention_deepsets, or ot_deepshe.")
    if requested_encoder == "descriptor" and bool(use_ot_prototypes):
        raise ValueError("--use-ot-prototypes requires a DeepSHE encoder, not descriptor mode.")
    deepshe_result = None
    model_path: Path | None = None
    prototype_distances = None
    prototype_posterior = None
    if requested_encoder == "descriptor":
        embedding = reduce_descriptor_embedding(
            descriptor.standardized,
            n_components=int(embedding_dim),
            method=str(embedding_method),
            random_state=int(seed),
        )
        cluster_features = embedding.values
        niche_source = f"descriptor_{cluster_method}"
    else:
        resolved_max_neighbors_per_radius = max(
            int(
                max_neighbors_per_radius
                if max_neighbors_per_radius is not None
                else min(int(max_neighbors), 64)
            ),
            1,
        )
        deep_config = CellNicheConfig(
            feature_obsm_key=str(feature_obsm_key),
            spatial_x_key=str(spatial_x_key),
            spatial_y_key=str(spatial_y_key),
            sample_obs_key=str(sample_obs_key) if sample_obs_key else "sample_id",
            radii_um=tuple(float(value) for value in radius_values),
            spatial_scale=float(spatial_scale),
            max_neighbors_per_radius=resolved_max_neighbors_per_radius,
            kernel=str(graph_kernel),
            molecular_dim=int(feature_pca_dim),
            state_codebook_size=int(state_codebook_size),
            radial_shells=int(radial_shells),
            descriptor_blocks=tuple(descriptor.metadata.get("requested_blocks", [])),
            covariance_dims=int(covariance_dims),
            encoder=requested_encoder,  # type: ignore[arg-type]
            embedding_dim=int(embedding_dim),
            token_dim=int(token_dim),
            hidden_dim=int(hidden_dim),
            use_ot_prototypes=bool(use_ot_prototypes or requested_encoder == "ot_deepshe"),
            n_ot_prototypes=int(n_ot_prototypes),
            prototype_support_size=int(prototype_support_size),
            ot_epsilon=float(ot_epsilon),
            ot_temperature=float(ot_temperature),
            ot_distance_feature_weight=float(ot_distance_feature_weight),
            batch_size=int(batch_size),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            device=str(device),
            context_reconstruction_weight=float(context_reconstruction_weight),
            ot_prototype_weight=float(ot_prototype_weight),
            prototype_balance_weight=float(prototype_balance_weight),
            variance_weight=float(variance_weight),
            decorrelation_weight=float(decorrelation_weight),
            cluster_method=str(cluster_method),  # type: ignore[arg-type]
            n_clusters=n_clusters,
            leiden_resolution=float(resolution),
            seed=int(seed),
        )
        deepshe_result = fit_deepshe_embedding(
            features=z,
            posteriors=codebook.posteriors,
            coords_um=coords_um,
            graphs=graphs,
            descriptor_targets=descriptor.standardized,
            config=deep_config,
        )
        embedding = EmbeddingResult(
            values=deepshe_result.embedding,
            metadata={
                "method": str(requested_encoder),
                "reduction": "mini_batch_deepshe_encoder",
                "requested_dim": int(embedding_dim),
                "embedding_dim": int(deepshe_result.embedding.shape[1]),
                "training": dict(deepshe_result.metadata),
            },
        )
        prototype_distances = deepshe_result.prototype_distances
        prototype_posterior = deepshe_result.prototype_posterior
        if prototype_distances is not None:
            cluster_features = np.hstack(
                [
                    embedding.values.astype(np.float32),
                    float(ot_distance_feature_weight)
                    * _standardize_for_feature_concat(prototype_distances),
                ]
            ).astype(np.float32)
            niche_source = f"deep_plus_ot_{cluster_method}"
        else:
            cluster_features = embedding.values
            niche_source = f"deep_embedding_{cluster_method}"
        model_path = out_dir / "cell_niche_model.pt"
        try:
            import torch

            torch.save(
                {
                    "state_dict": deepshe_result.model.state_dict()
                    if deepshe_result.model is not None
                    else None,
                    "config": deep_config.to_dict(),
                    "feature_space": dict(feature_space.metadata),
                    "codebook_centers": codebook.centers.astype(np.float32),
                    "embedding_metadata": dict(embedding.metadata),
                },
                model_path,
            )
        except Exception as exc:  # pragma: no cover - defensive serialization guard
            model_path = None
            embedding.metadata["model_save_error"] = str(exc)
    cluster = cluster_embeddings(
        cluster_features,
        method=str(cluster_method),
        n_clusters=n_clusters,
        resolution=float(resolution),
        candidate_resolutions=_parse_float_tuple(candidate_resolutions),
        n_neighbors=int(embedding_neighbors),
        random_state=int(seed),
    )
    instance_graph = _choose_instance_graph(graphs)
    instance_ids, instance_names = connected_components_by_label(
        instance_graph,
        cluster.labels,
    )

    graph_keys = _store_graphs(adata, graphs)
    niche_names = np.asarray([f"N{int(label)}" for label in cluster.labels], dtype=object)
    adata.obsm["X_spatial_heterogeneity_descriptor"] = descriptor.standardized.astype(
        np.float32
    )
    adata.obsm["X_spatial_heterogeneity"] = embedding.values.astype(np.float32)
    adata.obsm["X_spatial_heterogeneity_feature_space"] = z.astype(np.float32)
    adata.obsm["X_spatial_heterogeneity_state_posteriors"] = (
        codebook.posteriors.astype(np.float32)
    )
    if requested_encoder != "descriptor":
        adata.obsm["X_spatial_ot_deepshe"] = embedding.values.astype(np.float32)
        adata.obsm["X_spatial_ot_deepshe_cluster_features"] = cluster_features.astype(
            np.float32
        )
    if prototype_distances is not None:
        adata.obsm["X_spatial_ot_prototype_distances"] = prototype_distances.astype(
            np.float32
        )
    if prototype_posterior is not None:
        adata.obsm["X_spatial_ot_prototype_posterior"] = prototype_posterior.astype(
            np.float32
        )
    adata.obs["spatial_niche"] = pd.Categorical(niche_names)
    adata.obs["spatial_niche_int"] = cluster.labels.astype(np.int32)
    adata.obs["spatial_niche_confidence"] = cluster.confidence.astype(np.float32)
    adata.obs["spatial_niche_source"] = pd.Categorical(
        np.full(int(adata.n_obs), str(niche_source), dtype=object)
    )
    adata.obs["spatial_niche_instance"] = pd.Categorical(instance_names)
    adata.obs["spatial_niche_instance_int"] = instance_ids.astype(np.int32)
    for graph_key, graph in graphs.items():
        suffix = str(graph_key).replace("-", "_")
        adata.obs[f"local_density_{suffix}"] = np.diff(graph.connectivities.indptr).astype(
            np.float32
        )

    h5ad_path = out_dir / "cells_cell_niche.h5ad"
    descriptor_npz_path = out_dir / "cell_niche_descriptor_arrays.npz"
    summary_path = out_dir / "summary.json"
    spatial_plot_path = out_dir / "cell_niche_spatial_map.png"
    embedding_plot_path = out_dir / "cell_niche_embedding.png"
    arrays = {
        "descriptor": descriptor.standardized.astype(np.float32),
        "embedding": embedding.values.astype(np.float32),
        "cluster_features": cluster_features.astype(np.float32),
        "state_posteriors": codebook.posteriors.astype(np.float32),
        "labels": cluster.labels.astype(np.int32),
        "confidence": cluster.confidence.astype(np.float32),
        "instance_ids": instance_ids.astype(np.int32),
    }
    if prototype_distances is not None:
        arrays["prototype_distances"] = prototype_distances.astype(np.float32)
    if prototype_posterior is not None:
        arrays["prototype_posterior"] = prototype_posterior.astype(np.float32)
    np.savez_compressed(descriptor_npz_path, **arrays)

    summary: dict[str, object] = {
        "method_family": "cell_centered_spatial_heterogeneity_embedding",
        "active_path": "cell-niche",
        "primary_unit": "cell",
        "subregion_first_pipeline_role": "baseline_and_diagnostic_layer_only",
        "input_h5ad": str(input_path),
        "output_dir": str(out_dir),
        "n_cells": int(adata.n_obs),
        "feature_obsm_key": str(feature_source.get("feature_key", feature_obsm_key)),
        "feature_obsm_key_requested": str(feature_obsm_key),
        "feature_source": dict(feature_source),
        "feature_space": dict(feature_space.metadata),
        "spatial_x_key": str(spatial_x_key),
        "spatial_y_key": str(spatial_y_key),
        "sample_obs_key": str(sample_obs_key) if sample_obs_key else None,
        "spatial_scale": float(spatial_scale),
        "radii_um": [float(value) for value in radius_values],
        "knn_values": [int(value) for value in k_values],
        "max_neighbors": int(max_neighbors),
        "graph_kernel": str(graph_kernel),
        "density_correction": float(density_correction),
        "graph_obsp_keys": graph_keys,
        "graphs": {key: dict(graph.metadata) for key, graph in graphs.items()},
        "state_codebook": dict(codebook.metadata),
        "descriptor": dict(descriptor.metadata),
        "embedding": dict(embedding.metadata),
        "encoder": str(requested_encoder),
        "niche_source": str(niche_source),
        "deep_training": None
        if deepshe_result is None
        else {
            "metadata": dict(deepshe_result.metadata),
            "history": list(deepshe_result.history),
        },
        "ot_prototypes": None
        if prototype_distances is None
        else {
            "n_prototypes": int(prototype_distances.shape[1]),
            "distance_feature_weight": float(ot_distance_feature_weight),
            "mean_min_distance": float(np.mean(np.min(prototype_distances, axis=1))),
            "mean_assignment_entropy": float(
                np.mean(
                    -np.sum(
                        prototype_posterior.astype(np.float64)
                        * np.log(np.maximum(prototype_posterior, 1e-12)),
                        axis=1,
                    )
                    / max(float(np.log(max(prototype_posterior.shape[1], 2))), 1e-8)
                )
            )
            if prototype_posterior is not None
            else None,
        },
        "clustering": dict(cluster.metadata),
        "n_niches": int(np.unique(cluster.labels).size),
        "niche_counts": _cluster_counts(cluster.labels),
        "n_niche_instances": int(np.unique(instance_ids).size),
        "instance_graph_key": str(instance_graph.key),
        "self_weight": float(self_weight),
        "block_weights": block_weights,
        "run_null_checks_requested": bool(run_null_checks),
        "run_ablation_report_requested": bool(run_ablation_report),
        "validation_status": {
            "descriptor_mvp_implemented": True,
            "null_checks_implemented": False,
            "ablation_report_implemented": False,
            "learned_deepsets_encoder_implemented": requested_encoder != "descriptor",
            "ot_prototype_diagnostics_implemented": prototype_distances is not None,
            "ot_distillation_implemented": False,
            "transform_mode_implemented": False,
        },
        "method_layers": {
            "layer_1_cell_centered_context_measures": (
                "Each cell is represented by local same-sample neighborhoods over one or more radii/kNN scales."
            ),
            "layer_2_cell_heterogeneity_embedding": (
                "Composition, diversity, moments, radial organization, pair texture, COVET-like covariance, "
                "gradient/anisotropy, and optional intrinsic cell-state blocks are standardized and embedded at cell level. "
                "When a deep encoder is selected, the model reconstructs this descriptor target from mini-batch local measures."
            ),
            "layer_3_cell_niche_labels_and_instances": (
                "Global spatial_niche labels are reusable microenvironment motifs; connected components "
                "within the neighborhood graph are saved as spatial_niche_instance."
            ),
        },
        "method_notes": {
            "old_vs_new": (
                "Old spatial_ot: cells -> subregions -> subregion clusters -> projected cell labels. "
                "This path: cells -> local cell-centered context measures -> cell embeddings -> cell niche clusters."
            ),
            "self_weight_ablation": (
                "Set --self-weight 0 to test whether labels remain context-driven rather than disguised cell-state clusters."
            ),
            "coordinates": (
                "Raw absolute coordinates are used only to build same-sample neighborhood graphs and connected instances; "
                "they are not appended as clustering features."
            ),
            "ot_role": (
                "When enabled, balanced Sinkhorn distances to learned local-measure prototypes are used as cluster features. "
                "All-pairs cell-neighborhood OT is intentionally avoided."
            ),
        },
        "outputs": {
            "h5ad": str(h5ad_path),
            "descriptor_arrays": str(descriptor_npz_path),
            "summary": str(summary_path),
            "spatial_map": str(spatial_plot_path),
            "embedding_plot": str(embedding_plot_path),
            "model": None if model_path is None else str(model_path),
        },
        "seed": int(seed),
    }
    adata.uns["spatial_heterogeneity_config"] = {
        "radii_um": [float(value) for value in radius_values],
        "knn_values": [int(value) for value in k_values],
        "state_codebook_size": int(state_codebook_size),
        "descriptor_blocks": descriptor.metadata.get("requested_blocks", []),
        "covariance_dims": int(covariance_dims),
        "encoder": str(requested_encoder),
        "embedding_method": str(embedding_method),
        "cluster_method": str(cluster_method),
        "self_weight": float(self_weight),
        "block_weights": block_weights,
        "use_ot_prototypes": bool(prototype_distances is not None),
    }
    adata.uns["spatial_niche_summary"] = _sanitize_for_anndata(summary)

    _write_spatial_plot(
        coords=coords_um,
        labels=cluster.labels,
        output_path=spatial_plot_path,
        title="Cell-Centered Spatial Niche Labels",
    )
    _write_embedding_plot(
        embedding=embedding.values,
        labels=cluster.labels,
        output_path=embedding_plot_path,
    )
    adata.write_h5ad(h5ad_path, compression="gzip")
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    return summary


__all__ = ["run_cell_niche_on_h5ad"]
