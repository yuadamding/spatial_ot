from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ..feature_source import resolve_h5ad_features
from .cluster import (
    DEFAULT_CANDIDATE_N_CLUSTERS,
    cluster_from_distance,
    connected_components_by_label,
    ot_knn_affinity,
)
from .colors import assign_high_contrast_colors
from .config import PairwiseNicheConfig
from .distance_matrix import compute_pairwise_ot_distance_matrix
from .expression_embedding import fit_expression_embedding, save_expression_embedding_state
from .local_measure import build_instance_neighbor_indices, build_local_measures


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


def _sanitize(value):
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, dict) for item in value):
            return {str(idx): _sanitize(item) for idx, item in enumerate(value)}
        return [_sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return "none"
    return value


def _cluster_counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels, dtype=np.int32), return_counts=True)
    return {f"ON{int(label)}": int(count) for label, count in zip(values, counts, strict=False)}


def _distance_output_path(
    *,
    output_dir: Path,
    n_cells: int,
    distance_store: str,
) -> Path | None:
    requested = str(distance_store or "auto").strip().lower()
    if requested == "h5ad":
        return None
    if requested == "npy_memmap":
        return output_dir / "cell_ot_dissimilarity.npy"
    if requested == "auto":
        return None if int(n_cells) <= 5000 else output_dir / "cell_ot_dissimilarity.npy"
    raise ValueError("distance_store must be auto, h5ad, or npy_memmap.")


def _batch_embedding_note(
    *,
    embedding_method: str,
    expression_batch_key: str | None,
    sample_count: int,
) -> dict[str, object]:
    method = str(embedding_method or "").strip().lower()
    active_correction = False
    note = None
    if expression_batch_key and sample_count > 1 and method in {"pca", "svd"}:
        note = (
            "expression_batch_key is recorded for provenance only; PCA/SVD embedding "
            "does not apply batch correction. Use a precomputed batch-corrected latent "
            "space for corrected expression geometry."
        )
    return {
        "expression_batch_key": expression_batch_key,
        "batch_correction_applied_by_pairwise_niche": active_correction,
        "note": note,
    }


def run_pairwise_niche_on_h5ad(
    *,
    input_h5ad: str | Path,
    output_dir: str | Path,
    feature_obsm_key: str,
    spatial_x_key: str,
    spatial_y_key: str,
    sample_obs_key: str = "sample_id",
    spatial_scale: float = 1.0,
    embedding_method: str = "pca",
    embedding_dim: int = 32,
    expression_batch_key: str | None = None,
    standardize_precomputed: bool = True,
    allow_umap_as_feature: bool = False,
    radius_um: float = 50.0,
    max_neighbors: int = 32,
    include_anchor: bool = True,
    isolated_policy: str = "zero_dummy",
    graph_kernel: str = "gaussian",
    cap_mode: str = "radial_shell_state",
    cap_state_clusters: int = 16,
    radial_shells: int = 3,
    expression_weight: float = 1.0,
    spatial_weight: float = 0.25,
    distance_weight: float = 0.10,
    ground_cost_normalization: str = "sampled_median",
    ground_cost_sample_pairs: int = 10000,
    anchor_weight: float = 0.25,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 50,
    distance_mode: str = "debiased_entropic_transport",
    fgw_alpha: float = 0.5,
    fgw_iters: int = 5,
    pairwise_mode: str = "exact_blockwise",
    block_size: int = 64,
    device: str = "auto",
    max_exact_cells: int = 5000,
    max_ot_work_units: float = 5e11,
    force_large_exact_ot: bool = False,
    distance_store: str = "auto",
    cluster_method: str = "agglomerative",
    n_clusters: int | None = None,
    candidate_n_clusters: tuple[int, ...] | list[int] | None = None,
    model_selection_metrics: tuple[str, ...] | list[str] | None = None,
    ot_knn: int = 30,
    ot_affinity_scaling: str = "local",
    leiden_resolution: float = 1.0,
    candidate_resolutions: tuple[float, ...] | list[float] | None = None,
    instance_radius_um: float | None = None,
    instance_max_neighbors: int = 512,
    seed: int = 1337,
) -> dict[str, object]:
    """Run pairwise OT neighborhood dissimilarity clustering on a cell-level H5AD."""

    requested_pairwise = str(pairwise_mode or "exact_blockwise").strip().lower()
    if requested_pairwise not in {"exact", "exact_blockwise"}:
        raise ValueError("pairwise_mode currently supports exact or exact_blockwise.")
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
    sample_ids = (
        adata.obs[str(sample_obs_key)].astype(str).to_numpy()
        if sample_obs_key and str(sample_obs_key) in adata.obs
        else np.full(int(adata.n_obs), "sample_0", dtype=object)
    )
    coords_um = np.column_stack(
        [
            np.asarray(adata.obs[spatial_x_key], dtype=np.float32) * float(spatial_scale),
            np.asarray(adata.obs[spatial_y_key], dtype=np.float32) * float(spatial_scale),
        ]
    ).astype(np.float32)
    requested_cluster_method = str(cluster_method or "agglomerative").strip().lower()
    effective_candidate_n_clusters = candidate_n_clusters
    if (
        requested_cluster_method in {"agglomerative", "kmedoids", "pam"}
        and n_clusters is None
        and candidate_n_clusters is None
    ):
        effective_candidate_n_clusters = DEFAULT_CANDIDATE_N_CLUSTERS

    config = PairwiseNicheConfig(
        feature_obsm_key=str(feature_obsm_key),
        spatial_x_key=str(spatial_x_key),
        spatial_y_key=str(spatial_y_key),
        sample_obs_key=str(sample_obs_key),
        spatial_scale=float(spatial_scale),
        embedding_method=str(embedding_method),  # type: ignore[arg-type]
        embedding_dim=int(embedding_dim),
        expression_batch_key=expression_batch_key,
        standardize_precomputed=bool(standardize_precomputed),
        allow_umap_as_feature=bool(allow_umap_as_feature),
        radius_um=float(radius_um),
        max_neighbors=int(max_neighbors),
        include_anchor=bool(include_anchor),
        isolated_policy=str(isolated_policy),  # type: ignore[arg-type]
        graph_kernel=str(graph_kernel),  # type: ignore[arg-type]
        cap_mode=str(cap_mode),  # type: ignore[arg-type]
        cap_state_clusters=int(cap_state_clusters),
        radial_shells=int(radial_shells),
        expression_weight=float(expression_weight),
        spatial_weight=float(spatial_weight),
        distance_weight=float(distance_weight),
        ground_cost_normalization=str(ground_cost_normalization),  # type: ignore[arg-type]
        ground_cost_sample_pairs=int(ground_cost_sample_pairs),
        anchor_weight=float(anchor_weight),
        sinkhorn_epsilon=float(sinkhorn_epsilon),
        sinkhorn_iters=int(sinkhorn_iters),
        distance_mode=str(distance_mode),  # type: ignore[arg-type]
        fgw_alpha=float(fgw_alpha),
        fgw_iters=int(fgw_iters),
        pairwise_mode=requested_pairwise,  # type: ignore[arg-type]
        block_size=int(block_size),
        device=str(device),
        max_exact_cells=int(max_exact_cells),
        max_ot_work_units=float(max_ot_work_units),
        force_large_exact_ot=bool(force_large_exact_ot),
        distance_store=str(distance_store),  # type: ignore[arg-type]
        cluster_method=requested_cluster_method,  # type: ignore[arg-type]
        n_clusters=n_clusters,
        candidate_n_clusters=tuple(int(value) for value in effective_candidate_n_clusters)
        if effective_candidate_n_clusters
        else None,
        model_selection_metrics=tuple(str(value) for value in model_selection_metrics)
        if model_selection_metrics
        else ("silhouette", "calinski_harabasz", "davies_bouldin", "dunn"),
        ot_knn=int(ot_knn),
        ot_affinity_scaling=str(ot_affinity_scaling),  # type: ignore[arg-type]
        leiden_resolution=float(leiden_resolution),
        candidate_resolutions=tuple(float(value) for value in candidate_resolutions)
        if candidate_resolutions
        else None,
        instance_radius_um=instance_radius_um,
        instance_max_neighbors=int(instance_max_neighbors),
        seed=int(seed),
    )

    embedding = fit_expression_embedding(
        features,
        method=str(embedding_method),
        embedding_dim=int(embedding_dim),
        standardize_precomputed=bool(standardize_precomputed),
        random_state=int(seed),
    )
    measures = build_local_measures(
        expression_embedding=embedding.values,
        coords_um=coords_um,
        sample_ids=sample_ids,
        radius_um=float(radius_um),
        max_neighbors=int(max_neighbors),
        include_anchor=bool(include_anchor),
        isolated_policy=str(isolated_policy),
        graph_kernel=str(graph_kernel),
        cap_mode=str(cap_mode),
        cap_state_clusters=int(cap_state_clusters),
        radial_shells=int(radial_shells),
        expression_weight=float(expression_weight),
        spatial_weight=float(spatial_weight),
        distance_weight=float(distance_weight),
        ground_cost_normalization=str(ground_cost_normalization),
        ground_cost_sample_pairs=int(ground_cost_sample_pairs),
        seed=int(seed),
    )
    distance_path = _distance_output_path(
        output_dir=out_dir,
        n_cells=int(adata.n_obs),
        distance_store=str(distance_store),
    )
    distance, distance_metadata = compute_pairwise_ot_distance_matrix(
        measures=measures,
        anchor_embedding=embedding.values,
        output_path=distance_path,
        block_size=int(block_size),
        device=str(device),
        epsilon=float(sinkhorn_epsilon),
        n_iters=int(sinkhorn_iters),
        distance_mode=str(distance_mode),
        anchor_weight=float(anchor_weight),
        fgw_alpha=float(fgw_alpha),
        fgw_iters=int(fgw_iters),
        max_exact_cells=int(max_exact_cells),
        max_ot_work_units=float(max_ot_work_units),
        force_large_exact_ot=bool(force_large_exact_ot),
    )
    distance_array = np.asarray(distance, dtype=np.float32)
    cluster = cluster_from_distance(
        distance_array,
        method=requested_cluster_method,
        n_clusters=n_clusters,
        candidate_n_clusters=effective_candidate_n_clusters,
        model_selection_metrics=model_selection_metrics,
        ot_knn=int(ot_knn),
        ot_affinity_scaling=str(ot_affinity_scaling),
        leiden_resolution=float(leiden_resolution),
        candidate_resolutions=candidate_resolutions,
        random_state=int(seed),
    )
    instance_indices = build_instance_neighbor_indices(
        coords_um=coords_um,
        sample_ids=sample_ids,
        radius_um=float(instance_radius_um) if instance_radius_um is not None else float(radius_um),
        max_neighbors=int(instance_max_neighbors),
    )
    instance_ids, instance_names = connected_components_by_label(
        instance_indices,
        cluster.labels,
    )

    adata.obsm["X_gene_cohort"] = embedding.values.astype(np.float32)
    niche_categories = [f"ON{int(label)}" for label in sorted(np.unique(cluster.labels))]
    niche_names = np.asarray([f"ON{int(label)}" for label in cluster.labels], dtype=object)
    niche_colors = assign_high_contrast_colors(niche_categories)
    adata.obs["ot_niche"] = pd.Categorical(niche_names, categories=niche_categories)
    adata.obs["ot_niche_int"] = cluster.labels.astype(np.int32)
    adata.obs["ot_niche_assignment_score"] = cluster.assignment_score.astype(np.float32)
    adata.obs["ot_niche_instance"] = pd.Categorical(instance_names)
    adata.obs["ot_niche_instance_int"] = instance_ids.astype(np.int32)
    suffix = f"r{float(radius_um):g}".replace(".", "p")
    adata.obs[f"n_neighbors_full_{suffix}"] = measures.full_neighbor_counts.astype(np.int32)
    adata.obs[f"n_neighbors_retained_{suffix}"] = measures.retained_neighbor_counts.astype(np.int32)
    adata.obs[f"neighbor_retention_fraction_{suffix}"] = (
        measures.retained_neighbor_counts.astype(np.float32)
        / np.maximum(measures.full_neighbor_counts.astype(np.float32), 1.0)
    )
    density_area = np.pi * float(radius_um) ** 2
    adata.obs[f"local_density_full_per_um2_{suffix}"] = (
        measures.full_neighbor_counts.astype(np.float32) / max(density_area, 1e-12)
    )
    adata.obs[f"local_density_retained_per_um2_{suffix}"] = (
        measures.retained_neighbor_counts.astype(np.float32) / max(density_area, 1e-12)
    )
    affinity = ot_knn_affinity(
        distance_array,
        k=int(ot_knn),
        scaling=str(ot_affinity_scaling),
    )
    adata.obsp["cell_ot_affinity"] = affinity
    if distance_path is None:
        adata.obsp["cell_ot_dissimilarity"] = distance_array
        distance_store_path = None
    else:
        distance_store_path = str(distance_path)
        adata.uns["cell_ot_dissimilarity_store"] = distance_store_path

    model_dir = out_dir / "pairwise_niche_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    embedding_state_path = model_dir / "expression_embedding_state.npz"
    save_expression_embedding_state(embedding.state, embedding_state_path)

    sample_count = len(set(sample_ids.tolist()))
    batch_note = _batch_embedding_note(
        embedding_method=str(embedding_method),
        expression_batch_key=expression_batch_key,
        sample_count=sample_count,
    )
    anchor_enters_twice = bool(include_anchor) and float(anchor_weight) > 0.0
    summary = {
        "method_family": "pairwise_ot_cell_neighborhood_dissimilarity",
        "active_path": "pairwise-niche",
        "primary_unit": "cell",
        "input_h5ad": str(input_path),
        "output_dir": str(out_dir),
        "n_cells": int(adata.n_obs),
        "feature_source": dict(feature_source),
        "config": config.to_dict(),
        "expression_embedding": dict(embedding.metadata),
        "expression_embedding_state": str(embedding_state_path),
        "batch_embedding": batch_note,
        "local_measure": dict(measures.metadata),
        "distance_matrix": dict(distance_metadata),
        "clustering": dict(cluster.metadata),
        "niche_colors": dict(niche_colors),
        "method_semantics": {
            "anchor_in_measure": bool(include_anchor),
            "direct_anchor_cost_weight": float(anchor_weight),
            "anchor_expression_enters_twice": anchor_enters_twice,
            "instance_graph_source": "sample_isolated_radius_graph",
            "instance_radius_um": float(instance_radius_um)
            if instance_radius_um is not None
            else float(radius_um),
            "instance_max_neighbors": int(instance_max_neighbors),
        },
        "n_niches": int(np.unique(cluster.labels).size),
        "niche_counts": _cluster_counts(cluster.labels),
        "n_niche_instances": int(np.unique(instance_ids).size),
        "outputs": {
            "h5ad": str(out_dir / "cells_pairwise_niche.h5ad"),
            "summary": str(out_dir / "summary.json"),
            "niche_colors": str(out_dir / "ot_niche_colors.json"),
            "distance_matrix": distance_store_path,
            "expression_embedding_state": str(embedding_state_path),
        },
        "seed": int(seed),
    }
    adata.uns["pairwise_niche_config"] = _sanitize(config.to_dict())
    adata.uns["pairwise_niche_embedding_summary"] = _sanitize(embedding.metadata)
    adata.uns["pairwise_niche_distance_summary"] = _sanitize(distance_metadata)
    adata.uns["pairwise_niche_clustering_summary"] = _sanitize(cluster.metadata)
    adata.uns["pairwise_niche_color_map"] = _sanitize(niche_colors)
    adata.uns["ot_niche_colors"] = [
        niche_colors[str(category)] for category in niche_categories
    ]

    h5ad_path = out_dir / "cells_pairwise_niche.h5ad"
    summary_path = out_dir / "summary.json"
    color_path = out_dir / "ot_niche_colors.json"
    adata.write_h5ad(h5ad_path, compression="gzip")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    color_path.write_text(json.dumps(niche_colors, indent=2, sort_keys=True))
    return summary
