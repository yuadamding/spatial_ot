from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ..feature_source import resolve_h5ad_features
from .cluster import cluster_from_distance, connected_components_by_label, ot_knn_affinity
from .config import PairwiseNicheConfig
from .distance_matrix import compute_pairwise_ot_distance_matrix
from .expression_embedding import fit_expression_embedding
from .local_measure import build_local_measures


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
    radius_um: float = 50.0,
    max_neighbors: int = 32,
    include_anchor: bool = True,
    graph_kernel: str = "gaussian",
    cap_mode: str = "radial_shell_state",
    cap_state_clusters: int = 16,
    radial_shells: int = 3,
    expression_weight: float = 1.0,
    spatial_weight: float = 0.25,
    distance_weight: float = 0.10,
    anchor_weight: float = 0.25,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 50,
    distance_mode: str = "sinkhorn_divergence",
    pairwise_mode: str = "exact_blockwise",
    block_size: int = 64,
    device: str = "auto",
    max_exact_cells: int = 5000,
    distance_store: str = "auto",
    cluster_method: str = "agglomerative",
    n_clusters: int | None = None,
    ot_knn: int = 30,
    leiden_resolution: float = 1.0,
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
        allow_umap_as_feature=False,
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

    config = PairwiseNicheConfig(
        feature_obsm_key=str(feature_obsm_key),
        spatial_x_key=str(spatial_x_key),
        spatial_y_key=str(spatial_y_key),
        sample_obs_key=str(sample_obs_key),
        spatial_scale=float(spatial_scale),
        embedding_method=str(embedding_method),  # type: ignore[arg-type]
        embedding_dim=int(embedding_dim),
        expression_batch_key=expression_batch_key,
        radius_um=float(radius_um),
        max_neighbors=int(max_neighbors),
        include_anchor=bool(include_anchor),
        graph_kernel=str(graph_kernel),  # type: ignore[arg-type]
        cap_mode=str(cap_mode),  # type: ignore[arg-type]
        cap_state_clusters=int(cap_state_clusters),
        radial_shells=int(radial_shells),
        expression_weight=float(expression_weight),
        spatial_weight=float(spatial_weight),
        distance_weight=float(distance_weight),
        anchor_weight=float(anchor_weight),
        sinkhorn_epsilon=float(sinkhorn_epsilon),
        sinkhorn_iters=int(sinkhorn_iters),
        distance_mode=str(distance_mode),  # type: ignore[arg-type]
        pairwise_mode=requested_pairwise,  # type: ignore[arg-type]
        block_size=int(block_size),
        device=str(device),
        max_exact_cells=int(max_exact_cells),
        distance_store=str(distance_store),  # type: ignore[arg-type]
        cluster_method=str(cluster_method),  # type: ignore[arg-type]
        n_clusters=n_clusters,
        ot_knn=int(ot_knn),
        leiden_resolution=float(leiden_resolution),
        seed=int(seed),
    )

    embedding = fit_expression_embedding(
        features,
        method=str(embedding_method),
        embedding_dim=int(embedding_dim),
        random_state=int(seed),
    )
    measures = build_local_measures(
        expression_embedding=embedding.values,
        coords_um=coords_um,
        sample_ids=sample_ids,
        radius_um=float(radius_um),
        max_neighbors=int(max_neighbors),
        include_anchor=bool(include_anchor),
        graph_kernel=str(graph_kernel),
        cap_mode=str(cap_mode),
        cap_state_clusters=int(cap_state_clusters),
        radial_shells=int(radial_shells),
        expression_weight=float(expression_weight),
        spatial_weight=float(spatial_weight),
        distance_weight=float(distance_weight),
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
        max_exact_cells=int(max_exact_cells),
    )
    distance_array = np.asarray(distance, dtype=np.float32)
    cluster = cluster_from_distance(
        distance_array,
        method=str(cluster_method),
        n_clusters=n_clusters,
        ot_knn=int(ot_knn),
        leiden_resolution=float(leiden_resolution),
        random_state=int(seed),
    )
    instance_ids, instance_names = connected_components_by_label(
        measures.neighbor_indices,
        cluster.labels,
    )

    adata.obsm["X_gene_cohort"] = embedding.values.astype(np.float32)
    adata.obs["ot_niche"] = pd.Categorical(
        np.asarray([f"ON{int(label)}" for label in cluster.labels], dtype=object)
    )
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
    affinity = ot_knn_affinity(distance_array, k=int(ot_knn))
    adata.obsp["cell_ot_affinity"] = affinity
    if distance_path is None:
        adata.obsp["cell_ot_dissimilarity"] = distance_array
        distance_store_path = None
    else:
        distance_store_path = str(distance_path)
        adata.uns["cell_ot_dissimilarity_store"] = distance_store_path

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
        "local_measure": dict(measures.metadata),
        "distance_matrix": dict(distance_metadata),
        "clustering": dict(cluster.metadata),
        "n_niches": int(np.unique(cluster.labels).size),
        "niche_counts": _cluster_counts(cluster.labels),
        "n_niche_instances": int(np.unique(instance_ids).size),
        "outputs": {
            "h5ad": str(out_dir / "cells_pairwise_niche.h5ad"),
            "summary": str(out_dir / "summary.json"),
            "distance_matrix": distance_store_path,
        },
        "seed": int(seed),
    }
    adata.uns["pairwise_niche_config"] = _sanitize(config.to_dict())
    adata.uns["pairwise_niche_embedding_summary"] = _sanitize(embedding.metadata)
    adata.uns["pairwise_niche_distance_summary"] = _sanitize(distance_metadata)
    adata.uns["pairwise_niche_clustering_summary"] = _sanitize(cluster.metadata)

    h5ad_path = out_dir / "cells_pairwise_niche.h5ad"
    summary_path = out_dir / "summary.json"
    adata.write_h5ad(h5ad_path, compression="gzip")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return summary
