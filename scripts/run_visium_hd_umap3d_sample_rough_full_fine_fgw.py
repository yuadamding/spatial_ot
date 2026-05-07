from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_visium_hd_umap3d_landmark_fgw import (  # noqa: E402
    _build_compact_local_measures,
    _compute_landmark_fgw,
    _connected_instances_sparse,
    _json_default,
    _load_per_sample_umap,
    _log,
    _normalize_fgw_structures,
    _plot_outputs,
)
from spatial_ot.pairwise_niche.cluster import cluster_from_distance  # noqa: E402
from spatial_ot.pairwise_niche.colors import assign_high_contrast_colors  # noqa: E402
from spatial_ot.pairwise_niche.expression_embedding import (  # noqa: E402
    fit_expression_embedding,
    save_expression_embedding_state,
)
from spatial_ot.pairwise_niche.io import _sanitize  # noqa: E402


def _zscore(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    center = np.mean(values, axis=0, keepdims=True)
    scale = np.std(values, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return ((values - center) / scale).astype(np.float32, copy=False)


def _rough_cluster_per_sample(
    *,
    embedding: np.ndarray,
    coords_um: np.ndarray,
    full_neighbor_counts: np.ndarray,
    sample_ids: np.ndarray,
    rough_clusters_per_sample: int,
    rough_batch_size: int | None,
    rough_spatial_weight: float,
    rough_density_weight: float,
    seed: int,
    log_path: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    z = np.asarray(embedding, dtype=np.float32)
    coords = np.asarray(coords_um, dtype=np.float32)
    counts = np.asarray(full_neighbor_counts, dtype=np.float32)
    samples = np.asarray(sample_ids, dtype=object).astype(str)
    rough_id = np.full(z.shape[0], -1, dtype=np.int32)
    representative_indices: list[int] = []
    rows: list[dict[str, object]] = []
    next_id = 0
    for sample_pos, sample in enumerate(list(dict.fromkeys(samples.tolist()))):
        idx = np.flatnonzero(samples == sample)
        if idx.size == 0:
            continue
        sample_z = _zscore(z[idx])
        sample_xy = _zscore(coords[idx]) * np.sqrt(max(float(rough_spatial_weight), 0.0))
        sample_density = _zscore(np.log1p(counts[idx])[:, None]) * np.sqrt(
            max(float(rough_density_weight), 0.0)
        )
        rough_features = np.hstack([sample_z, sample_xy, sample_density]).astype(
            np.float32,
            copy=False,
        )
        k = min(max(int(rough_clusters_per_sample), 1), int(idx.size))
        batch_size = min(max(4096, k * 64), int(idx.size))
        if rough_batch_size is not None and int(rough_batch_size) > 0:
            batch_size = min(int(rough_batch_size), int(idx.size))
        model = MiniBatchKMeans(
            n_clusters=k,
            batch_size=batch_size,
            n_init=3,
            random_state=int(seed) + sample_pos,
        )
        local_labels = np.asarray(model.fit_predict(rough_features), dtype=np.int32)
        centers = np.asarray(model.cluster_centers_, dtype=np.float32)
        sample_reps: list[int] = []
        empty_local_clusters: list[int] = []
        for local_cluster in range(k):
            members_local = np.flatnonzero(local_labels == int(local_cluster))
            if members_local.size == 0:
                empty_local_clusters.append(int(local_cluster))
                continue
            center = centers[int(local_cluster)]
            distances = np.sum((rough_features[members_local] - center) ** 2, axis=1)
            rep = int(idx[members_local[int(np.argmin(distances))]])
            global_cluster = next_id
            global_members = idx[members_local]
            rough_id[global_members] = int(global_cluster)
            representative_indices.append(rep)
            sample_reps.append(rep)
            rows.append(
                {
                    "rough_cluster_id": int(global_cluster),
                    "sample_id": str(sample),
                    "sample_rough_cluster_id": int(local_cluster),
                    "n_cells": int(global_members.size),
                    "representative_index": int(rep),
                    "filler_representative": False,
                }
            )
            next_id += 1
        if empty_local_clusters:
            rng = np.random.default_rng(int(seed) + sample_pos + 1000003)
            used = np.asarray(sample_reps, dtype=np.int64)
            candidate_pool = np.setdiff1d(idx, used, assume_unique=False)
            if candidate_pool.size == 0:
                candidate_pool = idx
            extra = rng.choice(
                candidate_pool,
                size=len(empty_local_clusters),
                replace=candidate_pool.size < len(empty_local_clusters),
            )
            for local_cluster, rep in zip(empty_local_clusters, extra, strict=False):
                global_cluster = next_id
                representative_indices.append(int(rep))
                sample_reps.append(int(rep))
                rows.append(
                    {
                        "rough_cluster_id": int(global_cluster),
                        "sample_id": str(sample),
                        "sample_rough_cluster_id": int(local_cluster),
                        "n_cells": 0,
                        "representative_index": int(rep),
                        "filler_representative": True,
                    }
                )
                next_id += 1
        _log(
            log_path,
            f"  {sample}: rough_clusters={len(sample_reps):,}, cells={idx.size:,}, "
            f"median_cells_per_rough={float(np.median([row['n_cells'] for row in rows if row['sample_id'] == sample])):.1f}, "
            f"filler_representatives={len(empty_local_clusters):,}",
        )
    if np.any(rough_id < 0):
        raise RuntimeError("Internal error: some cells were not assigned to a rough sample cluster.")
    table = pd.DataFrame(rows)
    return rough_id, np.asarray(representative_indices, dtype=np.int64), table


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_visium_hd_umap3d_sample_rough_full_fine_fgw.log"
    if log_path.exists():
        log_path.unlink()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    t0 = time.time()
    _log(
        log_path,
        json.dumps(
            {
                "event": "start",
                "input_h5ad": str(args.input_h5ad),
                "output_dir": str(out_dir),
                "feature_obsm_key": str(args.umap_key),
                "stage_1": "rough_per_sample",
                "stage_2": "fine_full_cohort_representative_fgw",
                "device": str(device),
                "radius_um": float(args.radius_um),
                "max_neighbors": int(args.max_neighbors),
                "rough_clusters_per_sample": int(args.rough_clusters_per_sample),
            },
            indent=2,
        ),
    )
    _log(log_path, "[1/9] Loading pooled obs and per-sample 3D UMAP...")
    adata, umap_raw = _load_per_sample_umap(
        pooled_h5ad=Path(args.input_h5ad),
        input_dir=Path(args.input_dir),
        umap_key=str(args.umap_key),
        source_obs_key=str(args.source_obs_key),
    )
    sample_ids = adata.obs[str(args.sample_obs_key)].astype(str).to_numpy()
    coords = np.column_stack(
        [
            np.asarray(adata.obs[str(args.spatial_x_key)], dtype=np.float32),
            np.asarray(adata.obs[str(args.spatial_y_key)], dtype=np.float32),
        ]
    )
    _log(log_path, f"Input cells={adata.n_obs:,}, raw_umap_shape={umap_raw.shape}")

    _log(log_path, "[2/9] Standardizing 3D UMAP as graph node features...")
    embedding = fit_expression_embedding(
        umap_raw,
        method="precomputed",
        standardize_precomputed=not bool(args.no_standardize_umap),
        random_state=int(args.seed),
    )
    z = embedding.values.astype(np.float32, copy=False)
    np.save(out_dir / "X_cell_umap_3d_raw.npy", umap_raw.astype(np.float32, copy=False))
    np.save(out_dir / "X_cell_umap_3d.npy", z)
    save_expression_embedding_state(embedding.state, out_dir / "expression_embedding_state_umap3d.npz")
    _log(log_path, json.dumps(embedding.metadata, indent=2, default=_json_default))

    _log(log_path, "[3/9] Building compact same-sample local cell graphs...")
    measures = _build_compact_local_measures(
        expression_embedding=z,
        coords_um=coords,
        sample_ids=sample_ids,
        radius_um=float(args.radius_um),
        max_neighbors=int(args.max_neighbors),
        include_anchor=True,
        graph_kernel="gaussian",
        cap_mode="radial_shell_state",
        cap_state_clusters=int(args.cap_state_clusters),
        radial_shells=int(args.radial_shells),
        isolated_policy="anchor_fallback",
        fgw_structure_mode=str(args.fgw_structure_mode),
        fgw_structure_knn=int(args.fgw_structure_knn),
        fgw_structure_radius_fraction=float(args.fgw_structure_radius_fraction),
        expression_weight=float(args.expression_weight),
        spatial_weight=float(args.spatial_weight),
        distance_weight=float(args.distance_weight),
        ground_cost_normalization=str(args.ground_cost_normalization),
        ground_cost_sample_pairs=int(args.ground_cost_sample_pairs),
        seed=int(args.seed),
    )
    _log(log_path, json.dumps(measures.metadata, indent=2, default=_json_default))
    feature_width = z.shape[1]
    features = measures.tokens[:, :, :feature_width].astype(np.float32, copy=False)
    structures, structure_norm = _normalize_fgw_structures(
        np.asarray(measures.structure_matrices, dtype=np.float32),
        measures.mask,
        normalization=str(args.fgw_structure_normalization),
        n_pairs=int(args.fgw_structure_sample_pairs),
        seed=int(args.seed),
    )
    weights = measures.weights.astype(np.float32, copy=False)

    _log(log_path, "[4/9] Rough clustering independently inside each sample...")
    rough_id, representative_indices, rough_table = _rough_cluster_per_sample(
        embedding=z,
        coords_um=coords,
        full_neighbor_counts=measures.full_neighbor_counts,
        sample_ids=sample_ids,
        rough_clusters_per_sample=int(args.rough_clusters_per_sample),
        rough_batch_size=int(args.rough_batch_size) if args.rough_batch_size else None,
        rough_spatial_weight=float(args.rough_spatial_weight),
        rough_density_weight=float(args.rough_density_weight),
        seed=int(args.seed),
        log_path=log_path,
    )
    np.save(out_dir / "rough_sample_cluster_id.npy", rough_id)
    np.save(out_dir / "rough_representative_indices.npy", representative_indices)
    rough_table["representative_obs_name"] = adata.obs_names[representative_indices].to_numpy()
    rough_table.to_csv(out_dir / "rough_sample_clusters.csv", index=False)
    _log(log_path, f"  total_rough_clusters={representative_indices.size:,}")

    rep_graph_path = out_dir / "rough_representative_cell_graphs.npz"
    np.savez(
        rep_graph_path,
        representative_indices=representative_indices,
        representative_obs_names=adata.obs_names[representative_indices].to_numpy(dtype=str),
        sample_id=sample_ids[representative_indices].astype(str),
        rough_cluster_id=np.arange(representative_indices.size, dtype=np.int32),
        features=features[representative_indices],
        structures=structures[representative_indices],
        weights=weights[representative_indices],
        mask=measures.mask[representative_indices],
        full_neighbor_counts=measures.full_neighbor_counts[representative_indices],
        retained_neighbor_counts=measures.retained_neighbor_counts[representative_indices],
    )
    _log(log_path, f"  representative_graph_bundle={rep_graph_path}")
    if bool(args.representative_only):
        summary = {
            "active_path": "pairwise-niche-sample-rough-representative-graph-export-umap3d",
            "input_h5ad": str(args.input_h5ad),
            "output_dir": str(out_dir),
            "n_cells": int(adata.n_obs),
            "feature_obsm_key": str(args.umap_key),
            "sample_counts": {
                str(k): int(v)
                for k, v in adata.obs[str(args.sample_obs_key)].astype(str).value_counts().items()
            },
            "cell_feature_space": {
                "method": "precomputed_3d_umap",
                "standardized_for_fgw": not bool(args.no_standardize_umap),
                "allow_umap_as_feature": True,
                "uses_spatial_coordinates": False,
                "warning": "UMAP is used as requested for this exploratory run; it is not generally metric-preserving.",
            },
            "local_graph": measures.metadata,
            "rough_clustering_summary": {
                "method": "per_sample_minibatch_kmeans",
                "rough_feature_space": "standardized_umap3d_plus_within_sample_xy_plus_log1p_neighbor_count",
                "rough_clusters_per_sample": int(args.rough_clusters_per_sample),
                "rough_spatial_weight": float(args.rough_spatial_weight),
                "rough_density_weight": float(args.rough_density_weight),
                "n_rough_clusters": int(representative_indices.size),
                "representative_rule": "closest_cell_to_rough_cluster_centroid",
                "rough_cluster_table": str(out_dir / "rough_sample_clusters.csv"),
            },
            "representative_graphs": {
                "n_representative_cell_graphs": int(representative_indices.size),
                "radius_um": float(args.radius_um),
                "max_neighbors": int(args.max_neighbors),
                "feature_shape": list(features[representative_indices].shape),
                "structure_shape": list(structures[representative_indices].shape),
                "weight_shape": list(weights[representative_indices].shape),
                "mask_shape": list(measures.mask[representative_indices].shape),
                "fgw_structure_mode": str(measures.metadata.get("fgw_structure_mode")),
                **structure_norm,
            },
            "outputs": {
                "summary": str(out_dir / "summary.json"),
                "representative_graphs": str(rep_graph_path),
                "representative_indices": str(out_dir / "rough_representative_indices.npy"),
                "rough_cluster_ids": str(out_dir / "rough_sample_cluster_id.npy"),
                "rough_cluster_table": str(out_dir / "rough_sample_clusters.csv"),
                "cell_umap_3d": str(out_dir / "X_cell_umap_3d.npy"),
                "cell_umap_3d_raw": str(out_dir / "X_cell_umap_3d_raw.npy"),
            },
            "dense_pairwise_fgw_skipped": True,
            "skip_reason": "representative_only requested; 100,000 representative all-pairs FGW would require a dense 100000 x 100000 matrix and billions of pair solves.",
            "runtime_seconds": float(time.time() - t0),
            "seed": int(args.seed),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=_json_default)
        )
        _log(log_path, "[5/9] Representative-only requested; skipping dense fine FGW.")
        _log(log_path, f"Finished seconds: {time.time() - t0:.1f}")
        return summary

    _log(log_path, "[5/9] Computing FGW matrix among all rough-cluster representative cell graphs...")
    rep_features = features[representative_indices]
    rep_structures = structures[representative_indices]
    rep_weights = weights[representative_indices]
    representative_distance = _compute_landmark_fgw(
        features=rep_features,
        structures=rep_structures,
        weights=rep_weights,
        device=device,
        block_size=int(args.fine_block_size),
        alpha=float(args.fgw_alpha),
        epsilon=float(args.fgw_epsilon),
        sinkhorn_iters=int(args.fgw_sinkhorn_iters),
        fgw_iters=int(args.fgw_iters),
        log_path=log_path,
    )
    np.save(out_dir / "rough_representative_fgw_dissimilarity.npy", representative_distance)
    _log(
        log_path,
        json.dumps(
            {
                "shape": list(representative_distance.shape),
                "min": float(np.min(representative_distance)),
                "max": float(np.max(representative_distance)),
                "symmetric_max_abs_diff": float(
                    np.max(np.abs(representative_distance - representative_distance.T))
                ),
            },
            indent=2,
        ),
    )

    _log(log_path, "[6/9] Fine clustering pooled rough representatives with K=5..30...")
    cluster = cluster_from_distance(
        representative_distance,
        method="agglomerative",
        candidate_n_clusters=tuple(range(5, 31)),
        model_selection_metrics=(
            "silhouette",
            "pseudo_calinski_harabasz",
            "medoid_davies_bouldin",
            "percentile_dunn",
        ),
    )
    rep_fine_labels = cluster.labels.astype(np.int32, copy=False)
    full_labels = rep_fine_labels[rough_id].astype(np.int32, copy=False)
    full_scores = cluster.assignment_score[rough_id].astype(np.float32, copy=False)
    np.save(out_dir / "rough_representative_fine_labels.npy", rep_fine_labels)
    np.save(out_dir / "ot_niche_int.npy", full_labels)
    np.save(out_dir / "ot_niche_assignment_score.npy", full_scores)
    (out_dir / "fine_model_selection_k5_30.json").write_text(
        json.dumps(cluster.metadata["model_selection"], indent=2, sort_keys=True, default=_json_default)
    )
    rough_table["fine_niche_int"] = rep_fine_labels
    rough_table["fine_niche"] = [f"ON{int(value)}" for value in rep_fine_labels]
    rough_table.to_csv(out_dir / "rough_sample_clusters.csv", index=False)
    _log(log_path, json.dumps(cluster.metadata["model_selection"], indent=2, default=_json_default)[:5000])

    _log(log_path, "[7/9] Writing H5AD annotations and summaries...")
    categories = [f"ON{idx}" for idx in sorted(np.unique(full_labels))]
    label_names = np.asarray([f"ON{idx}" for idx in full_labels], dtype=object)
    colors = assign_high_contrast_colors(categories)
    adata.obsm["X_cell_umap_3d"] = z.astype(np.float32, copy=False)
    adata.obs["rough_sample_cluster_id"] = rough_id.astype(np.int32, copy=False)
    adata.obs["rough_sample_cluster"] = pd.Categorical(
        [f"RC{int(value)}" for value in rough_id],
        categories=[f"RC{int(value)}" for value in range(representative_indices.size)],
    )
    adata.obs["ot_niche"] = pd.Categorical(label_names, categories=categories)
    adata.obs["ot_niche_int"] = full_labels
    adata.obs["ot_niche_assignment_score"] = full_scores
    suffix = f"r{float(args.radius_um):g}".replace(".", "p")
    area = np.pi * float(args.radius_um) ** 2
    adata.obs[f"n_neighbors_full_{suffix}"] = measures.full_neighbor_counts
    adata.obs[f"n_neighbors_retained_{suffix}"] = measures.retained_neighbor_counts
    adata.obs[f"neighbor_retention_fraction_{suffix}"] = (
        measures.retained_neighbor_counts.astype(np.float32)
        / np.maximum(measures.full_neighbor_counts.astype(np.float32), 1.0)
    )
    adata.obs[f"local_density_full_per_um2_{suffix}"] = (
        measures.full_neighbor_counts.astype(np.float32) / max(area, 1e-12)
    )
    adata.obs[f"local_density_retained_per_um2_{suffix}"] = (
        measures.retained_neighbor_counts.astype(np.float32) / max(area, 1e-12)
    )
    adata.obs[f"is_isolated_{suffix}"] = measures.full_neighbor_counts == 0
    instance_ids, instance_names = _connected_instances_sparse(
        coords_um=coords,
        sample_ids=sample_ids,
        labels=full_labels,
        radius_um=float(args.radius_um),
        log_path=log_path,
    )
    adata.obs["ot_niche_instance"] = pd.Categorical(instance_names)
    adata.obs["ot_niche_instance_int"] = instance_ids.astype(np.int32)
    cluster_counts = {cat: int(np.sum(label_names == cat)) for cat in categories}
    summary = {
        "active_path": "pairwise-niche-sample-rough-full-cohort-fine-fgw-umap3d",
        "input_h5ad": str(args.input_h5ad),
        "output_dir": str(out_dir),
        "n_cells": int(adata.n_obs),
        "feature_obsm_key": str(args.umap_key),
        "cell_feature_space": {
            "method": "precomputed_3d_umap",
            "standardized_for_fgw": not bool(args.no_standardize_umap),
            "allow_umap_as_feature": True,
            "uses_spatial_coordinates": False,
            "warning": "UMAP is used as requested for this exploratory run; it is not generally metric-preserving.",
        },
        "sample_counts": {
            str(k): int(v)
            for k, v in adata.obs[str(args.sample_obs_key)].astype(str).value_counts().items()
        },
        "local_graph": measures.metadata,
        "rough_clustering_summary": {
            "method": "per_sample_minibatch_kmeans",
            "rough_feature_space": "standardized_umap3d_plus_within_sample_xy_plus_log1p_neighbor_count",
            "rough_clusters_per_sample": int(args.rough_clusters_per_sample),
            "rough_spatial_weight": float(args.rough_spatial_weight),
            "rough_density_weight": float(args.rough_density_weight),
            "n_rough_clusters": int(representative_indices.size),
            "representative_rule": "closest_cell_to_rough_cluster_centroid",
            "rough_cluster_table": str(out_dir / "rough_sample_clusters.csv"),
        },
        "distance_summary": {
            "mode": "representative_fused_gromov_wasserstein_after_sample_rough_clustering",
            "uses_graph_topology": bool(
                measures.metadata.get("fgw_structure_mode") != "complete_euclidean"
            ),
            "uses_complete_spatial_structure": bool(
                measures.metadata.get("fgw_structure_mode") == "complete_euclidean"
            ),
            "all_cell_pairs_materialized": False,
            "n_representative_cell_graphs": int(representative_indices.size),
            "radius_um": float(args.radius_um),
            "max_neighbors": int(args.max_neighbors),
            "fgw_alpha": float(args.fgw_alpha),
            "fgw_epsilon": float(args.fgw_epsilon),
            "fgw_sinkhorn_iters": int(args.fgw_sinkhorn_iters),
            "fgw_iters": int(args.fgw_iters),
            "node_feature_term": "standardized_3d_umap_only",
            "graph_structure_term": str(measures.metadata.get("fgw_structure_mode")),
            **structure_norm,
            "representative_distance_shape": list(representative_distance.shape),
            "representative_distance_min": float(np.min(representative_distance)),
            "representative_distance_max": float(np.max(representative_distance)),
        },
        "clustering_summary": {
            "method": "agglomerative_full_cohort_representative_model_selection",
            "candidate_n_clusters": list(range(5, 31)),
            "model_selection_metrics": [
                "silhouette",
                "pseudo_calinski_harabasz",
                "medoid_davies_bouldin",
                "percentile_dunn",
            ],
            "n_clusters": int(len(categories)),
            "cluster_counts": cluster_counts,
            "assignment_score_type": "rough_representative_full_cohort_pairwise_distance_margin",
            "fine_model_selection": cluster.metadata["model_selection"],
            "n_niche_instances": int(np.unique(instance_ids).size),
        },
        "niche_colors": colors,
        "outputs": {
            "h5ad": str(out_dir / "cells_pairwise_niche_sample_rough_full_fine_umap3d_fgw.h5ad"),
            "summary": str(out_dir / "summary.json"),
            "representative_distance": str(out_dir / "rough_representative_fgw_dissimilarity.npy"),
            "rough_cluster_table": str(out_dir / "rough_sample_clusters.csv"),
            "cell_umap_3d": str(out_dir / "X_cell_umap_3d.npy"),
            "cell_umap_3d_raw": str(out_dir / "X_cell_umap_3d_raw.npy"),
        },
        "runtime_seconds": float(time.time() - t0),
        "seed": int(args.seed),
    }
    adata.uns["pairwise_niche_color_map"] = colors
    adata.uns["ot_niche_colors"] = [colors[cat] for cat in categories]
    adata.uns["pairwise_niche_config"] = _sanitize(
        {
            "feature_obsm_key": str(args.umap_key),
            "embedding_method": "precomputed",
            "allow_umap_as_feature": True,
            "radius_um": float(args.radius_um),
            "max_neighbors": int(args.max_neighbors),
            "rough_clusters_per_sample": int(args.rough_clusters_per_sample),
            "candidate_n_clusters": list(range(5, 31)),
            "model_selection_metrics": [
                "silhouette",
                "pseudo_calinski_harabasz",
                "medoid_davies_bouldin",
                "percentile_dunn",
            ],
        }
    )
    adata.uns["pairwise_niche_rough_clustering_summary"] = _sanitize(
        summary["rough_clustering_summary"]
    )
    adata.uns["pairwise_niche_clustering_summary"] = _sanitize(summary["clustering_summary"])
    adata.uns["pairwise_niche_distance_summary"] = _sanitize(summary["distance_summary"])
    adata.uns["pairwise_niche_run_summary"] = _sanitize(summary)
    h5ad_path = out_dir / "cells_pairwise_niche_sample_rough_full_fine_umap3d_fgw.h5ad"
    adata.write_h5ad(h5ad_path, compression="gzip")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    (out_dir / "ot_niche_colors.json").write_text(json.dumps(colors, indent=2, sort_keys=True))
    (out_dir / "cluster_counts.json").write_text(json.dumps(cluster_counts, indent=2, sort_keys=True))

    _log(log_path, "[8/9] Generating full-cohort and per-sample plots...")
    _plot_outputs(
        adata=adata,
        out_dir=out_dir,
        plot_prefix="sample_rough_full_fine_umap3d_fgw",
    )

    _log(log_path, "[9/9] Validating saved output...")
    saved = ad.read_h5ad(h5ad_path, backed="r")
    try:
        _log(log_path, f"Saved H5AD shape: {saved.shape}")
        _log(log_path, f"Cluster counts: {json.dumps(cluster_counts, indent=2)}")
    finally:
        saved.file.close()
    _log(log_path, f"Finished seconds: {time.time() - t0:.1f}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5ad", default="/storage/hackathon_2026/spatial_ot_input/visium_hd_spatial_ot_input_pooled.h5ad")
    parser.add_argument("--input-dir", default="/storage/hackathon_2026/spatial_ot_input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--umap-key", default="X_umap_marker_genes_3d")
    parser.add_argument("--source-obs-key", default="source_h5ad")
    parser.add_argument("--sample-obs-key", default="sample_id")
    parser.add_argument("--spatial-x-key", default="cell_x")
    parser.add_argument("--spatial-y-key", default="cell_y")
    parser.add_argument("--radius-um", type=float, default=50.0)
    parser.add_argument("--max-neighbors", type=int, default=100)
    parser.add_argument("--rough-clusters-per-sample", type=int, default=200)
    parser.add_argument("--rough-batch-size", type=int, default=0)
    parser.add_argument("--rough-spatial-weight", type=float, default=0.25)
    parser.add_argument("--rough-density-weight", type=float, default=0.10)
    parser.add_argument("--cap-state-clusters", type=int, default=16)
    parser.add_argument("--radial-shells", type=int, default=3)
    parser.add_argument("--expression-weight", type=float, default=1.0)
    parser.add_argument("--spatial-weight", type=float, default=0.25)
    parser.add_argument("--distance-weight", type=float, default=0.10)
    parser.add_argument("--ground-cost-normalization", default="sampled_median")
    parser.add_argument("--ground-cost-sample-pairs", type=int, default=10000)
    parser.add_argument("--fgw-alpha", type=float, default=0.25)
    parser.add_argument(
        "--fgw-structure-mode",
        default="local_knn_shortest_path",
        choices=[
            "complete_euclidean",
            "local_knn_shortest_path",
            "radius_graph_shortest_path",
            "binary_edge_distance",
            "adjacency",
        ],
    )
    parser.add_argument("--fgw-structure-knn", type=int, default=6)
    parser.add_argument("--fgw-structure-radius-fraction", type=float, default=0.5)
    parser.add_argument("--fgw-structure-normalization", default="sampled_median")
    parser.add_argument("--fgw-structure-sample-pairs", type=int, default=10000)
    parser.add_argument("--fgw-epsilon", type=float, default=0.05)
    parser.add_argument("--fgw-sinkhorn-iters", type=int, default=8)
    parser.add_argument("--fgw-iters", type=int, default=2)
    parser.add_argument("--fine-block-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-standardize-umap", action="store_true")
    parser.add_argument(
        "--representative-only",
        action="store_true",
        help="Export rough representative cell graphs and skip dense representative FGW/clustering.",
    )
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
