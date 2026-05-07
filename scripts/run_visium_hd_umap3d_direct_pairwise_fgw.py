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


def _pairwise_feasibility(
    *,
    n_cells: int,
    support_size: int,
    sinkhorn_iters: int,
    fgw_iters: int,
    observed_pairs_per_second: float | None = None,
) -> dict[str, object]:
    n = int(n_cells)
    support = int(support_size)
    pairs_with_diag = n * (n + 1) // 2
    dense_square_bytes = n * n * np.dtype("float32").itemsize
    condensed_float32_bytes = pairs_with_diag * np.dtype("float32").itemsize
    condensed_float16_bytes = pairs_with_diag * np.dtype("float16").itemsize
    work_units = float(pairs_with_diag) * float(support) ** 2 * float(sinkhorn_iters) * float(fgw_iters)
    lower_bound_days = None
    if observed_pairs_per_second and observed_pairs_per_second > 0:
        lower_bound_days = float(pairs_with_diag / observed_pairs_per_second / 86400.0)
    return {
        "n_cells": n,
        "pairwise_comparison_type": "direct_all_cell_graph_pairs",
        "uses_landmarks_or_reference": False,
        "pairs_with_diagonal": int(pairs_with_diag),
        "off_diagonal_pairs": int(n * (n - 1) // 2),
        "support_size": support,
        "sinkhorn_iters": int(sinkhorn_iters),
        "fgw_iters": int(fgw_iters),
        "rough_fgw_work_units": float(work_units),
        "dense_float32_square_tib": float(dense_square_bytes / 1024**4),
        "condensed_float32_upper_tib": float(condensed_float32_bytes / 1024**4),
        "condensed_float16_upper_tib": float(condensed_float16_bytes / 1024**4),
        "observed_reference_pairs_per_second": (
            None if observed_pairs_per_second is None else float(observed_pairs_per_second)
        ),
        "lower_bound_runtime_days_at_observed_reference_rate": lower_bound_days,
    }


def _write_exact_outputs(
    *,
    adata: ad.AnnData,
    z: np.ndarray,
    full_labels: np.ndarray,
    assignment_score: np.ndarray,
    distance: np.ndarray,
    cluster_metadata: dict[str, object],
    measures_metadata: dict[str, object],
    structure_norm: dict[str, object],
    coords: np.ndarray,
    sample_ids: np.ndarray,
    out_dir: Path,
    args: argparse.Namespace,
    log_path: Path,
    runtime_seconds: float,
) -> dict[str, object]:
    categories = [f"ON{idx}" for idx in sorted(np.unique(full_labels))]
    label_names = np.asarray([f"ON{idx}" for idx in full_labels], dtype=object)
    colors = assign_high_contrast_colors(categories)
    adata.obsm["X_cell_umap_3d"] = z.astype(np.float32, copy=False)
    adata.obs["ot_niche"] = pd.Categorical(label_names, categories=categories)
    adata.obs["ot_niche_int"] = full_labels.astype(np.int32, copy=False)
    adata.obs["ot_niche_assignment_score"] = assignment_score.astype(np.float32, copy=False)
    instance_ids, instance_names = _connected_instances_sparse(
        coords_um=coords,
        sample_ids=sample_ids,
        labels=full_labels,
        radius_um=float(args.radius_um),
        log_path=log_path,
    )
    adata.obs["ot_niche_instance"] = pd.Categorical(instance_names)
    adata.obs["ot_niche_instance_int"] = instance_ids.astype(np.int32)
    adata.uns["pairwise_niche_color_map"] = colors
    adata.uns["ot_niche_colors"] = [colors[cat] for cat in categories]
    cluster_counts = {cat: int(np.sum(label_names == cat)) for cat in categories}
    summary = {
        "active_path": "pairwise-niche-direct-all-pairs-fgw-full-cohort-umap3d",
        "uses_landmarks_or_reference": False,
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
        "local_graph": measures_metadata,
        "distance_summary": {
            "mode": "direct_all_pairs_fused_gromov_wasserstein",
            "uses_graph_topology": bool(
                measures_metadata.get("fgw_structure_mode") != "complete_euclidean"
            ),
            "uses_complete_spatial_structure": bool(
                measures_metadata.get("fgw_structure_mode") == "complete_euclidean"
            ),
            "all_pairs_dense_distance_matrix_materialized": True,
            "distance_shape": list(distance.shape),
            "radius_um": float(args.radius_um),
            "max_neighbors": int(args.max_neighbors),
            "fgw_alpha": float(args.fgw_alpha),
            "fgw_epsilon": float(args.fgw_epsilon),
            "fgw_sinkhorn_iters": int(args.fgw_sinkhorn_iters),
            "fgw_iters": int(args.fgw_iters),
            "node_feature_term": "standardized_3d_umap_only",
            "graph_structure_term": str(measures_metadata.get("fgw_structure_mode")),
            **structure_norm,
        },
        "clustering_summary": {
            "method": "agglomerative_direct_full_pairwise_model_selection",
            "n_clusters": int(len(categories)),
            "cluster_counts": cluster_counts,
            "assignment_score_type": "full_pairwise_distance_margin",
            "model_selection": cluster_metadata.get("model_selection"),
            "n_niche_instances": int(np.unique(instance_ids).size),
        },
        "niche_colors": colors,
        "outputs": {
            "h5ad": str(out_dir / "cells_pairwise_niche_direct_umap3d_fgw.h5ad"),
            "summary": str(out_dir / "summary.json"),
            "distance_matrix": str(out_dir / "cell_fgw_dissimilarity.npy"),
        },
        "runtime_seconds": float(runtime_seconds),
        "seed": int(args.seed),
    }
    adata.uns["pairwise_niche_config"] = _sanitize(
        {
            "feature_obsm_key": str(args.umap_key),
            "embedding_method": "precomputed",
            "allow_umap_as_feature": True,
            "radius_um": float(args.radius_um),
            "max_neighbors": int(args.max_neighbors),
            "candidate_n_clusters": list(range(5, 31)),
            "model_selection_metrics": [
                "silhouette",
                "pseudo_calinski_harabasz",
                "medoid_davies_bouldin",
                "percentile_dunn",
            ],
            "uses_landmarks_or_reference": False,
        }
    )
    adata.uns["pairwise_niche_clustering_summary"] = _sanitize(summary["clustering_summary"])
    adata.uns["pairwise_niche_distance_summary"] = _sanitize(summary["distance_summary"])
    adata.uns["pairwise_niche_run_summary"] = _sanitize(summary)
    adata.write_h5ad(out_dir / "cells_pairwise_niche_direct_umap3d_fgw.h5ad", compression="gzip")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default)
    )
    (out_dir / "ot_niche_colors.json").write_text(json.dumps(colors, indent=2, sort_keys=True))
    _plot_outputs(
        adata=adata,
        out_dir=out_dir,
        plot_prefix="direct_full_pairwise_umap3d_fgw",
    )
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_visium_hd_umap3d_direct_pairwise_fgw.log"
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
                "direct_all_pairs": True,
                "uses_landmarks_or_reference": False,
                "device": str(device),
                "radius_um": float(args.radius_um),
                "max_neighbors": int(args.max_neighbors),
            },
            indent=2,
        ),
    )
    _log(log_path, "[1/6] Loading pooled obs and per-sample 3D UMAP...")
    adata, umap_raw = _load_per_sample_umap(
        pooled_h5ad=Path(args.input_h5ad),
        input_dir=Path(args.input_dir),
        umap_key=str(args.umap_key),
        source_obs_key=str(args.source_obs_key),
    )
    if args.max_cells is not None:
        max_cells = int(args.max_cells)
        if max_cells <= 0:
            raise ValueError("--max-cells must be positive when provided.")
        adata = adata[:max_cells].copy()
        umap_raw = umap_raw[:max_cells]
    sample_ids = adata.obs[str(args.sample_obs_key)].astype(str).to_numpy()
    coords = np.column_stack(
        [
            np.asarray(adata.obs[str(args.spatial_x_key)], dtype=np.float32),
            np.asarray(adata.obs[str(args.spatial_y_key)], dtype=np.float32),
        ]
    )
    _log(log_path, f"Input cells={adata.n_obs:,}, raw_umap_shape={umap_raw.shape}")

    _log(log_path, "[2/6] Using standardized 3D UMAP as graph node features...")
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

    _log(log_path, "[3/6] Building compact same-sample local cell graphs...")
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

    feasibility = _pairwise_feasibility(
        n_cells=int(adata.n_obs),
        support_size=int(features.shape[1]),
        sinkhorn_iters=int(args.fgw_sinkhorn_iters),
        fgw_iters=int(args.fgw_iters),
        observed_pairs_per_second=float(args.observed_pairs_per_second)
        if args.observed_pairs_per_second is not None
        else None,
    )
    feasibility.update(
        {
            "available_storage_note": "Check df -h for live storage before forcing exact all-pairs.",
            "status": "estimate_only",
            "force_exact": bool(args.force_exact),
            "max_direct_cells_without_force": int(args.max_direct_cells_without_force),
        }
    )
    (out_dir / "direct_all_pairs_feasibility.json").write_text(
        json.dumps(feasibility, indent=2, sort_keys=True, default=_json_default)
    )
    _log(log_path, json.dumps(feasibility, indent=2, default=_json_default))
    if int(adata.n_obs) > int(args.max_direct_cells_without_force) and not bool(args.force_exact):
        feasibility["status"] = "refused_exact_full_pairwise_without_force"
        feasibility["reason"] = (
            "Direct all-pairs FGW is the requested method, but this cohort exceeds the "
            "configured exact-cell guard. No landmark/reference approximation was run."
        )
        (out_dir / "direct_all_pairs_feasibility.json").write_text(
            json.dumps(feasibility, indent=2, sort_keys=True, default=_json_default)
        )
        _log(log_path, "[stop] Exact direct all-pairs FGW was not run; feasibility report written.")
        return feasibility

    _log(log_path, "[4/6] Computing direct all-cells pairwise FGW matrix...")
    distance = _compute_landmark_fgw(
        features=features,
        structures=structures,
        weights=weights,
        device=device,
        block_size=int(args.block_size),
        alpha=float(args.fgw_alpha),
        epsilon=float(args.fgw_epsilon),
        sinkhorn_iters=int(args.fgw_sinkhorn_iters),
        fgw_iters=int(args.fgw_iters),
        log_path=log_path,
    )
    np.save(out_dir / "cell_fgw_dissimilarity.npy", distance)

    _log(log_path, "[5/6] Clustering from the direct full pairwise FGW matrix with K=5..30...")
    cluster = cluster_from_distance(
        distance,
        method="agglomerative",
        candidate_n_clusters=tuple(range(5, 31)),
        model_selection_metrics=(
            "silhouette",
            "pseudo_calinski_harabasz",
            "medoid_davies_bouldin",
            "percentile_dunn",
        ),
    )
    labels = cluster.labels.astype(np.int32, copy=False)

    _log(log_path, "[6/6] Writing outputs...")
    summary = _write_exact_outputs(
        adata=adata,
        z=z,
        full_labels=labels,
        assignment_score=cluster.assignment_score,
        distance=distance,
        cluster_metadata=cluster.metadata,
        measures_metadata=measures.metadata,
        structure_norm=structure_norm,
        coords=coords,
        sample_ids=sample_ids,
        out_dir=out_dir,
        args=args,
        log_path=log_path,
        runtime_seconds=time.time() - t0,
    )
    _log(log_path, f"Finished seconds: {time.time() - t0:.1f}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-h5ad",
        default="/storage/hackathon_2026/spatial_ot_input/visium_hd_spatial_ot_input_pooled.h5ad",
    )
    parser.add_argument("--input-dir", default="/storage/hackathon_2026/spatial_ot_input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--umap-key", default="X_umap_marker_genes_3d")
    parser.add_argument("--source-obs-key", default="source_h5ad")
    parser.add_argument("--sample-obs-key", default="sample_id")
    parser.add_argument("--spatial-x-key", default="cell_x")
    parser.add_argument("--spatial-y-key", default="cell_y")
    parser.add_argument("--radius-um", type=float, default=50.0)
    parser.add_argument("--max-neighbors", type=int, default=100)
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
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-standardize-umap", action="store_true")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--force-exact", action="store_true")
    parser.add_argument("--max-direct-cells-without-force", type=int, default=5000)
    parser.add_argument(
        "--observed-pairs-per-second",
        type=float,
        default=179071.33303611234,
        help="Optional observed FGW pair rate used only for runtime feasibility estimates.",
    )
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
