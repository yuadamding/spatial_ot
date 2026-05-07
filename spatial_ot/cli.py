from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .feature_source import prepare_h5ad_feature_cache
from .pairwise_niche import run_pairwise_niche_on_h5ad
from .pooling import distribute_pooled_feature_cache_to_inputs, pool_h5ads_in_directory


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


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _parse_int_list(value: str | None) -> tuple[int, ...] | None:
    if value is None or str(value).strip() == "":
        return None
    out: list[int] = []
    for item in str(value).split(","):
        token = item.strip()
        if not token:
            continue
        separator = ":" if ":" in token else "-" if "-" in token[1:] else None
        if separator is None:
            out.append(int(token))
            continue
        left, right = token.split(separator, 1)
        start = int(left.strip())
        stop = int(right.strip())
        step = 1 if stop >= start else -1
        out.extend(range(start, stop + step, step))
    return tuple(out)


def _parse_float_list(value: str | None) -> tuple[float, ...] | None:
    if value is None or str(value).strip() == "":
        return None
    return tuple(float(item.strip()) for item in str(value).split(",") if item.strip())


def _parse_string_list(value: str | None) -> tuple[str, ...] | None:
    if value is None or str(value).strip() == "":
        return None
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _add_pairwise_niche_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "action",
        nargs="?",
        default="fit",
        choices=["fit"],
        help="Action to run. 'fit' builds pairwise OT neighborhoods and clusters distances.",
    )
    parser.add_argument("--input-h5ad", required=True, help="Input cell-level H5AD.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for pairwise OT niche artifacts.",
    )
    parser.add_argument(
        "--feature-obsm-key",
        default="X",
        help="Expression feature source for cohort embedding. Use an obsm key or 'X'.",
    )
    parser.add_argument("--spatial-x-key", required=True, help="obs key for x.")
    parser.add_argument("--spatial-y-key", required=True, help="obs key for y.")
    parser.add_argument(
        "--sample-obs-key",
        default="sample_id",
        help="obs key storing sample IDs. Local graphs never cross samples.",
    )
    parser.add_argument(
        "--spatial-scale",
        type=float,
        default=1.0,
        help="Multiplier converting coordinates into microns.",
    )
    parser.add_argument(
        "--embedding-method",
        default="pca",
        choices=["pca", "svd", "precomputed"],
        help="Cohort expression embedding method. Spatial coordinates are not used.",
    )
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument(
        "--expression-batch-key",
        default=None,
        help="Recorded for provenance; batch-corrected embeddings should be precomputed.",
    )
    parser.add_argument(
        "--standardize-precomputed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize precomputed expression embeddings before OT.",
    )
    parser.add_argument(
        "--allow-umap-as-feature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow a UMAP obsm key as the OT feature space for exploratory runs. "
            "UMAP is not generally metric-preserving."
        ),
    )
    parser.add_argument("--radius-um", type=float, default=50.0)
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=32,
        help="Maximum number of neighbors retained inside --radius-um for each cell graph.",
    )
    parser.add_argument(
        "--include-anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the anchor cell as a node in its local measure.",
    )
    parser.add_argument(
        "--isolated-policy",
        default="anchor_fallback",
        choices=["zero_dummy", "anchor_fallback"],
        help="How to represent no-neighbor cells when the anchor is not included.",
    )
    parser.add_argument(
        "--graph-kernel",
        default="gaussian",
        choices=["gaussian", "uniform", "inverse_distance", "binary", "inverse"],
    )
    parser.add_argument(
        "--cap-mode",
        default="radial_shell_state",
        choices=["radial_shell", "radial_shell_state"],
        help="Neighbor cap strategy for dense neighborhoods.",
    )
    parser.add_argument("--cap-state-clusters", type=int, default=16)
    parser.add_argument("--radial-shells", type=int, default=3)
    parser.add_argument("--expression-weight", type=float, default=1.0)
    parser.add_argument("--spatial-weight", type=float, default=0.25)
    parser.add_argument("--distance-weight", type=float, default=0.10)
    parser.add_argument(
        "--ground-cost-normalization",
        default="sampled_median",
        choices=["none", "dimension", "sampled_median"],
        help="Normalize expression/spatial/radial cost contributions before weighting.",
    )
    parser.add_argument("--ground-cost-sample-pairs", type=int, default=10000)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iters", type=int, default=50)
    parser.add_argument(
        "--distance-mode",
        default="debiased_entropic_transport",
        choices=[
            "sinkhorn",
            "debiased_entropic_transport",
            "sinkhorn_divergence",
            "fused_gromov_wasserstein",
        ],
    )
    parser.add_argument(
        "--fgw-alpha",
        type=float,
        default=0.25,
        help="Local structure weight for fused Gromov-Wasserstein.",
    )
    parser.add_argument(
        "--fgw-iters",
        type=int,
        default=5,
        help="Outer FGW coupling-refinement iterations.",
    )
    parser.add_argument(
        "--fgw-node-feature-mode",
        default="expression_only",
        choices=["expression_only", "expression_plus_radial", "full_token"],
        help="Node features used in FGW. expression_only avoids double-counting spatial geometry.",
    )
    parser.add_argument(
        "--fgw-structure-mode",
        default="local_knn_shortest_path",
        choices=[
            "complete_euclidean",
            "local_knn_shortest_path",
            "radius_graph_shortest_path",
            "adjacency",
        ],
        help="Local structure matrix used by FGW.",
    )
    parser.add_argument(
        "--fgw-structure-knn",
        type=int,
        default=6,
        help="Neighbor count for local_knn_shortest_path FGW structure.",
    )
    parser.add_argument(
        "--fgw-structure-radius-fraction",
        type=float,
        default=0.5,
        help="Radius fraction for radius_graph_shortest_path or adjacency FGW structure.",
    )
    parser.add_argument(
        "--fgw-structure-normalization",
        default="sampled_median",
        choices=["none", "sampled_median"],
        help="Normalize FGW structure-cost scale before applying --fgw-alpha.",
    )
    parser.add_argument(
        "--fgw-structure-sample-pairs",
        type=int,
        default=10000,
        help="Sample count for FGW structure-cost normalization.",
    )
    parser.add_argument(
        "--pairwise-mode",
        default="exact_blockwise",
        choices=["exact", "exact_blockwise"],
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-exact-cells",
        type=int,
        default=5000,
        help="Safety guard for exact all-pairs OT.",
    )
    parser.add_argument(
        "--max-ot-work-units",
        type=float,
        default=5e11,
        help="Safety guard for exact all-pairs Sinkhorn work.",
    )
    parser.add_argument(
        "--max-fgw-work-units",
        type=float,
        default=1e12,
        help="Safety guard for exact all-pairs FGW structure and Sinkhorn work.",
    )
    parser.add_argument(
        "--force-large-exact-ot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Override exact all-pairs size and work guards.",
    )
    parser.add_argument(
        "--distance-store",
        default="auto",
        choices=["auto", "h5ad", "npy_memmap"],
    )
    parser.add_argument(
        "--cluster-method",
        default="agglomerative",
        choices=["agglomerative", "kmedoids", "leiden_ot_knn"],
        help="Distance-based clustering method. KMeans is intentionally not available.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Required for agglomerative and k-medoids clustering.",
    )
    parser.add_argument(
        "--candidate-n-clusters",
        default=None,
        help=(
            "Comma-separated K values or inclusive ranges for model selection with "
            "agglomerative or k-medoids clustering. When omitted and --n-clusters is "
            "not set, candidates default to 5:30."
        ),
    )
    parser.add_argument(
        "--model-selection-metrics",
        default="silhouette,pseudo_calinski_harabasz,medoid_davies_bouldin,percentile_dunn",
        help=(
            "Comma-separated precomputed-distance model-selection metrics. Supported: "
            "silhouette, pseudo_calinski_harabasz, medoid_davies_bouldin, "
            "percentile_dunn, minimum_dunn. Legacy names are accepted as aliases."
        ),
    )
    parser.add_argument("--ot-knn", type=int, default=30)
    parser.add_argument(
        "--ot-affinity-scaling",
        default="local",
        choices=["local", "global"],
    )
    parser.add_argument("--leiden-resolution", type=float, default=1.0)
    parser.add_argument(
        "--candidate-resolutions",
        default=None,
        help=(
            "Comma-separated Leiden resolutions for model selection. Candidates are "
            "ranked with --model-selection-metrics."
        ),
    )
    parser.add_argument("--instance-radius-um", type=float, default=None)
    parser.add_argument("--instance-max-neighbors", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatial-ot",
        description="Pairwise OT spatial niche discovery utilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pairwise_niche = sub.add_parser(
        "pairwise-niche",
        help="Fit spatial niches from a pairwise OT neighborhood distance matrix.",
    )
    _add_pairwise_niche_args(pairwise_niche)

    pool_inputs = sub.add_parser(
        "pool-inputs",
        help="Pool sample H5AD files into a sample-aware cohort H5AD.",
    )
    pool_inputs.add_argument("--input-dir", required=True)
    pool_inputs.add_argument("--output-h5ad", required=True)
    pool_inputs.add_argument(
        "--feature-obsm-key",
        action="append",
        dest="feature_obsm_keys",
        required=True,
        help="Feature key to preserve. Repeat to keep multiple keys. Use 'X' for the gene matrix.",
    )
    pool_inputs.add_argument("--sample-glob", default="*_cells_marker_genes_umap3d.h5ad")
    pool_inputs.add_argument("--spatial-x-key", default="cell_x")
    pool_inputs.add_argument("--spatial-y-key", default="cell_y")
    pool_inputs.add_argument("--pooled-spatial-x-key", default="pooled_cell_x")
    pool_inputs.add_argument("--pooled-spatial-y-key", default="pooled_cell_y")
    pool_inputs.add_argument("--sample-obs-key", default="sample_id")
    pool_inputs.add_argument("--source-file-obs-key", default="source_h5ad")
    pool_inputs.add_argument("--sample-id-prefix", default="")
    pool_inputs.add_argument("--sample-id-suffix", default="_cells_marker_genes_umap3d")
    pool_inputs.add_argument(
        "--sample-id-case",
        default="preserve",
        choices=["preserve", "lower", "upper"],
    )
    pool_inputs.add_argument("--layout-columns", type=int, default=None)
    pool_inputs.add_argument("--layout-gap", type=float, default=None)

    prepare_inputs = sub.add_parser(
        "prepare-inputs",
        help="Create a reusable feature cache inside an H5AD.",
    )
    prepare_inputs.add_argument("--input-h5ad", required=True)
    prepare_inputs.add_argument("--output-h5ad", default=None)
    prepare_inputs.add_argument("--feature-obsm-key", default="X")
    prepare_inputs.add_argument("--output-obsm-key", default=None)
    prepare_inputs.add_argument("--overwrite", action="store_true")

    distribute = sub.add_parser(
        "distribute-prepared-inputs",
        help="Copy a pooled prepared feature cache back into source H5AD files.",
    )
    distribute.add_argument("--pooled-h5ad", required=True)
    distribute.add_argument("--input-dir", required=True)
    distribute.add_argument("--prepared-obsm-key", required=True)
    distribute.add_argument("--sample-glob", default="*_cells_marker_genes_umap3d.h5ad")
    distribute.add_argument("--source-file-obs-key", default="source_h5ad")
    distribute.add_argument("--overwrite", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "pairwise-niche":
        summary = run_pairwise_niche_on_h5ad(
            input_h5ad=args.input_h5ad,
            output_dir=args.output_dir,
            feature_obsm_key=args.feature_obsm_key,
            spatial_x_key=args.spatial_x_key,
            spatial_y_key=args.spatial_y_key,
            sample_obs_key=args.sample_obs_key,
            spatial_scale=args.spatial_scale,
            embedding_method=args.embedding_method,
            embedding_dim=args.embedding_dim,
            expression_batch_key=args.expression_batch_key,
            standardize_precomputed=args.standardize_precomputed,
            allow_umap_as_feature=args.allow_umap_as_feature,
            radius_um=args.radius_um,
            max_neighbors=args.max_neighbors,
            include_anchor=args.include_anchor,
            isolated_policy=args.isolated_policy,
            graph_kernel=args.graph_kernel,
            cap_mode=args.cap_mode,
            cap_state_clusters=args.cap_state_clusters,
            radial_shells=args.radial_shells,
            expression_weight=args.expression_weight,
            spatial_weight=args.spatial_weight,
            distance_weight=args.distance_weight,
            ground_cost_normalization=args.ground_cost_normalization,
            ground_cost_sample_pairs=args.ground_cost_sample_pairs,
            anchor_weight=args.anchor_weight,
            sinkhorn_epsilon=args.sinkhorn_epsilon,
            sinkhorn_iters=args.sinkhorn_iters,
            distance_mode=args.distance_mode,
            fgw_alpha=args.fgw_alpha,
            fgw_iters=args.fgw_iters,
            fgw_node_feature_mode=args.fgw_node_feature_mode,
            fgw_structure_mode=args.fgw_structure_mode,
            fgw_structure_knn=args.fgw_structure_knn,
            fgw_structure_radius_fraction=args.fgw_structure_radius_fraction,
            fgw_structure_normalization=args.fgw_structure_normalization,
            fgw_structure_sample_pairs=args.fgw_structure_sample_pairs,
            pairwise_mode=args.pairwise_mode,
            block_size=args.block_size,
            device=args.device,
            max_exact_cells=args.max_exact_cells,
            max_ot_work_units=args.max_ot_work_units,
            max_fgw_work_units=args.max_fgw_work_units,
            force_large_exact_ot=args.force_large_exact_ot,
            distance_store=args.distance_store,
            cluster_method=args.cluster_method,
            n_clusters=args.n_clusters,
            candidate_n_clusters=_parse_int_list(args.candidate_n_clusters),
            model_selection_metrics=_parse_string_list(args.model_selection_metrics),
            ot_knn=args.ot_knn,
            ot_affinity_scaling=args.ot_affinity_scaling,
            leiden_resolution=args.leiden_resolution,
            candidate_resolutions=_parse_float_list(args.candidate_resolutions),
            instance_radius_um=args.instance_radius_um,
            instance_max_neighbors=args.instance_max_neighbors,
            seed=args.seed,
        )
    elif args.command == "pool-inputs":
        summary = pool_h5ads_in_directory(
            input_dir=Path(args.input_dir),
            output_h5ad=Path(args.output_h5ad),
            feature_obsm_keys=list(args.feature_obsm_keys),
            sample_glob=args.sample_glob,
            spatial_x_key=args.spatial_x_key,
            spatial_y_key=args.spatial_y_key,
            pooled_spatial_x_key=args.pooled_spatial_x_key,
            pooled_spatial_y_key=args.pooled_spatial_y_key,
            sample_obs_key=args.sample_obs_key,
            source_file_obs_key=args.source_file_obs_key,
            sample_id_prefix=args.sample_id_prefix,
            sample_id_suffix=args.sample_id_suffix,
            sample_id_case=args.sample_id_case,
            layout_columns=args.layout_columns,
            layout_gap=args.layout_gap,
        )
    elif args.command == "prepare-inputs":
        summary = prepare_h5ad_feature_cache(
            input_h5ad=Path(args.input_h5ad),
            output_h5ad=Path(args.output_h5ad) if args.output_h5ad else None,
            feature_obsm_key=args.feature_obsm_key,
            output_obsm_key=args.output_obsm_key,
            overwrite=bool(args.overwrite),
        )
    elif args.command == "distribute-prepared-inputs":
        summary = distribute_pooled_feature_cache_to_inputs(
            pooled_h5ad=Path(args.pooled_h5ad),
            input_dir=Path(args.input_dir),
            prepared_obsm_key=args.prepared_obsm_key,
            sample_glob=args.sample_glob,
            source_file_obs_key=args.source_file_obs_key,
            overwrite=bool(args.overwrite),
        )
    else:  # pragma: no cover - argparse should prevent this.
        parser.error(f"Unknown command: {args.command}")
        return

    _print_json(summary)


if __name__ == "__main__":
    main()
