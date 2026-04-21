from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import (
    MultilevelExperimentConfig,
    load_config,
    load_multilevel_config,
    validate_multilevel_config,
)
from .deep import fit_deep_features_on_h5ad, transform_h5ad_with_deep_model
from .legacy.training import run_experiment
from .legacy.visualization import plot_preprocessed_inputs, plot_result_bundle
from .multilevel import plot_sample_niche_maps_from_run_dir, run_multilevel_ot_with_config
from .pooling import pool_h5ads_in_directory


def _configure_runtime_threads_from_env() -> None:
    torch_threads = os.environ.get("SPATIAL_OT_TORCH_NUM_THREADS")
    torch_interop_threads = os.environ.get("SPATIAL_OT_TORCH_NUM_INTEROP_THREADS")
    if torch_threads is None and torch_interop_threads is None:
        return
    try:
        import torch
    except Exception:
        return

    if torch_threads is not None:
        value = int(torch_threads)
        if value > 0:
            torch.set_num_threads(value)
    if torch_interop_threads is not None:
        value = int(torch_interop_threads)
        if value > 0:
            try:
                torch.set_num_interop_threads(value)
            except RuntimeError:
                pass


def _add_deep_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--deep-feature-method", default=None, choices=["none", "autoencoder", "graph_autoencoder"], help="Optional learned feature adapter before OT.")
    parser.add_argument("--deep-latent-dim", type=int, default=None, help="Latent dimension for the deep feature adapter.")
    parser.add_argument("--deep-hidden-dim", type=int, default=None, help="Hidden width for the deep feature adapter.")
    parser.add_argument("--deep-layers", type=int, default=None, help="Number of MLP layers in the deep feature adapter.")
    parser.add_argument("--deep-neighbor-k", type=int, default=None, help="k for neighborhood-summary self-supervision in the deep feature adapter.")
    parser.add_argument("--deep-radius-um", type=float, default=None, help="Optional radius for neighborhood-summary self-supervision. Overrides kNN if set.")
    parser.add_argument("--deep-short-radius-um", type=float, default=None, help="Short-range graph radius for graph_autoencoder.")
    parser.add_argument("--deep-mid-radius-um", type=float, default=None, help="Mid-range graph radius for graph_autoencoder.")
    parser.add_argument("--deep-graph-layers", type=int, default=None, help="Number of graph message-passing layers per scale for graph_autoencoder.")
    parser.add_argument("--deep-graph-max-neighbors", type=int, default=None, help="Maximum neighbors retained per node when building radius graphs for graph_autoencoder.")
    parser.add_argument("--deep-full-batch-max-cells", type=int, default=None, help="Maximum cells allowed for graph_autoencoder full-batch fit/transform. Use 0 to disable the guard.")
    parser.add_argument("--deep-validation-context-mode", default=None, choices=["inductive", "transductive"], help="Whether validation context targets are built from held-out cells only or from the full dataset.")
    parser.add_argument("--deep-epochs", type=int, default=None, help="Training epochs for the deep feature adapter.")
    parser.add_argument("--deep-batch-size", type=int, default=None, help="Batch size for the deep feature adapter.")
    parser.add_argument("--deep-lr", type=float, default=None, help="Learning rate for the deep feature adapter.")
    parser.add_argument("--deep-weight-decay", type=float, default=None, help="Weight decay for the deep feature adapter.")
    parser.add_argument("--deep-validation", default=None, choices=["none", "spatial_block", "sample_holdout"], help="Validation split mode for the deep feature adapter.")
    parser.add_argument("--deep-batch-key", default=None, help="Optional obs key used for sample-holdout validation and batch-aware metadata.")
    parser.add_argument("--deep-device", default=None, help="Torch device for the deep feature adapter, or 'auto'.")
    parser.add_argument("--deep-reconstruction-weight", type=float, default=None, help="Reconstruction loss weight for the deep feature adapter.")
    parser.add_argument("--deep-context-weight", type=float, default=None, help="Neighborhood-context prediction loss weight for the deep feature adapter.")
    parser.add_argument("--deep-contrastive-weight", type=float, default=None, help="Short-range graph contrastive loss weight for graph_autoencoder.")
    parser.add_argument("--deep-variance-weight", type=float, default=None, help="Variance regularization weight for the deep feature adapter.")
    parser.add_argument("--deep-decorrelation-weight", type=float, default=None, help="Decorrelation regularization weight for the deep feature adapter.")
    parser.add_argument("--deep-output-embedding", default=None, choices=["intrinsic", "context", "joint"], help="Which learned embedding to expose to the OT layer.")
    parser.add_argument(
        "--deep-allow-joint-ot-embedding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require explicit opt-in before using deep.output_embedding='joint' as the OT feature view.",
    )
    parser.add_argument("--deep-save-model", action=argparse.BooleanOptionalAction, default=None, help="Save the fitted deep feature model under the output directory.")
    parser.add_argument("--pretrained-deep-model", default=None, help="Path to a previously saved deep feature model for transform-only runs.")
    parser.add_argument("--deep-output-obsm-key", default=None, help="obsm key used to store the learned deep embedding.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run spatial_ot multilevel OT utilities and legacy scaffold commands.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run a staged training experiment.")
    train.add_argument("--config", required=True, help="Path to a TOML config file.")

    plot_inputs = sub.add_parser("plot-inputs", help="Render a 2D overview of the preprocessed input data.")
    plot_inputs.add_argument("--config", required=True, help="Path to a TOML config file.")
    plot_inputs.add_argument("--output", help="Optional output PNG path.")
    plot_inputs.add_argument("--cell-subset", type=int, help="Override cell subset size. Use 0 for all cells.")
    plot_inputs.add_argument("--bin-subset", type=int, help="Override bin subset size. Use 0 for all bins.")

    plot_results = sub.add_parser("plot-results", help="Render a visualization bundle from a finished run directory.")
    plot_results.add_argument("--run-dir", required=True, help="Path to a spatial_ot run directory with saved outputs.")
    plot_results.add_argument("--output-dir", help="Optional output directory for the figures.")

    plot_sample_niches = sub.add_parser(
        "plot-sample-niches",
        help="Render one spatial niche map per sample from a finished multilevel OT run directory.",
    )
    plot_sample_niches.add_argument("--run-dir", required=True, help="Path to a finished multilevel OT run directory.")
    plot_sample_niches.add_argument("--output-dir", help="Optional output directory for the sample niche PNG files.")
    plot_sample_niches.add_argument("--sample-obs-key", default="sample_id", help="obs key storing the sample identifier.")
    plot_sample_niches.add_argument("--source-file-obs-key", default="source_h5ad", help="obs key storing the source H5AD filename.")
    plot_sample_niches.add_argument("--cluster-obs-key", default="mlot_cluster_int", help="obs key storing integer niche labels.")
    plot_sample_niches.add_argument("--cluster-label-obs-key", default="mlot_cluster_id", help="obs key storing display niche labels.")
    plot_sample_niches.add_argument("--cluster-hex-obs-key", default="mlot_cluster_hex", help="obs key storing display colors.")
    plot_sample_niches.add_argument("--plot-spatial-x-key", default=None, help="Preferred obs key for x coordinates in per-sample plots.")
    plot_sample_niches.add_argument("--plot-spatial-y-key", default=None, help="Preferred obs key for y coordinates in per-sample plots.")
    plot_sample_niches.add_argument("--default-sample-id", default="all_cells", help="Sample label used if the requested sample obs key is absent.")
    plot_sample_niches.add_argument("--spatial-scale", type=float, default=None, help="Optional override for the coordinate scale applied before plotting.")

    pool_inputs = sub.add_parser(
        "pool-inputs",
        help="Pool multiple cohort H5AD files into one non-overlapping AnnData for joint latent learning and niche discovery.",
    )
    pool_inputs.add_argument("--input-dir", required=True, help="Directory containing the input H5AD files.")
    pool_inputs.add_argument("--output-h5ad", required=True, help="Output pooled H5AD path.")
    pool_inputs.add_argument(
        "--feature-obsm-key",
        action="append",
        required=True,
        help="Feature key to preserve in the pooled H5AD. Repeat to keep multiple keys. Use 'X' to preserve the full gene matrix.",
    )
    pool_inputs.add_argument("--sample-glob", default="*_cells_marker_genes_umap3d.h5ad", help="Glob used to select input H5AD files within --input-dir.")
    pool_inputs.add_argument("--spatial-x-key", default="cell_x", help="obs key for the original x coordinate.")
    pool_inputs.add_argument("--spatial-y-key", default="cell_y", help="obs key for the original y coordinate.")
    pool_inputs.add_argument("--pooled-spatial-x-key", default="pooled_cell_x", help="obs key used for the pooled, sample-separated x coordinate.")
    pool_inputs.add_argument("--pooled-spatial-y-key", default="pooled_cell_y", help="obs key used for the pooled, sample-separated y coordinate.")
    pool_inputs.add_argument("--sample-obs-key", default="sample_id", help="obs key storing the sample identifier.")
    pool_inputs.add_argument("--source-file-obs-key", default="source_h5ad", help="obs key storing the source H5AD filename.")
    pool_inputs.add_argument("--sample-id-suffix", default="_cells_marker_genes_umap3d", help="Filename suffix stripped when deriving sample IDs.")
    pool_inputs.add_argument("--layout-columns", type=int, default=None, help="Optional number of columns used to tile samples in pooled coordinate space.")
    pool_inputs.add_argument("--layout-gap", type=float, default=None, help="Optional gap between sample tiles in pooled coordinate units.")

    deep_fit = sub.add_parser(
        "deep-fit",
        help="Fit the active deep feature encoder on an H5AD and write an embedded H5AD plus saved model bundle.",
    )
    deep_fit.add_argument("--config", help="Optional TOML config for the active multilevel/deep path.")
    deep_fit.add_argument("--input-h5ad", help="Input cell-level H5AD.")
    deep_fit.add_argument("--output-dir", help="Output directory for deep feature artifacts.")
    deep_fit.add_argument("--feature-obsm-key", help="Feature source used as input to the deep encoder. Accepts an obsm key or 'X' for the full gene matrix.")
    deep_fit.add_argument("--spatial-x-key", default=None, help="obs key for the x coordinate.")
    deep_fit.add_argument("--spatial-y-key", default=None, help="obs key for the y coordinate.")
    deep_fit.add_argument("--spatial-scale", type=float, default=None, help="Multiply spatial coordinates by this value before building deep graph neighborhoods.")
    deep_fit.add_argument("--seed", type=int, default=None, help="Random seed.")
    _add_deep_args(deep_fit)

    deep_transform = sub.add_parser(
        "deep-transform",
        help="Apply a saved deep feature encoder bundle to a new H5AD and write an embedded H5AD.",
    )
    deep_transform.add_argument("--model", required=True, help="Path to a saved deep feature model bundle.")
    deep_transform.add_argument("--input-h5ad", required=True, help="Input cell-level H5AD.")
    deep_transform.add_argument("--output-h5ad", required=True, help="Output H5AD with the transformed embedding.")
    deep_transform.add_argument("--feature-obsm-key", required=True, help="Feature source used as input to the saved deep encoder. Accepts an obsm key or 'X' for the full gene matrix.")
    deep_transform.add_argument("--spatial-x-key", default="cell_x", help="obs key for the x coordinate.")
    deep_transform.add_argument("--spatial-y-key", default="cell_y", help="obs key for the y coordinate.")
    deep_transform.add_argument("--spatial-scale", type=float, default=1.0, help="Multiply spatial coordinates by this value before building deep graph neighborhoods.")
    deep_transform.add_argument("--output-obsm-key", default=None, help="Optional obsm key for the transformed embedding.")
    deep_transform.add_argument("--batch-size", type=int, default=None, help="Optional transform batch size for non-graph encoders.")

    multilevel = sub.add_parser(
        "multilevel-ot",
        help="Run shape-normalized cluster-specific semi-relaxed Wasserstein clustering on subregions built from cell-level features.",
    )
    multilevel.add_argument("--config", help="Optional TOML config for the active multilevel OT path.")
    multilevel.add_argument("--input-h5ad", help="Input cell-level H5AD.")
    multilevel.add_argument("--output-dir", help="Output directory for multilevel OT artifacts.")
    multilevel.add_argument("--feature-obsm-key", help="Feature source used for the OT ground cost. Accepts an obsm key or 'X' for the full gene matrix. Prefer full-gene, PCA, or standardized marker features; avoid UMAP unless exploratory.")
    multilevel.add_argument("--spatial-x-key", default=None, help="obs key for the x coordinate.")
    multilevel.add_argument("--spatial-y-key", default=None, help="obs key for the y coordinate.")
    multilevel.add_argument("--region-obs-key", help="Optional obs column defining explicit subregion membership. If set, spatial_ot clusters those regions instead of building radius windows.")
    multilevel.add_argument("--allow-umap-as-feature", action=argparse.BooleanOptionalAction, default=None, help="Allow UMAP coordinates as the OT feature space for exploratory runs.")
    multilevel.add_argument("--spatial-scale", type=float, default=None, help="Multiply spatial coordinates by this value to convert them into microns.")
    multilevel.add_argument("--n-clusters", type=int, default=None, help="Number of subregion clusters.")
    multilevel.add_argument("--atoms-per-cluster", type=int, default=None, help="Number of shared atoms per cluster.")
    multilevel.add_argument("--radius-um", type=float, default=None, help="Subregion radius in microns.")
    multilevel.add_argument("--stride-um", type=float, default=None, help="Subregion center stride in microns.")
    multilevel.add_argument("--basic-niche-size-um", type=float, default=None, help="Diameter in microns for the smallest basic niche blocks used to compose grid-built subregions. Set to 0 to disable niche composition.")
    multilevel.add_argument("--min-cells", type=int, default=None, help="Minimum cells required to keep a subregion.")
    multilevel.add_argument("--max-subregions", type=int, default=None, help="Maximum number of subregions to retain after grid construction.")
    multilevel.add_argument("--lambda-x", type=float, default=None, help="Weight on canonical spatial coordinates in the OT cost.")
    multilevel.add_argument("--lambda-y", type=float, default=None, help="Weight on feature coordinates in the OT cost.")
    multilevel.add_argument("--geometry-eps", type=float, default=None, help="Entropic OT regularization for geometry-only normalization into the reference domain.")
    multilevel.add_argument("--ot-eps", type=float, default=None, help="Entropic regularization for the semi-relaxed OT clustering objective.")
    multilevel.add_argument("--rho", type=float, default=None, help="Relaxation strength for the target marginal in the semi-relaxed OT objective.")
    multilevel.add_argument("--geometry-samples", type=int, default=None, help="Number of uniform geometry samples used to learn each subregion normalizer.")
    multilevel.add_argument("--compressed-support-size", type=int, default=None, help="Maximum number of compressed support points retained per subregion.")
    multilevel.add_argument("--align-iters", type=int, default=None, help="Number of residual similarity-alignment updates per subregion-cluster match.")
    multilevel.add_argument("--allow-reflection", action=argparse.BooleanOptionalAction, default=None, help="Allow reflections in the residual similarity alignment.")
    multilevel.add_argument("--allow-scale", action=argparse.BooleanOptionalAction, default=None, help="Allow scaling in the residual similarity alignment.")
    multilevel.add_argument("--min-scale", type=float, default=None, help="Lower bound on residual similarity scale when scaling is enabled.")
    multilevel.add_argument("--max-scale", type=float, default=None, help="Upper bound on residual similarity scale when scaling is enabled.")
    multilevel.add_argument("--scale-penalty", type=float, default=None, help="Penalty on residual scale drift from 1.0.")
    multilevel.add_argument("--shift-penalty", type=float, default=None, help="Penalty on residual translation magnitude in canonical space.")
    multilevel.add_argument("--n-init", type=int, default=None, help="Number of random restarts for the nonconvex multilevel OT fit.")
    multilevel.add_argument("--allow-observed-hull-geometry", action=argparse.BooleanOptionalAction, default=None, help="Allow observed-coordinate convex hull fallback when explicit region geometry is unavailable.")
    multilevel.add_argument("--max-iter", type=int, default=None, help="Maximum alternating-optimization iterations.")
    multilevel.add_argument("--tol", type=float, default=None, help="Support-shift tolerance for early stopping.")
    multilevel.add_argument("--seed", type=int, default=None, help="Random seed.")
    multilevel.add_argument("--compute-device", default=None, help="Torch compute device for the active multilevel OT path, or 'auto' to use CUDA when available.")
    _add_deep_args(multilevel)
    return parser


def _set_if_not_none(target, name: str, value) -> None:
    if value is not None:
        setattr(target, name, value)


def _resolve_multilevel_config_from_args(args: argparse.Namespace) -> MultilevelExperimentConfig:
    config = load_multilevel_config(args.config) if args.config else MultilevelExperimentConfig()
    _set_if_not_none(config.paths, "input_h5ad", args.input_h5ad)
    _set_if_not_none(config.paths, "output_dir", args.output_dir)
    _set_if_not_none(config.paths, "feature_obsm_key", args.feature_obsm_key)
    _set_if_not_none(config.paths, "spatial_x_key", args.spatial_x_key)
    _set_if_not_none(config.paths, "spatial_y_key", args.spatial_y_key)
    _set_if_not_none(config.paths, "spatial_scale", args.spatial_scale)
    _set_if_not_none(config.paths, "region_obs_key", args.region_obs_key)
    _set_if_not_none(config.paths, "allow_umap_as_feature", args.allow_umap_as_feature)
    _set_if_not_none(config.paths, "spatial_scale", args.spatial_scale)

    for name in [
        "n_clusters",
        "atoms_per_cluster",
        "radius_um",
        "stride_um",
        "basic_niche_size_um",
        "min_cells",
        "max_subregions",
        "lambda_x",
        "lambda_y",
        "geometry_eps",
        "ot_eps",
        "rho",
        "geometry_samples",
        "compressed_support_size",
        "align_iters",
        "allow_reflection",
        "allow_scale",
        "min_scale",
        "max_scale",
        "scale_penalty",
        "shift_penalty",
        "n_init",
        "allow_observed_hull_geometry",
        "max_iter",
        "tol",
        "seed",
        "compute_device",
    ]:
        attr = "allow_convex_hull_fallback" if name == "allow_observed_hull_geometry" else name
        _set_if_not_none(config.ot, attr, getattr(args, name))
    if config.ot.basic_niche_size_um is not None and float(config.ot.basic_niche_size_um) <= 0:
        config.ot.basic_niche_size_um = None

    deep_mapping = {
        "deep_feature_method": "method",
        "deep_latent_dim": "latent_dim",
        "deep_hidden_dim": "hidden_dim",
        "deep_layers": "layers",
        "deep_neighbor_k": "neighbor_k",
        "deep_radius_um": "radius_um",
        "deep_short_radius_um": "short_radius_um",
        "deep_mid_radius_um": "mid_radius_um",
        "deep_graph_layers": "graph_layers",
        "deep_graph_max_neighbors": "graph_max_neighbors",
        "deep_full_batch_max_cells": "full_batch_max_cells",
        "deep_validation_context_mode": "validation_context_mode",
        "deep_epochs": "epochs",
        "deep_batch_size": "batch_size",
        "deep_lr": "learning_rate",
        "deep_weight_decay": "weight_decay",
        "deep_validation": "validation",
        "deep_batch_key": "batch_key",
        "deep_device": "device",
        "deep_reconstruction_weight": "reconstruction_weight",
        "deep_context_weight": "context_weight",
        "deep_contrastive_weight": "contrastive_weight",
        "deep_variance_weight": "variance_weight",
        "deep_decorrelation_weight": "decorrelation_weight",
        "deep_output_embedding": "output_embedding",
        "deep_allow_joint_ot_embedding": "allow_joint_ot_embedding",
        "deep_save_model": "save_model",
        "pretrained_deep_model": "pretrained_model",
        "deep_output_obsm_key": "output_obsm_key",
    }
    for arg_name, cfg_name in deep_mapping.items():
        _set_if_not_none(config.deep, cfg_name, getattr(args, arg_name))
    return validate_multilevel_config(config)


def _resolve_deep_fit_config_from_args(args: argparse.Namespace) -> tuple[MultilevelExperimentConfig, int]:
    config = load_multilevel_config(args.config) if args.config else MultilevelExperimentConfig()
    _set_if_not_none(config.paths, "input_h5ad", args.input_h5ad)
    _set_if_not_none(config.paths, "output_dir", args.output_dir)
    _set_if_not_none(config.paths, "feature_obsm_key", args.feature_obsm_key)
    _set_if_not_none(config.paths, "spatial_x_key", args.spatial_x_key)
    _set_if_not_none(config.paths, "spatial_y_key", args.spatial_y_key)
    _set_if_not_none(config.paths, "spatial_scale", args.spatial_scale)

    deep_mapping = {
        "deep_feature_method": "method",
        "deep_latent_dim": "latent_dim",
        "deep_hidden_dim": "hidden_dim",
        "deep_layers": "layers",
        "deep_neighbor_k": "neighbor_k",
        "deep_radius_um": "radius_um",
        "deep_short_radius_um": "short_radius_um",
        "deep_mid_radius_um": "mid_radius_um",
        "deep_graph_layers": "graph_layers",
        "deep_graph_max_neighbors": "graph_max_neighbors",
        "deep_full_batch_max_cells": "full_batch_max_cells",
        "deep_validation_context_mode": "validation_context_mode",
        "deep_epochs": "epochs",
        "deep_batch_size": "batch_size",
        "deep_lr": "learning_rate",
        "deep_weight_decay": "weight_decay",
        "deep_validation": "validation",
        "deep_batch_key": "batch_key",
        "deep_device": "device",
        "deep_reconstruction_weight": "reconstruction_weight",
        "deep_context_weight": "context_weight",
        "deep_contrastive_weight": "contrastive_weight",
        "deep_variance_weight": "variance_weight",
        "deep_decorrelation_weight": "decorrelation_weight",
        "deep_output_embedding": "output_embedding",
        "deep_allow_joint_ot_embedding": "allow_joint_ot_embedding",
        "deep_save_model": "save_model",
        "pretrained_deep_model": "pretrained_model",
        "deep_output_obsm_key": "output_obsm_key",
    }
    for arg_name, cfg_name in deep_mapping.items():
        _set_if_not_none(config.deep, cfg_name, getattr(args, arg_name))
    config = validate_multilevel_config(config)
    resolved_seed = int(args.seed if args.seed is not None else config.ot.seed)
    return config, resolved_seed


def main() -> None:
    _configure_runtime_threads_from_env()
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
        summary = run_experiment(config)
        print(json.dumps(summary, indent=2))
    elif args.command == "plot-inputs":
        config = load_config(args.config)
        output_path = plot_preprocessed_inputs(
            config=config,
            cell_subset=args.cell_subset,
            bin_subset=args.bin_subset,
            output_path=args.output,
        )
        print(json.dumps({"output_path": str(output_path)}, indent=2))
    elif args.command == "plot-results":
        manifest = plot_result_bundle(run_dir=Path(args.run_dir), output_dir=Path(args.output_dir) if args.output_dir else None)
        print(json.dumps(manifest, indent=2))
    elif args.command == "plot-sample-niches":
        manifest = plot_sample_niche_maps_from_run_dir(
            run_dir=Path(args.run_dir),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            sample_obs_key=args.sample_obs_key,
            source_file_obs_key=args.source_file_obs_key,
            cluster_obs_key=args.cluster_obs_key,
            cluster_label_obs_key=args.cluster_label_obs_key,
            cluster_hex_obs_key=args.cluster_hex_obs_key,
            plot_spatial_x_key=args.plot_spatial_x_key,
            plot_spatial_y_key=args.plot_spatial_y_key,
            default_sample_id=args.default_sample_id,
            spatial_scale=args.spatial_scale,
        )
        print(json.dumps(manifest, indent=2))
    elif args.command == "pool-inputs":
        summary = pool_h5ads_in_directory(
            input_dir=args.input_dir,
            output_h5ad=args.output_h5ad,
            feature_obsm_keys=list(args.feature_obsm_key),
            sample_glob=args.sample_glob,
            spatial_x_key=args.spatial_x_key,
            spatial_y_key=args.spatial_y_key,
            pooled_spatial_x_key=args.pooled_spatial_x_key,
            pooled_spatial_y_key=args.pooled_spatial_y_key,
            sample_obs_key=args.sample_obs_key,
            source_file_obs_key=args.source_file_obs_key,
            sample_id_suffix=args.sample_id_suffix,
            layout_columns=args.layout_columns,
            layout_gap=args.layout_gap,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "deep-fit":
        config, seed = _resolve_deep_fit_config_from_args(args)
        summary = fit_deep_features_on_h5ad(
            input_h5ad=config.paths.input_h5ad,
            output_dir=config.paths.output_dir,
            feature_obsm_key=config.paths.feature_obsm_key,
            spatial_x_key=config.paths.spatial_x_key,
            spatial_y_key=config.paths.spatial_y_key,
            spatial_scale=config.paths.spatial_scale,
            config=config.deep,
            seed=seed,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "deep-transform":
        summary = transform_h5ad_with_deep_model(
            model_path=args.model,
            input_h5ad=args.input_h5ad,
            output_h5ad=args.output_h5ad,
            feature_obsm_key=args.feature_obsm_key,
            spatial_x_key=args.spatial_x_key,
            spatial_y_key=args.spatial_y_key,
            spatial_scale=args.spatial_scale,
            output_obsm_key=args.output_obsm_key,
            batch_size=args.batch_size,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "multilevel-ot":
        config = _resolve_multilevel_config_from_args(args)
        summary = run_multilevel_ot_with_config(config)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
