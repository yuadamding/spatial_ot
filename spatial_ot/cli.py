from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    MultilevelExperimentConfig,
    load_config,
    load_multilevel_config,
    validate_multilevel_config,
)
from .legacy.training import run_experiment
from .legacy.visualization import plot_preprocessed_inputs, plot_result_bundle
from .multilevel import run_multilevel_ot_with_config


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

    multilevel = sub.add_parser(
        "multilevel-ot",
        help="Run shape-normalized cluster-specific semi-relaxed Wasserstein clustering on subregions built from cell-level features.",
    )
    multilevel.add_argument("--config", help="Optional TOML config for the active multilevel OT path.")
    multilevel.add_argument("--input-h5ad", help="Input cell-level H5AD.")
    multilevel.add_argument("--output-dir", help="Output directory for multilevel OT artifacts.")
    multilevel.add_argument("--feature-obsm-key", help="obsm key containing the feature embedding used for the OT ground cost. Prefer PCA or standardized markers; avoid UMAP unless exploratory.")
    multilevel.add_argument("--spatial-x-key", default=None, help="obs key for the x coordinate.")
    multilevel.add_argument("--spatial-y-key", default=None, help="obs key for the y coordinate.")
    multilevel.add_argument("--region-obs-key", help="Optional obs column defining explicit subregion membership. If set, spatial_ot clusters those regions instead of building radius windows.")
    multilevel.add_argument("--spatial-scale", type=float, default=None, help="Multiply spatial coordinates by this value to convert them into microns.")
    multilevel.add_argument("--n-clusters", type=int, default=None, help="Number of subregion clusters.")
    multilevel.add_argument("--atoms-per-cluster", type=int, default=None, help="Number of shared atoms per cluster.")
    multilevel.add_argument("--radius-um", type=float, default=None, help="Subregion radius in microns.")
    multilevel.add_argument("--stride-um", type=float, default=None, help="Subregion center stride in microns.")
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
    multilevel.add_argument("--deep-feature-method", default=None, choices=["none", "autoencoder", "graph_autoencoder"], help="Optional learned feature adapter before OT.")
    multilevel.add_argument("--deep-latent-dim", type=int, default=None, help="Latent dimension for the deep feature adapter.")
    multilevel.add_argument("--deep-hidden-dim", type=int, default=None, help="Hidden width for the deep feature adapter.")
    multilevel.add_argument("--deep-layers", type=int, default=None, help="Number of MLP layers in the deep feature adapter.")
    multilevel.add_argument("--deep-neighbor-k", type=int, default=None, help="k for neighborhood-summary self-supervision in the deep feature adapter.")
    multilevel.add_argument("--deep-radius-um", type=float, default=None, help="Optional radius for neighborhood-summary self-supervision. Overrides kNN if set.")
    multilevel.add_argument("--deep-short-radius-um", type=float, default=None, help="Short-range graph radius for graph_autoencoder.")
    multilevel.add_argument("--deep-mid-radius-um", type=float, default=None, help="Mid-range graph radius for graph_autoencoder.")
    multilevel.add_argument("--deep-graph-layers", type=int, default=None, help="Number of graph message-passing layers per scale for graph_autoencoder.")
    multilevel.add_argument("--deep-validation-context-mode", default=None, choices=["inductive", "transductive"], help="Whether validation context targets are built from held-out cells only or from the full dataset.")
    multilevel.add_argument("--deep-epochs", type=int, default=None, help="Training epochs for the deep feature adapter.")
    multilevel.add_argument("--deep-batch-size", type=int, default=None, help="Batch size for the deep feature adapter.")
    multilevel.add_argument("--deep-lr", type=float, default=None, help="Learning rate for the deep feature adapter.")
    multilevel.add_argument("--deep-weight-decay", type=float, default=None, help="Weight decay for the deep feature adapter.")
    multilevel.add_argument("--deep-validation", default=None, choices=["none", "spatial_block", "sample_holdout"], help="Validation split mode for the deep feature adapter.")
    multilevel.add_argument("--deep-batch-key", default=None, help="Optional obs key used for sample-holdout validation and batch-aware metadata.")
    multilevel.add_argument("--deep-device", default=None, help="Torch device for the deep feature adapter, or 'auto'.")
    multilevel.add_argument("--deep-reconstruction-weight", type=float, default=None, help="Reconstruction loss weight for the deep feature adapter.")
    multilevel.add_argument("--deep-context-weight", type=float, default=None, help="Neighborhood-context prediction loss weight for the deep feature adapter.")
    multilevel.add_argument("--deep-contrastive-weight", type=float, default=None, help="Short-range graph contrastive loss weight for graph_autoencoder.")
    multilevel.add_argument("--deep-variance-weight", type=float, default=None, help="Variance regularization weight for the deep feature adapter.")
    multilevel.add_argument("--deep-decorrelation-weight", type=float, default=None, help="Decorrelation regularization weight for the deep feature adapter.")
    multilevel.add_argument("--deep-output-embedding", default=None, choices=["intrinsic", "context", "joint"], help="Which learned embedding to expose to the OT layer.")
    multilevel.add_argument("--deep-save-model", action=argparse.BooleanOptionalAction, default=None, help="Save the fitted deep feature model under the output directory.")
    multilevel.add_argument("--pretrained-deep-model", default=None, help="Path to a previously saved deep feature model for transform-only runs.")
    multilevel.add_argument("--deep-output-obsm-key", default=None, help="obsm key used to store the learned deep embedding.")
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
    _set_if_not_none(config.paths, "region_obs_key", args.region_obs_key)
    _set_if_not_none(config.paths, "spatial_scale", args.spatial_scale)

    for name in [
        "n_clusters",
        "atoms_per_cluster",
        "radius_um",
        "stride_um",
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
        "deep_save_model": "save_model",
        "pretrained_deep_model": "pretrained_model",
        "deep_output_obsm_key": "output_obsm_key",
    }
    for arg_name, cfg_name in deep_mapping.items():
        _set_if_not_none(config.deep, cfg_name, getattr(args, arg_name))
    return validate_multilevel_config(config)


def main() -> None:
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
    elif args.command == "multilevel-ot":
        config = _resolve_multilevel_config_from_args(args)
        summary = run_multilevel_ot_with_config(config)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
