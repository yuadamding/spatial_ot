from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .multilevel_ot import run_multilevel_ot_on_h5ad
from .training import run_experiment
from .visualization import plot_preprocessed_inputs, plot_result_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Spatial OT teacher-student niche model.")
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
        help="Run cluster-specific shared-support multilevel OT on subregions built from cell-level features.",
    )
    multilevel.add_argument("--input-h5ad", required=True, help="Input cell-level H5AD.")
    multilevel.add_argument("--output-dir", required=True, help="Output directory for multilevel OT artifacts.")
    multilevel.add_argument("--feature-obsm-key", default="X_umap_marker_genes_3d", help="obsm key containing cell-level feature coordinates.")
    multilevel.add_argument("--spatial-x-key", default="cell_x", help="obs key for the x coordinate.")
    multilevel.add_argument("--spatial-y-key", default="cell_y", help="obs key for the y coordinate.")
    multilevel.add_argument("--spatial-scale", type=float, default=1.0, help="Multiply spatial coordinates by this value to convert them into microns.")
    multilevel.add_argument("--n-clusters", type=int, default=8, help="Number of subregion clusters.")
    multilevel.add_argument("--atoms-per-cluster", type=int, default=8, help="Number of shared atoms per cluster.")
    multilevel.add_argument("--radius-um", type=float, default=100.0, help="Subregion radius in microns.")
    multilevel.add_argument("--stride-um", type=float, default=100.0, help="Subregion center stride in microns.")
    multilevel.add_argument("--min-cells", type=int, default=25, help="Minimum cells required to keep a subregion.")
    multilevel.add_argument("--max-subregions", type=int, default=1500, help="Maximum number of subregions to retain after grid construction.")
    multilevel.add_argument("--lambda-x", type=float, default=0.5, help="Weight on canonical spatial coordinates in the OT cost.")
    multilevel.add_argument("--lambda-y", type=float, default=1.0, help="Weight on feature coordinates in the OT cost.")
    multilevel.add_argument("--geometry-eps", type=float, default=0.03, help="Entropic OT regularization for geometry-only normalization into the reference domain.")
    multilevel.add_argument("--ot-eps", type=float, default=0.03, help="Entropic regularization for the semi-relaxed OT clustering objective.")
    multilevel.add_argument("--rho", type=float, default=0.5, help="Relaxation strength for the target marginal in the semi-relaxed OT objective.")
    multilevel.add_argument("--geometry-samples", type=int, default=192, help="Number of uniform geometry samples used to learn each subregion normalizer.")
    multilevel.add_argument("--compressed-support-size", type=int, default=96, help="Maximum number of compressed support points retained per subregion.")
    multilevel.add_argument("--align-iters", type=int, default=4, help="Number of residual similarity-alignment updates per subregion-cluster match.")
    multilevel.add_argument("--no-reflection", action="store_true", help="Disallow reflections in the residual similarity alignment.")
    multilevel.add_argument("--no-scale", action="store_true", help="Disallow scaling in the residual similarity alignment.")
    multilevel.add_argument("--max-iter", type=int, default=10, help="Maximum alternating-optimization iterations.")
    multilevel.add_argument("--tol", type=float, default=1e-4, help="Support-shift tolerance for early stopping.")
    multilevel.add_argument("--seed", type=int, default=1337, help="Random seed.")
    return parser


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
        summary = run_multilevel_ot_on_h5ad(
            input_h5ad=args.input_h5ad,
            output_dir=args.output_dir,
            feature_obsm_key=args.feature_obsm_key,
            spatial_x_key=args.spatial_x_key,
            spatial_y_key=args.spatial_y_key,
            spatial_scale=args.spatial_scale,
            n_clusters=args.n_clusters,
            atoms_per_cluster=args.atoms_per_cluster,
            radius_um=args.radius_um,
            stride_um=args.stride_um,
            min_cells=args.min_cells,
            max_subregions=args.max_subregions,
            lambda_x=args.lambda_x,
            lambda_y=args.lambda_y,
            geometry_eps=args.geometry_eps,
            ot_eps=args.ot_eps,
            rho=args.rho,
            geometry_samples=args.geometry_samples,
            compressed_support_size=args.compressed_support_size,
            align_iters=args.align_iters,
            allow_reflection=not args.no_reflection,
            allow_scale=not args.no_scale,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
