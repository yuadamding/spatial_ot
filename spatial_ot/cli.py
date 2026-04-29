from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import (
    MultilevelExperimentConfig,
    load_multilevel_config,
    validate_multilevel_config,
)
from .deep import fit_deep_features_on_h5ad, transform_h5ad_with_deep_model
from .feature_source import prepare_h5ad_feature_cache
from .multilevel import (
    plot_sample_niche_maps_from_run_dir,
    plot_sample_spot_latent_maps_from_run_dir,
    run_multilevel_ot_with_config,
    spatial_niche_validation_report,
    write_concern_resolution_report,
)
from .optimal_search import run_multilevel_optimal_search
from .pooling import distribute_pooled_feature_cache_to_inputs, pool_h5ads_in_directory


def _configure_runtime_threads_from_env() -> None:
    torch_threads = os.environ.get("SPATIAL_OT_TORCH_NUM_THREADS") or os.environ.get(
        "OMP_NUM_THREADS"
    )
    torch_interop_threads = os.environ.get("SPATIAL_OT_TORCH_NUM_INTEROP_THREADS")
    if torch_threads is None and torch_interop_threads is None:
        return
    try:
        from .multilevel.runtime import configure_local_thread_budget
    except Exception:
        return
    threads = (
        int(torch_threads)
        if torch_threads is not None
        else max(int(os.cpu_count() or 1), 1)
    )
    interop_threads = (
        int(torch_interop_threads) if torch_interop_threads is not None else 4
    )
    configure_local_thread_budget(threads, interop_threads)


def _add_deep_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--deep-feature-method",
        default=None,
        choices=["none", "autoencoder", "graph_autoencoder"],
        help="Optional learned feature adapter before OT.",
    )
    parser.add_argument(
        "--deep-latent-dim",
        type=int,
        default=None,
        help="Latent dimension for the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-hidden-dim",
        type=int,
        default=None,
        help="Hidden width for the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-layers",
        type=int,
        default=None,
        help="Number of MLP layers in the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-neighbor-k",
        type=int,
        default=None,
        help="k for neighborhood-summary self-supervision in the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-radius-um",
        type=float,
        default=None,
        help="Optional radius for neighborhood-summary self-supervision. Overrides kNN if set.",
    )
    parser.add_argument(
        "--deep-short-radius-um",
        type=float,
        default=None,
        help="Short-range graph radius for graph_autoencoder.",
    )
    parser.add_argument(
        "--deep-mid-radius-um",
        type=float,
        default=None,
        help="Mid-range graph radius for graph_autoencoder.",
    )
    parser.add_argument(
        "--deep-graph-layers",
        type=int,
        default=None,
        help="Number of graph message-passing layers per scale for graph_autoencoder.",
    )
    parser.add_argument(
        "--deep-graph-aggr",
        default=None,
        choices=["mean"],
        help="Graph aggregation mode for graph_autoencoder.",
    )
    parser.add_argument(
        "--deep-graph-max-neighbors",
        type=int,
        default=None,
        help="Maximum neighbors retained per node when building radius graphs for graph_autoencoder.",
    )
    parser.add_argument(
        "--deep-full-batch-max-cells",
        type=int,
        default=None,
        help="Maximum cells allowed for graph_autoencoder full-batch fit/transform. Use 0 to disable the guard.",
    )
    parser.add_argument(
        "--deep-validation-context-mode",
        default=None,
        choices=["inductive", "transductive"],
        help="Whether validation context targets are built from held-out cells only or from the full dataset.",
    )
    parser.add_argument(
        "--deep-epochs",
        type=int,
        default=None,
        help="Training epochs for the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-batch-size",
        type=int,
        default=None,
        help="Batch size for the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-lr",
        type=float,
        default=None,
        help="Learning rate for the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-weight-decay",
        type=float,
        default=None,
        help="Weight decay for the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-validation",
        default=None,
        choices=["none", "spatial_block", "sample_holdout"],
        help="Validation split mode for the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-batch-key",
        default=None,
        help="Optional obs key used for sample-holdout validation and batch-aware metadata.",
    )
    parser.add_argument(
        "--deep-count-layer",
        default=None,
        help="Count matrix source used only as a denoising reconstruction target. Use 'X' or a layer name such as 'counts'.",
    )
    parser.add_argument(
        "--deep-count-decoder-rank",
        type=int,
        default=None,
        help="Low-rank decoder width used for count-aware reconstruction.",
    )
    parser.add_argument(
        "--deep-count-chunk-size",
        type=int,
        default=None,
        help="Number of genes reconstructed per optimization step when count-aware reconstruction is enabled.",
    )
    parser.add_argument(
        "--deep-count-loss-weight",
        type=float,
        default=None,
        help="Count-reconstruction loss weight for the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-device",
        default=None,
        help="Torch device for the deep feature adapter, or 'auto'.",
    )
    parser.add_argument(
        "--deep-reconstruction-weight",
        type=float,
        default=None,
        help="Reconstruction loss weight for the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-context-weight",
        type=float,
        default=None,
        help="Neighborhood-context prediction loss weight for the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-contrastive-weight",
        type=float,
        default=None,
        help="Short-range graph contrastive loss weight for graph_autoencoder.",
    )
    parser.add_argument(
        "--deep-variance-weight",
        type=float,
        default=None,
        help="Variance regularization weight for the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-decorrelation-weight",
        type=float,
        default=None,
        help="Decorrelation regularization weight for the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-independence-weight",
        type=float,
        default=None,
        help="Cross-embedding independence regularization weight for the deep feature adapter.",
    )
    parser.add_argument(
        "--deep-output-embedding",
        default=None,
        choices=["intrinsic", "context", "joint"],
        help="Which learned embedding to expose to the OT layer.",
    )
    parser.add_argument(
        "--deep-allow-joint-ot-embedding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require explicit opt-in before using deep.output_embedding='joint' as the OT feature view.",
    )
    parser.add_argument(
        "--deep-early-stopping-patience",
        type=int,
        default=None,
        help="Epochs without validation improvement before early stopping.",
    )
    parser.add_argument(
        "--deep-min-delta",
        type=float,
        default=None,
        help="Minimum validation improvement required to reset early stopping patience.",
    )
    parser.add_argument(
        "--deep-restore-best",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restore the best validation checkpoint after deep feature training.",
    )
    parser.add_argument(
        "--deep-save-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save the fitted deep feature model under the output directory.",
    )
    parser.add_argument(
        "--pretrained-deep-model",
        default=None,
        help="Path to a previously saved deep feature model for transform-only runs.",
    )
    parser.add_argument(
        "--deep-output-obsm-key",
        default=None,
        help="obsm key used to store the learned deep embedding.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run spatial_ot multilevel OT utilities."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser(
        "doctor",
        help="Report package / torch / CUDA state and check scripts/run.sh shell defaults against dataclass defaults.",
    )
    doctor.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status when the report status is not 'ok'.",
    )

    plot_sample_niches = sub.add_parser(
        "plot-sample-niches",
        help="Render one spatial niche map per sample from a finished multilevel OT run directory.",
    )
    plot_sample_niches.add_argument(
        "--run-dir",
        required=True,
        help="Path to a finished multilevel OT run directory.",
    )
    plot_sample_niches.add_argument(
        "--output-dir", help="Optional output directory for the sample niche PNG files."
    )
    plot_sample_niches.add_argument(
        "--sample-obs-key",
        default="sample_id",
        help="obs key storing the sample identifier.",
    )
    plot_sample_niches.add_argument(
        "--source-file-obs-key",
        default="source_h5ad",
        help="obs key storing the source H5AD filename.",
    )
    plot_sample_niches.add_argument(
        "--cluster-obs-key",
        default="mlot_cluster_int",
        help="obs key storing integer niche labels.",
    )
    plot_sample_niches.add_argument(
        "--cluster-label-obs-key",
        default="mlot_cluster_id",
        help="obs key storing display niche labels.",
    )
    plot_sample_niches.add_argument(
        "--cluster-hex-obs-key",
        default="mlot_cluster_hex",
        help="obs key storing display colors.",
    )
    plot_sample_niches.add_argument(
        "--spot-latent-npz",
        default=None,
        help="Optional occurrence-level NPZ with fitted subregion IDs. Defaults to spot_level_latent_multilevel_ot.npz under --run-dir.",
    )
    plot_sample_niches.add_argument(
        "--plot-spatial-x-key",
        default=None,
        help="Preferred obs key for x coordinates in per-sample plots.",
    )
    plot_sample_niches.add_argument(
        "--plot-spatial-y-key",
        default=None,
        help="Preferred obs key for y coordinates in per-sample plots.",
    )
    plot_sample_niches.add_argument(
        "--default-sample-id",
        default="all_cells",
        help="Sample label used if the requested sample obs key is absent.",
    )
    plot_sample_niches.add_argument(
        "--spatial-scale",
        type=float,
        default=None,
        help="Optional override for the coordinate scale applied before plotting.",
    )

    plot_sample_spot_latent = sub.add_parser(
        "plot-sample-spot-latent",
        help="Render one spot-level latent field plot per sample from a finished multilevel OT run directory.",
    )
    plot_sample_spot_latent.add_argument(
        "--run-dir",
        required=True,
        help="Path to a finished multilevel OT run directory.",
    )
    plot_sample_spot_latent.add_argument(
        "--output-dir",
        help="Optional output directory for the sample latent PNG files.",
    )
    plot_sample_spot_latent.add_argument(
        "--sample-obs-key",
        default="sample_id",
        help="obs key storing the sample identifier.",
    )
    plot_sample_spot_latent.add_argument(
        "--source-file-obs-key",
        default="source_h5ad",
        help="obs key storing the source H5AD filename.",
    )
    plot_sample_spot_latent.add_argument(
        "--spot-latent-npz",
        default=None,
        help="Optional occurrence-level spot latent NPZ. Defaults to spot_level_latent_multilevel_ot.npz under --run-dir.",
    )
    plot_sample_spot_latent.add_argument(
        "--latent-obsm-key",
        default="mlot_spot_latent_coords",
        help="obsm key storing 2D spot-level latent coordinates.",
    )
    plot_sample_spot_latent.add_argument(
        "--latent-cluster-obs-key",
        default="mlot_spot_latent_cluster_int",
        help="obs key storing integer latent-chart cluster labels.",
    )
    plot_sample_spot_latent.add_argument(
        "--plot-spatial-x-key",
        default=None,
        help="Preferred obs key for x coordinates in per-sample plots.",
    )
    plot_sample_spot_latent.add_argument(
        "--plot-spatial-y-key",
        default=None,
        help="Preferred obs key for y coordinates in per-sample plots.",
    )
    plot_sample_spot_latent.add_argument(
        "--default-sample-id",
        default="all_cells",
        help="Sample label used if the requested sample obs key is absent.",
    )
    plot_sample_spot_latent.add_argument(
        "--spatial-scale",
        type=float,
        default=None,
        help="Optional override for the coordinate scale applied before plotting.",
    )
    plot_sample_spot_latent.add_argument(
        "--max-occurrences-per-sample",
        "--max-occurrences-per-cluster",
        dest="max_occurrences_per_sample",
        type=int,
        default=0,
        help="Maximum occurrence points drawn in each whole-sample latent map. Use 0 for all.",
    )

    validate_run = sub.add_parser(
        "validate-run-concerns",
        help="Write a concern-resolution report for a finished multilevel OT run.",
    )
    validate_run.add_argument(
        "--run-dir", required=True, help="Finished multilevel OT run directory."
    )
    validate_run.add_argument(
        "--coordinate-baseline-run-dir",
        default=None,
        help="Optional coordinate-only baseline run directory used to address boundary-circularity concerns.",
    )
    validate_run.add_argument(
        "--stability-run-dir",
        action="append",
        default=[],
        help="Optional fixed-K stability run directory. Repeat for multiple runs.",
    )
    validate_run.add_argument(
        "--leakage-ablation-run-dir",
        action="append",
        default=[],
        help="Optional shape/density leakage-ablation run directory. Repeat for multiple runs.",
    )
    validate_run.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit with status 1 when blocking primary-claim validation concerns remain.",
    )
    validate_run.add_argument(
        "--output-json", default=None, help="Optional JSON report path."
    )
    validate_run.add_argument(
        "--output-md", default=None, help="Optional Markdown report path."
    )

    spatial_validation = sub.add_parser(
        "spatial-niche-validation",
        help="Compute spatial-organization QC for pooled subregion clusters from a finished run.",
    )
    spatial_validation.add_argument(
        "--run-dir", required=True, help="Finished multilevel OT run directory."
    )
    spatial_validation.add_argument(
        "--output-json", default=None, help="Optional JSON report path."
    )
    spatial_validation.add_argument(
        "--output-cluster-csv",
        default=None,
        help="Optional cluster-statistics CSV path.",
    )
    spatial_validation.add_argument(
        "--sample-obs-key",
        default=None,
        help="Sample column used for within-sample spatial validation. Defaults to summary.json sample_obs_key or sample_id.",
    )
    spatial_validation.add_argument(
        "--allow-missing-sample-key",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow cohort-wide validation when the requested sample column is absent.",
    )
    spatial_validation.add_argument(
        "--max-subregions",
        type=int,
        default=50000,
        help="Maximum subregions sampled for the kNN spatial homophily graph. Use 0 for all.",
    )
    spatial_validation.add_argument(
        "--knn",
        type=int,
        default=6,
        help="Neighbors per subregion for spatial homophily.",
    )
    spatial_validation.add_argument(
        "--permutations",
        type=int,
        default=20,
        help="Within-sample label permutations for the spatial homophily null.",
    )
    spatial_validation.add_argument(
        "--seed", type=int, default=0, help="Random seed for subsampling/permutation."
    )

    select_k = sub.add_parser(
        "select-k",
        help="Run comprehensive K selection on pooled subregion latent embeddings from a finished run.",
    )
    select_k.add_argument(
        "--run-dir", required=True, help="Finished multilevel OT run directory."
    )
    select_k.add_argument(
        "--diagnostics-npz",
        default=None,
        help="Optional diagnostics NPZ containing subregion_latent_embeddings. Defaults to multilevel_ot_candidate_cost_diagnostics.npz under --run-dir.",
    )
    select_k.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for K-selection artifacts. Defaults to comprehensive_k_selection under --run-dir.",
    )
    select_k.add_argument(
        "--candidate-n-clusters",
        default="15-25",
        help="Candidate K values, e.g. '15-25' or '15,16,17'.",
    )
    select_k.add_argument(
        "--seeds",
        default="1337-1356",
        help="KMeans seeds used for multistart stability, e.g. '1337-1356'.",
    )
    select_k.add_argument(
        "--n-init", type=int, default=10, help="KMeans n_init for each seed and K."
    )
    select_k.add_argument(
        "--min-subregions-per-cluster",
        type=int,
        default=50,
        help="Minimum number of subregions per cluster when feasible.",
    )
    select_k.add_argument(
        "--gap-references",
        type=int,
        default=16,
        help="Reference resamples for the gap statistic.",
    )
    select_k.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=12,
        help="Bootstrap stability repeats per K.",
    )
    select_k.add_argument(
        "--bootstrap-fraction",
        type=float,
        default=0.8,
        help="Fraction of subregions sampled per bootstrap repeat.",
    )
    select_k.add_argument(
        "--max-silhouette-subregions",
        type=int,
        default=0,
        help="Maximum subregions used for silhouette. Use 0 for all subregions.",
    )
    select_k.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for sampling/reference draws.",
    )

    pool_inputs = sub.add_parser(
        "pool-inputs",
        help="Pool multiple cohort H5AD files into one non-overlapping AnnData for joint latent learning and niche discovery.",
    )
    pool_inputs.add_argument(
        "--input-dir", required=True, help="Directory containing the input H5AD files."
    )
    pool_inputs.add_argument(
        "--output-h5ad", required=True, help="Output pooled H5AD path."
    )
    pool_inputs.add_argument(
        "--feature-obsm-key",
        action="append",
        required=True,
        help="Feature key to preserve in the pooled H5AD. Repeat to keep multiple keys. Use 'X' to preserve the full gene matrix.",
    )
    pool_inputs.add_argument(
        "--sample-glob",
        default="*_cells_marker_genes_umap3d.h5ad",
        help="Glob used to select input H5AD files within --input-dir.",
    )
    pool_inputs.add_argument(
        "--spatial-x-key",
        default="cell_x",
        help="obs key for the original x coordinate.",
    )
    pool_inputs.add_argument(
        "--spatial-y-key",
        default="cell_y",
        help="obs key for the original y coordinate.",
    )
    pool_inputs.add_argument(
        "--pooled-spatial-x-key",
        default="pooled_cell_x",
        help="obs key used for the pooled, sample-separated x coordinate.",
    )
    pool_inputs.add_argument(
        "--pooled-spatial-y-key",
        default="pooled_cell_y",
        help="obs key used for the pooled, sample-separated y coordinate.",
    )
    pool_inputs.add_argument(
        "--sample-obs-key",
        default="sample_id",
        help="obs key storing the sample identifier.",
    )
    pool_inputs.add_argument(
        "--source-file-obs-key",
        default="source_h5ad",
        help="obs key storing the source H5AD filename.",
    )
    pool_inputs.add_argument(
        "--sample-id-suffix",
        default="_cells_marker_genes_umap3d",
        help="Filename suffix stripped when deriving sample IDs.",
    )
    pool_inputs.add_argument(
        "--layout-columns",
        type=int,
        default=None,
        help="Optional number of columns used to tile samples in pooled coordinate space.",
    )
    pool_inputs.add_argument(
        "--layout-gap",
        type=float,
        default=None,
        help="Optional gap between sample tiles in pooled coordinate units.",
    )

    prepare_inputs = sub.add_parser(
        "prepare-inputs",
        help="Precompute reusable CPU-side feature caches inside an H5AD before GPU-heavy runs.",
    )
    prepare_inputs.add_argument(
        "--input-h5ad", required=True, help="Input H5AD to update or copy."
    )
    prepare_inputs.add_argument(
        "--output-h5ad",
        help="Optional output H5AD path. Defaults to updating --input-h5ad in place.",
    )
    prepare_inputs.add_argument(
        "--feature-obsm-key",
        default="X",
        help="Feature source to prepare. Currently only 'X' is supported for cached full-gene preprocessing.",
    )
    prepare_inputs.add_argument(
        "--output-obsm-key",
        help="Optional obsm key used to store the prepared feature cache.",
    )
    prepare_inputs.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute the prepared feature cache even if a matching one already exists.",
    )

    distribute_inputs = sub.add_parser(
        "distribute-prepared-inputs",
        help="Copy a prepared pooled feature cache back into each source H5AD so single-sample runs can reuse the same cohort-aligned feature space.",
    )
    distribute_inputs.add_argument(
        "--pooled-h5ad",
        required=True,
        help="Pooled H5AD containing the prepared feature cache.",
    )
    distribute_inputs.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the source sample H5AD files.",
    )
    distribute_inputs.add_argument(
        "--prepared-obsm-key",
        required=True,
        help="Prepared pooled obsm key to distribute back into the source files.",
    )
    distribute_inputs.add_argument(
        "--sample-glob",
        default="*_cells_marker_genes_umap3d.h5ad",
        help="Glob used to select source H5AD files within --input-dir.",
    )
    distribute_inputs.add_argument(
        "--sample-obs-key",
        default="sample_id",
        help="obs key storing the sample identifier.",
    )
    distribute_inputs.add_argument(
        "--source-file-obs-key",
        default="source_h5ad",
        help="obs key in the pooled H5AD storing the source H5AD filename.",
    )
    distribute_inputs.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rewrite the distributed feature cache even if a matching pooled-derived cache already exists in a source file.",
    )

    deep_fit = sub.add_parser(
        "deep-fit",
        help="Fit the active deep feature encoder on an H5AD and write an embedded H5AD plus saved model bundle.",
    )
    deep_fit.add_argument(
        "--config", help="Optional TOML config for the active multilevel/deep path."
    )
    deep_fit.add_argument("--input-h5ad", help="Input cell-level H5AD.")
    deep_fit.add_argument(
        "--output-dir", help="Output directory for deep feature artifacts."
    )
    deep_fit.add_argument(
        "--feature-obsm-key",
        help="Feature source used as input to the deep encoder. Accepts an obsm key or 'X' for the full gene matrix.",
    )
    deep_fit.add_argument(
        "--spatial-x-key", default=None, help="obs key for the x coordinate."
    )
    deep_fit.add_argument(
        "--spatial-y-key", default=None, help="obs key for the y coordinate."
    )
    deep_fit.add_argument(
        "--spatial-scale",
        type=float,
        default=None,
        help="Multiply spatial coordinates by this value before building deep graph neighborhoods.",
    )
    deep_fit.add_argument("--seed", type=int, default=None, help="Random seed.")
    _add_deep_args(deep_fit)

    deep_transform = sub.add_parser(
        "deep-transform",
        help="Apply a saved deep feature encoder bundle to a new H5AD and write an embedded H5AD.",
    )
    deep_transform.add_argument(
        "--model", required=True, help="Path to a saved deep feature model bundle."
    )
    deep_transform.add_argument(
        "--input-h5ad", required=True, help="Input cell-level H5AD."
    )
    deep_transform.add_argument(
        "--output-h5ad",
        required=True,
        help="Output H5AD with the transformed embedding.",
    )
    deep_transform.add_argument(
        "--feature-obsm-key",
        required=True,
        help="Feature source used as input to the saved deep encoder. Accepts an obsm key or 'X' for the full gene matrix.",
    )
    deep_transform.add_argument(
        "--spatial-x-key", default="cell_x", help="obs key for the x coordinate."
    )
    deep_transform.add_argument(
        "--spatial-y-key", default="cell_y", help="obs key for the y coordinate."
    )
    deep_transform.add_argument(
        "--spatial-scale",
        type=float,
        default=1.0,
        help="Multiply spatial coordinates by this value before building deep graph neighborhoods.",
    )
    deep_transform.add_argument(
        "--output-obsm-key",
        default=None,
        help="Optional obsm key for the transformed embedding.",
    )
    deep_transform.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional transform batch size for non-graph encoders.",
    )

    multilevel = sub.add_parser(
        "multilevel-ot",
        help="Run shape-normalized cluster-specific semi-relaxed Wasserstein clustering on subregions built from cell-level features.",
    )
    multilevel.add_argument(
        "--config", help="Optional TOML config for the active multilevel OT path."
    )
    multilevel.add_argument("--input-h5ad", help="Input cell-level H5AD.")
    multilevel.add_argument(
        "--output-dir", help="Output directory for multilevel OT artifacts."
    )
    multilevel.add_argument(
        "--feature-obsm-key",
        help="Feature source used for the OT ground cost. Accepts an obsm key or 'X' for the full gene matrix. Prefer full-gene, PCA, or standardized marker features; avoid UMAP unless exploratory.",
    )
    multilevel.add_argument(
        "--spatial-x-key", default=None, help="obs key for the x coordinate."
    )
    multilevel.add_argument(
        "--spatial-y-key", default=None, help="obs key for the y coordinate."
    )
    multilevel.add_argument(
        "--sample-obs-key",
        default=None,
        help="obs key storing sample identifiers used for sample-aware subregion latent shrinkage.",
    )
    multilevel.add_argument(
        "--region-obs-key",
        help="Optional obs column defining explicit mutually exclusive subregion membership. If set, spatial_ot clusters those regions instead of learning data-driven spatial subregions.",
    )
    multilevel.add_argument(
        "--region-geometry-json",
        help="Optional JSON file with explicit polygon/mask geometry keyed by --region-obs-key values.",
    )
    multilevel.add_argument(
        "--allow-umap-as-feature",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow UMAP coordinates as the OT feature space for exploratory runs.",
    )
    multilevel.add_argument(
        "--spatial-scale",
        type=float,
        default=None,
        help="Multiply spatial coordinates by this value to convert them into microns.",
    )
    multilevel.add_argument(
        "--n-clusters", type=int, default=None, help="Number of subregion clusters."
    )
    multilevel.add_argument(
        "--atoms-per-cluster",
        type=int,
        default=None,
        help="Number of shared atoms per cluster.",
    )
    multilevel.add_argument(
        "--radius-um",
        type=float,
        default=None,
        help="Compatibility graph/diagnostic radius in microns; generated data-driven subregions do not use it as a fixed membership radius.",
    )
    multilevel.add_argument(
        "--stride-um",
        type=float,
        default=None,
        help="Subregion center stride in microns.",
    )
    multilevel.add_argument(
        "--basic-niche-size-um",
        type=float,
        default=None,
        help="Target scale in microns for data-driven atomic membership seeds. Set to 0 to disable this scale hint.",
    )
    multilevel.add_argument(
        "--min-cells",
        type=int,
        default=None,
        help="Minimum cells required to keep a subregion.",
    )
    multilevel.add_argument(
        "--max-subregions",
        type=int,
        default=None,
        help="Maximum number of data-driven subregions to retain after minimum-size merging.",
    )
    multilevel.add_argument(
        "--max-subregion-area-um2",
        type=float,
        default=None,
        help="Optional soft QC target for observed point-cloud area in square microns; generated subregions are not hard-split by this value.",
    )
    multilevel.add_argument(
        "--lambda-x",
        type=float,
        default=None,
        help="Weight on canonical spatial coordinates in the OT cost.",
    )
    multilevel.add_argument(
        "--lambda-y",
        type=float,
        default=None,
        help="Weight on feature coordinates in the OT cost.",
    )
    multilevel.add_argument(
        "--geometry-eps",
        type=float,
        default=None,
        help="Entropic OT regularization for geometry-only normalization into the reference domain.",
    )
    multilevel.add_argument(
        "--ot-eps",
        type=float,
        default=None,
        help="Entropic regularization for the semi-relaxed OT clustering objective.",
    )
    multilevel.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Relaxation strength for the target marginal in the semi-relaxed OT objective.",
    )
    multilevel.add_argument(
        "--geometry-samples",
        type=int,
        default=None,
        help="Maximum number of data-driven geometry samples used to learn each subregion normalizer.",
    )
    multilevel.add_argument(
        "--compressed-support-size",
        type=int,
        default=None,
        help="Maximum number of compressed support points retained per subregion.",
    )
    multilevel.add_argument(
        "--align-iters",
        type=int,
        default=None,
        help="Number of residual similarity-alignment updates per subregion-cluster match.",
    )
    multilevel.add_argument(
        "--allow-reflection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow reflections in the residual similarity alignment.",
    )
    multilevel.add_argument(
        "--allow-scale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow scaling in the residual similarity alignment.",
    )
    multilevel.add_argument(
        "--min-scale",
        type=float,
        default=None,
        help="Lower bound on residual similarity scale when scaling is enabled.",
    )
    multilevel.add_argument(
        "--max-scale",
        type=float,
        default=None,
        help="Upper bound on residual similarity scale when scaling is enabled.",
    )
    multilevel.add_argument(
        "--scale-penalty",
        type=float,
        default=None,
        help="Penalty on residual scale drift from 1.0.",
    )
    multilevel.add_argument(
        "--shift-penalty",
        type=float,
        default=None,
        help="Penalty on residual translation magnitude in canonical space.",
    )
    multilevel.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="Number of random restarts for the nonconvex multilevel OT fit.",
    )
    multilevel.add_argument(
        "--overlap-consistency-weight",
        type=float,
        default=None,
        help="Penalty weight encouraging high-overlap subregions with low feature contrast to keep similar cluster assignments.",
    )
    multilevel.add_argument(
        "--overlap-jaccard-min",
        type=float,
        default=None,
        help="Minimum subregion-overlap Jaccard retained by the overlap-consistency graph.",
    )
    multilevel.add_argument(
        "--overlap-contrast-scale",
        type=float,
        default=None,
        help="Contrast scale for gating the overlap-consistency penalty.",
    )
    multilevel.add_argument(
        "--allow-observed-hull-geometry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow observed-coordinate convex hull fallback when explicit region geometry is unavailable.",
    )
    multilevel.add_argument(
        "--subregion-construction-method",
        default=None,
        choices=["data_driven", "deep_segmentation", "joint_refinement"],
        help="Generated subregion construction mode. joint_refinement starts from deep segmentation and then performs constrained cluster-aware boundary refinement.",
    )
    multilevel.add_argument(
        "--subregion-feature-weight",
        type=float,
        default=None,
        help="Opt-in weight for using the OT feature view during data-driven boundary construction. Default 0 keeps coordinate-only boundaries.",
    )
    multilevel.add_argument(
        "--subregion-feature-dims",
        type=int,
        default=None,
        help="Leading feature dimensions used for opt-in data-driven feature-aware boundary construction.",
    )
    multilevel.add_argument(
        "--deep-segmentation-knn",
        type=int,
        default=None,
        help="Spatial kNN degree used by deep graph subregion segmentation.",
    )
    multilevel.add_argument(
        "--deep-segmentation-feature-dims",
        type=int,
        default=None,
        help="Leading learned feature dimensions used for deep graph segmentation affinities.",
    )
    multilevel.add_argument(
        "--deep-segmentation-feature-weight",
        type=float,
        default=None,
        help="Weight on learned-feature edge contrast during deep graph segmentation.",
    )
    multilevel.add_argument(
        "--deep-segmentation-spatial-weight",
        type=float,
        default=None,
        help="Weight on spatial edge length during deep graph segmentation.",
    )
    multilevel.add_argument(
        "--joint-refinement-iters",
        type=int,
        default=None,
        help="Outer boundary-refinement sweeps for joint segmentation-clustering mode.",
    )
    multilevel.add_argument(
        "--joint-refinement-knn",
        type=int,
        default=None,
        help="Spatial kNN degree used to propose adjacent boundary moves in joint refinement.",
    )
    multilevel.add_argument(
        "--joint-refinement-feature-dims",
        type=int,
        default=None,
        help="Leading feature dimensions used to score cluster-coherence boundary moves.",
    )
    multilevel.add_argument(
        "--joint-refinement-cluster-weight",
        type=float,
        default=None,
        help="Weight on pooled-latent cluster-prototype coherence during joint refinement.",
    )
    multilevel.add_argument(
        "--joint-refinement-spatial-weight",
        type=float,
        default=None,
        help="Weight discouraging boundary moves that worsen spatial compactness.",
    )
    multilevel.add_argument(
        "--joint-refinement-cut-weight",
        type=float,
        default=None,
        help="Weight discouraging boundary moves that increase local kNN label cuts.",
    )
    multilevel.add_argument(
        "--joint-refinement-max-move-fraction",
        type=float,
        default=None,
        help="Maximum fraction of cells that may move during each joint-refinement sweep.",
    )
    multilevel.add_argument(
        "--subregion-clustering-method",
        default=None,
        choices=[
            "pooled_subregion_latent",
            "heterogeneity_descriptor_niche",
            "heterogeneity_ot_niche",
            "ot_dictionary",
        ],
        help="How fitted subregions receive niche labels. heterogeneity_descriptor_niche clusters block-normalized internal spatial-cell-state motif descriptors; heterogeneity_ot_niche is a legacy alias until true fused-OT/FGW modes are implemented; pooled_subregion_latent is the composition/distribution-summary baseline; ot_dictionary keeps the historical OT-dictionary assignment.",
    )
    multilevel.add_argument(
        "--subregion-latent-embedding-mode",
        default=None,
        choices=[
            "mean_std",
            "mean_std_shrunk",
            "mean_std_skew_count",
            "mean_std_quantile",
            "codebook_histogram",
            "mean_std_codebook",
        ],
        help="Raw member-cell distribution summary clustered by pooled_subregion_latent. The default shrinkage mode reduces low-cell-count variance noise without adding spatial information.",
    )
    multilevel.add_argument(
        "--subregion-latent-shrinkage-tau",
        type=float,
        default=None,
        help="Pseudo-count strength for reliability shrinkage in subregion latent summaries.",
    )
    multilevel.add_argument(
        "--subregion-latent-heterogeneity-weight",
        type=float,
        default=None,
        help="Weight applied to within-subregion heterogeneity blocks such as shrunk standard deviations.",
    )
    multilevel.add_argument(
        "--subregion-latent-sample-prior-weight",
        type=float,
        default=None,
        help="Weight on the matched sample prior inside hierarchical sample/cohort shrinkage.",
    )
    multilevel.add_argument(
        "--subregion-latent-codebook-size",
        type=int,
        default=None,
        help="Cell-state codebook size for soft codebook subregion latent modes.",
    )
    multilevel.add_argument(
        "--subregion-latent-codebook-sample-size",
        type=int,
        default=None,
        help="Maximum member cells sampled to fit the whitened codebook.",
    )
    multilevel.add_argument(
        "--heterogeneity-composition-weight",
        type=float,
        default=None,
        help="Descriptor block weight for soft cell-state composition.",
    )
    multilevel.add_argument(
        "--heterogeneity-diversity-weight",
        type=float,
        default=None,
        help="Descriptor block weight for diversity and multimodality statistics.",
    )
    multilevel.add_argument(
        "--heterogeneity-spatial-field-weight",
        type=float,
        default=None,
        help="Descriptor block weight for canonical spatial-state density fields.",
    )
    multilevel.add_argument(
        "--heterogeneity-pair-cooccurrence-weight",
        type=float,
        default=None,
        help="Descriptor block weight for state-pair co-occurrence enrichment.",
    )
    multilevel.add_argument(
        "--heterogeneity-pair-distance-bins",
        default=None,
        help="Comma-separated canonical distance bin upper bounds for pair motifs.",
    )
    multilevel.add_argument(
        "--heterogeneity-pair-graph-mode",
        default=None,
        choices=["all_pairs", "knn", "radius"],
        help="Pairs used for co-occurrence motifs: all_pairs, local kNN, or radius graph.",
    )
    multilevel.add_argument(
        "--heterogeneity-pair-graph-k",
        type=int,
        default=None,
        help="k for heterogeneity pair_graph_mode=knn.",
    )
    multilevel.add_argument(
        "--heterogeneity-pair-graph-radius",
        type=float,
        default=None,
        help="Canonical-coordinate radius for heterogeneity pair_graph_mode=radius.",
    )
    multilevel.add_argument(
        "--heterogeneity-pair-bin-normalization",
        default=None,
        choices=["per_bin", "global"],
        help="Whether pair co-occurrence tensors are normalized per distance bin or globally.",
    )
    multilevel.add_argument(
        "--shape-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run shape-leakage random-forest diagnostics after fitting.",
    )
    multilevel.add_argument(
        "--shape-leakage-permutations",
        type=int,
        default=None,
        help="Number of permutations used for the shape-leakage baseline.",
    )
    multilevel.add_argument(
        "--compute-spot-latent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute and save occurrence-level OT atom-barycentric spot latent diagnostic charts.",
    )
    multilevel.add_argument(
        "--auto-n-clusters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Select the number of subregion clusters automatically from pilot OT-landmark geometry.",
    )
    multilevel.add_argument(
        "--candidate-n-clusters",
        default=None,
        help="Candidate K values for --auto-n-clusters, e.g. '15-25' or '15,16,17,18'.",
    )
    multilevel.add_argument(
        "--auto-k-max-score-subregions",
        type=int,
        default=None,
        help="Maximum sampled subregions used to score automatic K selection; 0 scores all.",
    )
    multilevel.add_argument(
        "--auto-k-gap-references",
        type=int,
        default=None,
        help="Reference resamples used by the automatic-K gap statistic.",
    )
    multilevel.add_argument(
        "--auto-k-mds-components",
        type=int,
        default=None,
        help="MDS embedding dimensions used by CH/DB/Gap automatic-K scores.",
    )
    multilevel.add_argument(
        "--auto-k-pilot-n-init",
        type=int,
        default=None,
        help="Restart count for the automatic-K pilot fit.",
    )
    multilevel.add_argument(
        "--auto-k-pilot-max-iter",
        type=int,
        default=None,
        help="Maximum iterations for the automatic-K pilot fit.",
    )
    multilevel.add_argument(
        "--min-subregions-per-cluster",
        type=int,
        default=None,
        help="Minimum number of subregions assigned to each subregion cluster when feasible.",
    )
    multilevel.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Maximum alternating-optimization iterations.",
    )
    multilevel.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Support-shift tolerance for early stopping.",
    )
    multilevel.add_argument("--seed", type=int, default=None, help="Random seed.")
    multilevel.add_argument(
        "--compute-device",
        default=None,
        help="Torch compute device for the active multilevel OT path, or 'auto' to use CUDA when available.",
    )
    _add_deep_args(multilevel)

    optimal_search = sub.add_parser(
        "optimal-search",
        help="Run a staged multilevel OT parameter search using summary-level compactness, boundary, and OT-reliability diagnostics.",
    )
    optimal_search.add_argument(
        "--config", help="Optional TOML config used as the search baseline."
    )
    optimal_search.add_argument("--input-h5ad", help="Input cell-level H5AD.")
    optimal_search.add_argument(
        "--output-dir", help="Output directory for search artifacts and candidate runs."
    )
    optimal_search.add_argument(
        "--feature-obsm-key",
        help="Feature source used for the OT ground cost. Defaults to the pooled prepared feature cache when omitted.",
    )
    optimal_search.add_argument(
        "--spatial-x-key", default=None, help="obs key for the x coordinate."
    )
    optimal_search.add_argument(
        "--spatial-y-key", default=None, help="obs key for the y coordinate."
    )
    optimal_search.add_argument(
        "--region-obs-key",
        help="Optional obs column defining explicit subregion membership.",
    )
    optimal_search.add_argument(
        "--region-geometry-json",
        help="Optional JSON file with explicit polygon/mask geometry keyed by --region-obs-key values.",
    )
    optimal_search.add_argument(
        "--allow-umap-as-feature",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow UMAP coordinates as the OT feature space for exploratory runs.",
    )
    optimal_search.add_argument(
        "--spatial-scale",
        type=float,
        default=None,
        help="Multiply spatial coordinates by this value to convert them into microns.",
    )
    optimal_search.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Baseline number of subregion clusters used to seed the search.",
    )
    optimal_search.add_argument(
        "--atoms-per-cluster",
        type=int,
        default=None,
        help="Number of shared atoms per cluster.",
    )
    optimal_search.add_argument(
        "--radius-um",
        type=float,
        default=None,
        help="Compatibility graph/diagnostic radius in microns; generated data-driven subregions do not use it as a fixed membership radius.",
    )
    optimal_search.add_argument(
        "--stride-um",
        type=float,
        default=None,
        help="Baseline subregion center stride in microns.",
    )
    optimal_search.add_argument(
        "--basic-niche-size-um",
        type=float,
        default=None,
        help="Target scale in microns for data-driven atomic membership seeds.",
    )
    optimal_search.add_argument(
        "--min-cells",
        type=int,
        default=None,
        help="Minimum cells required to keep a subregion.",
    )
    optimal_search.add_argument(
        "--max-subregions",
        type=int,
        default=None,
        help="Maximum number of data-driven subregions retained after minimum-size merging.",
    )
    optimal_search.add_argument(
        "--max-subregion-area-um2",
        type=float,
        default=None,
        help="Optional soft QC target for observed point-cloud area in square microns; generated subregions are not hard-split by this value.",
    )
    optimal_search.add_argument(
        "--lambda-x",
        type=float,
        default=None,
        help="Weight on canonical spatial coordinates in the OT cost.",
    )
    optimal_search.add_argument(
        "--lambda-y",
        type=float,
        default=None,
        help="Weight on feature coordinates in the OT cost.",
    )
    optimal_search.add_argument(
        "--geometry-eps",
        type=float,
        default=None,
        help="Entropic OT regularization for geometry-only normalization.",
    )
    optimal_search.add_argument(
        "--ot-eps",
        type=float,
        default=None,
        help="Entropic regularization for the semi-relaxed OT clustering objective.",
    )
    optimal_search.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Relaxation strength for the target marginal.",
    )
    optimal_search.add_argument(
        "--geometry-samples",
        type=int,
        default=None,
        help="Uniform geometry samples used to learn each subregion normalizer.",
    )
    optimal_search.add_argument(
        "--compressed-support-size",
        type=int,
        default=None,
        help="Maximum compressed support points retained per subregion.",
    )
    optimal_search.add_argument(
        "--align-iters",
        type=int,
        default=None,
        help="Residual similarity-alignment updates per subregion-cluster match.",
    )
    optimal_search.add_argument(
        "--allow-reflection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow reflections in the residual similarity alignment.",
    )
    optimal_search.add_argument(
        "--allow-scale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow scaling in the residual similarity alignment.",
    )
    optimal_search.add_argument(
        "--min-scale",
        type=float,
        default=None,
        help="Lower bound on residual similarity scale when scaling is enabled.",
    )
    optimal_search.add_argument(
        "--max-scale",
        type=float,
        default=None,
        help="Upper bound on residual similarity scale when scaling is enabled.",
    )
    optimal_search.add_argument(
        "--scale-penalty",
        type=float,
        default=None,
        help="Penalty on residual scale drift from 1.0.",
    )
    optimal_search.add_argument(
        "--shift-penalty",
        type=float,
        default=None,
        help="Penalty on residual translation magnitude in canonical space.",
    )
    optimal_search.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="Random restarts for the final confirmatory fits.",
    )
    optimal_search.add_argument(
        "--overlap-consistency-weight",
        type=float,
        default=None,
        help="Penalty weight encouraging high-overlap subregions with low feature contrast to keep similar cluster assignments.",
    )
    optimal_search.add_argument(
        "--overlap-jaccard-min",
        type=float,
        default=None,
        help="Minimum subregion-overlap Jaccard retained by the overlap-consistency graph.",
    )
    optimal_search.add_argument(
        "--overlap-contrast-scale",
        type=float,
        default=None,
        help="Contrast scale for gating the overlap-consistency penalty.",
    )
    optimal_search.add_argument(
        "--allow-observed-hull-geometry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow observed-coordinate convex hull fallback when explicit region geometry is unavailable.",
    )
    optimal_search.add_argument(
        "--subregion-construction-method",
        default=None,
        choices=["data_driven", "deep_segmentation", "joint_refinement"],
        help="Generated subregion construction mode. joint_refinement starts from deep segmentation and then performs constrained cluster-aware boundary refinement.",
    )
    optimal_search.add_argument(
        "--subregion-feature-weight",
        type=float,
        default=None,
        help="Opt-in weight for using the OT feature view during data-driven boundary construction. Default 0 keeps coordinate-only boundaries.",
    )
    optimal_search.add_argument(
        "--subregion-feature-dims",
        type=int,
        default=None,
        help="Leading feature dimensions used for opt-in data-driven feature-aware boundary construction.",
    )
    optimal_search.add_argument(
        "--deep-segmentation-knn",
        type=int,
        default=None,
        help="Spatial kNN degree used by deep graph subregion segmentation.",
    )
    optimal_search.add_argument(
        "--deep-segmentation-feature-dims",
        type=int,
        default=None,
        help="Leading learned feature dimensions used for deep graph segmentation affinities.",
    )
    optimal_search.add_argument(
        "--deep-segmentation-feature-weight",
        type=float,
        default=None,
        help="Weight on learned-feature edge contrast during deep graph segmentation.",
    )
    optimal_search.add_argument(
        "--deep-segmentation-spatial-weight",
        type=float,
        default=None,
        help="Weight on spatial edge length during deep graph segmentation.",
    )
    optimal_search.add_argument(
        "--joint-refinement-iters",
        type=int,
        default=None,
        help="Outer boundary-refinement sweeps for joint segmentation-clustering mode.",
    )
    optimal_search.add_argument(
        "--joint-refinement-knn",
        type=int,
        default=None,
        help="Spatial kNN degree used to propose adjacent boundary moves in joint refinement.",
    )
    optimal_search.add_argument(
        "--joint-refinement-feature-dims",
        type=int,
        default=None,
        help="Leading feature dimensions used to score cluster-coherence boundary moves.",
    )
    optimal_search.add_argument(
        "--joint-refinement-cluster-weight",
        type=float,
        default=None,
        help="Weight on pooled-latent cluster-prototype coherence during joint refinement.",
    )
    optimal_search.add_argument(
        "--joint-refinement-spatial-weight",
        type=float,
        default=None,
        help="Weight discouraging boundary moves that worsen spatial compactness.",
    )
    optimal_search.add_argument(
        "--joint-refinement-cut-weight",
        type=float,
        default=None,
        help="Weight discouraging boundary moves that increase local kNN label cuts.",
    )
    optimal_search.add_argument(
        "--joint-refinement-max-move-fraction",
        type=float,
        default=None,
        help="Maximum fraction of cells that may move during each joint-refinement sweep.",
    )
    optimal_search.add_argument(
        "--subregion-clustering-method",
        default=None,
        choices=[
            "pooled_subregion_latent",
            "heterogeneity_descriptor_niche",
            "heterogeneity_ot_niche",
            "ot_dictionary",
        ],
        help="How fitted subregions receive niche labels. heterogeneity_descriptor_niche clusters block-normalized internal spatial-cell-state motif descriptors; heterogeneity_ot_niche is a legacy alias until true fused-OT/FGW modes are implemented; pooled_subregion_latent is the composition/distribution-summary baseline; ot_dictionary keeps the historical OT-dictionary assignment.",
    )
    optimal_search.add_argument(
        "--subregion-latent-embedding-mode",
        default=None,
        choices=[
            "mean_std",
            "mean_std_shrunk",
            "mean_std_skew_count",
            "mean_std_quantile",
            "codebook_histogram",
            "mean_std_codebook",
        ],
        help="Raw member-cell distribution summary clustered by pooled_subregion_latent.",
    )
    optimal_search.add_argument(
        "--subregion-latent-shrinkage-tau",
        type=float,
        default=None,
        help="Pseudo-count strength for reliability shrinkage in subregion latent summaries.",
    )
    optimal_search.add_argument(
        "--subregion-latent-heterogeneity-weight",
        type=float,
        default=None,
        help="Weight applied to within-subregion heterogeneity blocks such as shrunk standard deviations.",
    )
    optimal_search.add_argument(
        "--subregion-latent-sample-prior-weight",
        type=float,
        default=None,
        help="Weight on the matched sample prior inside hierarchical sample/cohort shrinkage.",
    )
    optimal_search.add_argument(
        "--subregion-latent-codebook-size",
        type=int,
        default=None,
        help="Cell-state codebook size for soft codebook subregion latent modes.",
    )
    optimal_search.add_argument(
        "--subregion-latent-codebook-sample-size",
        type=int,
        default=None,
        help="Maximum member cells sampled to fit the whitened codebook.",
    )
    optimal_search.add_argument(
        "--heterogeneity-composition-weight",
        type=float,
        default=None,
        help="Descriptor block weight for soft cell-state composition.",
    )
    optimal_search.add_argument(
        "--heterogeneity-diversity-weight",
        type=float,
        default=None,
        help="Descriptor block weight for diversity and multimodality statistics.",
    )
    optimal_search.add_argument(
        "--heterogeneity-spatial-field-weight",
        type=float,
        default=None,
        help="Descriptor block weight for canonical spatial-state density fields.",
    )
    optimal_search.add_argument(
        "--heterogeneity-pair-cooccurrence-weight",
        type=float,
        default=None,
        help="Descriptor block weight for state-pair co-occurrence enrichment.",
    )
    optimal_search.add_argument(
        "--heterogeneity-pair-distance-bins",
        default=None,
        help="Comma-separated canonical distance bin upper bounds for pair motifs.",
    )
    optimal_search.add_argument(
        "--heterogeneity-pair-graph-mode",
        default=None,
        choices=["all_pairs", "knn", "radius"],
        help="Pairs used for co-occurrence motifs: all_pairs, local kNN, or radius graph.",
    )
    optimal_search.add_argument(
        "--heterogeneity-pair-graph-k",
        type=int,
        default=None,
        help="k for heterogeneity pair_graph_mode=knn.",
    )
    optimal_search.add_argument(
        "--heterogeneity-pair-graph-radius",
        type=float,
        default=None,
        help="Canonical-coordinate radius for heterogeneity pair_graph_mode=radius.",
    )
    optimal_search.add_argument(
        "--heterogeneity-pair-bin-normalization",
        default=None,
        choices=["per_bin", "global"],
        help="Whether pair co-occurrence tensors are normalized per distance bin or globally.",
    )
    optimal_search.add_argument(
        "--shape-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run shape-leakage random-forest diagnostics after fitting.",
    )
    optimal_search.add_argument(
        "--shape-leakage-permutations",
        type=int,
        default=None,
        help="Number of permutations used for the shape-leakage baseline.",
    )
    optimal_search.add_argument(
        "--compute-spot-latent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute and save occurrence-level OT atom-barycentric spot latent diagnostic charts.",
    )
    optimal_search.add_argument(
        "--auto-n-clusters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Select the number of subregion clusters automatically from pilot OT-landmark geometry.",
    )
    optimal_search.add_argument(
        "--candidate-n-clusters",
        default=None,
        help="Candidate K values for --auto-n-clusters, e.g. '15-25' or '15,16,17,18'.",
    )
    optimal_search.add_argument(
        "--auto-k-max-score-subregions",
        type=int,
        default=None,
        help="Maximum sampled subregions used to score automatic K selection; 0 scores all.",
    )
    optimal_search.add_argument(
        "--auto-k-gap-references",
        type=int,
        default=None,
        help="Reference resamples used by the automatic-K gap statistic.",
    )
    optimal_search.add_argument(
        "--auto-k-mds-components",
        type=int,
        default=None,
        help="MDS embedding dimensions used by CH/DB/Gap automatic-K scores.",
    )
    optimal_search.add_argument(
        "--auto-k-pilot-n-init",
        type=int,
        default=None,
        help="Restart count for the automatic-K pilot fit.",
    )
    optimal_search.add_argument(
        "--auto-k-pilot-max-iter",
        type=int,
        default=None,
        help="Maximum iterations for the automatic-K pilot fit.",
    )
    optimal_search.add_argument(
        "--min-subregions-per-cluster",
        type=int,
        default=None,
        help="Minimum number of subregions assigned to each subregion cluster when feasible.",
    )
    optimal_search.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Maximum alternating-optimization iterations for final confirmatory fits.",
    )
    optimal_search.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Support-shift tolerance for early stopping.",
    )
    optimal_search.add_argument("--seed", type=int, default=None, help="Random seed.")
    optimal_search.add_argument(
        "--compute-device",
        default=None,
        help="Torch compute device for the active multilevel OT path, or 'auto' to use CUDA when available.",
    )
    optimal_search.add_argument(
        "--time-budget-hours",
        type=float,
        default=20.0,
        help="Maximum wall-clock budget for the staged search.",
    )
    optimal_search.add_argument(
        "--keep-top-k",
        type=int,
        default=3,
        help="Number of highest-scoring candidate directories that keep full artifacts during the search.",
    )
    optimal_search.add_argument(
        "--sample-obs-key",
        default="sample_id",
        help="obs key used when writing one best-run niche map per sample.",
    )
    optimal_search.add_argument(
        "--source-file-obs-key",
        default="source_h5ad",
        help="obs key tracking the source H5AD used for pooled runs.",
    )
    optimal_search.add_argument(
        "--plot-spatial-x-key",
        default="cell_x",
        help="obs x key used for best-run per-sample niche plots.",
    )
    optimal_search.add_argument(
        "--plot-spatial-y-key",
        default="cell_y",
        help="obs y key used for best-run per-sample niche plots.",
    )
    optimal_search.add_argument(
        "--default-sample-id",
        default="pooled_cohort",
        help="Fallback sample id for single-sample plotting.",
    )
    optimal_search.add_argument(
        "--plot-best-sample-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render one spatial niche PNG per sample for the highest-scoring run.",
    )
    _add_deep_args(optimal_search)
    return parser


def _set_if_not_none(target, name: str, value) -> None:
    if value is not None:
        setattr(target, name, value)


def _resolve_multilevel_config_from_args(
    args: argparse.Namespace,
) -> MultilevelExperimentConfig:
    config = (
        load_multilevel_config(args.config)
        if args.config
        else MultilevelExperimentConfig()
    )
    _set_if_not_none(config.paths, "input_h5ad", args.input_h5ad)
    _set_if_not_none(config.paths, "output_dir", args.output_dir)
    _set_if_not_none(config.paths, "feature_obsm_key", args.feature_obsm_key)
    _set_if_not_none(config.paths, "spatial_x_key", args.spatial_x_key)
    _set_if_not_none(config.paths, "spatial_y_key", args.spatial_y_key)
    _set_if_not_none(
        config.paths, "sample_obs_key", getattr(args, "sample_obs_key", None)
    )
    _set_if_not_none(config.paths, "spatial_scale", args.spatial_scale)
    _set_if_not_none(config.paths, "region_obs_key", args.region_obs_key)
    _set_if_not_none(config.paths, "region_geometry_json", args.region_geometry_json)
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
        "max_subregion_area_um2",
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
        "overlap_consistency_weight",
        "overlap_jaccard_min",
        "overlap_contrast_scale",
        "allow_observed_hull_geometry",
        "subregion_construction_method",
        "subregion_feature_weight",
        "subregion_feature_dims",
        "deep_segmentation_knn",
        "deep_segmentation_feature_dims",
        "deep_segmentation_feature_weight",
        "deep_segmentation_spatial_weight",
        "joint_refinement_iters",
        "joint_refinement_knn",
        "joint_refinement_feature_dims",
        "joint_refinement_cluster_weight",
        "joint_refinement_spatial_weight",
        "joint_refinement_cut_weight",
        "joint_refinement_max_move_fraction",
        "subregion_clustering_method",
        "subregion_latent_embedding_mode",
        "subregion_latent_shrinkage_tau",
        "subregion_latent_heterogeneity_weight",
        "subregion_latent_sample_prior_weight",
        "subregion_latent_codebook_size",
        "subregion_latent_codebook_sample_size",
        "heterogeneity_composition_weight",
        "heterogeneity_diversity_weight",
        "heterogeneity_spatial_field_weight",
        "heterogeneity_pair_cooccurrence_weight",
        "heterogeneity_pair_distance_bins",
        "heterogeneity_pair_graph_mode",
        "heterogeneity_pair_graph_k",
        "heterogeneity_pair_graph_radius",
        "heterogeneity_pair_bin_normalization",
        "shape_diagnostics",
        "shape_leakage_permutations",
        "compute_spot_latent",
        "auto_n_clusters",
        "candidate_n_clusters",
        "auto_k_max_score_subregions",
        "auto_k_gap_references",
        "auto_k_mds_components",
        "auto_k_pilot_n_init",
        "auto_k_pilot_max_iter",
        "min_subregions_per_cluster",
        "max_iter",
        "tol",
        "seed",
        "compute_device",
    ]:
        attr = (
            "allow_convex_hull_fallback"
            if name == "allow_observed_hull_geometry"
            else name
        )
        _set_if_not_none(config.ot, attr, getattr(args, name))
    if (
        config.ot.basic_niche_size_um is not None
        and float(config.ot.basic_niche_size_um) <= 0
    ):
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
        "deep_graph_aggr": "graph_aggr",
        "deep_graph_max_neighbors": "graph_max_neighbors",
        "deep_full_batch_max_cells": "full_batch_max_cells",
        "deep_validation_context_mode": "validation_context_mode",
        "deep_epochs": "epochs",
        "deep_batch_size": "batch_size",
        "deep_lr": "learning_rate",
        "deep_weight_decay": "weight_decay",
        "deep_validation": "validation",
        "deep_batch_key": "batch_key",
        "deep_count_layer": "count_layer",
        "deep_count_decoder_rank": "count_decoder_rank",
        "deep_count_chunk_size": "count_chunk_size",
        "deep_count_loss_weight": "count_loss_weight",
        "deep_device": "device",
        "deep_reconstruction_weight": "reconstruction_weight",
        "deep_context_weight": "context_weight",
        "deep_contrastive_weight": "contrastive_weight",
        "deep_variance_weight": "variance_weight",
        "deep_decorrelation_weight": "decorrelation_weight",
        "deep_independence_weight": "independence_weight",
        "deep_output_embedding": "output_embedding",
        "deep_allow_joint_ot_embedding": "allow_joint_ot_embedding",
        "deep_early_stopping_patience": "early_stopping_patience",
        "deep_min_delta": "min_delta",
        "deep_restore_best": "restore_best",
        "deep_save_model": "save_model",
        "pretrained_deep_model": "pretrained_model",
        "deep_output_obsm_key": "output_obsm_key",
    }
    for arg_name, cfg_name in deep_mapping.items():
        _set_if_not_none(config.deep, cfg_name, getattr(args, arg_name))
    return validate_multilevel_config(config)


def _resolve_deep_fit_config_from_args(
    args: argparse.Namespace,
) -> tuple[MultilevelExperimentConfig, int]:
    config = (
        load_multilevel_config(args.config)
        if args.config
        else MultilevelExperimentConfig()
    )
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
        "deep_graph_aggr": "graph_aggr",
        "deep_graph_max_neighbors": "graph_max_neighbors",
        "deep_full_batch_max_cells": "full_batch_max_cells",
        "deep_validation_context_mode": "validation_context_mode",
        "deep_epochs": "epochs",
        "deep_batch_size": "batch_size",
        "deep_lr": "learning_rate",
        "deep_weight_decay": "weight_decay",
        "deep_validation": "validation",
        "deep_batch_key": "batch_key",
        "deep_count_layer": "count_layer",
        "deep_count_decoder_rank": "count_decoder_rank",
        "deep_count_chunk_size": "count_chunk_size",
        "deep_count_loss_weight": "count_loss_weight",
        "deep_device": "device",
        "deep_reconstruction_weight": "reconstruction_weight",
        "deep_context_weight": "context_weight",
        "deep_contrastive_weight": "contrastive_weight",
        "deep_variance_weight": "variance_weight",
        "deep_decorrelation_weight": "decorrelation_weight",
        "deep_independence_weight": "independence_weight",
        "deep_output_embedding": "output_embedding",
        "deep_allow_joint_ot_embedding": "allow_joint_ot_embedding",
        "deep_early_stopping_patience": "early_stopping_patience",
        "deep_min_delta": "min_delta",
        "deep_restore_best": "restore_best",
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
    if args.command == "doctor":
        from .doctor import run_doctor

        report = run_doctor(verbose=True)
        if args.strict and report.get("status") != "ok":
            raise SystemExit(1)
        return
    if args.command == "plot-sample-niches":
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
            spot_latent_npz=args.spot_latent_npz,
        )
        print(json.dumps(manifest, indent=2))
    elif args.command == "plot-sample-spot-latent":
        manifest = plot_sample_spot_latent_maps_from_run_dir(
            run_dir=Path(args.run_dir),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            sample_obs_key=args.sample_obs_key,
            source_file_obs_key=args.source_file_obs_key,
            spot_latent_npz=args.spot_latent_npz,
            latent_obsm_key=args.latent_obsm_key,
            latent_cluster_obs_key=args.latent_cluster_obs_key,
            plot_spatial_x_key=args.plot_spatial_x_key,
            plot_spatial_y_key=args.plot_spatial_y_key,
            default_sample_id=args.default_sample_id,
            spatial_scale=args.spatial_scale,
            max_occurrences_per_cluster=args.max_occurrences_per_sample,
        )
        print(json.dumps(manifest, indent=2))
    elif args.command == "validate-run-concerns":
        report = write_concern_resolution_report(
            run_dir=Path(args.run_dir),
            coordinate_baseline_run_dir=Path(args.coordinate_baseline_run_dir)
            if args.coordinate_baseline_run_dir
            else None,
            stability_run_dirs=[Path(item) for item in args.stability_run_dir],
            leakage_ablation_run_dirs=[
                Path(item) for item in args.leakage_ablation_run_dir
            ],
            output_json=Path(args.output_json) if args.output_json else None,
            output_md=Path(args.output_md) if args.output_md else None,
        )
        print(json.dumps(report, indent=2))
        if args.strict and report.get("blocking_concerns"):
            raise SystemExit(1)
    elif args.command == "spatial-niche-validation":
        report = spatial_niche_validation_report(
            run_dir=Path(args.run_dir),
            output_json=Path(args.output_json) if args.output_json else None,
            output_cluster_csv=Path(args.output_cluster_csv)
            if args.output_cluster_csv
            else None,
            max_subregions=args.max_subregions,
            knn=args.knn,
            n_permutations=args.permutations,
            random_state=args.seed,
            sample_obs_key=args.sample_obs_key,
            allow_missing_sample_key=args.allow_missing_sample_key,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "select-k":
        import csv

        import matplotlib
        import numpy as np

        from .multilevel.model_selection import (
            comprehensive_select_k_from_latent_embeddings,
            fit_kmeans_on_latent_embeddings,
        )

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _json_default(value):
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            raise TypeError(
                f"Object of type {type(value).__name__} is not JSON serializable"
            )

        def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
            keys: list[str] = []
            seen: set[str] = set()
            for row in rows:
                for key in row:
                    if key not in seen:
                        seen.add(key)
                        keys.append(key)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)

        run_dir = Path(args.run_dir)
        diagnostics_npz = (
            Path(args.diagnostics_npz)
            if args.diagnostics_npz
            else run_dir / "multilevel_ot_candidate_cost_diagnostics.npz"
        )
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else run_dir / "comprehensive_k_selection"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with np.load(diagnostics_npz, allow_pickle=False) as data:
            if "subregion_latent_embeddings" not in data:
                raise ValueError(
                    f"{diagnostics_npz} does not contain subregion_latent_embeddings."
                )
            latent_embeddings = np.asarray(
                data["subregion_latent_embeddings"], dtype=np.float32
            )

        selection = comprehensive_select_k_from_latent_embeddings(
            latent_embeddings,
            candidate_n_clusters=args.candidate_n_clusters,
            fallback_n_clusters=15,
            seeds=args.seeds,
            n_init=args.n_init,
            min_cluster_size=args.min_subregions_per_cluster,
            gap_references=args.gap_references,
            bootstrap_repeats=args.bootstrap_repeats,
            bootstrap_fraction=args.bootstrap_fraction,
            max_silhouette_subregions=args.max_silhouette_subregions,
            random_state=args.seed,
        )
        scores = list(selection.get("scores", []))
        seed_runs = list(selection.get("seed_runs", []))
        selected_k = int(selection["selected_k"])
        selected_seed_rows = [
            row for row in seed_runs if int(row["n_clusters"]) == selected_k
        ]
        selected_seed = (
            int(min(selected_seed_rows, key=lambda row: float(row["inertia"]))["seed"])
            if selected_seed_rows
            else int(args.seed)
        )
        recommended_fit = fit_kmeans_on_latent_embeddings(
            latent_embeddings,
            n_clusters=selected_k,
            n_init=args.n_init,
            min_cluster_size=args.min_subregions_per_cluster,
            random_state=selected_seed + 104729 * selected_k,
        )

        summary_path = output_dir / "comprehensive_k_selection.json"
        score_csv = output_dir / "comprehensive_k_selection_scores.csv"
        seed_csv = output_dir / "comprehensive_k_selection_seed_runs.csv"
        labels_npz = output_dir / "recommended_k_labels.npz"
        selection["output_dir"] = str(output_dir)
        selection["diagnostics_npz"] = str(diagnostics_npz)
        selection["recommended_seed"] = int(selected_seed)
        selection["recommended_k_labels_npz"] = str(labels_npz)
        summary_path.write_text(
            json.dumps(selection, indent=2, default=_json_default) + "\n"
        )
        _write_csv(score_csv, scores)
        _write_csv(seed_csv, seed_runs)
        np.savez_compressed(
            labels_npz,
            labels=np.asarray(recommended_fit["labels"], dtype=np.int32),
            centers=np.asarray(recommended_fit["centers"], dtype=np.float32),
            costs=np.asarray(recommended_fit["costs"], dtype=np.float32),
            forced_label_mask=np.asarray(
                recommended_fit["forced_label_mask"], dtype=bool
            ),
            standardized_embeddings=np.asarray(
                recommended_fit["standardized_embeddings"], dtype=np.float32
            ),
            selected_k=np.asarray(selected_k, dtype=np.int32),
            selected_seed=np.asarray(selected_seed, dtype=np.int32),
        )

        if scores:
            ks = np.asarray([int(row["n_clusters"]) for row in scores], dtype=np.int32)
            metric_panels = [
                ("silhouette", "Silhouette", True),
                ("gap", "Gap", True),
                ("calinski_harabasz", "Calinski-Harabasz", True),
                ("davies_bouldin", "Davies-Bouldin", False),
                ("seed_ari_mean", "Seed ARI", True),
                ("bootstrap_ari_mean", "Bootstrap ARI", True),
                ("forced_repair_fraction_mean", "Forced Repair Fraction", False),
                ("best_seed_inertia", "Best Inertia", False),
            ]
            fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
            for ax, (metric, title, higher) in zip(
                axes.ravel(), metric_panels, strict=True
            ):
                values = np.asarray(
                    [
                        np.nan if row.get(metric) is None else float(row.get(metric))
                        for row in scores
                    ],
                    dtype=np.float64,
                )
                ax.plot(ks, values, marker="o", linewidth=1.8)
                ax.axvline(selected_k, color="#d62728", linewidth=1.2, linestyle="--")
                ax.set_title(f"{title} ({'higher' if higher else 'lower'} better)")
                ax.set_xlabel("K")
                ax.grid(alpha=0.25)
            fig.suptitle(
                "Comprehensive K Selection on Pooled Subregion Latent Embeddings"
            )
            fig.savefig(output_dir / "comprehensive_k_selection_scores.png", dpi=180)
            plt.close(fig)

        print(json.dumps(selection, indent=2, default=_json_default))
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
    elif args.command == "prepare-inputs":
        summary = prepare_h5ad_feature_cache(
            input_h5ad=args.input_h5ad,
            output_h5ad=args.output_h5ad,
            feature_obsm_key=args.feature_obsm_key,
            output_obsm_key=args.output_obsm_key,
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "distribute-prepared-inputs":
        summary = distribute_pooled_feature_cache_to_inputs(
            pooled_h5ad=args.pooled_h5ad,
            input_dir=args.input_dir,
            prepared_obsm_key=args.prepared_obsm_key,
            sample_glob=args.sample_glob,
            sample_obs_key=args.sample_obs_key,
            source_file_obs_key=args.source_file_obs_key,
            overwrite=bool(args.overwrite),
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
    elif args.command == "optimal-search":
        config = _resolve_multilevel_config_from_args(args)
        summary = run_multilevel_optimal_search(
            config=config,
            search_output_dir=args.output_dir or config.paths.output_dir,
            sample_obs_key=args.sample_obs_key,
            source_file_obs_key=args.source_file_obs_key,
            plot_spatial_x_key=args.plot_spatial_x_key,
            plot_spatial_y_key=args.plot_spatial_y_key,
            default_sample_id=args.default_sample_id,
            time_budget_hours=args.time_budget_hours,
            keep_top_k=args.keep_top_k,
            plot_best_sample_maps=bool(args.plot_best_sample_maps),
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
