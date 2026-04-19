# spatial_ot

`spatial_ot` is a compact research package for multilevel, shape-normalized semi-relaxed Wasserstein clustering on spatial subregions.

The current primary path realizes:

- geometry-only OT normalization of each subregion into a shared reference domain
- an optional learned deep feature adapter with fit/transform/save/load behavior
- compressed empirical measures over canonical coordinates plus local features
- cluster-specific shared heterogeneity atoms with subregion-specific mixture weights
- semi-relaxed unbalanced OT matching with residual similarity alignment
- shape-leakage diagnostics so boundary geometry can be checked explicitly
- cell-level projection and visualization from the fitted subregion clusters

The package also still contains the earlier teacher-student Visium HD scaffold for legacy experiments:

- an `8 µm` teacher branch trained on binned Visium HD counts
- an intrinsic cell branch that learns `z`
- a multi-view context/program branch that learns `s`
- teacher-to-cell distillation through `2 µm -> 8 µm -> cell` overlap mappings
- OT niche fitting on neighborhood objects built from state atoms, program activity, and shell profiles
- a COMMOT-style communication head that estimates directional sender->receiver flows over learned programs

The implementation is intentionally stage-wise:

1. teacher pretraining on `8 µm` bins
2. intrinsic cell pretraining
3. context/program training with asymmetric independence regularization
4. OT niche fitting on frozen latents
5. communication-flow fitting and residual attribution

This is a staged approximation of the proposed stack, not a claim that the exact combined method is already a published or fully benchmarked model.

Current notes:

- the recommended active path is `multilevel-ot`
- the legacy `train` path is still available, but it is not the main model surface anymore
- outputs should be written outside the package directory to keep the package compact

## Layout

- `spatial_ot/multilevel/`: active multilevel OT path, split into a dedicated namespace
- `spatial_ot/deep/`: reusable deep feature adapter namespace
- `spatial_ot/legacy/`: earlier teacher-student scaffold, kept for backward-compatible legacy experiments
- `spatial_ot/`: top-level compatibility shims and shared config/CLI entrypoints
- `configs/`: config files and demo prior programs
- `tests/`: regression and multilevel OT tests

## Legacy train path

The smoke config points at the already-prepared `P2 CRC` Visium HD sample in this workspace and keeps the subset/epoch sizes intentionally small.

```bash
cd spatial_ot
conda run -n ml1 python -m spatial_ot train --config configs/p2_crc_smoke.toml
```

Use an output path outside `spatial_ot` if you want to keep the package directory clean.

## Input visualization

You can render a 2D overview of the preprocessed inputs used by a config with:

```bash
cd spatial_ot
conda run -n ml1 python -m spatial_ot plot-inputs --config configs/p2_crc_smoke.toml
```

By default this writes `input_2d_overview.png` under the configured output directory.

## Multilevel OT on Cell-Level Features

`spatial_ot` now also exposes a redesigned multilevel OT path for cases where the core input is already a cell-level feature embedding.

Prefer PCA, standardized marker expression, or another calibrated latent space for serious OT runs. UMAP can be useful for exploratory clustering and visualization, but its Euclidean geometry is not generally metric-preserving.

The current deep feature adapter is still a research-stage component, but it is stronger than a plain denoising MLP:

- `deep.method = "autoencoder"` learns a reusable feature adapter before OT
- `deep.method = "graph_autoencoder"` adds native Torch multi-scale message passing over short/mid radius graphs before OT
- `validation_context_mode = "inductive"` keeps train and validation neighborhood targets separated to reduce transductive leakage
- `batch_key` currently supports validation/sample-holdout bookkeeping, not true batch correction
- `count_layer` is reserved but not implemented for count reconstruction yet
- the active multilevel OT path now has its own Torch compute device via `--compute-device` / `ot.compute_device`; `auto` uses CUDA when available for cost matrices, projection, and the semi-relaxed Sinkhorn solver

Current deep-path capability snapshot:

| Capability | Status |
| --- | --- |
| `autoencoder` | implemented |
| `graph_autoencoder` | implemented |
| graph density cap via `graph_max_neighbors` | implemented |
| deep-only `fit` / `transform` lifecycle | implemented |
| mini-batch graph training | not implemented |
| count reconstruction | not implemented |
| batch adversarial correction | not implemented |
| OT-aware fine-tuning | not implemented |
| multilevel OT prediction bundle for new samples | not implemented |

Two multilevel OT modes are supported:

- grid-window discovery: `spatial_ot` builds overlapping radius windows from cells for exploratory local pattern discovery
- explicit-region clustering: pass true region membership through `--region-obs-key` or the Python API and, ideally, supply explicit geometry objects so boundary shape can be treated as nuisance rather than signal

Without explicit masks or polygons, boundary-shape invariance is not guaranteed. The observed-coordinate convex-hull fallback is now opt-in and should be treated as exploratory only.

This path:

- can learn a reusable feature adapter before OT through `--deep-feature-method autoencoder` or `--deep-feature-method graph_autoencoder`
- builds overlapping spatial subregions from cells
- learns a geometry-only OT map that normalizes each subregion into a shared reference domain
- compresses each subregion into a smaller empirical measure over canonical coordinates plus features
- learns cluster-specific shared heterogeneity atoms and subregion-specific mixture weights
- matches subregions to clusters with semi-relaxed unbalanced OT plus residual similarity alignment
- projects the learned labels back to cells for visualization
- reports shape-leakage diagnostics so we can check whether boundary geometry is still driving cluster labels

Validated PCA-style run:

```bash
cd spatial_ot
conda run -n ml1 python -m spatial_ot multilevel-ot \
  --input-h5ad ../data/cells.h5ad \
  --output-dir ../work/spatial_ot_runs/p2_crc_multilevel_pca \
  --feature-obsm-key X_pca \
  --spatial-x-key cell_x \
  --spatial-y-key cell_y \
  --spatial-scale 1.0 \
  --compute-device auto \
  --n-clusters 8 \
  --atoms-per-cluster 8 \
  --radius-um 100 \
  --stride-um 150 \
  --min-cells 20 \
  --max-subregions 2000 \
  --lambda-x 0.5 \
  --lambda-y 1.0 \
  --geometry-eps 0.03 \
  --ot-eps 0.03 \
  --rho 0.5 \
  --geometry-samples 192 \
  --compressed-support-size 96 \
  --align-iters 4 \
  --n-init 5 \
  --deep-feature-method graph_autoencoder
```

Exploratory UMAP run when only a UMAP embedding is available:

```bash
cd spatial_ot
conda run -n ml1 python -m spatial_ot multilevel-ot \
  --input-h5ad ../work/visium_hd_p2_crc/exports/p2_crc_cells_marker_genes_umap3d_rgb.h5ad \
  --output-dir ../work/spatial_ot_runs/p2_crc_multilevel_umap_exploratory \
  --feature-obsm-key X_umap_marker_genes_3d \
  --spatial-x-key cell_x \
  --spatial-y-key cell_y \
  --spatial-scale 0.2737012522439323 \
  --compute-device auto \
  --n-clusters 8 \
  --atoms-per-cluster 8 \
  --radius-um 100 \
  --stride-um 150 \
  --min-cells 20 \
  --max-subregions 2000 \
  --allow-observed-hull-geometry
```

The active path now also supports a TOML config surface. A portable example lives at `configs/multilevel_deep_example.toml`, and it now demonstrates the graph-aware encoder:

```bash
cd spatial_ot
conda run -n ml1 python -m spatial_ot multilevel-ot \
  --config configs/multilevel_deep_example.toml
```

You can still override config values from the CLI, for example:

```bash
conda run -n ml1 python -m spatial_ot multilevel-ot \
  --config configs/multilevel_deep_example.toml \
  --input-h5ad ../data/cells.h5ad \
  --output-dir ../work/spatial_ot_runs/example_output \
  --feature-obsm-key X_pca \
  --deep-feature-method graph_autoencoder
```

You can now also run the deep encoder as its own reusable stage:

```bash
cd spatial_ot
conda run -n ml1 python -m spatial_ot deep-fit \
  --config configs/multilevel_deep_example.toml \
  --input-h5ad ../data/cells.h5ad \
  --output-dir ../work/spatial_ot_runs/deep_encoder_only \
  --feature-obsm-key X_pca
```

That writes:

- `cells_deep_features.h5ad`
- `deep_feature_model.pt`
- `deep_feature_history.csv`
- `deep_feature_config.json`
- `summary.json`

You can then reuse the saved encoder on a new H5AD:

```bash
cd spatial_ot
conda run -n ml1 python -m spatial_ot deep-transform \
  --model ../work/spatial_ot_runs/deep_encoder_only/deep_feature_model.pt \
  --input-h5ad ../data/new_cells.h5ad \
  --output-h5ad ../work/spatial_ot_runs/deep_encoder_only/new_cells_embedded.h5ad \
  --feature-obsm-key X_pca \
  --spatial-x-key cell_x \
  --spatial-y-key cell_y
```

Key artifacts from this path:

- `cells_multilevel_ot.h5ad`
- `subregions_multilevel_ot.parquet`
- `cluster_supports_multilevel_ot.npz`
- `deep_feature_model.pt` when deep features are enabled and model saving is on
- `deep_feature_history.csv` and `deep_feature_config.json` when deep features are enabled
- `multilevel_ot_spatial_map.png`
- `multilevel_ot_subregion_embedding.png`
- `multilevel_ot_atom_layouts.png`
- `summary.json`

The saved summary now includes:

- a `deep_features` block describing whether the active path learned a feature adapter before OT
- restart summaries and the selected restart
- geometry-source counts and convex-hull fallback frequency
- assigned OT fallback frequency and the effective entropy values actually used by the solver
- the requested and resolved Torch compute device for the active multilevel path
- package version, git SHA, and summary schema version for reproducibility
- graph usage metadata such as whether the deep encoder used a spatial graph and its training-graph degree statistics
- graph density controls such as `graph_max_neighbors`, so radius graphs can be capped instead of growing without bound in dense tissue
- a `boundary_invariance_claim` field showing whether explicit geometry supported the run; when observed-hull fallback is used the claim is explicitly exploratory
- random-fold and spatial-block shape-leakage diagnostics
- canonical-normalizer radius / interpolation diagnostics

Helper scripts:

- `run_p2_crc_multilevel_ot.sh`: safer default helper using `X_pca`
- `run_p2_crc_multilevel_ot_exploratory_umap.sh`: explicit exploratory helper when only a UMAP feature space is available

## Config notes

- `subset_strategy` should currently be `spatial_grid` or `stratified`
- the active `multilevel-ot` path now has its own TOML config surface via `load_multilevel_config`
- overlap fallback from cells to the nearest `8 µm` bin is controlled by `allow_nearest_overlap_fallback`

## Pilot config

`configs/p2_crc_pilot.toml` is a larger but still subset-based configuration intended as the next step after the smoke run.

## Prior programs

`configs/crc_demo_programs.json` is a starter library so the full path can run now. It is a demo prior set, not the final biology for your CRC project.

Once marker panels arrive, replace that file with a curated program library and rerun the same CLI.
