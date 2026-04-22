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
- the active production path is `multilevel-ot`
- the deep `autoencoder` / `graph_autoencoder` layer is an experimental feature adapter, not yet a full generative niche model
- the legacy `train` path is still available, but it is not the main model surface anymore
- outputs should be written outside the package directory to keep the package compact

Method hierarchy:

- active production path:
  `multilevel-ot`
- experimental deep feature path:
  `autoencoder` / `graph_autoencoder`
- legacy path:
  teacher-student Visium HD scaffold

## Layout

- `spatial_ot/multilevel/`: active multilevel OT path, split into a dedicated namespace
- `spatial_ot/deep/`: reusable deep feature adapter namespace
- `spatial_ot/legacy/`: canonical home of the earlier teacher-student scaffold
- `spatial_ot/`: shared config/CLI entrypoints plus thin backward-compatible facades for older import paths such as `spatial_ot.training`, `spatial_ot.preprocessing`, and `spatial_ot.multilevel_ot`
- `configs/`: config files and demo prior programs
- `tests/`: regression and multilevel OT tests

## Environment Setup

Create or refresh a virtual environment under `../.venv`, so it lives next to both `spatial_ot/` and `spatial_ot_input/`, and install the package in editable mode:

```bash
cd spatial_ot
bash install_env.sh
```

You can override the Python executable, target virtual environment directory, or extras:

```bash
cd spatial_ot
PYTHON_BIN=python3.10 VENV_DIR=../.venv EXTRAS=dev,viz,geometry,parallel bash install_env.sh
```

By default `install_env.sh` now searches for a compatible interpreter in this order: `python3.13`, `python3.12`, `python3.11`, `python3.10`, then `python3`. It only selects interpreters that can actually create a pip-enabled virtual environment. If an existing `../.venv` was created with an incompatible Python, the script recreates it automatically.

The zip-ready runnable path in this package expects the cohort H5AD inputs to live in the sibling directory `../spatial_ot_input/`. The legacy `train` configs remain in the repo for reference, but they still depend on extra Visium HD workspace files beyond that packaged H5AD staging directory.

## Legacy train path

The smoke config points at the already-prepared `P2 CRC` Visium HD sample in this workspace and keeps the subset/epoch sizes intentionally small.

```bash
cd spatial_ot
../.venv/bin/python -m spatial_ot train --config configs/p2_crc_smoke.toml
```

Use an output path outside `spatial_ot` if you want to keep the package directory clean.

## Input visualization

You can render a 2D overview of the preprocessed inputs used by a config with:

```bash
cd spatial_ot
../.venv/bin/python -m spatial_ot plot-inputs --config configs/p2_crc_smoke.toml
```

By default this writes `input_2d_overview.png` under the configured output directory.

## Multilevel OT on Cell-Level Features

`spatial_ot` now also exposes a redesigned multilevel OT path for cases where the core input is already a cell-level feature embedding.

Prefer PCA, standardized marker expression, or another calibrated latent space for serious OT runs. UMAP can be useful for exploratory clustering and visualization, but its Euclidean geometry is not generally metric-preserving.

The current deep feature adapter is still a research-stage component, but it is stronger than a plain denoising MLP:

- `deep.method = "autoencoder"` learns a reusable feature adapter before OT
- `deep.method = "graph_autoencoder"` adds native Torch multi-scale message passing over short/mid radius graphs before OT
- `output_embedding` must now be set explicitly whenever a deep encoder is active
- using `output_embedding = "joint"` as the OT feature view now requires explicit opt-in; the example config uses `context` instead
- `validation_context_mode = "inductive"` keeps train and validation neighborhood targets separated to reduce transductive leakage
- `batch_key` currently supports validation/sample-holdout bookkeeping, not true batch correction
- `count_layer` now enables count-aware denoising against `X` or a named count layer through a low-rank chunked decoder
- the packaged shell runner now requests `cuda` explicitly by default, while the Python API and TOML config surface still accept `auto` or an explicit device string such as `cuda:0`
- the packaged `run.sh` path now defaults to `deep.method = "autoencoder"` with `output_embedding = "context"` over the prepared full-gene SVD cache; set `DEEP_FEATURE_METHOD=none` to disable the deep adapter
- `graph_autoencoder` remains full-batch only, and `deep.full_batch_max_cells` now guards against accidental oversize runs

Current deep-path capability snapshot:

| Capability | Status |
| --- | --- |
| `autoencoder` | implemented |
| `graph_autoencoder` | implemented, full-batch only |
| graph density cap via `graph_max_neighbors` | implemented |
| deep-only `fit` / `transform` lifecycle | implemented |
| explicit opt-in required for `joint` OT view | implemented |
| count reconstruction via `count_layer` | implemented |
| mini-batch graph training | not implemented |
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

Portable run against the packaged `spatial_ot_input` H5AD files:

```bash
cd spatial_ot
../.venv/bin/python -m spatial_ot multilevel-ot \
  --input-h5ad ../spatial_ot_input/p2_crc_cells_marker_genes_umap3d.h5ad \
  --output-dir ../outputs/spatial_ot/p2_crc_multilevel_ot \
  --feature-obsm-key X \
  --spatial-x-key cell_x \
  --spatial-y-key cell_y \
  --spatial-scale 0.2737012522439323 \
  --compute-device cuda \
  --n-clusters 8 \
  --atoms-per-cluster 8 \
  --radius-um 100 \
  --stride-um 150 \
  --basic-niche-size-um 200 \
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
  --allow-observed-hull-geometry
```

The packaged input bundle includes both the full sparse gene matrix in `X` and an exploratory `obsm["X_umap_marker_genes_3d"]`. The fastest repeated GPU path is now:

1. pool all samples once into `../spatial_ot_input/spatial_ot_input_pooled.h5ad`
2. precompute the CPU-heavy `X -> log1p-normalized TruncatedSVD` feature cache once into that same pooled H5AD
3. fit the default deep `autoencoder` adapter on that cached cohort-level feature matrix and pass the learned `context` embedding into OT
4. point repeated OT runs at the cached / learned feature matrices instead of recomputing the CPU-heavy preprocessing every run

The UMAP path remains available only through explicit exploratory opt-in.

Treat `../spatial_ot_input/` as the canonical staging directory for those H5AD files and keep it next to the unzipped `spatial_ot/` directory.

The reusable cohort-level pooled input now also lives in that same sibling staging directory by default:

```bash
cd spatial_ot
bash pool_spatial_ot_input.sh
```

That writes `../spatial_ot_input/spatial_ot_input_pooled.h5ad` plus its JSON summary once, so later cohort runs can reuse the same pooled input without rebuilding it.

To also precompute the CPU-heavy full-gene feature transform up front:

```bash
cd spatial_ot
bash prepare_spatial_ot_input.sh
```

That reuses or refreshes `../spatial_ot_input/spatial_ot_input_pooled.h5ad`, then writes a prepared shared feature cache such as `obsm["X_spatial_ot_x_svd_512"]` into that pooled cohort H5AD from the pooled full-gene matrix `X`. This is the recommended staging step before long GPU runs because the pooled run needs one cohort-aligned feature basis rather than separate per-sample SVD bases.

If you also want every original sample H5AD in `../spatial_ot_input/` to carry the same prepared cache for single-sample runs, stage them all in one pass:

```bash
cd spatial_ot
bash prepare_all_spatial_ot_input.sh
```

That first prepares the pooled cohort H5AD, then copies the resulting pooled shared feature cache back into each `*_cells_marker_genes_umap3d.h5ad` in place, and finally verifies that every sample now contains the prepared cache. This avoids recomputing separate per-sample SVD spaces while still letting single-sample runs reuse a precomputed feature matrix.

In this packaged workspace, `../spatial_ot_input/*.h5ad` may also be managed by an external sync job. Because of that, `prepare_all_spatial_ot_input.sh` defaults to the safe pooled-only staging path and leaves those source files untouched unless you explicitly opt in:

```bash
cd spatial_ot
WRITE_BACK_TO_SOURCE_INPUTS=1 bash prepare_all_spatial_ot_input.sh
```

Use that opt-in only when the source H5AD files are not being overwritten by another process.

The default `bash run.sh` path now reuses `../spatial_ot_input/spatial_ot_input_pooled.h5ad` when it already exists, reuses the prepared full-gene cache when it already exists, and only recomputes those CPU-side steps when the file is missing or when you explicitly refresh them:

- `sample_id` and `source_h5ad` are written into `obs`
- original per-sample coordinates remain available in `cell_x` / `cell_y` and explicit `original_cell_x` / `original_cell_y`
- pooled sample-separated coordinates are written to `pooled_cell_x` / `pooled_cell_y`
- the default pooled-run feature key is now the prepared cache `X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}` rather than recomputing TruncatedSVD from `X` inside each OT run
- `prepare_all_spatial_ot_input.sh` stages the pooled shared cache first, and can optionally write that same cohort-aligned cache back into each sample H5AD when `WRITE_BACK_TO_SOURCE_INPUTS=1`
- when `POOL_ALL_INPUTS=0` and `PREPARE_INPUTS_AHEAD=1`, the same prepared cache key is also used for single-sample runs
- the pooled coordinates place each sample on its own non-overlapping tile, so samples contribute jointly to latent/OT learning without being treated as one physically continuous tissue section
- the packaged runner now defaults to `MIN_CELLS=1`, `MAX_SUBREGIONS=0`, `STRIDE_UM=RADIUS_UM`, and `REQUIRE_FULL_CELL_COVERAGE=1`, so it does not cap or subsample candidate subregions, uses the full pooled cell table, and fails the run if any cells are left outside all analyzed subregions
- after the OT fit finishes, `run.sh` also writes one spatial niche PNG per sample under `../outputs/spatial_ot/cohort_multilevel_ot/sample_niche_plots/`
- those per-sample plots use the original within-sample spatial coordinates rather than the pooled tiled coordinates, so each specimen is visualized in its native layout

If you want to force a pooled-input rebuild before running OT:

```bash
cd spatial_ot
REFRESH_POOLED_INPUT=1 bash run.sh
```

If you want to force the prepared full-gene cache to be recomputed too:

```bash
cd spatial_ot
REFRESH_PREPARED_FEATURES=1 bash run.sh
```

The active path now also supports a TOML config surface. A portable example lives at `configs/multilevel_deep_example.toml`, and it now demonstrates the graph-aware encoder:

```bash
cd spatial_ot
../.venv/bin/python -m spatial_ot multilevel-ot \
  --config configs/multilevel_deep_example.toml
```

You can still override config values from the CLI, for example:

```bash
../.venv/bin/python -m spatial_ot multilevel-ot \
  --config configs/multilevel_deep_example.toml \
  --input-h5ad ../spatial_ot_input/p5_crc_cells_marker_genes_umap3d.h5ad \
  --output-dir ../outputs/spatial_ot/p5_crc_multilevel_ot \
  --feature-obsm-key X \
  --deep-feature-method graph_autoencoder \
  --deep-output-embedding joint
```

You can now also run the deep encoder as its own reusable stage:

```bash
cd spatial_ot
../.venv/bin/python -m spatial_ot deep-fit \
  --config configs/multilevel_deep_example.toml \
  --input-h5ad ../spatial_ot_input/p2_crc_cells_marker_genes_umap3d.h5ad \
  --output-dir ../outputs/spatial_ot/deep_encoder_only \
  --feature-obsm-key X \
  --spatial-scale 0.2737012522439323
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
../.venv/bin/python -m spatial_ot deep-transform \
  --model ../outputs/spatial_ot/deep_encoder_only/deep_feature_model.pt \
  --input-h5ad ../spatial_ot_input/p5_crc_cells_marker_genes_umap3d.h5ad \
  --output-h5ad ../outputs/spatial_ot/deep_encoder_only/p5_crc_embedded.h5ad \
  --feature-obsm-key X \
  --spatial-x-key cell_x \
  --spatial-y-key cell_y \
  --spatial-scale 0.2737012522439323
```

For graph-based deep runs, `spatial_scale` matters: graph radii such as `radius_um`, `short_radius_um`, and `mid_radius_um` are interpreted after applying that scale, so pixel-space coordinates should be converted to microns before training or transform.

The packaged `run.sh` path now requests `cuda` explicitly so GPU-equipped environments use CUDA by default. The underlying Python API and TOML config surface still accept `auto`, `cuda`, `cuda:0`, `cuda:1`, or `cpu`.

The packaged `run.sh` path also assumes a `28`-thread CPU by default and exports:

- `OMP_NUM_THREADS=28`
- `MKL_NUM_THREADS=28`
- `OPENBLAS_NUM_THREADS=28`
- `NUMEXPR_NUM_THREADS=28`
- `BLIS_NUM_THREADS=28`
- `VECLIB_MAXIMUM_THREADS=28`
- `SPATIAL_OT_TORCH_NUM_THREADS=28`
- `SPATIAL_OT_TORCH_NUM_INTEROP_THREADS=4`
- `SPATIAL_OT_X_SVD_COMPONENTS=512`
- `SPATIAL_OT_X_TARGET_SUM=10000`

Override these with `CPU_THREADS`, `TORCH_INTRAOP_THREADS`, `TORCH_INTEROP_THREADS`, `X_FEATURE_COMPONENTS`, `X_TARGET_SUM`, or the underlying library-specific environment variables if your node layout is different.

The active multilevel OT runtime now also supports multi-GPU restart parallelism for the outer `n_init` loop. By default the packaged runner exports:

- `SPATIAL_OT_CUDA_DEVICE_LIST=all`
- `SPATIAL_OT_PARALLEL_RESTARTS=auto`
- `SPATIAL_OT_CUDA_TARGET_VRAM_GB=50`

This means:

- all visible CUDA devices are eligible for restart workers
- independent restarts are spread across those GPUs when `n_init > 1`
- GPU-side pairwise kernels aim for roughly `50 GB` of working-set memory per device, but are automatically capped to about `80%` of the actual visible GPU memory on smaller cards
- when restart workers are launched in parallel, the Torch/BLAS CPU thread budget is divided across those workers automatically instead of letting every worker claim all `28` threads

The current multilevel OT and deep-feature implementations are still single-GPU within any one restart, not tensor-parallel across devices. On a 2-GPU node such as a dual-H100 machine, the package now speeds up by distributing independent restarts across both GPUs. If you want to pin a run to a subset of devices, use `CUDA_VISIBLE_DEVICES` or override `CUDA_DEVICE_LIST`.

Examples:

```bash
cd spatial_ot
CUDA_VISIBLE_DEVICES=0 COMPUTE_DEVICE=cuda bash run.sh
```

```bash
cd spatial_ot
CUDA_VISIBLE_DEVICES=1 COMPUTE_DEVICE=cuda SAMPLE_KEY=p2_crc POOL_ALL_INPUTS=0 bash run.sh
```

If you need CPU fallback explicitly, set `COMPUTE_DEVICE=cpu`.

You can rerun just the per-sample niche maps from an existing multilevel OT output directory with:

```bash
cd spatial_ot
../.venv/bin/python -m spatial_ot plot-sample-niches \
  --run-dir ../outputs/spatial_ot/cohort_multilevel_ot \
  --output-dir ../outputs/spatial_ot/cohort_multilevel_ot/sample_niche_plots \
  --sample-obs-key sample_id \
  --plot-spatial-x-key cell_x \
  --plot-spatial-y-key cell_y
```

For grid-built multilevel OT runs, `basic_niche_size_um` sets the smallest building block used to compose larger subregions. The packaged default now uses a `50 µm` basic niche diameter so the smallest building block stays local and the larger OT subregions are composed from finer units.

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

- `run.sh`: canonical packaged runner; pools all H5ADs under `../spatial_ot_input/` into one cohort input by default
  It now requests `COMPUTE_DEVICE=cuda` by default.
  It also defaults to `CPU_THREADS=28` and configures BLAS plus Torch CPU threading for that budget.
  It also defaults to `CUDA_DEVICE_LIST=all`, `PARALLEL_RESTARTS=auto`, and `CUDA_TARGET_VRAM_GB=50` to push the OT runtime harder on multi-GPU H100-class nodes.
  It defaults to `FEATURE_OBSM_KEY=X`, so pooled runs learn from the full gene matrix instead of the packaged UMAP coordinates.
  It now defaults to `BASIC_NICHE_SIZE_UM=50`, `MIN_CELLS=25`, and `MAX_SUBREGIONS=1500` so the packaged run is less likely to chase single-cell noise or blow up GPU memory.
  It now keeps observed-hull geometry fallback off by default; explicitly opt in only for exploratory runs where boundary-shape invariance is not required.
- `run_optimal_setting_search.sh`: packaged launcher for the staged pooled-cohort search; it uses the prepared pooled feature cache, writes ranked candidate summaries under `../work/spatial_ot_runs/cohort_optimal_search/`, and keeps one full best-run niche map per sample
- `run_spatial_ot_input.sh`: backward-compatible alias for `run.sh`
- `run_p2_crc_multilevel_ot.sh`: pinned single-sample alias that defaults `SAMPLE_KEY=p2_crc`
- `run_p2_crc_multilevel_ot_exploratory_umap.sh`: backward-compatible exploratory single-sample alias for `p2_crc`; it explicitly switches `FEATURE_OBSM_KEY` back to `X_umap_marker_genes_3d`

Example:

```bash
cd spatial_ot
bash run.sh
```

Explicit 2-GPU / 28-thread-per-job examples:

```bash
cd spatial_ot
CUDA_VISIBLE_DEVICES=0 CPU_THREADS=28 bash run.sh
```

```bash
cd spatial_ot
CUDA_VISIBLE_DEVICES=1 CPU_THREADS=28 POOL_ALL_INPUTS=0 SAMPLE_KEY=p2_crc bash run.sh
```

## Config notes

- `subset_strategy` should currently be `spatial_grid` or `stratified`
- the active `multilevel-ot` path now has its own TOML config surface via `load_multilevel_config`
- overlap fallback from cells to the nearest `8 µm` bin is controlled by `allow_nearest_overlap_fallback`

## Pilot config

`configs/p2_crc_pilot.toml` is a larger but still subset-based configuration intended as the next step after the smoke run.

## Prior programs

`configs/crc_demo_programs.json` is a starter library so the full path can run now. It is a demo prior set, not the final biology for your CRC project.

Once marker panels arrive, replace that file with a curated program library and rerun the same CLI.
