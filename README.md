# spatial_ot

`spatial_ot` is a compact research package for multilevel, shape-normalized semi-relaxed Wasserstein clustering on spatial subregions.

The active path realizes:

- geometry-only OT normalization of each subregion into a shared reference domain
- an optional learned deep feature adapter with fit/transform/save/load behavior
- compressed empirical measures over canonical coordinates plus local features
- cluster-specific shared heterogeneity atoms with subregion-specific mixture weights
- semi-relaxed unbalanced OT matching with residual similarity alignment
- shape-leakage diagnostics so boundary geometry can be checked explicitly
- cell-level projection and visualization from the fitted subregion clusters

The package also retains a legacy teacher-student Visium HD scaffold (`train` subcommand, see [Legacy train path](#legacy-train-path)).

The deep `autoencoder` / `graph_autoencoder` layer is an experimental feature adapter, not yet a full generative niche model.

Always write outputs outside the package directory to keep it compact.

## Layout

- `spatial_ot/multilevel/`: active multilevel OT path
- `spatial_ot/deep/`: reusable deep feature adapter
- `spatial_ot/legacy/`: canonical home of the teacher-student scaffold
- `spatial_ot/`: shared config/CLI plus thin backward-compatible facades for older import paths (`spatial_ot.training`, `spatial_ot.preprocessing`, `spatial_ot.multilevel_ot`, …)
- `configs/`: TOML configs and demo prior programs
- `tests/`: regression and multilevel OT tests

## Environment setup

`install_env.sh` creates or refreshes `../.venv` (next to `spatial_ot/` and `spatial_ot_input/`) and installs the package editable:

```bash
cd spatial_ot
bash install_env.sh
```

Override interpreter / venv / extras via env vars:

```bash
PYTHON_BIN=python3.10 VENV_DIR=../.venv EXTRAS=dev,viz,geometry,parallel bash install_env.sh
```

By default the script searches for a compatible Python (3.13 → 3.12 → 3.11 → 3.10 → 3) and recreates `../.venv` if the existing one is incompatible.

The packaged path expects cohort H5AD inputs under the sibling directory `../spatial_ot_input/`.

## Multilevel OT — primary path

`spatial_ot multilevel-ot` runs shape-normalized cluster-specific semi-relaxed Wasserstein clustering on subregions built from cell-level features.

Prefer PCA, standardized marker expression, or another calibrated latent space. UMAP is exploratory only — its Euclidean geometry is not metric-preserving.

Two modes:

- **grid-window discovery** (default): overlapping radius windows over cells
- **explicit-region clustering**: pass `--region-obs-key` and ideally explicit geometry objects so boundary shape can be treated as nuisance

Without explicit masks or polygons, boundary-shape invariance is not guaranteed. The observed-coordinate convex-hull fallback is opt-in via `--allow-observed-hull-geometry` and is exploratory only.

### Quick run against packaged inputs

```bash
cd spatial_ot
../.venv/bin/python -m spatial_ot multilevel-ot \
  --input-h5ad ../spatial_ot_input/p2_crc_cells_marker_genes_umap3d.h5ad \
  --output-dir ../outputs/spatial_ot/p2_crc_multilevel_ot \
  --feature-obsm-key X \
  --spatial-x-key cell_x --spatial-y-key cell_y \
  --spatial-scale 0.2737012522439323 \
  --compute-device cuda \
  --n-clusters 8 --atoms-per-cluster 8 \
  --radius-um 100 --stride-um 150 --basic-niche-size-um 200 \
  --min-cells 20 --max-subregions 2000 \
  --lambda-x 0.5 --lambda-y 1.0 \
  --geometry-eps 0.03 --ot-eps 0.03 --rho 0.5 \
  --geometry-samples 192 --compressed-support-size 96 \
  --align-iters 4 --n-init 5 \
  --allow-observed-hull-geometry
```

### TOML config

`configs/multilevel_deep_example.toml` is a portable example that demonstrates the graph-aware encoder:

```bash
../.venv/bin/python -m spatial_ot multilevel-ot --config configs/multilevel_deep_example.toml
```

CLI flags override config values — for example switching to a different sample and the graph encoder:

```bash
../.venv/bin/python -m spatial_ot multilevel-ot \
  --config configs/multilevel_deep_example.toml \
  --input-h5ad ../spatial_ot_input/p5_crc_cells_marker_genes_umap3d.h5ad \
  --output-dir ../outputs/spatial_ot/p5_crc_multilevel_ot \
  --deep-feature-method graph_autoencoder \
  --deep-output-embedding joint
```

### Deep feature adapter

- `deep.method = "autoencoder"`: learns a reusable feature adapter before OT
- `deep.method = "graph_autoencoder"`: native Torch multi-scale message passing over short/mid radius graphs (full-batch only; `deep.full_batch_max_cells` guards against accidental oversize runs)
- `output_embedding` must be set explicitly when a deep encoder is active. Using `joint` as the OT view requires explicit opt-in
- `validation_context_mode = "inductive"` keeps train and validation neighborhood targets separated
- `count_layer` enables count-aware NB denoising against `X` or a named layer through a low-rank chunked decoder
- `batch_key` supports validation/sample-holdout bookkeeping only — not true batch correction

| Capability | Status |
| --- | --- |
| `autoencoder` | implemented |
| `graph_autoencoder` | implemented, full-batch only |
| graph density cap (`graph_max_neighbors`) | implemented |
| deep-only `fit`/`transform` lifecycle | implemented |
| explicit opt-in for `joint` OT view | implemented |
| count reconstruction (`count_layer`) | implemented |
| mini-batch graph training | not implemented |
| batch adversarial correction | not implemented |
| OT-aware fine-tuning | not implemented |
| multilevel OT prediction bundle for new samples | not implemented |

### Standalone deep encoder

```bash
../.venv/bin/python -m spatial_ot deep-fit \
  --config configs/multilevel_deep_example.toml \
  --input-h5ad ../spatial_ot_input/p2_crc_cells_marker_genes_umap3d.h5ad \
  --output-dir ../outputs/spatial_ot/deep_encoder_only \
  --feature-obsm-key X \
  --spatial-scale 0.2737012522439323
```

Writes `cells_deep_features.h5ad`, `deep_feature_model.pt`, `deep_feature_history.csv`, `deep_feature_config.json`, `summary.json`.

Reuse the saved encoder on a new H5AD:

```bash
../.venv/bin/python -m spatial_ot deep-transform \
  --model ../outputs/spatial_ot/deep_encoder_only/deep_feature_model.pt \
  --input-h5ad ../spatial_ot_input/p5_crc_cells_marker_genes_umap3d.h5ad \
  --output-h5ad ../outputs/spatial_ot/deep_encoder_only/p5_crc_embedded.h5ad \
  --feature-obsm-key X \
  --spatial-x-key cell_x --spatial-y-key cell_y \
  --spatial-scale 0.2737012522439323
```

For graph-based runs, `spatial_scale` matters: graph radii (`radius_um`, `short_radius_um`, `mid_radius_um`) are interpreted after applying that scale, so pixel-space coordinates need conversion to microns.

## Input staging pipeline

The fastest repeated GPU path is:

1. pool all samples once into `../spatial_ot_input/spatial_ot_input_pooled.h5ad`
2. precompute the CPU-heavy `X → log1p-normalized TruncatedSVD` feature cache once into the pooled H5AD
3. fit the deep `autoencoder` adapter on that cached cohort matrix and feed the learned `context` embedding into OT
4. point repeated OT runs at the cached/learned matrices instead of recomputing CPU-heavy preprocessing

Helper scripts:

```bash
bash pool_spatial_ot_input.sh           # pool only → ../spatial_ot_input/spatial_ot_input_pooled.h5ad
bash prepare_spatial_ot_input.sh        # pool + precompute X → SVD into the pooled H5AD
bash prepare_all_spatial_ot_input.sh    # also verify each sample H5AD has the prepared cache
WRITE_BACK_TO_SOURCE_INPUTS=1 bash prepare_all_spatial_ot_input.sh
                                        # opt in to copying the cohort cache back into source files
```

Pooled coordinates place each sample on its own non-overlapping tile, so samples contribute jointly to latent/OT learning without being treated as one continuous tissue section.

`obs` columns written by pooling:

- `sample_id`, `source_h5ad`
- `cell_x`, `cell_y` and explicit `original_cell_x`, `original_cell_y` (per-sample coordinates)
- `pooled_cell_x`, `pooled_cell_y` (pooled, sample-separated coordinates)

The default pooled feature key is `X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}` (the prepared cache), not a fresh per-run TruncatedSVD.

In this packaged workspace the input H5ADs may be managed by an external sync job, so `prepare_all_spatial_ot_input.sh` defaults to leaving source files untouched and writes only the pooled cohort cache.

## Packaged runner: `run.sh`

```bash
bash run.sh                                                             # default cohort run
CUDA_VISIBLE_DEVICES=0 bash run.sh                                       # pin to GPU 0
CUDA_VISIBLE_DEVICES=1 SAMPLE_KEY=p2_crc POOL_ALL_INPUTS=0 bash run.sh   # single-sample on GPU 1
COMPUTE_DEVICE=cpu bash run.sh                                           # CPU fallback
REFRESH_POOLED_INPUT=1 bash run.sh                                       # rebuild pooled input
REFRESH_PREPARED_FEATURES=1 bash run.sh                                  # rebuild prepared cache
```

`run.sh` reuses `../spatial_ot_input/spatial_ot_input_pooled.h5ad` and the prepared full-gene cache when present; only recomputes when missing or explicitly refreshed. After OT it writes one spatial niche PNG per sample under `../outputs/spatial_ot/cohort_multilevel_ot/sample_niche_plots/` using each sample's native within-sample coordinates.

Defaults relevant to safety / cost (override with the matching env var):

- `COMPUTE_DEVICE=cuda`, `FEATURE_OBSM_KEY=X`
- `BASIC_NICHE_SIZE_UM=50`, `MIN_CELLS=25`, `MAX_SUBREGIONS=1500`, `STRIDE_UM=$RADIUS_UM`
- `REQUIRE_FULL_CELL_COVERAGE=1` (fails the run if any cell falls outside all analyzed subregions)
- `ALLOW_OBSERVED_HULL_GEOMETRY=0` (disabled — opt in only for exploratory runs)
- `DEEP_FEATURE_METHOD=autoencoder`, `DEEP_OUTPUT_EMBEDDING=context`, `DEEP_DEVICE=cuda`. Set `DEEP_FEATURE_METHOD=none` to skip the deep adapter.

### Compute resources

`run.sh` assumes a 28-thread CPU and exports a coordinated thread budget:

- `OMP_NUM_THREADS=28`, `MKL_NUM_THREADS=28`, `OPENBLAS_NUM_THREADS=28`, `NUMEXPR_NUM_THREADS=28`, `BLIS_NUM_THREADS=28`, `VECLIB_MAXIMUM_THREADS=28`
- `SPATIAL_OT_TORCH_NUM_THREADS=28`, `SPATIAL_OT_TORCH_NUM_INTEROP_THREADS=4`
- `SPATIAL_OT_X_SVD_COMPONENTS=512`, `SPATIAL_OT_X_TARGET_SUM=10000`

Override with `CPU_THREADS`, `TORCH_INTRAOP_THREADS`, `TORCH_INTEROP_THREADS`, `X_FEATURE_COMPONENTS`, `X_TARGET_SUM`, or the underlying library env vars.

For multi-GPU restart parallelism in the outer `n_init` loop:

- `SPATIAL_OT_CUDA_DEVICE_LIST=all` (eligible devices)
- `SPATIAL_OT_PARALLEL_RESTARTS=auto` (spread independent restarts across GPUs when `n_init > 1`)
- `SPATIAL_OT_CUDA_TARGET_VRAM_GB=50` (per-device working set, capped to ~80% of visible memory)

When restart workers run in parallel, the Torch/BLAS thread budget is divided across them automatically. The multilevel OT and deep-feature implementations are still single-GPU within any one restart.

The Python API and TOML config still accept `auto`, `cuda`, `cuda:0`, `cuda:1`, or `cpu`.

### Aliases

- `run_spatial_ot_input.sh`: backward-compatible alias for `run.sh`
- `run_p2_crc_multilevel_ot.sh`: pinned single-sample alias (`SAMPLE_KEY=p2_crc`)
- `run_p2_crc_multilevel_ot_exploratory_umap.sh`: same, but switches `FEATURE_OBSM_KEY` back to `X_umap_marker_genes_3d`
- `run_optimal_setting_search.sh`: launches a staged pooled-cohort parameter search; results land under `../work/spatial_ot_runs/cohort_optimal_search/`. Explicitly enables observed-hull fallback because the pooled bundle ships no explicit polygons — treat geometry as exploratory.

## Replotting per-sample niche maps

```bash
../.venv/bin/python -m spatial_ot plot-sample-niches \
  --run-dir ../outputs/spatial_ot/cohort_multilevel_ot \
  --output-dir ../outputs/spatial_ot/cohort_multilevel_ot/sample_niche_plots \
  --sample-obs-key sample_id \
  --plot-spatial-x-key cell_x --plot-spatial-y-key cell_y
```

For grid-built runs, `basic_niche_size_um` sets the smallest building block used to compose larger subregions (default `50 µm`).

## Output artifacts

Multilevel OT writes:

- `cells_multilevel_ot.h5ad`
- `subregions_multilevel_ot.parquet`
- `cluster_supports_multilevel_ot.npz`
- `multilevel_ot_spatial_map.png`, `multilevel_ot_subregion_embedding.png`, `multilevel_ot_atom_layouts.png`
- `summary.json`
- `deep_feature_model.pt`, `deep_feature_history.csv`, `deep_feature_config.json` (when deep features are enabled)

`summary.json` includes:

- a `deep_features` block describing whether a feature adapter was learned before OT
- restart summaries and the selected restart
- geometry-source counts and convex-hull fallback frequency
- assigned OT fallback frequency and the effective entropy values actually used by the solver
- requested vs resolved Torch compute device
- package version, git SHA, summary schema version
- graph usage metadata (whether the deep encoder used a spatial graph; training-graph degree statistics; `graph_max_neighbors`)
- `boundary_invariance_claim` (explicitly exploratory when observed-hull fallback was used)
- random-fold and spatial-block shape-leakage diagnostics
- canonical-normalizer radius / interpolation diagnostics

## Legacy train path

```bash
../.venv/bin/python -m spatial_ot train --config configs/p2_crc_smoke.toml
```

The smoke config points at the already-prepared P2 CRC Visium HD sample with intentionally small subset/epoch sizes. `configs/p2_crc_pilot.toml` is a larger but still subset-based variant.

```bash
../.venv/bin/python -m spatial_ot plot-inputs --config configs/p2_crc_smoke.toml
```

By default this writes `input_2d_overview.png` under the configured output directory.

## Config notes

- `subset_strategy` should be `spatial_grid` or `stratified`
- the active `multilevel-ot` path has its own TOML surface via `load_multilevel_config`
- overlap fallback from cells to the nearest 8 µm bin is controlled by `allow_nearest_overlap_fallback`

## Prior programs

`configs/crc_demo_programs.json` is a starter program library so the legacy path can run today. It is a demo prior set, not the final biology for any specific CRC project — replace with a curated panel and rerun the same CLI when marker panels arrive.
