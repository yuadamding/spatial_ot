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
- `scripts/`: canonical operational shell/Python helpers; root-level shell files are tiny compatibility wrappers
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

New operational helpers should live under `scripts/`; the root wrappers remain only so existing commands like `bash run.sh` and `bash install_env.sh` keep working.

## Multilevel OT — primary path

`spatial_ot multilevel-ot` runs shape-normalized cluster-specific semi-relaxed Wasserstein clustering on subregions built from cell-level features.

Prefer PCA, standardized marker expression, or another calibrated latent space. UMAP is exploratory only — its Euclidean geometry is not metric-preserving.

### Three-layer method

The active method should be interpreted as three linked layers:

- **Layer 1: subregion formation.** The biological unit is a mutually exclusive subregion: a small tissue region containing many cells, not a single cell or spot. Subregion boundaries can be coordinate-only, feature-aware, deep-graph refined, or supplied explicitly; realized size, shape, density, and minimum-size constraints are part of the method.
- **Layer 2: subregion heterogeneity clustering.** Each subregion is represented as a compressed empirical measure over canonical within-subregion coordinates and cell-level features. Clustering compares the internal heterogeneity of these subregion measures with semi-relaxed OT against shared cluster atoms. This is the primary niche-clustering layer.
- **Layer 3: projection and visualization.** Cell labels, spot-level latent fields, and sample maps are downstream projections from fitted subregion clusters. They are for interpretation and QC; they do not redefine the fitted subregion labels.

Supported modes:

- **coordinate-only data-driven subregion discovery** (default): mutually exclusive spatial subregions learned from observed coordinates, with sparse connected pieces merged to satisfy `min_cells`; the fitted boundary/shape geometry is taken from the observed member-cell point cloud rather than a hand-coded template
- **feature-aware data-driven subregion discovery**: set `--subregion-feature-weight > 0` to let the OT feature view influence generated boundaries; this is useful for sensitivity analysis but should be compared against the coordinate-only baseline because the same feature signal may later enter OT clustering
- **deep graph segmentation**: set `--subregion-construction-method deep_segmentation` and provide a learned feature view, typically `--deep-feature-method autoencoder --deep-output-embedding context`; many coordinate seeds give full tissue coverage, then a spatial kNN boundary-refinement pass moves boundaries using learned embedding affinity before minimum-size merging
- **explicit-region clustering**: pass `--region-obs-key` and ideally explicit geometry objects so boundary shape can be treated as nuisance

For generated subregions, membership and boundary shape are data-driven from the observed member cells. `--radius-um` is not a fixed ball/window membership radius in this mode; it is retained for compatibility and graph diagnostics. The realized subregion scale comes from `--basic-niche-size-um` or `--stride-um`, `--min-cells`, `--max-subregions`, spatial connectivity, and optionally feature-aware boundary refinement or deep graph segmentation. For explicit-region runs without masks or polygons, boundary-shape invariance is not guaranteed unless you intentionally opt into the observed-coordinate convex-hull fallback via `--allow-observed-hull-geometry`.

For CLI explicit-region runs, pass `--region-geometry-json` with a JSON object containing `regions`, each with `region_id` plus `polygon_vertices`, `polygon_components`, or `mask`. Polygon coordinates are interpreted in scaled microns by default; set `"coordinate_units": "obs"` to multiply them by `--spatial-scale`.

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
  --radius-um 100 --stride-um 100 --basic-niche-size-um 50 \
  --min-cells 20 --max-subregions 2000 \
  --lambda-x 0.5 --lambda-y 1.0 \
  --geometry-eps 0.03 --ot-eps 0.03 --rho 0.5 \
  --geometry-samples 192 --compressed-support-size 96 \
  --align-iters 4 --n-init 5
```

### Automatic K Selection

`multilevel-ot` can choose the number of subregion clusters before the final fit:

```bash
../.venv/bin/python -m spatial_ot multilevel-ot \
  --input-h5ad ../spatial_ot_input/spatial_ot_input_pooled.h5ad \
  --output-dir ../outputs/spatial_ot/cohort_multilevel_ot_auto_k \
  --feature-obsm-key X_spatial_ot_x_svd_512 \
  --spatial-x-key pooled_cell_x --spatial-y-key pooled_cell_y \
  --spatial-scale 0.2737012522439323 \
  --auto-n-clusters --candidate-n-clusters 15-25
```

The selector runs one pilot OT fit at the largest candidate `K`, builds a scalable OT-landmark distance between subregions from the pilot fused transport-cost profiles, then scores candidate `K` values with Silhouette on that precomputed distance plus Gap / Calinski-Harabasz / Davies-Bouldin on a classical MDS embedding. The final `K` is chosen by majority vote and the model is refit at that selected `K`. This avoids all-pairs subregion OT and avoids running a full final fit for every candidate `K`, but it should be treated as exploratory model selection until stability and ablation checks are run around the selected `K`.

The cluster-size constraint is defined on fitted subregions, not on cells or spots. Set `MIN_SUBREGIONS_PER_CLUSTER=50` or pass `--min-subregions-per-cluster 50` to require each subregion cluster to contain at least that many subregions when feasible; projected cell and spot labels remain downstream summaries.

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
  --deep-output-embedding joint \
  --deep-allow-joint-ot-embedding
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
bash run_prepared_cohort_gpu.sh         # verify the prepared pooled H5AD, then launch the remaining GPU OT run
bash run_deep_segmentation_cohort_gpu.sh
                                        # run the prepared cohort with autoencoder context features and deep graph segmentation
```

Pooled coordinates place each sample on its own non-overlapping tile, so samples contribute jointly to latent/OT learning without being treated as one continuous tissue section.

`run_prepared_cohort_gpu.sh` intentionally disables pooling and feature-cache refresh by default. It only accepts a pooled input that already has `X_spatial_ot_x_svd_512`, then delegates to `run.sh` with `COMPUTE_DEVICE=cuda` and `AUTO_N_CLUSTERS=1`.

`run_deep_segmentation_cohort_gpu.sh` is the deep-boundary variant: it trains/uses an autoencoder context embedding, sets `SUBREGION_CONSTRUCTION_METHOD=deep_segmentation`, and then delegates to the same prepared-cohort runner.

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

`run.sh` reuses `../spatial_ot_input/spatial_ot_input_pooled.h5ad` and the prepared full-gene cache when present; only recomputes when missing or explicitly refreshed. After OT it writes one spatial niche PNG per sample under `../outputs/spatial_ot/cohort_multilevel_ot/sample_niche_plots/` using each sample's native within-sample coordinates. These niche maps show both fitted mutually exclusive subregion polygons and cell-wise inherited labels with the same cluster colors.

Defaults relevant to safety / cost (override with the matching env var):

- `COMPUTE_DEVICE=cuda`, `FEATURE_OBSM_KEY=X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}` when `PREPARE_INPUTS_AHEAD=1` (`X` otherwise)
- `BASIC_NICHE_SIZE_UM=50`, `MIN_CELLS=25`, `MAX_SUBREGIONS=5000`, `STRIDE_UM=$RADIUS_UM`
- `AUTO_N_CLUSTERS=0` by default. Set `AUTO_N_CLUSTERS=1` with `CANDIDATE_N_CLUSTERS=15-25` to run pilot-based model selection before the final fit.
- `MIN_SUBREGIONS_PER_CLUSTER=50` constrains the number of subregions per selected cluster; it does not constrain projected cell or spot counts.
- `SUBREGION_FEATURE_WEIGHT=0`, `SUBREGION_FEATURE_DIMS=16` keep generated data-driven boundaries coordinate-only by default. Set a positive weight only for feature-aware boundary sensitivity runs.
- `SUBREGION_CONSTRUCTION_METHOD=deep_segmentation` switches generated subregion detection to learned-affinity graph segmentation. `DEEP_SEGMENTATION_KNN=12`, `DEEP_SEGMENTATION_FEATURE_DIMS=32`, `DEEP_SEGMENTATION_FEATURE_WEIGHT=1.0`, and `DEEP_SEGMENTATION_SPATIAL_WEIGHT=0.05` control the graph cut. Treat this as an opt-in boundary model and compare it against coordinate-only construction.
- `N_INIT=2`, `MAX_ITER=5`, `ALIGN_ITERS=2`, `GEOMETRY_SAMPLES=64`, `COMPRESSED_SUPPORT_SIZE=48` for the packaged cohort runner; raise these for a final high-depth confirmation run.
- `SINKHORN_MAX_ITER=256`, `SINKHORN_TOL=1e-4` for CUDA OT solves in the packaged runner.
- `SHAPE_LEAKAGE_PERMUTATIONS=16` keeps the default diagnostic pass light; raise it for publication-quality QC.
- `SPATIAL_OT_LEAKAGE_RF_ESTIMATORS=120` controls the random-forest size used for shape and size/density leakage diagnostics.
- `LIGHT_CELL_H5AD=1`, `H5AD_COMPRESSION=lzf`, `WRITE_SAMPLE_SPATIAL_MAPS=0` keep the default cohort output fast and visualization-focused; set `LIGHT_CELL_H5AD=0` to copy the full input matrix into `cells_multilevel_ot.h5ad`.
- `PROGRESS_LOG=1` prints major fit stages and restart completion times during long cohort runs.
- `REQUIRE_FULL_CELL_COVERAGE=0` (capped cohort runs report incomplete analyzed-subregion coverage instead of failing; set to `1` after tuning `STRIDE_UM`, `BASIC_NICHE_SIZE_UM`, and `MAX_SUBREGIONS` when every spot must have a niche-context latent occurrence)
- `ALLOW_OBSERVED_HULL_GEOMETRY=0` (disabled — opt in only for exploratory runs)
- `DEEP_FEATURE_METHOD=none` by default for fast pooled-cohort iteration. Set `DEEP_FEATURE_METHOD=autoencoder` or `graph_autoencoder` when you intentionally want to retrain the experimental deep adapter; set `DEEP_ALLOW_JOINT_OT_EMBEDDING=1` when intentionally using the `joint` deep embedding as the OT view.

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
- `run_optimal_setting_search.sh`: launches a staged pooled-cohort parameter search on the same prepared-feature + deep-context path used by `run.sh`; results land under `../work/spatial_ot_runs/cohort_optimal_search/`. Generated runs use data-driven observed-coordinate membership and geometry, so observed-hull fallback stays off unless you opt in for exploratory explicit-region runs missing geometry.

## Replotting per-sample niche maps

```bash
../.venv/bin/python -m spatial_ot plot-sample-niches \
  --run-dir ../outputs/spatial_ot/cohort_multilevel_ot \
  --output-dir ../outputs/spatial_ot/cohort_multilevel_ot/sample_niche_plots \
  --sample-obs-key sample_id \
  --plot-spatial-x-key cell_x --plot-spatial-y-key cell_y
```

The niche-map command writes a paired figure for each sample: a subregion-wise filled polygon panel and a cell-wise inherited-label scatter panel. It reads `mlot_subregion_id` from `cells_multilevel_ot.h5ad` when available, otherwise recovers subregion membership from `spot_level_latent_multilevel_ot.npz`.

Per-sample spot-latent fields are rendered as one whole-slide map per sample. The default stored spot latent is now an OT-grounded atom-barycentric MDS chart: fitted cluster atom measures define global anchors, and each spot/cell occurrence is placed inside its assigned cluster by barycentering that cluster's atom embedding with a cost-gap-calibrated OT atom posterior. Raw aligned x/y coordinates are not concatenated into the default chart features, and cluster-local variance is not forced to a fixed display radius. The legacy supervised Fisher/local-PCA chart remains available only as an explicit diagnostic mode through `SPATIAL_OT_SPOT_LATENT_MODE=diagnostic_fisher_current`. The slide map rescales colors within each niche/cluster before RGB conversion, overlays subregion outlines, and uses the side key to show the model-grounded latent layout. This remains diagnostic visualization, not independent validation of niche discovery; use posterior entropy, atom-argmax maps, effective-temperature summaries, stability, leakage, and held-out-sample checks before interpreting biological heterogeneity.

```bash
../.venv/bin/python -m spatial_ot plot-sample-spot-latent \
  --run-dir ../outputs/spatial_ot/cohort_multilevel_ot \
  --output-dir ../outputs/spatial_ot/cohort_multilevel_ot/sample_spot_latent_plots \
  --sample-obs-key sample_id \
  --plot-spatial-x-key cell_x --plot-spatial-y-key cell_y
```

For generated runs, `basic_niche_size_um` is only a target scale hint for data-driven atomic membership seeds (default `50 µm`). Fitted subregions are mutually exclusive by construction. By default, seed boundaries are learned from spatial position only; feature-aware or deep boundaries are explicit opt-in modes and should be supported by coordinate-only, feature-weight, and held-out-sample ablations. Sparse connected pieces are merged to satisfy `min_cells`, and OT boundary geometry is learned from the observed coordinates of the cells inside each final subregion. The package reports realized size/shape/density statistics because those realized subregions, not a nominal radius, are the biological units being clustered.

## Output artifacts

Multilevel OT writes:

- `cells_multilevel_ot.h5ad`
- `subregions_multilevel_ot.parquet`
- `cluster_supports_multilevel_ot.npz`
- `spot_level_latent_multilevel_ot.npz` with occurrence-level `(subregion, spot)` OT atom-barycentric latent coordinates plus posterior entropy, atom argmax, effective temperature, cluster anchors, and atom embeddings
- `multilevel_ot_spatial_map.png`, `multilevel_ot_subregion_embedding.png`, `multilevel_ot_atom_layouts.png`
- `summary.json`
- `deep_feature_model.pt`, `deep_feature_history.csv`, `deep_feature_config.json` (when deep features are enabled)

`summary.json` includes:

- a `deep_features` block describing whether a feature adapter was learned before OT
- restart summaries and the selected restart
- geometry-source counts and convex-hull fallback frequency
- `subregion_construction`, `radius_um_semantics`, and `realized_subregion_statistics` blocks describing the actual clustered subregion units
- assigned OT fallback frequency and the effective entropy values actually used by the solver
- requested vs resolved Torch compute device
- package version, git SHA, summary schema version
- graph usage metadata (whether the deep encoder used a spatial graph; training-graph degree statistics; `graph_max_neighbors`)
- `boundary_invariance_claim` (explicit geometry supports the strongest claim; observed point-cloud geometry is reported as normalized but not fully shape-invariant, and observed-hull fallback is exploratory)
- random-fold and spatial-block shape-leakage diagnostics plus size/density leakage diagnostics
- canonical-normalizer radius / interpolation diagnostics

## Concern Resolution

Deep/feature-aware boundaries, shape/density leakage, and automatic K selection are reported as validation concerns rather than hidden. After a run, write an explicit remediation report:

```bash
../.venv/bin/python -m spatial_ot validate-run-concerns \
  --run-dir ../outputs/spatial_ot/cohort_multilevel_ot \
  --coordinate-baseline-run-dir ../outputs/spatial_ot/cohort_multilevel_ot_coordinate_only_baseline \
  --stability-run-dir ../outputs/spatial_ot/cohort_multilevel_ot_fixed_k16_seed1338 \
  --leakage-ablation-run-dir ../outputs/spatial_ot/cohort_multilevel_ot_shape_ablation \
  --leakage-ablation-run-dir ../outputs/spatial_ot/cohort_multilevel_ot_density_ablation \
  --strict
```

The report writes `concern_resolution_report.json` and `.md` under the run directory. Packaged runs write this report by default; set `WRITE_CONCERN_REPORT=0` only for quick local debugging. Use `STRICT_CONCERN_REPORT=1` in shell runs, or `--strict` in the CLI, to fail when blockers remain. If the primary run used deep/feature-aware boundaries, the report gives a coordinate-only baseline command. If auto-K was used, it gives fixed-K stability commands around the selected K. Shape/density leakage ablation runs can be passed back into the report with repeated `--leakage-ablation-run-dir` arguments. Primary biological claims should be made only after the coordinate-only baseline, leakage/null checks, OT cost-comparability checks, and fixed-K stability runs agree.

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
