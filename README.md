# spatial_ot

`spatial_ot` is a compact research package for multilevel spatial subregion discovery and pooled feature-latent niche clustering.

The active path realizes:

- geometry-only OT normalization of each subregion into a shared reference domain for fixed-label atom/projection diagnostics
- an optional learned deep feature adapter with fit/transform/save/load behavior
- raw member-cell feature-distribution latent embeddings for each fitted subregion, pooled across the cohort for niche clustering
- cluster-specific shared heterogeneity atoms with subregion-specific mixture weights for downstream diagnostics
- semi-relaxed unbalanced OT matching with residual similarity alignment after labels are fixed, used for atom/projection QC rather than primary label assignment
- shape-leakage diagnostics so boundary geometry can be checked explicitly
- cell-level projection and visualization from the fitted subregion clusters

The deep `autoencoder` / `graph_autoencoder` layer is an experimental feature adapter, not yet a full generative niche model.

Always write outputs outside the package directory to keep it compact.

## Layout

- `spatial_ot/multilevel/`: active multilevel OT path
- `spatial_ot/deep/`: reusable deep feature adapter
- `spatial_ot/`: shared config/CLI, pooling, feature preparation, and compatibility entrypoints for the active method
- `scripts/`: canonical operational shell/Python helpers
- `configs/`: active multilevel OT TOML examples
- `tests/`: regression and multilevel OT tests

## Environment setup

`scripts/install_spatial_ot.sh` is the short install entrypoint; it delegates to `scripts/install_env.sh`, which creates or refreshes `../.venv` (next to `spatial_ot/` and `spatial_ot_input/`) and installs the package editable with parallel build/runtime thread defaults:

```bash
cd spatial_ot
bash scripts/install_spatial_ot.sh
```

Override interpreter / venv / extras via env vars:

```bash
PYTHON_BIN=python3.10 VENV_DIR=../.venv EXTRAS=dev,viz,geometry,parallel bash scripts/install_env.sh
```

By default the script searches for a compatible Python (3.13 → 3.12 → 3.11 → 3.10 → 3) and recreates `../.venv` if the existing one is incompatible.

The packaged path expects cohort H5AD inputs under the sibling directory `../spatial_ot_input/`.

Operational helpers live under `scripts/`; root-level shell wrappers were removed to keep the package compact.

## Multilevel OT — primary path

`spatial_ot multilevel-ot` forms spatial subregions, converts every subregion to either a cohort-comparable heterogeneity descriptor, a baseline feature-distribution embedding, or a transport-measure object, and clusters subregions with the selected representation.

Prefer PCA, standardized marker expression, or another calibrated latent space. UMAP is exploratory only — its Euclidean geometry is not metric-preserving.

The scalable default target mode remains `heterogeneity_descriptor_niche`: each subregion is represented as a block-normalized internal heterogeneity motif over compressed canonical coordinates and soft cell-state codebook posteriors. The clustered blocks are soft state composition, diversity/multimodality, a canonical spatial-state density field, and within-subregion state-pair co-occurrence enrichment. Block weights are exposed through CLI/config settings, and pair motifs can use all support pairs, local kNN pairs, or canonical-radius pairs with explicit distance bins. This mode uses internal shape-normalized arrangement, but not raw tissue position, subregion centers, sample labels, or shape/density descriptors. The older `pooled_subregion_latent` path remains as a composition/distribution-summary baseline; its default latent is `mean_std_shrunk`, with alternatives `mean_std`, `mean_std_skew_count`, `mean_std_quantile`, `codebook_histogram`, and `mean_std_codebook`. The legacy name `heterogeneity_ot_niche` is still accepted as an alias for the descriptor mode.

Two true transport-defined modes are also available for fixed-K validation-scale runs. `heterogeneity_fused_ot_niche` builds attributed subregion support measures and compares subregions with balanced OT over a fused feature-plus-canonical-coordinate ground cost. `heterogeneity_fgw_niche` compares the same measured support objects with fused Gromov-Wasserstein using point feature costs plus within-subregion canonical structure matrices. These modes set `uses_ot_costs=true`, build an all-pairs precomputed transport distance matrix, fit global feature/coordinate/structure cost scales from sampled subregion pairs, and cluster that matrix with average-linkage agglomerative clustering plus medoid diagnostics. They are intentionally guarded by `--heterogeneity-transport-max-subregions` because all-pairs fused OT/FGW is much more expensive than descriptor clustering. Hellinger feature costs are only valid for `--heterogeneity-transport-feature-mode soft_codebook`; use `sqeuclidean` for signed whitened feature modes, and use `split_marker_codebook` with `whitened_features_plus_soft_codebook` to combine squared-Euclidean marker costs with Hellinger codebook-posterior costs. Mixed marker/codebook features require the split cost, and balanced FGW defaults to three starts (`outer_product,feature_ot,coordinate_ot`) so initialization sensitivity is visible in distance-matrix metadata.

### Three-layer method

The active method should be interpreted as three linked layers:

- **Layer 1: subregion formation.** The biological unit is a mutually exclusive subregion: a small tissue region containing many cells, not a single cell or spot. Subregion boundaries can be coordinate-only, feature-aware, deep-graph refined, jointly refined after clustering, or supplied explicitly; realized size, shape, density, and minimum-size constraints are part of the method.
- **Layer 2: subregion heterogeneity clustering.** Each subregion is represented by either a fast heterogeneity descriptor, a true transport measure over canonical coordinate plus cell-state support points, or a baseline pooled feature-distribution summary. Shape/density/sample metadata are used for QC, not as clustering features. The `pooled_subregion_latent` composition-summary path remains available as a baseline.
- **Layer 3: projection and visualization.** Cell labels, spot-level latent fields, and sample maps are downstream projections from fitted subregion clusters. They are for interpretation and QC; they do not redefine the fitted subregion labels.

Supported modes:

- **coordinate-only data-driven subregion discovery** (default): mutually exclusive spatial subregions learned from observed coordinates, with sparse connected pieces merged to satisfy `min_cells`; the fitted boundary/shape geometry is taken from the observed member-cell point cloud rather than a hand-coded template
- **feature-aware data-driven subregion discovery**: set `--subregion-feature-weight > 0` to let the feature view influence generated boundaries; this is useful for sensitivity analysis but should be compared against the coordinate-only baseline because the same feature signal may later enter subregion heterogeneity clustering and OT diagnostics
- **deep graph segmentation**: set `--subregion-construction-method deep_segmentation` and provide a learned feature view, typically `--deep-feature-method autoencoder --deep-output-embedding context`; many coordinate seeds give full tissue coverage, then a spatial kNN boundary-refinement pass moves boundaries using learned embedding affinity before connected minimum-size merging
- **joint segmentation-clustering refinement**: set `--subregion-construction-method joint_refinement`; the method initializes with deep graph segmentation, fits preliminary subregion clusters using the active embedding target, then allows only adjacent boundary-cell moves that improve cluster-prototype coherence by at least `--joint-refinement-acceptance-margin` after spatial and kNN-cut penalties. Source connectivity and `min_cells` are checked during moves, then disconnected/small pieces are repaired before final clustering.
- **explicit-region clustering**: pass `--region-obs-key` and ideally explicit geometry objects so boundary shape can be treated as nuisance

The historical OT-dictionary assignment is still available as `--subregion-clustering-method ot_dictionary`, and the composition-summary baseline is available as `--subregion-clustering-method pooled_subregion_latent`. The packaged cohort workflow defaults to `heterogeneity_descriptor_niche` for scale, while `heterogeneity_fused_ot_niche` and `heterogeneity_fgw_niche` are intended for fixed-K validation, medoid checks, and smaller cohorts or landmarked subregion sets. OT atoms and spot-latent charts remain downstream diagnostics/projections conditioned on the fitted subregion labels.

For generated subregions, membership and boundary shape are data-driven from the observed member cells. `--radius-um` is not a fixed ball/window membership radius in this mode; it is retained for compatibility and graph diagnostics. The realized subregion scale comes from `--basic-niche-size-um` or `--stride-um`, `--min-cells`, `--max-subregions`, spatial connectivity, and optionally feature-aware boundary refinement or deep graph segmentation. `--max-subregion-area-um2` is a soft QC target, not a hard splitter: generated subregions are allowed to exceed it when that preserves connected data-driven regions and the `min_cells` constraint. For explicit-region runs without masks or polygons, boundary-shape invariance is not guaranteed unless you intentionally opt into the observed-coordinate convex-hull fallback via `--allow-observed-hull-geometry`.

For CLI explicit-region runs, pass `--region-geometry-json` with a JSON object containing `regions`, each with `region_id` plus `polygon_vertices`, `polygon_components`, or `mask`. Polygon coordinates are interpreted in scaled microns by default; set `"coordinate_units": "obs"` to multiply them by `--spatial-scale`.

### Quick run against packaged inputs

```bash
cd spatial_ot
../.venv/bin/python -m spatial_ot multilevel-ot \
  --input-h5ad ../spatial_ot_input/visium_hd_spatial_ot_input_pooled.h5ad \
  --output-dir ../outputs/spatial_ot/cohort_multilevel_ot \
  --feature-obsm-key X_spatial_ot_x_svd_512 \
  --spatial-x-key pooled_cell_x --spatial-y-key pooled_cell_y \
  --sample-obs-key sample_id \
  --spatial-scale 0.2737012522439323 \
  --compute-device cuda \
  --n-clusters 15 --atoms-per-cluster 8 \
  --radius-um 100 --stride-um 100 --basic-niche-size-um 50 \
  --min-cells 20 --max-subregions 5000 \
  --subregion-clustering-method heterogeneity_descriptor_niche \
  --heterogeneity-composition-weight 0.20 \
  --heterogeneity-diversity-weight 0.15 \
  --heterogeneity-spatial-field-weight 0.35 \
  --heterogeneity-pair-cooccurrence-weight 0.30 \
  --heterogeneity-pair-graph-mode all_pairs \
  --subregion-latent-embedding-mode mean_std_shrunk \
  --subregion-latent-heterogeneity-weight 0.5 \
  --subregion-latent-sample-prior-weight 0.5 \
  --lambda-x 0.5 --lambda-y 1.0 \
  --geometry-eps 0.03 --ot-eps 0.03 --rho 0.5 \
  --geometry-samples 192 --compressed-support-size 96 \
  --align-iters 4 --n-init 5
```

### Automatic K Selection

`multilevel-ot` can choose the number of subregion clusters before the final fit:

```bash
../.venv/bin/python -m spatial_ot multilevel-ot \
  --input-h5ad ../spatial_ot_input/visium_hd_spatial_ot_input_pooled.h5ad \
  --output-dir ../outputs/spatial_ot/cohort_multilevel_ot_auto_k \
  --feature-obsm-key X_spatial_ot_x_svd_512 \
  --spatial-x-key pooled_cell_x --spatial-y-key pooled_cell_y \
  --spatial-scale 0.2737012522439323 \
  --auto-n-clusters --candidate-n-clusters 15-25
```

Under `heterogeneity_descriptor_niche` and `pooled_subregion_latent`, the selector scores candidate `K` values directly on the corresponding subregion embedding using Silhouette, Gap, Calinski-Harabasz, Davies-Bouldin, seed stability, bootstrap stability, and forced-repair penalties. The historical OT-landmark selector is retained for `--subregion-clustering-method ot_dictionary`. Auto-K is intentionally not enabled for `heterogeneity_fused_ot_niche` or `heterogeneity_fgw_niche`; choose a fixed `K` from descriptor/model-selection runs, then run the true transport mode as a validation or refinement check. Auto-K should still be treated as exploratory until full fixed-K stability, leakage, heterogeneity-block ablations, and boundary-mode checks are run around the selected `K`.

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

1. pool all samples once into `../spatial_ot_input/visium_hd_spatial_ot_input_pooled.h5ad`
2. precompute the CPU-heavy `X → log1p-normalized TruncatedSVD` feature cache once into the pooled H5AD
3. fit the deep `autoencoder` adapter on that cached cohort matrix and feed the learned `context` embedding into OT
4. point repeated OT runs at the cached/learned matrices instead of recomputing CPU-heavy preprocessing

Helper scripts:

```bash
bash scripts/pool_spatial_ot_input.sh           # pool only → ../spatial_ot_input/visium_hd_spatial_ot_input_pooled.h5ad
bash scripts/prepare_spatial_ot_input.sh        # pool + precompute X → SVD into the pooled H5AD
bash scripts/prepare_all_spatial_ot_input.sh    # also verify each sample H5AD has the prepared cache
WRITE_BACK_TO_SOURCE_INPUTS=1 bash scripts/prepare_all_spatial_ot_input.sh
                                        # opt in to copying the cohort cache back into source files
bash scripts/prepare_xenium_spatial_ot_input.sh
                                        # pool all processed Xenium samples into one cohort H5AD
bash scripts/run_prepared_cohort_gpu.sh         # verify the prepared pooled H5AD, then launch the remaining GPU OT run
bash scripts/run_visium_hd_cohort_gpu.sh        # run the all-Visium HD pooled cohort high-VRAM deep-expression profile
bash scripts/run_deep_segmentation_cohort_gpu.sh
                                        # run the prepared cohort with autoencoder context features and joint refinement
bash scripts/run_xenium_cohort_gpu.sh           # run the all-Xenium pooled cohort profile
```

Pooled coordinates place each sample on its own non-overlapping tile, so samples contribute jointly to latent/OT learning without being treated as one continuous tissue section.

`scripts/run_prepared_cohort_gpu.sh` intentionally disables pooling and feature-cache refresh by default. It only accepts a pooled input that already has `X_spatial_ot_x_svd_512`, then delegates to `scripts/run.sh` with `COMPUTE_DEVICE=cuda` and `AUTO_N_CLUSTERS=1`.

`scripts/run_visium_hd_cohort_gpu.sh` and `scripts/run_xenium_cohort_gpu.sh` are separate dataset-specific high-VRAM deep-expression profiles. `scripts/run_deep_segmentation_cohort_gpu.sh` is the Visium HD deep-boundary profile: it trains/uses an autoencoder context embedding, sets `SUBREGION_CONSTRUCTION_METHOD=joint_refinement`, and then delegates to the same prepared-cohort runner. Use these profiles for the current deep-learning segmentation plus constrained segmentation-clustering feedback path; use the coordinate-only prepared run as the baseline/ablation.

`../spatial_ot_input/` is the default input address for both source H5ADs and direct pooled H5ADs. The processed Visium HD and Xenium source files are staged there, and the direct-use pooled inputs are `visium_hd_spatial_ot_input_pooled.h5ad` and `xenium_spatial_ot_input_pooled.h5ad`.

`scripts/prepare_xenium_spatial_ot_input.sh` mirrors the Visium HD cohort staging for the processed Xenium files under `../spatial_ot_input`: it matches `xenium_*_processed.h5ad`, strips the `xenium_` prefix and `_processed` suffix, lowercases sample IDs to `p1_crc`/`p2_crc`/`p5_crc`, uses `x_centroid`/`y_centroid`, and writes `../spatial_ot_input/xenium_spatial_ot_input_pooled.h5ad`. `scripts/run_xenium_cohort_gpu.sh` uses that pooled input with `SPATIAL_SCALE=1.0`, reads the prepared full-panel expression cache `X_spatial_ot_x_svd_421`, trains an intrinsic autoencoder latent, then feeds `X_spatial_ot_deep_expression_autoencoder` into the multilevel OT fit.

`obs` columns written by pooling:

- `sample_id`, `source_h5ad`
- `cell_x`, `cell_y` and explicit `original_cell_x`, `original_cell_y` (per-sample coordinates)
- `pooled_cell_x`, `pooled_cell_y` (pooled, sample-separated coordinates)

The default pooled feature key is `X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}` (the prepared cache), not a fresh per-run TruncatedSVD.

In this packaged workspace the input H5ADs may be managed by an external sync job, so `prepare_all_spatial_ot_input.sh` defaults to leaving source files untouched and writes only the pooled cohort cache.

## Packaged runner: `scripts/run.sh`

```bash
bash scripts/run.sh                                                             # default cohort run
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh                                       # pin to GPU 0
CUDA_VISIBLE_DEVICES=1 SAMPLE_KEY=p2_crc POOL_ALL_INPUTS=0 bash scripts/run.sh   # single-sample on GPU 1
COMPUTE_DEVICE=cpu bash scripts/run.sh                                           # CPU fallback
REFRESH_POOLED_INPUT=1 bash scripts/run.sh                                       # rebuild pooled input
REFRESH_PREPARED_FEATURES=1 bash scripts/run.sh                                  # rebuild prepared cache
```

`scripts/run.sh` reuses `../spatial_ot_input/visium_hd_spatial_ot_input_pooled.h5ad` and the prepared full-gene cache when present; only recomputes when missing or explicitly refreshed. After OT it writes one spatial niche PNG per sample under `../outputs/spatial_ot/cohort_multilevel_ot/sample_niche_plots/` using each sample's native within-sample coordinates. These niche maps show both fitted mutually exclusive subregion polygons and cell-wise inherited labels with the same cluster colors.

Defaults relevant to safety / cost (override with the matching env var):

- `COMPUTE_DEVICE=cuda`, `FEATURE_OBSM_KEY=X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}` when `PREPARE_INPUTS_AHEAD=1` (`X` otherwise)
- `BASIC_NICHE_SIZE_UM=50`, `MIN_CELLS=25`, `MAX_SUBREGIONS=5000`, `STRIDE_UM=$RADIUS_UM`
- `AUTO_N_CLUSTERS=0` by default. Set `AUTO_N_CLUSTERS=1` with `CANDIDATE_N_CLUSTERS=15-25` to run pilot-based model selection before the final fit.
- `MIN_SUBREGIONS_PER_CLUSTER=50` constrains the number of subregions per selected cluster; it does not constrain projected cell or spot counts.
- `MAX_SUBREGION_AREA_UM2`, when set, is reported as a soft area QC target. It does not force final connected regions to split and it does not override `MIN_CELLS`.
- `SUBREGION_FEATURE_WEIGHT=0`, `SUBREGION_FEATURE_DIMS=16` keep generated data-driven boundaries coordinate-only by default. Set a positive weight only for feature-aware boundary sensitivity runs.
- `SUBREGION_CONSTRUCTION_METHOD=joint_refinement` starts from learned-affinity deep graph segmentation, clusters the active embedding target, then runs bounded adjacent boundary moves with `JOINT_REFINEMENT_ITERS=2`, `JOINT_REFINEMENT_KNN=12`, `JOINT_REFINEMENT_MAX_MOVE_FRACTION=0.05`, and `JOINT_REFINEMENT_ACCEPTANCE_MARGIN=1e-3`. `SUBREGION_CONSTRUCTION_METHOD=deep_segmentation` remains available to disable the cluster-aware feedback step. Treat both as opt-in boundary models and compare them against coordinate-only construction.
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

`run.sh` detects the available CPU core count with `getconf`/`nproc` and exports a coordinated thread budget:

- `OMP_NUM_THREADS=$CPU_THREADS`, `MKL_NUM_THREADS=$CPU_THREADS`, `OPENBLAS_NUM_THREADS=$CPU_THREADS`, `NUMEXPR_NUM_THREADS=$CPU_THREADS`, `BLIS_NUM_THREADS=$CPU_THREADS`, `VECLIB_MAXIMUM_THREADS=$CPU_THREADS`, `NUMBA_NUM_THREADS=$CPU_THREADS`
- `SPATIAL_OT_TORCH_NUM_THREADS=$TORCH_INTRAOP_THREADS`, `SPATIAL_OT_TORCH_NUM_INTEROP_THREADS=4`
- `SPATIAL_OT_X_SVD_COMPONENTS=512`, `SPATIAL_OT_X_TARGET_SUM=10000`

Override with `CPU_THREADS`, `TORCH_INTRAOP_THREADS`, `TORCH_INTEROP_THREADS`, `X_FEATURE_COMPONENTS`, `X_TARGET_SUM`, or the underlying library env vars.

For multi-GPU restart parallelism in the outer `n_init` loop:

- `SPATIAL_OT_CUDA_DEVICE_LIST=all` (eligible devices)
- `SPATIAL_OT_PARALLEL_RESTARTS=auto` (spread independent restarts across GPUs when `n_init > 1`)
- `SPATIAL_OT_CUDA_TARGET_VRAM_GB=70` (per-device working set, capped to `SPATIAL_OT_CUDA_MAX_TARGET_FRACTION`, default `0.92` in high-VRAM profiles, of visible memory)

The high-VRAM profiles source `scripts/_high_vram_deep_profile.sh`, which defaults to `DEEP_TARGET_VRAM_GB=70`, `DEEP_HIDDEN_DIM=8192`, `DEEP_LAYERS=6`, `DEEP_LATENT_DIM=256`, and `DEEP_BATCH_SIZE=131072`. The guard checks visible CUDA memory before launch and fails early on undersized GPUs; set `CHECK_HIGH_VRAM_GPU=0` only for dry-run configuration checks, or lower the deep-model env vars for a smaller GPU. Each CUDA run writes `runtime_memory_qc` into `summary.json` with the requested target and observed peak reserved memory.

When restart workers run in parallel, the Torch/BLAS thread budget is divided across them automatically. The multilevel OT and deep-feature implementations are still single-GPU within any one restart.

The Python API and TOML config still accept `auto`, `cuda`, `cuda:0`, `cuda:1`, or `cpu`.

### Additional Scripts

- `scripts/run_optimal_setting_search.sh`: launches a staged pooled-cohort parameter search on the same prepared-feature + deep-context path used by `scripts/run.sh`; results land under `../work/spatial_ot_runs/cohort_optimal_search/`. Generated runs use data-driven observed-coordinate membership and geometry, so observed-hull fallback stays off unless you opt in for exploratory explicit-region runs missing geometry.
- For single-sample runs, use `POOL_ALL_INPUTS=0 SAMPLE_KEY=p2_crc bash scripts/run.sh`.
- For exploratory UMAP runs, set `FEATURE_OBSM_KEY=X_umap_marker_genes_3d ALLOW_UMAP_AS_FEATURE=1` explicitly.

## Replotting per-sample niche maps

```bash
../.venv/bin/python -m spatial_ot plot-sample-niches \
  --run-dir ../outputs/spatial_ot/cohort_multilevel_ot \
  --output-dir ../outputs/spatial_ot/cohort_multilevel_ot/sample_niche_plots \
  --sample-obs-key sample_id \
  --plot-spatial-x-key cell_x --plot-spatial-y-key cell_y
```

The niche-map command writes a paired figure for each sample: a subregion-wise filled polygon panel and a cell-wise inherited-label scatter panel. It reads `mlot_subregion_id` from `cells_multilevel_ot.h5ad` when available, otherwise recovers subregion membership from `spot_level_latent_multilevel_ot.npz`.

Per-sample spot-latent fields are rendered as one whole-slide map per sample. The default stored spot latent is an OT-grounded atom-barycentric MDS chart: fitted cluster atom measures define global anchors using balanced OT distances between atom measures, and each spot/cell occurrence is placed inside its assigned cluster by barycentering that cluster's atom embedding with an entropy-calibrated OT atom posterior. Raw aligned x/y coordinates are not concatenated into the default chart features, and cluster-local variance is not forced to a fixed display radius. The supervised Fisher/local-PCA chart remains available only as an explicit diagnostic mode through `SPATIAL_OT_SPOT_LATENT_MODE=diagnostic_fisher_current`. The slide map uses global latent color scaling by default so colors are comparable across clusters and samples; within-cluster RGB rescaling is available only through `SPATIAL_OT_SPOT_LATENT_COLOR_SCALE=within_cluster` as a diagnostic. The side key is rendered as a 3D global latent plot: when anchor distances are saved, the key uses 3D MDS of those model-grounded distances and overlays within-cluster barycentric residuals. Anchor-distance fallbacks are recorded explicitly; any balanced/sinkhorn anchor OT fallback blocks within-niche latent-heterogeneity claims. This remains diagnostic visualization, not independent validation of niche discovery; use MDS stress/eigenvalue diagnostics, posterior entropy, atom-argmax maps, fixed/cost-gap/entropy-calibrated temperature summaries, stability, leakage, and held-out-sample checks before interpreting biological heterogeneity.

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
- `joint_refinement_energy.csv`, `joint_refinement_moves.parquet`, `cell_to_subregion_initial.npy`, `cell_to_subregion_refined.npy`, `cluster_labels_initial.npy`, and `cluster_labels_refined.npy` when `joint_refinement` is enabled
- `cluster_supports_multilevel_ot.npz`
- `spot_level_latent_multilevel_ot.npz` with occurrence-level `(subregion, spot)` OT atom-barycentric latent coordinates plus posterior entropy, atom argmax, effective/cost-gap/fixed temperature diagnostics, balanced-OT cluster-anchor distances, anchor-OT fallback diagnostics, MDS diagnostics, cluster anchors, and atom embeddings
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

The report writes `concern_resolution_report.json` and `.md` under the run directory. Packaged runs write this report by default; set `WRITE_CONCERN_REPORT=0` only for quick local debugging. Use `STRICT_CONCERN_REPORT=1` in shell runs, or `--strict` in the CLI, to fail when blockers remain. If the primary run used deep/feature-aware boundaries, the report gives a coordinate-only baseline command. If auto-K was used, it gives fixed-K stability commands around the selected K. Shape/density leakage ablation runs can be passed back into the report with repeated `--leakage-ablation-run-dir` arguments.

The report also includes a `validation_suite` block with executable commands for fixed-K stability, shrinkage-τ sensitivity, sample-prior-weight sensitivity, heterogeneity-weight sensitivity, codebook-size sensitivity, scale/leakage ablations, and spatial niche validation. The spatial validation command can be run directly:

```bash
../.venv/bin/python -m spatial_ot spatial-niche-validation \
  --run-dir ../outputs/spatial_ot/cohort_multilevel_ot \
  --sample-obs-key sample_id \
  --max-subregions 50000 \
  --knn 6
```

This writes `spatial_niche_validation.json` and `spatial_niche_cluster_statistics.csv`, including per-cluster cell-count/area/shrinkage summaries, sample mixing, within-sample kNN label homophily with permutation p-values/z-scores, and connected-component fragmentation summaries. Primary biological claims should be made only after the coordinate-only baseline, leakage/null checks, OT cost-comparability checks, fixed-K stability runs, shrinkage/heterogeneity/codebook sensitivity, and spatial organization checks agree.

## Config notes

- The active `multilevel-ot` path uses the TOML surface loaded by `load_multilevel_config`.
- The old teacher-student configs and program-prior demos were removed from the compact package layout.
