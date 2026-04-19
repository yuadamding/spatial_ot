# spatial_ot

`spatial_ot` is a compact research package for multilevel, shape-normalized semi-relaxed Wasserstein clustering on spatial subregions.

The current primary path realizes:

- geometry-only OT normalization of each subregion into a shared reference domain
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

- `spatial_ot/`: package code
- `configs/`: config files and demo prior programs
- `tests/`: regression and multilevel OT tests

## Legacy train path

The smoke config points at the already-prepared `P2 CRC` Visium HD sample in this workspace and keeps the subset/epoch sizes intentionally small.

```bash
cd /storage/hackathon_2026/spatial_ot
conda run -n ml1 python -m spatial_ot train --config configs/p2_crc_smoke.toml
```

Use an output path outside `/storage/hackathon_2026/spatial_ot` if you want to keep the package directory clean.

## Input visualization

You can render a 2D overview of the preprocessed inputs used by a config with:

```bash
cd /storage/hackathon_2026/spatial_ot
conda run -n ml1 python -m spatial_ot plot-inputs --config configs/p2_crc_smoke.toml
```

By default this writes `input_2d_overview.png` under the configured output directory.

## Multilevel OT on Cell-Level Features

`spatial_ot` now also exposes a redesigned multilevel OT path for cases where the core input is already a cell-level feature embedding, such as UMAP coordinates.

Prefer PCA, standardized marker expression, or another calibrated latent space for serious OT runs. UMAP can be useful for exploratory clustering and visualization, but its Euclidean geometry is not generally metric-preserving.

This path:

- builds overlapping spatial subregions from cells
- learns a geometry-only OT map that normalizes each subregion into a shared reference domain
- compresses each subregion into a smaller empirical measure over canonical coordinates plus features
- learns cluster-specific shared heterogeneity atoms and subregion-specific mixture weights
- matches subregions to clusters with semi-relaxed unbalanced OT plus residual similarity alignment
- projects the learned labels back to cells for visualization
- reports shape-leakage diagnostics so we can check whether boundary geometry is still driving cluster labels

Example on the `P2 CRC` cell-level marker-gene UMAP object:

```bash
cd /storage/hackathon_2026/spatial_ot
conda run -n ml1 python -m spatial_ot multilevel-ot \
  --input-h5ad /storage/hackathon_2026/work/visium_hd_p2_crc/exports/p2_crc_cells_marker_genes_umap3d_rgb.h5ad \
  --output-dir /storage/hackathon_2026/work/spatial_ot_runs/p2_crc_multilevel_umap \
  --feature-obsm-key X_umap_marker_genes_3d \
  --spatial-x-key cell_x \
  --spatial-y-key cell_y \
  --spatial-scale 0.2737012522439323 \
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
  --n-init 5
```

Key artifacts from this path:

- `cells_multilevel_ot.h5ad`
- `subregions_multilevel_ot.parquet`
- `cluster_supports_multilevel_ot.npz`
- `multilevel_ot_spatial_map.png`
- `multilevel_ot_subregion_embedding.png`
- `multilevel_ot_atom_layouts.png`
- `summary.json`

The saved summary now includes:

- restart summaries and the selected restart
- geometry-source counts and convex-hull fallback frequency
- random-fold and spatial-block shape-leakage diagnostics
- canonical-normalizer radius / interpolation diagnostics

## Config notes

- `subset_strategy` should currently be `spatial_grid` or `stratified`
- the public TOML config surface currently applies to the legacy `train` path
- overlap fallback from cells to the nearest `8 µm` bin is controlled by `allow_nearest_overlap_fallback`

## Pilot config

`configs/p2_crc_pilot.toml` is a larger but still subset-based configuration intended as the next step after the smoke run.

## Prior programs

`configs/crc_demo_programs.json` is a starter library so the full path can run now. It is a demo prior set, not the final biology for your CRC project.

Once marker panels arrive, replace that file with a curated program library and rerun the same CLI.
