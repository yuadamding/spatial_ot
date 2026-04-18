# spatial_ot

`spatial_ot` is a concrete research scaffold for a teacher-student spatial niche model on segmented Visium HD CRC.

The package currently realizes:

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

Current active-path notes:

- the active teacher branch consumes the `8 µm` bins
- the smoke and pilot configs are subset-based development runs, not full-sample production runs
- count validation expects integer-like raw counts and keeps both full-library and panel-library summaries internally

## Layout

- `spatial_ot/`: package code
- `configs/`: runnable config files and demo prior programs
- `runs/`: experiment outputs

## Smoke run

The smoke config points at the already-prepared `P2 CRC` Visium HD sample in this workspace and keeps the subset/epoch sizes intentionally small.

```bash
cd /storage/hackathon_2026/spatial_ot
conda run -n ml1 python -m spatial_ot train --config configs/p2_crc_smoke.toml
```

Expected outputs land under:

`/storage/hackathon_2026/spatial_ot/runs/p2_crc_smoke/`

Key artifacts:

- `cells_output.h5ad`
- `teacher_bins_output.h5ad`
- `summary.json`
- `niche_flows.csv`
- `top_flow_edges.parquet`
- stage checkpoints under `checkpoints/`

## Input visualization

You can render a 2D overview of the preprocessed inputs used by a config with:

```bash
cd /storage/hackathon_2026/spatial_ot
conda run -n ml1 python -m spatial_ot plot-inputs --config configs/p2_crc_smoke.toml
```

By default this writes `input_2d_overview.png` under the configured output directory.

## Multilevel OT on Cell-Level Features

`spatial_ot` now also exposes a redesigned multilevel OT path for cases where the core input is already a cell-level feature embedding, such as UMAP coordinates.

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
  --output-dir /storage/hackathon_2026/spatial_ot/runs/p2_crc_multilevel_umap \
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
  --align-iters 4
```

Key artifacts from this path:

- `cells_multilevel_ot.h5ad`
- `subregions_multilevel_ot.parquet`
- `cluster_supports_multilevel_ot.npz`
- `multilevel_ot_spatial_map.png`
- `multilevel_ot_subregion_embedding.png`
- `multilevel_ot_atom_layouts.png`
- `summary.json`

## Config notes

- `subset_strategy` should currently be `spatial_grid` or `stratified`
- the public config surface currently exposes only the active `8 µm` teacher path
- overlap fallback from cells to the nearest `8 µm` bin is controlled by `allow_nearest_overlap_fallback`

## Pilot config

`configs/p2_crc_pilot.toml` is a larger but still subset-based configuration intended as the next step after the smoke run.

## Prior programs

`configs/crc_demo_programs.json` is a starter library so the full path can run now. It is a demo prior set, not the final biology for your CRC project.

Once marker panels arrive, replace that file with a curated program library and rerun the same CLI.
