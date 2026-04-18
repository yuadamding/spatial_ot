#!/usr/bin/env bash
set -euo pipefail

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
  --align-iters 4 \
  --max-iter 10 \
  --tol 1e-4 \
  --seed 1337
