#!/usr/bin/env bash
set -euo pipefail

cd /storage/hackathon_2026/spatial_ot

INPUT_H5AD="${INPUT_H5AD:-/storage/hackathon_2026/work/visium_hd_p2_crc/exports/p2_crc_cells_marker_genes_umap3d_rgb.h5ad}"
OUTPUT_DIR="${OUTPUT_DIR:-/storage/hackathon_2026/work/spatial_ot_runs/p2_crc_multilevel_umap}"
FEATURE_OBSM_KEY="${FEATURE_OBSM_KEY:-X_umap_marker_genes_3d}"
SPATIAL_X_KEY="${SPATIAL_X_KEY:-cell_x}"
SPATIAL_Y_KEY="${SPATIAL_Y_KEY:-cell_y}"
SPATIAL_SCALE="${SPATIAL_SCALE:-0.2737012522439323}"
N_CLUSTERS="${N_CLUSTERS:-8}"
ATOMS_PER_CLUSTER="${ATOMS_PER_CLUSTER:-8}"

conda run -n ml1 python -m spatial_ot multilevel-ot \
  --input-h5ad "$INPUT_H5AD" \
  --output-dir "$OUTPUT_DIR" \
  --feature-obsm-key "$FEATURE_OBSM_KEY" \
  --spatial-x-key "$SPATIAL_X_KEY" \
  --spatial-y-key "$SPATIAL_Y_KEY" \
  --spatial-scale "$SPATIAL_SCALE" \
  --n-clusters "$N_CLUSTERS" \
  --atoms-per-cluster "$ATOMS_PER_CLUSTER" \
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
  --max-iter 10 \
  --tol 1e-4 \
  --seed 1337
