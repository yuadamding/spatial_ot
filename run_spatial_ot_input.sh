#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-ml1}"
SAMPLE_KEY="${SAMPLE_KEY:-p2_crc}"
INPUT_H5AD="${INPUT_H5AD:-../spatial_ot_input/${SAMPLE_KEY}_cells_marker_genes_umap3d.h5ad}"
OUTPUT_DIR="${OUTPUT_DIR:-../work/spatial_ot_runs/${SAMPLE_KEY}_multilevel_umap_exploratory}"
FEATURE_OBSM_KEY="${FEATURE_OBSM_KEY:-X_umap_marker_genes_3d}"
SPATIAL_X_KEY="${SPATIAL_X_KEY:-cell_x}"
SPATIAL_Y_KEY="${SPATIAL_Y_KEY:-cell_y}"
SPATIAL_SCALE="${SPATIAL_SCALE:-0.2737012522439323}"
N_CLUSTERS="${N_CLUSTERS:-8}"
ATOMS_PER_CLUSTER="${ATOMS_PER_CLUSTER:-8}"
COMPUTE_DEVICE="${COMPUTE_DEVICE:-cuda}"
BASIC_NICHE_SIZE_UM="${BASIC_NICHE_SIZE_UM:-200}"

if [[ ! -f "$INPUT_H5AD" ]]; then
  echo "Missing input H5AD: $INPUT_H5AD" >&2
  exit 1
fi

conda run -n "$CONDA_ENV" python -m spatial_ot multilevel-ot \
  --input-h5ad "$INPUT_H5AD" \
  --output-dir "$OUTPUT_DIR" \
  --feature-obsm-key "$FEATURE_OBSM_KEY" \
  --spatial-x-key "$SPATIAL_X_KEY" \
  --spatial-y-key "$SPATIAL_Y_KEY" \
  --spatial-scale "$SPATIAL_SCALE" \
  --compute-device "$COMPUTE_DEVICE" \
  --n-clusters "$N_CLUSTERS" \
  --atoms-per-cluster "$ATOMS_PER_CLUSTER" \
  --radius-um 100 \
  --stride-um 150 \
  --basic-niche-size-um "$BASIC_NICHE_SIZE_UM" \
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
  --seed 1337 \
  --allow-observed-hull-geometry
