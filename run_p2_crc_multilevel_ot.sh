#!/usr/bin/env bash
set -euo pipefail

# Validated-ish default helper for local development.
# Prefer a metric-stable feature space such as X_pca and avoid
# observed-hull geometry fallback unless you explicitly want an
# exploratory local-pattern scan.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_H5AD="${INPUT_H5AD:-../data/cells.h5ad}"
OUTPUT_DIR="${OUTPUT_DIR:-../work/spatial_ot_runs/p2_crc_multilevel_pca}"
FEATURE_OBSM_KEY="${FEATURE_OBSM_KEY:-X_pca}"
SPATIAL_X_KEY="${SPATIAL_X_KEY:-cell_x}"
SPATIAL_Y_KEY="${SPATIAL_Y_KEY:-cell_y}"
SPATIAL_SCALE="${SPATIAL_SCALE:-0.2737012522439323}"
N_CLUSTERS="${N_CLUSTERS:-8}"
ATOMS_PER_CLUSTER="${ATOMS_PER_CLUSTER:-8}"
COMPUTE_DEVICE="${COMPUTE_DEVICE:-auto}"
BASIC_NICHE_SIZE_UM="${BASIC_NICHE_SIZE_UM:-200}"
ALLOW_OBSERVED_HULL_GEOMETRY="${ALLOW_OBSERVED_HULL_GEOMETRY:-0}"

EXTRA_FLAGS=()
if [[ "$ALLOW_OBSERVED_HULL_GEOMETRY" == "1" ]]; then
  EXTRA_FLAGS+=(--allow-observed-hull-geometry)
fi

conda run -n ml1 python -m spatial_ot multilevel-ot \
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
  "${EXTRA_FLAGS[@]}"
