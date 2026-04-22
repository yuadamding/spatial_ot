#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-../.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
INPUT_H5AD="${INPUT_H5AD:-../spatial_ot_input/spatial_ot_input_pooled.h5ad}"
OUTPUT_DIR="${OUTPUT_DIR:-../work/spatial_ot_runs/cohort_optimal_search}"
FEATURE_OBSM_KEY="${FEATURE_OBSM_KEY:-X_spatial_ot_x_svd_512}"
SPATIAL_X_KEY="${SPATIAL_X_KEY:-pooled_cell_x}"
SPATIAL_Y_KEY="${SPATIAL_Y_KEY:-pooled_cell_y}"
SPATIAL_SCALE="${SPATIAL_SCALE:-0.2737012522439323}"
COMPUTE_DEVICE="${COMPUTE_DEVICE:-cuda}"
TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-20}"
N_CLUSTERS="${N_CLUSTERS:-8}"
ATOMS_PER_CLUSTER="${ATOMS_PER_CLUSTER:-8}"
RADIUS_UM="${RADIUS_UM:-80}"
STRIDE_UM="${STRIDE_UM:-80}"
BASIC_NICHE_SIZE_UM="${BASIC_NICHE_SIZE_UM:-50}"
MIN_CELLS="${MIN_CELLS:-25}"
MAX_SUBREGIONS="${MAX_SUBREGIONS:-1500}"
LAMBDA_X="${LAMBDA_X:-0.35}"
LAMBDA_Y="${LAMBDA_Y:-1.0}"
GEOMETRY_EPS="${GEOMETRY_EPS:-0.03}"
OT_EPS="${OT_EPS:-0.03}"
RHO="${RHO:-0.5}"
GEOMETRY_SAMPLES="${GEOMETRY_SAMPLES:-192}"
COMPRESSED_SUPPORT_SIZE="${COMPRESSED_SUPPORT_SIZE:-96}"
ALIGN_ITERS="${ALIGN_ITERS:-4}"
N_INIT="${N_INIT:-5}"
OVERLAP_CONSISTENCY_WEIGHT="${OVERLAP_CONSISTENCY_WEIGHT:-0.05}"
OVERLAP_JACCARD_MIN="${OVERLAP_JACCARD_MIN:-0.15}"
OVERLAP_CONTRAST_SCALE="${OVERLAP_CONTRAST_SCALE:-1.0}"
ALLOW_OBSERVED_HULL_GEOMETRY="${ALLOW_OBSERVED_HULL_GEOMETRY:-0}"
KEEP_TOP_K="${KEEP_TOP_K:-3}"
DEEP_FEATURE_METHOD="${DEEP_FEATURE_METHOD:-none}"
DEEP_OUTPUT_EMBEDDING="${DEEP_OUTPUT_EMBEDDING:-context}"
DEEP_DEVICE="${DEEP_DEVICE:-cuda}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtual environment python not found: $PYTHON_BIN" >&2
  echo "Run 'bash install_env.sh' from the spatial_ot directory first." >&2
  exit 1
fi

EXTRA_FLAGS=()
if [[ "$ALLOW_OBSERVED_HULL_GEOMETRY" == "1" ]]; then
  EXTRA_FLAGS+=(--allow-observed-hull-geometry)
else
  EXTRA_FLAGS+=(--no-allow-observed-hull-geometry)
fi

DEEP_FLAGS=()
if [[ "$DEEP_FEATURE_METHOD" != "none" ]]; then
  DEEP_FLAGS+=(
    --deep-feature-method "$DEEP_FEATURE_METHOD"
    --deep-output-embedding "$DEEP_OUTPUT_EMBEDDING"
    --deep-device "$DEEP_DEVICE"
  )
fi

"$PYTHON_BIN" -m spatial_ot optimal-search \
  --input-h5ad "$INPUT_H5AD" \
  --output-dir "$OUTPUT_DIR" \
  --feature-obsm-key "$FEATURE_OBSM_KEY" \
  --spatial-x-key "$SPATIAL_X_KEY" \
  --spatial-y-key "$SPATIAL_Y_KEY" \
  --spatial-scale "$SPATIAL_SCALE" \
  --compute-device "$COMPUTE_DEVICE" \
  --time-budget-hours "$TIME_BUDGET_HOURS" \
  --keep-top-k "$KEEP_TOP_K" \
  --n-clusters "$N_CLUSTERS" \
  --atoms-per-cluster "$ATOMS_PER_CLUSTER" \
  --radius-um "$RADIUS_UM" \
  --stride-um "$STRIDE_UM" \
  --basic-niche-size-um "$BASIC_NICHE_SIZE_UM" \
  --min-cells "$MIN_CELLS" \
  --max-subregions "$MAX_SUBREGIONS" \
  --lambda-x "$LAMBDA_X" \
  --lambda-y "$LAMBDA_Y" \
  --geometry-eps "$GEOMETRY_EPS" \
  --ot-eps "$OT_EPS" \
  --rho "$RHO" \
  --geometry-samples "$GEOMETRY_SAMPLES" \
  --compressed-support-size "$COMPRESSED_SUPPORT_SIZE" \
  --align-iters "$ALIGN_ITERS" \
  --n-init "$N_INIT" \
  --overlap-consistency-weight "$OVERLAP_CONSISTENCY_WEIGHT" \
  --overlap-jaccard-min "$OVERLAP_JACCARD_MIN" \
  --overlap-contrast-scale "$OVERLAP_CONTRAST_SCALE" \
  "${EXTRA_FLAGS[@]}" \
  "${DEEP_FLAGS[@]}"
