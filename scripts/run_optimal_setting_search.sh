#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

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
MIN_SUBREGIONS_PER_CLUSTER="${MIN_SUBREGIONS_PER_CLUSTER:-50}"
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
SUBREGION_CONSTRUCTION_METHOD="${SUBREGION_CONSTRUCTION_METHOD:-data_driven}"
SUBREGION_FEATURE_WEIGHT="${SUBREGION_FEATURE_WEIGHT:-0}"
SUBREGION_FEATURE_DIMS="${SUBREGION_FEATURE_DIMS:-16}"
DEEP_SEGMENTATION_KNN="${DEEP_SEGMENTATION_KNN:-12}"
DEEP_SEGMENTATION_FEATURE_DIMS="${DEEP_SEGMENTATION_FEATURE_DIMS:-32}"
DEEP_SEGMENTATION_FEATURE_WEIGHT="${DEEP_SEGMENTATION_FEATURE_WEIGHT:-1.0}"
DEEP_SEGMENTATION_SPATIAL_WEIGHT="${DEEP_SEGMENTATION_SPATIAL_WEIGHT:-0.05}"
ALLOW_OBSERVED_HULL_GEOMETRY="${ALLOW_OBSERVED_HULL_GEOMETRY:-0}"
REGION_OBS_KEY="${REGION_OBS_KEY:-}"
REGION_GEOMETRY_JSON="${REGION_GEOMETRY_JSON:-}"
SHAPE_DIAGNOSTICS="${SHAPE_DIAGNOSTICS:-1}"
SHAPE_LEAKAGE_PERMUTATIONS="${SHAPE_LEAKAGE_PERMUTATIONS:-64}"
KEEP_TOP_K="${KEEP_TOP_K:-3}"
DEEP_FEATURE_METHOD="${DEEP_FEATURE_METHOD:-autoencoder}"
DEEP_OUTPUT_EMBEDDING="${DEEP_OUTPUT_EMBEDDING:-context}"
DEEP_OUTPUT_OBSM_KEY="${DEEP_OUTPUT_OBSM_KEY:-X_spatial_ot_deep_context_autoencoder}"
DEEP_DEVICE="${DEEP_DEVICE:-cuda}"
DEEP_LATENT_DIM="${DEEP_LATENT_DIM:-64}"
DEEP_HIDDEN_DIM="${DEEP_HIDDEN_DIM:-1024}"
DEEP_LAYERS="${DEEP_LAYERS:-3}"
DEEP_NEIGHBOR_K="${DEEP_NEIGHBOR_K:-8}"
DEEP_GRAPH_LAYERS="${DEEP_GRAPH_LAYERS:-2}"
DEEP_GRAPH_MAX_NEIGHBORS="${DEEP_GRAPH_MAX_NEIGHBORS:-64}"
DEEP_FULL_BATCH_MAX_CELLS="${DEEP_FULL_BATCH_MAX_CELLS:-50000}"
DEEP_EPOCHS="${DEEP_EPOCHS:-30}"
DEEP_BATCH_SIZE="${DEEP_BATCH_SIZE:-32768}"
DEEP_LR="${DEEP_LR:-0.001}"
DEEP_WEIGHT_DECAY="${DEEP_WEIGHT_DECAY:-0.0001}"
DEEP_VALIDATION="${DEEP_VALIDATION:-spatial_block}"
DEEP_VALIDATION_CONTEXT_MODE="${DEEP_VALIDATION_CONTEXT_MODE:-inductive}"
DEEP_RECONSTRUCTION_WEIGHT="${DEEP_RECONSTRUCTION_WEIGHT:-1.0}"
DEEP_CONTEXT_WEIGHT="${DEEP_CONTEXT_WEIGHT:-0.5}"
DEEP_CONTRASTIVE_WEIGHT="${DEEP_CONTRASTIVE_WEIGHT:-0.1}"
DEEP_VARIANCE_WEIGHT="${DEEP_VARIANCE_WEIGHT:-0.1}"
DEEP_DECORRELATION_WEIGHT="${DEEP_DECORRELATION_WEIGHT:-0.01}"
DEEP_ALLOW_JOINT_OT_EMBEDDING="${DEEP_ALLOW_JOINT_OT_EMBEDDING:-0}"
DEEP_SAVE_MODEL="${DEEP_SAVE_MODEL:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtual environment python not found: $PYTHON_BIN" >&2
  echo "Run 'bash scripts/install_env.sh' from the spatial_ot directory first." >&2
  exit 1
fi

EXTRA_FLAGS=()
if [[ "$ALLOW_OBSERVED_HULL_GEOMETRY" == "1" ]]; then
  EXTRA_FLAGS+=(--allow-observed-hull-geometry)
else
  EXTRA_FLAGS+=(--no-allow-observed-hull-geometry)
fi
if [[ "$SHAPE_DIAGNOSTICS" == "1" ]]; then
  EXTRA_FLAGS+=(--shape-diagnostics)
else
  EXTRA_FLAGS+=(--no-shape-diagnostics)
fi
EXTRA_FLAGS+=(--shape-leakage-permutations "$SHAPE_LEAKAGE_PERMUTATIONS")
if [[ -n "$REGION_OBS_KEY" ]]; then
  EXTRA_FLAGS+=(--region-obs-key "$REGION_OBS_KEY")
fi
if [[ -n "$REGION_GEOMETRY_JSON" ]]; then
  EXTRA_FLAGS+=(--region-geometry-json "$REGION_GEOMETRY_JSON")
fi

DEEP_FLAGS=()
if [[ "$DEEP_FEATURE_METHOD" != "none" ]]; then
  DEEP_FLAGS+=(
    --deep-feature-method "$DEEP_FEATURE_METHOD"
    --deep-output-embedding "$DEEP_OUTPUT_EMBEDDING"
    --deep-output-obsm-key "$DEEP_OUTPUT_OBSM_KEY"
    --deep-device "$DEEP_DEVICE"
    --deep-latent-dim "$DEEP_LATENT_DIM"
    --deep-hidden-dim "$DEEP_HIDDEN_DIM"
    --deep-layers "$DEEP_LAYERS"
    --deep-neighbor-k "$DEEP_NEIGHBOR_K"
    --deep-graph-layers "$DEEP_GRAPH_LAYERS"
    --deep-graph-max-neighbors "$DEEP_GRAPH_MAX_NEIGHBORS"
    --deep-full-batch-max-cells "$DEEP_FULL_BATCH_MAX_CELLS"
    --deep-epochs "$DEEP_EPOCHS"
    --deep-batch-size "$DEEP_BATCH_SIZE"
    --deep-lr "$DEEP_LR"
    --deep-weight-decay "$DEEP_WEIGHT_DECAY"
    --deep-validation "$DEEP_VALIDATION"
    --deep-validation-context-mode "$DEEP_VALIDATION_CONTEXT_MODE"
    --deep-reconstruction-weight "$DEEP_RECONSTRUCTION_WEIGHT"
    --deep-context-weight "$DEEP_CONTEXT_WEIGHT"
    --deep-contrastive-weight "$DEEP_CONTRASTIVE_WEIGHT"
    --deep-variance-weight "$DEEP_VARIANCE_WEIGHT"
    --deep-decorrelation-weight "$DEEP_DECORRELATION_WEIGHT"
  )
  if [[ "$DEEP_ALLOW_JOINT_OT_EMBEDDING" == "1" ]]; then
    DEEP_FLAGS+=(--deep-allow-joint-ot-embedding)
  else
    DEEP_FLAGS+=(--no-deep-allow-joint-ot-embedding)
  fi
  if [[ "$DEEP_SAVE_MODEL" == "1" ]]; then
    DEEP_FLAGS+=(--deep-save-model)
  else
    DEEP_FLAGS+=(--no-deep-save-model)
  fi
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
  --min-subregions-per-cluster "$MIN_SUBREGIONS_PER_CLUSTER" \
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
  --subregion-construction-method "$SUBREGION_CONSTRUCTION_METHOD" \
  --subregion-feature-weight "$SUBREGION_FEATURE_WEIGHT" \
  --subregion-feature-dims "$SUBREGION_FEATURE_DIMS" \
  --deep-segmentation-knn "$DEEP_SEGMENTATION_KNN" \
  --deep-segmentation-feature-dims "$DEEP_SEGMENTATION_FEATURE_DIMS" \
  --deep-segmentation-feature-weight "$DEEP_SEGMENTATION_FEATURE_WEIGHT" \
  --deep-segmentation-spatial-weight "$DEEP_SEGMENTATION_SPATIAL_WEIGHT" \
  "${EXTRA_FLAGS[@]}" \
  "${DEEP_FLAGS[@]}"
