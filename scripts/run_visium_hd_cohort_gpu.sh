#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

export INPUT_DIR="${VISIUM_HD_INPUT_DIR:-../spatial_ot_input}"
export INPUT_H5AD="${VISIUM_HD_INPUT_H5AD:-../spatial_ot_input/visium_hd_spatial_ot_input_pooled.h5ad}"
export POOLED_INPUT_NAME="${POOLED_INPUT_NAME:-visium_hd_spatial_ot_input_pooled.h5ad}"
export OUTPUT_DIR="${OUTPUT_DIR:-../outputs/spatial_ot/visium_hd_cohort_multilevel_ot_deep_expression_${RUN_STAMP}}"
export SAMPLE_GLOB="${SAMPLE_GLOB:-hd_*_processed.h5ad}"
export SAMPLE_ID_PREFIX="${SAMPLE_ID_PREFIX:-hd_}"
export SAMPLE_ID_SUFFIX="${SAMPLE_ID_SUFFIX:-_processed}"
export SAMPLE_ID_CASE="${SAMPLE_ID_CASE:-lower}"
export ORIGINAL_SPATIAL_X_KEY="${ORIGINAL_SPATIAL_X_KEY:-cell_x}"
export ORIGINAL_SPATIAL_Y_KEY="${ORIGINAL_SPATIAL_Y_KEY:-cell_y}"
export SPATIAL_SCALE="${SPATIAL_SCALE:-0.2737012522439323}"
export X_FEATURE_COMPONENTS="${X_FEATURE_COMPONENTS:-512}"
export PREPARED_FEATURE_OBSM_KEY="${PREPARED_FEATURE_OBSM_KEY:-X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}}"
export FEATURE_OBSM_KEY="${FEATURE_OBSM_KEY:-X}"
export SPATIAL_OT_X_USE_SVD="${SPATIAL_OT_X_USE_SVD:-0}"
export POOL_ALL_INPUTS="${POOL_ALL_INPUTS:-1}"
export PREPARE_INPUTS_AHEAD="${PREPARE_INPUTS_AHEAD:-0}"
export REFRESH_POOLED_INPUT="${REFRESH_POOLED_INPUT:-0}"
export COMPUTE_DEVICE="${COMPUTE_DEVICE:-cuda}"
export AUTO_N_CLUSTERS="${AUTO_N_CLUSTERS:-1}"
export CANDIDATE_N_CLUSTERS="${CANDIDATE_N_CLUSTERS:-15-25}"
export MIN_SUBREGIONS_PER_CLUSTER="${MIN_SUBREGIONS_PER_CLUSTER:-50}"
export MAX_SUBREGIONS="${MAX_SUBREGIONS:-12000}"
export BASIC_NICHE_SIZE_UM="${BASIC_NICHE_SIZE_UM:-50}"
export MIN_CELLS="${MIN_CELLS:-25}"
export RADIUS_UM="${RADIUS_UM:-100}"
export STRIDE_UM="${STRIDE_UM:-100}"
export DEFAULT_PLOT_SAMPLE_ID="${DEFAULT_PLOT_SAMPLE_ID:-visium_hd_cohort}"

export DEEP_OUTPUT_EMBEDDING="${DEEP_OUTPUT_EMBEDDING:-intrinsic}"
export DEEP_OUTPUT_OBSM_KEY="${DEEP_OUTPUT_OBSM_KEY:-X_spatial_ot_deep_expression_autoencoder}"
export DEEP_CONTEXT_WEIGHT="${DEEP_CONTEXT_WEIGHT:-0.0}"
export DEEP_CONTRASTIVE_WEIGHT="${DEEP_CONTRASTIVE_WEIGHT:-0.0}"
export DEEP_INDEPENDENCE_WEIGHT="${DEEP_INDEPENDENCE_WEIGHT:-0.0}"
export DEEP_CHECKPOINT_EVERY_EPOCHS="${DEEP_CHECKPOINT_EVERY_EPOCHS:-5}"

# shellcheck source=scripts/_high_vram_deep_profile.sh
source "$SCRIPT_DIR/_high_vram_deep_profile.sh"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  env | sort | grep -E '^(INPUT_H5AD|OUTPUT_DIR|FEATURE_OBSM_KEY|DEEP_[A-Z0-9_]*|CUDA_[A-Z0-9_]*|SPATIAL_OT_CUDA_[A-Z0-9_]*|SPATIAL_OT_X_[A-Z0-9_]*|CPU_THREADS|TORCH_[A-Z0-9_]*|PYTORCH_[A-Z0-9_]*|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS)='
  exit 0
fi

exec bash "$SCRIPT_DIR/run.sh"
