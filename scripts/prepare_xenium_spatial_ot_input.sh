#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

export INPUT_DIR="${XENIUM_INPUT_DIR:-../data_review/results}"
export OUTPUT_H5AD="${XENIUM_OUTPUT_H5AD:-../spatial_ot_input/xenium_spatial_ot_input_pooled.h5ad}"
export SAMPLE_GLOB="${SAMPLE_GLOB:-xenium_*_processed.h5ad}"
export SAMPLE_ID_PREFIX="${SAMPLE_ID_PREFIX:-xenium_}"
export SAMPLE_ID_SUFFIX="${SAMPLE_ID_SUFFIX:-_processed}"
export SAMPLE_ID_CASE="${SAMPLE_ID_CASE:-lower}"
export ORIGINAL_SPATIAL_X_KEY="${ORIGINAL_SPATIAL_X_KEY:-x_centroid}"
export ORIGINAL_SPATIAL_Y_KEY="${ORIGINAL_SPATIAL_Y_KEY:-y_centroid}"
export POOLED_SPATIAL_X_KEY="${POOLED_SPATIAL_X_KEY:-pooled_cell_x}"
export POOLED_SPATIAL_Y_KEY="${POOLED_SPATIAL_Y_KEY:-pooled_cell_y}"
export SAMPLE_OBS_KEY="${SAMPLE_OBS_KEY:-sample_id}"
export SOURCE_FILE_OBS_KEY="${SOURCE_FILE_OBS_KEY:-source_h5ad}"
export X_FEATURE_COMPONENTS="${X_FEATURE_COMPONENTS:-256}"
export PREPARED_FEATURE_OBSM_KEY="${PREPARED_FEATURE_OBSM_KEY:-X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}}"

exec bash "$SCRIPT_DIR/prepare_spatial_ot_input.sh"
