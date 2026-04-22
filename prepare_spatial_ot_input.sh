#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-../.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
INPUT_DIR="${INPUT_DIR:-../spatial_ot_input}"
OUTPUT_H5AD="${OUTPUT_H5AD:-${INPUT_DIR}/spatial_ot_input_pooled.h5ad}"
REFRESH_POOLED_INPUT="${REFRESH_POOLED_INPUT:-0}"
REFRESH_PREPARED_FEATURES="${REFRESH_PREPARED_FEATURES:-0}"
SAMPLE_GLOB="${SAMPLE_GLOB:-*_cells_marker_genes_umap3d.h5ad}"
SAMPLE_ID_SUFFIX="${SAMPLE_ID_SUFFIX:-_cells_marker_genes_umap3d}"
X_FEATURE_COMPONENTS="${X_FEATURE_COMPONENTS:-512}"
FEATURE_OBSM_KEY="${FEATURE_OBSM_KEY:-X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}}"
PREPARED_FEATURE_OBSM_KEY="${PREPARED_FEATURE_OBSM_KEY:-X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}}"
ORIGINAL_SPATIAL_X_KEY="${ORIGINAL_SPATIAL_X_KEY:-cell_x}"
ORIGINAL_SPATIAL_Y_KEY="${ORIGINAL_SPATIAL_Y_KEY:-cell_y}"
POOLED_SPATIAL_X_KEY="${POOLED_SPATIAL_X_KEY:-pooled_cell_x}"
POOLED_SPATIAL_Y_KEY="${POOLED_SPATIAL_Y_KEY:-pooled_cell_y}"
SAMPLE_OBS_KEY="${SAMPLE_OBS_KEY:-sample_id}"
SOURCE_FILE_OBS_KEY="${SOURCE_FILE_OBS_KEY:-source_h5ad}"
X_TARGET_SUM="${X_TARGET_SUM:-10000}"
X_SVD_N_ITER="${X_SVD_N_ITER:-7}"
X_SVD_RANDOM_STATE="${X_SVD_RANDOM_STATE:-1337}"

export SPATIAL_OT_X_SVD_COMPONENTS="${SPATIAL_OT_X_SVD_COMPONENTS:-$X_FEATURE_COMPONENTS}"
export SPATIAL_OT_X_TARGET_SUM="${SPATIAL_OT_X_TARGET_SUM:-$X_TARGET_SUM}"
export SPATIAL_OT_X_SVD_N_ITER="${SPATIAL_OT_X_SVD_N_ITER:-$X_SVD_N_ITER}"
export SPATIAL_OT_X_SVD_RANDOM_STATE="${SPATIAL_OT_X_SVD_RANDOM_STATE:-$X_SVD_RANDOM_STATE}"

pooled_has_prepared_key() {
  "$PYTHON_BIN" - "$OUTPUT_H5AD" "$PREPARED_FEATURE_OBSM_KEY" <<'PY'
from pathlib import Path
import os
import sys

import anndata as ad

path = Path(sys.argv[1])
prepared_key = sys.argv[2]
if not path.exists():
    raise SystemExit(1)
adata = ad.read_h5ad(path, backed="r")
try:
    if prepared_key not in adata.obsm:
        raise SystemExit(1)
    prepared_uns = dict(adata.uns.get("spatial_ot_prepared_features", {}))
    metadata = dict(prepared_uns.get(prepared_key, {}))
    if not metadata:
        raise SystemExit(1)
    expected_components = int(os.environ["SPATIAL_OT_X_SVD_COMPONENTS"])
    expected_target_sum = float(os.environ["SPATIAL_OT_X_TARGET_SUM"])
    expected_n_iter = int(os.environ["SPATIAL_OT_X_SVD_N_ITER"])
    expected_random_state = int(os.environ["SPATIAL_OT_X_SVD_RANDOM_STATE"])
    if str(metadata.get("input_feature_key", "")) != "X":
        raise SystemExit(1)
    if int(metadata.get("svd_components_requested", -1)) != expected_components:
        raise SystemExit(1)
    if float(metadata.get("target_sum", float("nan"))) != expected_target_sum:
        raise SystemExit(1)
    if int(metadata.get("svd_n_iter", -1)) != expected_n_iter:
        raise SystemExit(1)
    if int(metadata.get("svd_random_state", -1)) != expected_random_state:
        raise SystemExit(1)
finally:
    adata.file.close()
PY
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtual environment python not found: $PYTHON_BIN" >&2
  echo "Run 'bash install_env.sh' from the spatial_ot directory first." >&2
  exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Missing input directory: $INPUT_DIR" >&2
  exit 1
fi

if [[ "$REFRESH_POOLED_INPUT" == "1" || ! -f "$OUTPUT_H5AD" ]]; then
  mkdir -p "$(dirname -- "$OUTPUT_H5AD")"
  "$PYTHON_BIN" -m spatial_ot pool-inputs \
    --input-dir "$INPUT_DIR" \
    --output-h5ad "$OUTPUT_H5AD" \
    --feature-obsm-key X \
    --sample-glob "$SAMPLE_GLOB" \
    --spatial-x-key "$ORIGINAL_SPATIAL_X_KEY" \
    --spatial-y-key "$ORIGINAL_SPATIAL_Y_KEY" \
    --pooled-spatial-x-key "$POOLED_SPATIAL_X_KEY" \
    --pooled-spatial-y-key "$POOLED_SPATIAL_Y_KEY" \
    --sample-obs-key "$SAMPLE_OBS_KEY" \
    --source-file-obs-key "$SOURCE_FILE_OBS_KEY" \
    --sample-id-suffix "$SAMPLE_ID_SUFFIX"
fi

PREPARE_ARGS=()
if [[ "$REFRESH_PREPARED_FEATURES" == "1" ]]; then
  PREPARE_ARGS+=(--overwrite)
fi

if [[ "$REFRESH_PREPARED_FEATURES" == "1" ]] || ! pooled_has_prepared_key; then
  "$PYTHON_BIN" -m spatial_ot prepare-inputs \
    --input-h5ad "$OUTPUT_H5AD" \
    --feature-obsm-key X \
    --output-obsm-key "$PREPARED_FEATURE_OBSM_KEY" \
    "${PREPARE_ARGS[@]}"
fi

if ! pooled_has_prepared_key; then
  echo "Prepared feature cache '$PREPARED_FEATURE_OBSM_KEY' is missing from $OUTPUT_H5AD after staging." >&2
  exit 1
fi
