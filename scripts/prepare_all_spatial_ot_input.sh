#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

VENV_DIR="${VENV_DIR:-../.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
INPUT_DIR="${INPUT_DIR:-../spatial_ot_input}"
SAMPLE_GLOB="${SAMPLE_GLOB:-*_cells_marker_genes_umap3d.h5ad}"
SAMPLE_ID_PREFIX="${SAMPLE_ID_PREFIX:-}"
SAMPLE_ID_SUFFIX="${SAMPLE_ID_SUFFIX:-_cells_marker_genes_umap3d}"
SAMPLE_ID_CASE="${SAMPLE_ID_CASE:-preserve}"
PREPARE_POOLED_INPUT="${PREPARE_POOLED_INPUT:-1}"
REFRESH_PREPARED_FEATURES="${REFRESH_PREPARED_FEATURES:-0}"
REFRESH_POOLED_INPUT="${REFRESH_POOLED_INPUT:-0}"
POOLED_INPUT_NAME="${POOLED_INPUT_NAME:-spatial_ot_input_pooled.h5ad}"
X_FEATURE_COMPONENTS="${X_FEATURE_COMPONENTS:-512}"
PREPARED_FEATURE_OBSM_KEY="${PREPARED_FEATURE_OBSM_KEY:-X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}}"
VERIFY_PREPARED_FEATURES="${VERIFY_PREPARED_FEATURES:-1}"
WRITE_BACK_TO_SOURCE_INPUTS="${WRITE_BACK_TO_SOURCE_INPUTS:-0}"
SAMPLE_OBS_KEY="${SAMPLE_OBS_KEY:-sample_id}"
SOURCE_FILE_OBS_KEY="${SOURCE_FILE_OBS_KEY:-source_h5ad}"
X_TARGET_SUM="${X_TARGET_SUM:-10000}"
X_SVD_N_ITER="${X_SVD_N_ITER:-7}"
X_SVD_RANDOM_STATE="${X_SVD_RANDOM_STATE:-1337}"

export SPATIAL_OT_X_SVD_COMPONENTS="${SPATIAL_OT_X_SVD_COMPONENTS:-$X_FEATURE_COMPONENTS}"
export SPATIAL_OT_X_TARGET_SUM="${SPATIAL_OT_X_TARGET_SUM:-$X_TARGET_SUM}"
export SPATIAL_OT_X_SVD_N_ITER="${SPATIAL_OT_X_SVD_N_ITER:-$X_SVD_N_ITER}"
export SPATIAL_OT_X_SVD_RANDOM_STATE="${SPATIAL_OT_X_SVD_RANDOM_STATE:-$X_SVD_RANDOM_STATE}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtual environment python not found: $PYTHON_BIN" >&2
  echo "Run 'bash scripts/install_env.sh' from the spatial_ot directory first." >&2
  exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Missing input directory: $INPUT_DIR" >&2
  exit 1
fi

shopt -s nullglob
inputs=("$INPUT_DIR"/$SAMPLE_GLOB)
shopt -u nullglob

if [[ "${#inputs[@]}" -eq 0 ]]; then
  echo "No input H5AD files matched '$SAMPLE_GLOB' under $INPUT_DIR" >&2
  exit 1
fi

if [[ "$PREPARE_POOLED_INPUT" != "1" ]]; then
  echo "prepare_all_spatial_ot_input.sh requires PREPARE_POOLED_INPUT=1 so that sample-level caches are derived from one pooled shared feature space." >&2
  exit 1
fi

export OUTPUT_H5AD="${INPUT_DIR}/${POOLED_INPUT_NAME}"
export REFRESH_POOLED_INPUT
export REFRESH_PREPARED_FEATURES
export PREPARED_FEATURE_OBSM_KEY
export SAMPLE_GLOB
export SAMPLE_ID_PREFIX
export SAMPLE_ID_SUFFIX
export SAMPLE_ID_CASE
bash "$SCRIPT_DIR/prepare_spatial_ot_input.sh"

if [[ "$WRITE_BACK_TO_SOURCE_INPUTS" != "1" ]]; then
  exit 0
fi

DISTRIBUTE_ARGS=()
if [[ "$REFRESH_PREPARED_FEATURES" == "1" ]]; then
  DISTRIBUTE_ARGS+=(--overwrite)
fi

"$PYTHON_BIN" -m spatial_ot distribute-prepared-inputs \
  --pooled-h5ad "$OUTPUT_H5AD" \
  --input-dir "$INPUT_DIR" \
  --prepared-obsm-key "$PREPARED_FEATURE_OBSM_KEY" \
  --sample-glob "$SAMPLE_GLOB" \
  --sample-obs-key "$SAMPLE_OBS_KEY" \
  --source-file-obs-key "$SOURCE_FILE_OBS_KEY" \
  "${DISTRIBUTE_ARGS[@]}"

if [[ "$VERIFY_PREPARED_FEATURES" == "1" ]]; then
  "$PYTHON_BIN" - "$INPUT_DIR" "$SAMPLE_GLOB" "$PREPARED_FEATURE_OBSM_KEY" <<'PY'
from pathlib import Path
import sys

import anndata as ad

input_dir = Path(sys.argv[1])
sample_glob = sys.argv[2]
prepared_key = sys.argv[3]
paths = sorted(input_dir.glob(sample_glob))
missing: list[str] = []
for path in paths:
    adata = ad.read_h5ad(path, backed="r")
    try:
        if prepared_key not in adata.obsm:
            missing.append(path.name)
    finally:
        adata.file.close()
if missing:
    raise SystemExit("Missing prepared feature cache '%s' in: %s" % (prepared_key, ", ".join(missing)))
PY
fi
