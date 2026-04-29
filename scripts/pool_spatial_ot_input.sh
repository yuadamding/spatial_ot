#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

VENV_DIR="${VENV_DIR:-../.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
INPUT_DIR="${INPUT_DIR:-../spatial_ot_input}"
OUTPUT_H5AD="${OUTPUT_H5AD:-${INPUT_DIR}/spatial_ot_input_pooled.h5ad}"
FEATURE_OBSM_KEY="${FEATURE_OBSM_KEY:-X}"
SAMPLE_GLOB="${SAMPLE_GLOB:-*_cells_marker_genes_umap3d.h5ad}"
SAMPLE_ID_SUFFIX="${SAMPLE_ID_SUFFIX:-_cells_marker_genes_umap3d}"
ORIGINAL_SPATIAL_X_KEY="${ORIGINAL_SPATIAL_X_KEY:-cell_x}"
ORIGINAL_SPATIAL_Y_KEY="${ORIGINAL_SPATIAL_Y_KEY:-cell_y}"
POOLED_SPATIAL_X_KEY="${POOLED_SPATIAL_X_KEY:-pooled_cell_x}"
POOLED_SPATIAL_Y_KEY="${POOLED_SPATIAL_Y_KEY:-pooled_cell_y}"
SAMPLE_OBS_KEY="${SAMPLE_OBS_KEY:-sample_id}"
SOURCE_FILE_OBS_KEY="${SOURCE_FILE_OBS_KEY:-source_h5ad}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtual environment python not found: $PYTHON_BIN" >&2
  echo "Run 'bash install_env.sh' from the spatial_ot directory first." >&2
  exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Missing input directory: $INPUT_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname -- "$OUTPUT_H5AD")"

"$PYTHON_BIN" -m spatial_ot pool-inputs \
  --input-dir "$INPUT_DIR" \
  --output-h5ad "$OUTPUT_H5AD" \
  --feature-obsm-key "$FEATURE_OBSM_KEY" \
  --sample-glob "$SAMPLE_GLOB" \
  --spatial-x-key "$ORIGINAL_SPATIAL_X_KEY" \
  --spatial-y-key "$ORIGINAL_SPATIAL_Y_KEY" \
  --pooled-spatial-x-key "$POOLED_SPATIAL_X_KEY" \
  --pooled-spatial-y-key "$POOLED_SPATIAL_Y_KEY" \
  --sample-obs-key "$SAMPLE_OBS_KEY" \
  --source-file-obs-key "$SOURCE_FILE_OBS_KEY" \
  --sample-id-suffix "$SAMPLE_ID_SUFFIX"
