#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-../.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
INPUT_DIR="${INPUT_DIR:-../spatial_ot_input}"
POOL_ALL_INPUTS="${POOL_ALL_INPUTS:-1}"
SAMPLE_GLOB="${SAMPLE_GLOB:-*_cells_marker_genes_umap3d.h5ad}"
SAMPLE_ID_SUFFIX="${SAMPLE_ID_SUFFIX:-_cells_marker_genes_umap3d}"
SAMPLE_OBS_KEY="${SAMPLE_OBS_KEY:-sample_id}"
SOURCE_FILE_OBS_KEY="${SOURCE_FILE_OBS_KEY:-source_h5ad}"
SAMPLE_KEY="${SAMPLE_KEY:-p2_crc}"
FEATURE_OBSM_KEY="${FEATURE_OBSM_KEY:-X}"
ORIGINAL_SPATIAL_X_KEY="${ORIGINAL_SPATIAL_X_KEY:-cell_x}"
ORIGINAL_SPATIAL_Y_KEY="${ORIGINAL_SPATIAL_Y_KEY:-cell_y}"
POOLED_SPATIAL_X_KEY="${POOLED_SPATIAL_X_KEY:-pooled_cell_x}"
POOLED_SPATIAL_Y_KEY="${POOLED_SPATIAL_Y_KEY:-pooled_cell_y}"
SPATIAL_SCALE="${SPATIAL_SCALE:-0.2737012522439323}"
N_CLUSTERS="${N_CLUSTERS:-8}"
ATOMS_PER_CLUSTER="${ATOMS_PER_CLUSTER:-8}"
COMPUTE_DEVICE="${COMPUTE_DEVICE:-cuda}"
BASIC_NICHE_SIZE_UM="${BASIC_NICHE_SIZE_UM:-200}"
MIN_CELLS="${MIN_CELLS:-1}"
MAX_SUBREGIONS="${MAX_SUBREGIONS:-0}"
ALLOW_UMAP_AS_FEATURE="${ALLOW_UMAP_AS_FEATURE:-0}"
ALLOW_OBSERVED_HULL_GEOMETRY="${ALLOW_OBSERVED_HULL_GEOMETRY:-1}"
PLOT_SAMPLE_NICHES="${PLOT_SAMPLE_NICHES:-1}"
CPU_THREADS="${CPU_THREADS:-28}"
TORCH_INTRAOP_THREADS="${TORCH_INTRAOP_THREADS:-$CPU_THREADS}"
TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-4}"
CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-all}"
PARALLEL_RESTARTS="${PARALLEL_RESTARTS:-auto}"
CUDA_TARGET_VRAM_GB="${CUDA_TARGET_VRAM_GB:-50}"
X_FEATURE_COMPONENTS="${X_FEATURE_COMPONENTS:-512}"
X_TARGET_SUM="${X_TARGET_SUM:-10000}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$CPU_THREADS}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-$CPU_THREADS}"
export SPATIAL_OT_TORCH_NUM_THREADS="${SPATIAL_OT_TORCH_NUM_THREADS:-$TORCH_INTRAOP_THREADS}"
export SPATIAL_OT_TORCH_NUM_INTEROP_THREADS="${SPATIAL_OT_TORCH_NUM_INTEROP_THREADS:-$TORCH_INTEROP_THREADS}"
export SPATIAL_OT_CUDA_DEVICE_LIST="${SPATIAL_OT_CUDA_DEVICE_LIST:-$CUDA_DEVICE_LIST}"
export SPATIAL_OT_PARALLEL_RESTARTS="${SPATIAL_OT_PARALLEL_RESTARTS:-$PARALLEL_RESTARTS}"
export SPATIAL_OT_CUDA_TARGET_VRAM_GB="${SPATIAL_OT_CUDA_TARGET_VRAM_GB:-$CUDA_TARGET_VRAM_GB}"
export SPATIAL_OT_X_SVD_COMPONENTS="${SPATIAL_OT_X_SVD_COMPONENTS:-$X_FEATURE_COMPONENTS}"
export SPATIAL_OT_X_TARGET_SUM="${SPATIAL_OT_X_TARGET_SUM:-$X_TARGET_SUM}"

if [[ "$POOL_ALL_INPUTS" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-../outputs/spatial_ot/cohort_multilevel_ot}"
  INPUT_H5AD="${INPUT_H5AD:-${OUTPUT_DIR}/pooled_inputs/spatial_ot_input_pooled.h5ad}"
  SPATIAL_X_KEY="${SPATIAL_X_KEY:-$POOLED_SPATIAL_X_KEY}"
  SPATIAL_Y_KEY="${SPATIAL_Y_KEY:-$POOLED_SPATIAL_Y_KEY}"
  DEFAULT_PLOT_SAMPLE_ID="${DEFAULT_PLOT_SAMPLE_ID:-pooled_cohort}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-../outputs/spatial_ot/${SAMPLE_KEY}_multilevel_ot}"
  INPUT_H5AD="${INPUT_H5AD:-${INPUT_DIR}/${SAMPLE_KEY}_cells_marker_genes_umap3d.h5ad}"
  SPATIAL_X_KEY="${SPATIAL_X_KEY:-$ORIGINAL_SPATIAL_X_KEY}"
  SPATIAL_Y_KEY="${SPATIAL_Y_KEY:-$ORIGINAL_SPATIAL_Y_KEY}"
  DEFAULT_PLOT_SAMPLE_ID="${DEFAULT_PLOT_SAMPLE_ID:-$SAMPLE_KEY}"
fi
SAMPLE_NICHE_PLOT_DIR="${SAMPLE_NICHE_PLOT_DIR:-${OUTPUT_DIR}/sample_niche_plots}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtual environment python not found: $PYTHON_BIN" >&2
  echo "Run 'bash install_env.sh' from the spatial_ot directory first." >&2
  exit 1
fi

if [[ "$POOL_ALL_INPUTS" == "1" ]]; then
  mkdir -p "$(dirname -- "$INPUT_H5AD")"
  "$PYTHON_BIN" -m spatial_ot pool-inputs \
    --input-dir "$INPUT_DIR" \
    --output-h5ad "$INPUT_H5AD" \
    --feature-obsm-key "$FEATURE_OBSM_KEY" \
    --sample-glob "$SAMPLE_GLOB" \
    --spatial-x-key "$ORIGINAL_SPATIAL_X_KEY" \
    --spatial-y-key "$ORIGINAL_SPATIAL_Y_KEY" \
    --pooled-spatial-x-key "$POOLED_SPATIAL_X_KEY" \
    --pooled-spatial-y-key "$POOLED_SPATIAL_Y_KEY" \
    --sample-obs-key "$SAMPLE_OBS_KEY" \
    --source-file-obs-key "$SOURCE_FILE_OBS_KEY" \
    --sample-id-suffix "$SAMPLE_ID_SUFFIX"
elif [[ ! -f "$INPUT_H5AD" ]]; then
  echo "Missing input H5AD: $INPUT_H5AD" >&2
  if [[ -d "$INPUT_DIR" ]]; then
    echo "Available sample keys under $INPUT_DIR:" >&2
    shopt -s nullglob
    for candidate in "$INPUT_DIR"/*_cells_marker_genes_umap3d.h5ad; do
      sample_name="$(basename -- "$candidate")"
      echo "  - ${sample_name%_cells_marker_genes_umap3d.h5ad}" >&2
    done
    shopt -u nullglob
  fi
  exit 1
fi

EXTRA_FLAGS=()
if [[ "$ALLOW_UMAP_AS_FEATURE" == "1" ]]; then
  EXTRA_FLAGS+=(--allow-umap-as-feature)
fi
if [[ "$ALLOW_OBSERVED_HULL_GEOMETRY" == "1" ]]; then
  EXTRA_FLAGS+=(--allow-observed-hull-geometry)
fi

"$PYTHON_BIN" -m spatial_ot multilevel-ot \
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
  --min-cells "$MIN_CELLS" \
  --max-subregions "$MAX_SUBREGIONS" \
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

if [[ "$PLOT_SAMPLE_NICHES" == "1" ]]; then
  "$PYTHON_BIN" -m spatial_ot plot-sample-niches \
    --run-dir "$OUTPUT_DIR" \
    --output-dir "$SAMPLE_NICHE_PLOT_DIR" \
    --sample-obs-key "$SAMPLE_OBS_KEY" \
    --source-file-obs-key "$SOURCE_FILE_OBS_KEY" \
    --plot-spatial-x-key "$ORIGINAL_SPATIAL_X_KEY" \
    --plot-spatial-y-key "$ORIGINAL_SPATIAL_Y_KEY" \
    --default-sample-id "$DEFAULT_PLOT_SAMPLE_ID"
fi
