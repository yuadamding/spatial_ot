#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-../.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
INPUT_DIR="${INPUT_DIR:-../spatial_ot_input}"
POOL_ALL_INPUTS="${POOL_ALL_INPUTS:-1}"
REFRESH_POOLED_INPUT="${REFRESH_POOLED_INPUT:-0}"
POOLED_INPUT_NAME="${POOLED_INPUT_NAME:-spatial_ot_input_pooled.h5ad}"
PREPARE_INPUTS_AHEAD="${PREPARE_INPUTS_AHEAD:-1}"
REFRESH_PREPARED_FEATURES="${REFRESH_PREPARED_FEATURES:-0}"
SAMPLE_GLOB="${SAMPLE_GLOB:-*_cells_marker_genes_umap3d.h5ad}"
SAMPLE_ID_SUFFIX="${SAMPLE_ID_SUFFIX:-_cells_marker_genes_umap3d}"
SAMPLE_OBS_KEY="${SAMPLE_OBS_KEY:-sample_id}"
SOURCE_FILE_OBS_KEY="${SOURCE_FILE_OBS_KEY:-source_h5ad}"
SAMPLE_KEY="${SAMPLE_KEY:-p2_crc}"
ORIGINAL_SPATIAL_X_KEY="${ORIGINAL_SPATIAL_X_KEY:-cell_x}"
ORIGINAL_SPATIAL_Y_KEY="${ORIGINAL_SPATIAL_Y_KEY:-cell_y}"
POOLED_SPATIAL_X_KEY="${POOLED_SPATIAL_X_KEY:-pooled_cell_x}"
POOLED_SPATIAL_Y_KEY="${POOLED_SPATIAL_Y_KEY:-pooled_cell_y}"
SPATIAL_SCALE="${SPATIAL_SCALE:-0.2737012522439323}"
N_CLUSTERS="${N_CLUSTERS:-8}"
AUTO_N_CLUSTERS="${AUTO_N_CLUSTERS:-0}"
CANDIDATE_N_CLUSTERS="${CANDIDATE_N_CLUSTERS:-15-25}"
AUTO_K_MAX_SCORE_SUBREGIONS="${AUTO_K_MAX_SCORE_SUBREGIONS:-2500}"
AUTO_K_GAP_REFERENCES="${AUTO_K_GAP_REFERENCES:-8}"
AUTO_K_MDS_COMPONENTS="${AUTO_K_MDS_COMPONENTS:-8}"
AUTO_K_PILOT_N_INIT="${AUTO_K_PILOT_N_INIT:-1}"
AUTO_K_PILOT_MAX_ITER="${AUTO_K_PILOT_MAX_ITER:-3}"
MIN_SUBREGIONS_PER_CLUSTER="${MIN_SUBREGIONS_PER_CLUSTER:-50}"
ATOMS_PER_CLUSTER="${ATOMS_PER_CLUSTER:-8}"
COMPUTE_DEVICE="${COMPUTE_DEVICE:-cuda}"
RADIUS_UM="${RADIUS_UM:-100}"
STRIDE_UM="${STRIDE_UM:-$RADIUS_UM}"
BASIC_NICHE_SIZE_UM="${BASIC_NICHE_SIZE_UM:-50}"
MIN_CELLS="${MIN_CELLS:-25}"
MAX_SUBREGIONS="${MAX_SUBREGIONS:-5000}"
SUBREGION_FEATURE_WEIGHT="${SUBREGION_FEATURE_WEIGHT:-0}"
SUBREGION_FEATURE_DIMS="${SUBREGION_FEATURE_DIMS:-16}"
SUBREGION_CONSTRUCTION_METHOD="${SUBREGION_CONSTRUCTION_METHOD:-data_driven}"
DEEP_SEGMENTATION_KNN="${DEEP_SEGMENTATION_KNN:-12}"
DEEP_SEGMENTATION_FEATURE_DIMS="${DEEP_SEGMENTATION_FEATURE_DIMS:-32}"
DEEP_SEGMENTATION_FEATURE_WEIGHT="${DEEP_SEGMENTATION_FEATURE_WEIGHT:-1.0}"
DEEP_SEGMENTATION_SPATIAL_WEIGHT="${DEEP_SEGMENTATION_SPATIAL_WEIGHT:-0.05}"
DEEP_SEGMENTATION_REFINEMENT_ITERS="${DEEP_SEGMENTATION_REFINEMENT_ITERS:-6}"
REQUIRE_FULL_CELL_COVERAGE="${REQUIRE_FULL_CELL_COVERAGE:-0}"
ALLOW_UMAP_AS_FEATURE="${ALLOW_UMAP_AS_FEATURE:-0}"
ALLOW_OBSERVED_HULL_GEOMETRY="${ALLOW_OBSERVED_HULL_GEOMETRY:-0}"
REGION_OBS_KEY="${REGION_OBS_KEY:-}"
REGION_GEOMETRY_JSON="${REGION_GEOMETRY_JSON:-}"
SHAPE_DIAGNOSTICS="${SHAPE_DIAGNOSTICS:-1}"
SHAPE_LEAKAGE_PERMUTATIONS="${SHAPE_LEAKAGE_PERMUTATIONS:-16}"
COMPUTE_SPOT_LATENT="${COMPUTE_SPOT_LATENT:-1}"
PLOT_SAMPLE_NICHES="${PLOT_SAMPLE_NICHES:-1}"
PLOT_SAMPLE_SPOT_LATENT="${PLOT_SAMPLE_SPOT_LATENT:-1}"
MAX_SPOT_LATENT_PLOT_OCCURRENCES="${MAX_SPOT_LATENT_PLOT_OCCURRENCES:-150000}"
LIGHT_CELL_H5AD="${LIGHT_CELL_H5AD:-1}"
H5AD_COMPRESSION="${H5AD_COMPRESSION:-lzf}"
WRITE_SAMPLE_SPATIAL_MAPS="${WRITE_SAMPLE_SPATIAL_MAPS:-0}"
PROGRESS_LOG="${PROGRESS_LOG:-1}"
CPU_THREADS="${CPU_THREADS:-28}"
TORCH_INTRAOP_THREADS="${TORCH_INTRAOP_THREADS:-$CPU_THREADS}"
TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-4}"
CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-all}"
PARALLEL_RESTARTS="${PARALLEL_RESTARTS:-auto}"
CUDA_TARGET_VRAM_GB="${CUDA_TARGET_VRAM_GB:-50}"
X_FEATURE_COMPONENTS="${X_FEATURE_COMPONENTS:-512}"
X_TARGET_SUM="${X_TARGET_SUM:-10000}"
PREPARED_FEATURE_OBSM_KEY="${PREPARED_FEATURE_OBSM_KEY:-X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}}"
FEATURE_OBSM_KEY="${FEATURE_OBSM_KEY:-}"
DEEP_FEATURE_METHOD="${DEEP_FEATURE_METHOD:-none}"
DEEP_OUTPUT_EMBEDDING="${DEEP_OUTPUT_EMBEDDING:-context}"
DEEP_OUTPUT_OBSM_KEY="${DEEP_OUTPUT_OBSM_KEY:-X_spatial_ot_deep_context_autoencoder}"
DEEP_DEVICE="${DEEP_DEVICE:-cuda}"
DEEP_LATENT_DIM="${DEEP_LATENT_DIM:-64}"
DEEP_HIDDEN_DIM="${DEEP_HIDDEN_DIM:-1024}"
DEEP_LAYERS="${DEEP_LAYERS:-3}"
DEEP_NEIGHBOR_K="${DEEP_NEIGHBOR_K:-8}"
DEEP_RADIUS_UM="${DEEP_RADIUS_UM:-}"
DEEP_SHORT_RADIUS_UM="${DEEP_SHORT_RADIUS_UM:-}"
DEEP_MID_RADIUS_UM="${DEEP_MID_RADIUS_UM:-}"
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
DEEP_PRETRAINED_MODEL="${DEEP_PRETRAINED_MODEL:-}"
LAMBDA_X="${LAMBDA_X:-0.5}"
LAMBDA_Y="${LAMBDA_Y:-1.0}"
GEOMETRY_EPS="${GEOMETRY_EPS:-0.03}"
OT_EPS="${OT_EPS:-0.03}"
RHO="${RHO:-0.5}"
GEOMETRY_SAMPLES="${GEOMETRY_SAMPLES:-64}"
COMPRESSED_SUPPORT_SIZE="${COMPRESSED_SUPPORT_SIZE:-48}"
ALIGN_ITERS="${ALIGN_ITERS:-2}"
ALLOW_REFLECTION="${ALLOW_REFLECTION:-0}"
ALLOW_SCALE="${ALLOW_SCALE:-0}"
MIN_SCALE="${MIN_SCALE:-0.75}"
MAX_SCALE="${MAX_SCALE:-1.33}"
SCALE_PENALTY="${SCALE_PENALTY:-0.05}"
SHIFT_PENALTY="${SHIFT_PENALTY:-0.05}"
N_INIT="${N_INIT:-2}"
OVERLAP_CONSISTENCY_WEIGHT="${OVERLAP_CONSISTENCY_WEIGHT:-0.05}"
OVERLAP_JACCARD_MIN="${OVERLAP_JACCARD_MIN:-0.15}"
OVERLAP_CONTRAST_SCALE="${OVERLAP_CONTRAST_SCALE:-1.0}"
MAX_ITER="${MAX_ITER:-5}"
TOL="${TOL:-1e-4}"
SINKHORN_MAX_ITER="${SINKHORN_MAX_ITER:-256}"
SINKHORN_TOL="${SINKHORN_TOL:-1e-4}"
CPU_SINKHORN_MAX_ITER="${CPU_SINKHORN_MAX_ITER:-1200}"
CPU_SINKHORN_TOL="${CPU_SINKHORN_TOL:-1e-7}"

if [[ -z "$FEATURE_OBSM_KEY" ]]; then
  if [[ "$PREPARE_INPUTS_AHEAD" == "1" ]]; then
    FEATURE_OBSM_KEY="$PREPARED_FEATURE_OBSM_KEY"
  else
    FEATURE_OBSM_KEY="X"
  fi
fi

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
export SPATIAL_OT_SINKHORN_MAX_ITER="${SPATIAL_OT_SINKHORN_MAX_ITER:-$SINKHORN_MAX_ITER}"
export SPATIAL_OT_SINKHORN_TOL="${SPATIAL_OT_SINKHORN_TOL:-$SINKHORN_TOL}"
export SPATIAL_OT_CPU_SINKHORN_MAX_ITER="${SPATIAL_OT_CPU_SINKHORN_MAX_ITER:-$CPU_SINKHORN_MAX_ITER}"
export SPATIAL_OT_CPU_SINKHORN_TOL="${SPATIAL_OT_CPU_SINKHORN_TOL:-$CPU_SINKHORN_TOL}"
export SPATIAL_OT_SUBREGION_FEATURE_WEIGHT="${SPATIAL_OT_SUBREGION_FEATURE_WEIGHT:-$SUBREGION_FEATURE_WEIGHT}"
export SPATIAL_OT_SUBREGION_FEATURE_DIMS="${SPATIAL_OT_SUBREGION_FEATURE_DIMS:-$SUBREGION_FEATURE_DIMS}"
export SPATIAL_OT_DEEP_SEGMENTATION_KNN="${SPATIAL_OT_DEEP_SEGMENTATION_KNN:-$DEEP_SEGMENTATION_KNN}"
export SPATIAL_OT_DEEP_SEGMENTATION_FEATURE_DIMS="${SPATIAL_OT_DEEP_SEGMENTATION_FEATURE_DIMS:-$DEEP_SEGMENTATION_FEATURE_DIMS}"
export SPATIAL_OT_DEEP_SEGMENTATION_FEATURE_WEIGHT="${SPATIAL_OT_DEEP_SEGMENTATION_FEATURE_WEIGHT:-$DEEP_SEGMENTATION_FEATURE_WEIGHT}"
export SPATIAL_OT_DEEP_SEGMENTATION_SPATIAL_WEIGHT="${SPATIAL_OT_DEEP_SEGMENTATION_SPATIAL_WEIGHT:-$DEEP_SEGMENTATION_SPATIAL_WEIGHT}"
export SPATIAL_OT_DEEP_SEGMENTATION_REFINEMENT_ITERS="${SPATIAL_OT_DEEP_SEGMENTATION_REFINEMENT_ITERS:-$DEEP_SEGMENTATION_REFINEMENT_ITERS}"
export SPATIAL_OT_LIGHT_CELL_H5AD="${SPATIAL_OT_LIGHT_CELL_H5AD:-$LIGHT_CELL_H5AD}"
export SPATIAL_OT_H5AD_COMPRESSION="${SPATIAL_OT_H5AD_COMPRESSION:-$H5AD_COMPRESSION}"
export SPATIAL_OT_WRITE_SAMPLE_SPATIAL_MAPS="${SPATIAL_OT_WRITE_SAMPLE_SPATIAL_MAPS:-$WRITE_SAMPLE_SPATIAL_MAPS}"
export SPATIAL_OT_PROGRESS="${SPATIAL_OT_PROGRESS:-$PROGRESS_LOG}"

if [[ "$POOL_ALL_INPUTS" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-../outputs/spatial_ot/cohort_multilevel_ot}"
  INPUT_H5AD="${INPUT_H5AD:-${INPUT_DIR}/${POOLED_INPUT_NAME}}"
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
SAMPLE_SPOT_LATENT_PLOT_DIR="${SAMPLE_SPOT_LATENT_PLOT_DIR:-${OUTPUT_DIR}/sample_spot_latent_plots}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtual environment python not found: $PYTHON_BIN" >&2
  echo "Run 'bash install_env.sh' from the spatial_ot directory first." >&2
  exit 1
fi

if [[ "$REQUIRE_FULL_CELL_COVERAGE" == "1" ]]; then
  "$PYTHON_BIN" - "$RADIUS_UM" "$STRIDE_UM" <<'PY'
import sys

radius = float(sys.argv[1])
stride = float(sys.argv[2])
if stride > radius + 1e-9:
    raise SystemExit(
        f"REQUIRE_FULL_CELL_COVERAGE=1 requires STRIDE_UM <= RADIUS_UM, "
        f"but got STRIDE_UM={stride:g} and RADIUS_UM={radius:g}."
    )
PY
fi

if [[ "$POOL_ALL_INPUTS" == "1" ]]; then
  if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Missing input directory: $INPUT_DIR" >&2
    exit 1
  fi
  if [[ "$PREPARE_INPUTS_AHEAD" == "1" && "$FEATURE_OBSM_KEY" == "$PREPARED_FEATURE_OBSM_KEY" ]]; then
    export OUTPUT_H5AD="$INPUT_H5AD"
    export FEATURE_OBSM_KEY
    export PREPARED_FEATURE_OBSM_KEY
    export REFRESH_POOLED_INPUT
    export REFRESH_PREPARED_FEATURES
    bash "$SCRIPT_DIR/prepare_spatial_ot_input.sh"
  elif [[ "$REFRESH_POOLED_INPUT" == "1" || ! -f "$INPUT_H5AD" ]]; then
    mkdir -p "$(dirname -- "$INPUT_H5AD")"
    "$PYTHON_BIN" -m spatial_ot pool-inputs \
      --input-dir "$INPUT_DIR" \
      --output-h5ad "$INPUT_H5AD" \
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

if [[ "$POOL_ALL_INPUTS" != "1" && "$PREPARE_INPUTS_AHEAD" == "1" && "$FEATURE_OBSM_KEY" == "$PREPARED_FEATURE_OBSM_KEY" ]]; then
  PREPARE_ARGS=()
  if [[ "$REFRESH_PREPARED_FEATURES" == "1" ]]; then
    PREPARE_ARGS+=(--overwrite)
  fi
  "$PYTHON_BIN" -m spatial_ot prepare-inputs \
    --input-h5ad "$INPUT_H5AD" \
    --feature-obsm-key X \
    --output-obsm-key "$PREPARED_FEATURE_OBSM_KEY" \
    "${PREPARE_ARGS[@]}"
fi

EXTRA_FLAGS=()
if [[ "$ALLOW_UMAP_AS_FEATURE" == "1" ]]; then
  EXTRA_FLAGS+=(--allow-umap-as-feature)
fi
if [[ "$ALLOW_OBSERVED_HULL_GEOMETRY" == "1" ]]; then
  EXTRA_FLAGS+=(--allow-observed-hull-geometry)
else
  EXTRA_FLAGS+=(--no-allow-observed-hull-geometry)
fi
if [[ "$ALLOW_REFLECTION" == "1" ]]; then
  EXTRA_FLAGS+=(--allow-reflection)
else
  EXTRA_FLAGS+=(--no-allow-reflection)
fi
if [[ "$ALLOW_SCALE" == "1" ]]; then
  EXTRA_FLAGS+=(--allow-scale)
else
  EXTRA_FLAGS+=(--no-allow-scale)
fi
if [[ "$SHAPE_DIAGNOSTICS" == "1" ]]; then
  EXTRA_FLAGS+=(--shape-diagnostics)
else
  EXTRA_FLAGS+=(--no-shape-diagnostics)
fi
EXTRA_FLAGS+=(--shape-leakage-permutations "$SHAPE_LEAKAGE_PERMUTATIONS")
if [[ "$COMPUTE_SPOT_LATENT" == "1" ]]; then
  EXTRA_FLAGS+=(--compute-spot-latent)
else
  EXTRA_FLAGS+=(--no-compute-spot-latent)
fi
if [[ "$AUTO_N_CLUSTERS" == "1" ]]; then
  EXTRA_FLAGS+=(
    --auto-n-clusters
    --candidate-n-clusters "$CANDIDATE_N_CLUSTERS"
    --auto-k-max-score-subregions "$AUTO_K_MAX_SCORE_SUBREGIONS"
    --auto-k-gap-references "$AUTO_K_GAP_REFERENCES"
    --auto-k-mds-components "$AUTO_K_MDS_COMPONENTS"
    --auto-k-pilot-n-init "$AUTO_K_PILOT_N_INIT"
    --auto-k-pilot-max-iter "$AUTO_K_PILOT_MAX_ITER"
  )
else
  EXTRA_FLAGS+=(--no-auto-n-clusters)
fi
EXTRA_FLAGS+=(--min-subregions-per-cluster "$MIN_SUBREGIONS_PER_CLUSTER")
EXTRA_FLAGS+=(
  --subregion-construction-method "$SUBREGION_CONSTRUCTION_METHOD"
  --subregion-feature-weight "$SUBREGION_FEATURE_WEIGHT"
  --subregion-feature-dims "$SUBREGION_FEATURE_DIMS"
  --deep-segmentation-knn "$DEEP_SEGMENTATION_KNN"
  --deep-segmentation-feature-dims "$DEEP_SEGMENTATION_FEATURE_DIMS"
  --deep-segmentation-feature-weight "$DEEP_SEGMENTATION_FEATURE_WEIGHT"
  --deep-segmentation-spatial-weight "$DEEP_SEGMENTATION_SPATIAL_WEIGHT"
)
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
  if [[ -n "$DEEP_RADIUS_UM" ]]; then
    DEEP_FLAGS+=(--deep-radius-um "$DEEP_RADIUS_UM")
  fi
  if [[ -n "$DEEP_SHORT_RADIUS_UM" ]]; then
    DEEP_FLAGS+=(--deep-short-radius-um "$DEEP_SHORT_RADIUS_UM")
  fi
  if [[ -n "$DEEP_MID_RADIUS_UM" ]]; then
    DEEP_FLAGS+=(--deep-mid-radius-um "$DEEP_MID_RADIUS_UM")
  fi
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
  if [[ -n "$DEEP_PRETRAINED_MODEL" ]]; then
    DEEP_FLAGS+=(--pretrained-deep-model "$DEEP_PRETRAINED_MODEL")
  fi
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
  --min-scale "$MIN_SCALE" \
  --max-scale "$MAX_SCALE" \
  --scale-penalty "$SCALE_PENALTY" \
  --shift-penalty "$SHIFT_PENALTY" \
  --n-init "$N_INIT" \
  --overlap-consistency-weight "$OVERLAP_CONSISTENCY_WEIGHT" \
  --overlap-jaccard-min "$OVERLAP_JACCARD_MIN" \
  --overlap-contrast-scale "$OVERLAP_CONTRAST_SCALE" \
  --max-iter "$MAX_ITER" \
  --tol "$TOL" \
  --seed 1337 \
  "${DEEP_FLAGS[@]}" \
  "${EXTRA_FLAGS[@]}"

if [[ "$REQUIRE_FULL_CELL_COVERAGE" == "1" ]]; then
  "$PYTHON_BIN" - "$OUTPUT_DIR/summary.json" <<'PY'
from pathlib import Path
import json
import sys

summary_path = Path(sys.argv[1])
if not summary_path.exists():
    raise SystemExit(f"Expected run summary was not written: {summary_path}")
summary = json.loads(summary_path.read_text())
uncovered = int(summary.get("uncovered_cell_count", -1))
coverage = float(summary.get("cell_subregion_coverage_fraction", -1.0))
if uncovered != 0 or coverage < 1.0 - 1e-8:
    raise SystemExit(
        "Run completed, but not all cells were covered by at least one analyzed subregion. "
        f"uncovered_cell_count={uncovered}, cell_subregion_coverage_fraction={coverage:.6f}. "
        "Reduce STRIDE_UM, increase RADIUS_UM, or set REQUIRE_FULL_CELL_COVERAGE=0 to override."
    )
PY
else
  "$PYTHON_BIN" - "$OUTPUT_DIR/summary.json" <<'PY'
from pathlib import Path
import json
import sys

summary_path = Path(sys.argv[1])
if not summary_path.exists():
    raise SystemExit(f"Expected run summary was not written: {summary_path}")
summary = json.loads(summary_path.read_text())
uncovered = int(summary.get("uncovered_cell_count", 0))
coverage = float(summary.get("cell_subregion_coverage_fraction", 1.0))
if uncovered > 0:
    print(
        "Run completed with incomplete analyzed-subregion coverage: "
        f"uncovered_cell_count={uncovered}, cell_subregion_coverage_fraction={coverage:.6f}. "
        "Set REQUIRE_FULL_CELL_COVERAGE=1 after tuning STRIDE_UM/RADIUS_UM/MAX_SUBREGIONS "
        "when every spot must have a niche-context latent occurrence.",
        file=sys.stderr,
    )
PY
fi

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

if [[ "$PLOT_SAMPLE_SPOT_LATENT" == "1" && "$COMPUTE_SPOT_LATENT" == "1" ]]; then
  "$PYTHON_BIN" -m spatial_ot plot-sample-spot-latent \
    --run-dir "$OUTPUT_DIR" \
    --output-dir "$SAMPLE_SPOT_LATENT_PLOT_DIR" \
    --sample-obs-key "$SAMPLE_OBS_KEY" \
    --source-file-obs-key "$SOURCE_FILE_OBS_KEY" \
    --plot-spatial-x-key "$ORIGINAL_SPATIAL_X_KEY" \
    --plot-spatial-y-key "$ORIGINAL_SPATIAL_Y_KEY" \
    --default-sample-id "$DEFAULT_PLOT_SAMPLE_ID" \
    --max-occurrences-per-cluster "$MAX_SPOT_LATENT_PLOT_OCCURRENCES"
fi
