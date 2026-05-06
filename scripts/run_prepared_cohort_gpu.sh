#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

default_cpu_threads() {
  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN 2>/dev/null && return
  fi
  if command -v nproc >/dev/null 2>&1; then
    nproc 2>/dev/null && return
  fi
  echo 1
}

VENV_DIR="${VENV_DIR:-../.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
INPUT_DIR="${INPUT_DIR:-../spatial_ot_input}"
POOLED_INPUT_NAME="${POOLED_INPUT_NAME:-visium_hd_spatial_ot_input_pooled.h5ad}"
X_FEATURE_COMPONENTS="${X_FEATURE_COMPONENTS:-512}"
PREPARED_FEATURE_OBSM_KEY="${PREPARED_FEATURE_OBSM_KEY:-X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}}"
INPUT_H5AD="${INPUT_H5AD:-${INPUT_DIR}/${POOLED_INPUT_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-../outputs/spatial_ot/cohort_multilevel_ot_prepared_gpu}"
CPU_THREADS="${CPU_THREADS:-$(default_cpu_threads)}"
TORCH_INTRAOP_THREADS="${TORCH_INTRAOP_THREADS:-$CPU_THREADS}"
TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-4}"

export CPU_THREADS
export TORCH_INTRAOP_THREADS
export TORCH_INTEROP_THREADS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$CPU_THREADS}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-$CPU_THREADS}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-$CPU_THREADS}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"
export SPATIAL_OT_TORCH_NUM_THREADS="${SPATIAL_OT_TORCH_NUM_THREADS:-$TORCH_INTRAOP_THREADS}"
export SPATIAL_OT_TORCH_NUM_INTEROP_THREADS="${SPATIAL_OT_TORCH_NUM_INTEROP_THREADS:-$TORCH_INTEROP_THREADS}"
export SPATIAL_OT_CPU_THREADS="${SPATIAL_OT_CPU_THREADS:-$CPU_THREADS}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtual environment python not found: $PYTHON_BIN" >&2
  echo "Run 'bash scripts/install_env.sh' from the spatial_ot directory first." >&2
  exit 1
fi

"$PYTHON_BIN" - "$INPUT_H5AD" "$PREPARED_FEATURE_OBSM_KEY" <<'PY'
from pathlib import Path
import sys

import anndata as ad

path = Path(sys.argv[1])
prepared_key = sys.argv[2]
if not path.exists():
    raise SystemExit(f"Missing pooled cohort input: {path}")
adata = ad.read_h5ad(path, backed="r")
try:
    missing = []
    if prepared_key not in adata.obsm:
        missing.append(f"obsm['{prepared_key}']")
    for obs_key in ("sample_id", "source_h5ad", "pooled_cell_x", "pooled_cell_y"):
        if obs_key not in adata.obs:
            missing.append(f"obs['{obs_key}']")
    if missing:
        raise SystemExit(
            "Prepared cohort input is incomplete. Missing "
            + ", ".join(missing)
            + ". Run 'bash scripts/prepare_all_spatial_ot_input.sh' first."
        )
finally:
    adata.file.close()
PY

export VENV_DIR
export PYTHON_BIN
export INPUT_DIR
export POOLED_INPUT_NAME
export INPUT_H5AD
export OUTPUT_DIR
export X_FEATURE_COMPONENTS
export PREPARED_FEATURE_OBSM_KEY
export FEATURE_OBSM_KEY="${FEATURE_OBSM_KEY:-$PREPARED_FEATURE_OBSM_KEY}"
export POOL_ALL_INPUTS="${POOL_ALL_INPUTS:-1}"
export PREPARE_INPUTS_AHEAD="${PREPARE_INPUTS_AHEAD:-0}"
export REFRESH_POOLED_INPUT="${REFRESH_POOLED_INPUT:-0}"
export REFRESH_PREPARED_FEATURES="${REFRESH_PREPARED_FEATURES:-0}"
export COMPUTE_DEVICE="${COMPUTE_DEVICE:-cuda}"
export AUTO_N_CLUSTERS="${AUTO_N_CLUSTERS:-1}"
export CANDIDATE_N_CLUSTERS="${CANDIDATE_N_CLUSTERS:-15-25}"
export MIN_SUBREGIONS_PER_CLUSTER="${MIN_SUBREGIONS_PER_CLUSTER:-50}"

exec bash "$SCRIPT_DIR/run.sh"
