#!/usr/bin/env bash
# Shared high-VRAM deep-expression profile for cohort-scale GPU runs.
# Source this file after dataset-specific INPUT_H5AD/OUTPUT_DIR settings.

DEEP_TARGET_VRAM_GB="${DEEP_TARGET_VRAM_GB:-70}"
CUDA_MAX_TARGET_FRACTION="${CUDA_MAX_TARGET_FRACTION:-0.92}"
CUDA_TARGET_VRAM_GB="${CUDA_TARGET_VRAM_GB:-$DEEP_TARGET_VRAM_GB}"

DEEP_FEATURE_METHOD="${DEEP_FEATURE_METHOD:-autoencoder}"
DEEP_DEVICE="${DEEP_DEVICE:-cuda}"
DEEP_LATENT_DIM="${DEEP_LATENT_DIM:-256}"
DEEP_HIDDEN_DIM="${DEEP_HIDDEN_DIM:-8192}"
DEEP_LAYERS="${DEEP_LAYERS:-6}"
DEEP_EPOCHS="${DEEP_EPOCHS:-20}"
DEEP_BATCH_SIZE="${DEEP_BATCH_SIZE:-131072}"
DEEP_VALIDATION="${DEEP_VALIDATION:-none}"
DEEP_RECONSTRUCTION_WEIGHT="${DEEP_RECONSTRUCTION_WEIGHT:-1.0}"
DEEP_VARIANCE_WEIGHT="${DEEP_VARIANCE_WEIGHT:-0.1}"
DEEP_DECORRELATION_WEIGHT="${DEEP_DECORRELATION_WEIGHT:-0.01}"
DEEP_SAVE_MODEL="${DEEP_SAVE_MODEL:-1}"

export DEEP_TARGET_VRAM_GB
export CUDA_MAX_TARGET_FRACTION
export CUDA_TARGET_VRAM_GB
export DEEP_FEATURE_METHOD
export DEEP_DEVICE
export DEEP_LATENT_DIM
export DEEP_HIDDEN_DIM
export DEEP_LAYERS
export DEEP_EPOCHS
export DEEP_BATCH_SIZE
export DEEP_VALIDATION
export DEEP_RECONSTRUCTION_WEIGHT
export DEEP_VARIANCE_WEIGHT
export DEEP_DECORRELATION_WEIGHT
export DEEP_SAVE_MODEL
export SPATIAL_OT_CUDA_TARGET_VRAM_GB="${SPATIAL_OT_CUDA_TARGET_VRAM_GB:-$CUDA_TARGET_VRAM_GB}"
export SPATIAL_OT_CUDA_MAX_TARGET_FRACTION="${SPATIAL_OT_CUDA_MAX_TARGET_FRACTION:-$CUDA_MAX_TARGET_FRACTION}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [[ "${CHECK_HIGH_VRAM_GPU:-1}" == "1" && "$DEEP_DEVICE" == cuda* ]]; then
  "${PYTHON_BIN:-../.venv/bin/python}" - "$DEEP_TARGET_VRAM_GB" "$CUDA_MAX_TARGET_FRACTION" <<'PY'
import sys

target_gb = float(sys.argv[1])
max_fraction = float(sys.argv[2])
try:
    import torch
except Exception as exc:
    raise SystemExit(f"Cannot check CUDA capacity because torch import failed: {exc}")

if not torch.cuda.is_available():
    raise SystemExit(
        "The high-VRAM deep profile requested CUDA, but torch.cuda.is_available() is False. "
        "Set CHECK_HIGH_VRAM_GPU=0 only if this is a dry run."
    )

usable_gb = max(
    torch.cuda.get_device_properties(i).total_memory / (1024.0**3) * max_fraction
    for i in range(torch.cuda.device_count())
)
if usable_gb + 0.5 < target_gb:
    raise SystemExit(
        f"The high-VRAM profile targets {target_gb:.1f} GiB, but the largest visible GPU "
        f"has about {usable_gb:.1f} GiB usable after CUDA_MAX_TARGET_FRACTION={max_fraction:g}. "
        "Run on a larger GPU, lower DEEP_TARGET_VRAM_GB/DEEP_BATCH_SIZE/DEEP_HIDDEN_DIM, "
        "or set CHECK_HIGH_VRAM_GPU=0 for a dry-run configuration check."
    )
PY
fi
